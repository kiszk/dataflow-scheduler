//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// AddressAssignment: Analyze memory allocations and compute sizes for
// address assignment.
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Analysis/Presburger/IntegerRelation.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>  // IWYU pragma: keep
#include <mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/AlignmentAttrInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ValueBoundsOpInterface.h>
#include <mlir/Pass/Pass.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>

#include "dataflow-scheduler/Analysis/Mapping.h"
#include "dataflow-scheduler/Analysis/MemoryTrackerAnalysis.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "dataflow-scheduler/Transforms/AllocationPlan.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "ktir/Dialect/KTDP/KTDP.h"

#define PASS_NAME "address-assignment"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Address Assignment pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_ADDRESSASSIGNMENTPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Tries to determine the upper bound of @p shape given @p dynamic_sizes .
[[nodiscard]] auto getUpperBound(llvm::ArrayRef<int64_t> shape,
                                 mlir::ValueRange dynamic_sizes)
    -> llvm::FailureOr<llvm::SmallVector<int64_t>> {
  llvm::SmallVector<int64_t> result;

  unsigned dynamic_index = 0U;
  for (auto [idx, dim] : llvm::enumerate(shape)) {
    if (mlir::ShapedType::isStatic(dim)) {
      result.push_back(dim);
      continue;
    }

    const auto size = dynamic_sizes[dynamic_index++];
    LDBG(1) << "  Dynamic dim " << idx << ": SSA value " << size;

    const auto maybe_bound =
        mlir::ValueBoundsConstraintSet::computeConstantBound(
            mlir::presburger::BoundType::UB, size, nullptr, true);
    if (llvm::failed(maybe_bound)) {
      LDBG(1) << "    Failed to compute constant upper bound";
      LDBG(1)
          << "    Note: ValueBoundsConstraintSet could not resolve the bound "
             "to a constant value.";
      LDBG(1) << "    This may require more sophisticated analysis or manual "
                 "annotation.";
      return llvm::failure();
    }

    LDBG(1) << "    Computed upper bound: " << *maybe_bound;
    result.push_back(*maybe_bound);
  }

  return std::move(result);
}

/// Named constraint for an allocation that is processed by this pass.
struct AllocOp
    : mlir::Op<AllocOp, mlir::OpTrait::ZeroRegions, mlir::OpTrait::OneResult,
               mlir::OpTrait::OneTypedResult<mlir::MemRefType>::Impl,
               mlir::MemoryEffectOpInterface::Trait> {
  [[nodiscard]] static auto classof(mlir::Operation* op) -> bool {
    return llvm::isa<mlir::memref::AllocOp, mlir::memref::AllocaOp>(op);
  }

  using Op::Op;

  /*implicit*/ AllocOp(mlir::memref::AllocOp alloc) : Op(alloc) {}
  /*implicit*/ AllocOp(mlir::memref::AllocaOp alloc) : Op(alloc) {}

  void getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                      mlir::MemoryEffects::Effect>>& effects) {
    llvm::cast<mlir::MemoryEffectOpInterface>(getOperation())
        .getEffects(effects);
  }

  /// Tries to get the dynamic sizes associated with this allocation, if any.
  [[nodiscard]] auto getDynamicSizes() -> llvm::FailureOr<mlir::OperandRange> {
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(getOperation())) {
      return alloc.getDynamicSizes();
    }
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocaOp>(getOperation())) {
      return alloc.getDynamicSizes();
    }
    return llvm::failure();
  }

  /// Tries to get the static upper bound for the allocated memref shape.
  [[nodiscard]] auto getUpperShapeBound()
      -> llvm::FailureOr<llvm::SmallVector<int64_t>> {
    if (getType().hasStaticShape()) {
      return llvm::to_vector(getType().getShape());
    }

    const auto maybe_sizes = getDynamicSizes();
    if (llvm::failed(maybe_sizes)) {
      return llvm::failure();
    }
    return getUpperBound(getType().getShape(), *maybe_sizes);
  }

  /// Tries to get the static upper bound for the largest element offset.
  [[nodiscard]] auto getUpperOffsetBound() -> llvm::FailureOr<size_t> {
    auto maybe_shape = getUpperShapeBound();
    if (llvm::failed(maybe_shape)) {
      return llvm::failure();
    }

    llvm::SmallVector<int64_t> strides;
    int64_t offset;
    if (llvm::failed(getType().getLayout().getStridesAndOffset(
            *maybe_shape, strides, offset)) ||
        offset < 0 ||
        llvm::any_of(strides, [](int64_t stride) { return stride < 0; })) {
      LDBG() << "unsupported layout: " << getType().getLayout();
      return llvm::failure();
    }

    auto result = static_cast<size_t>(offset);
    for (auto [sz, stride] : llvm::zip_equal(*maybe_shape, strides)) {
      size_t temp;
      if (__builtin_mul_overflow(sz - 1, stride, &temp) ||
          __builtin_add_overflow(result, temp, &result)) {
        LDBG() << "allocation too large";
        return llvm::failure();
      }
    }
    return result;
  }

  /// Tries to get the static lower bound for the allocation size.
  [[nodiscard]] auto getSize() -> llvm::FailureOr<size_t> {
    LDBG(1) << "Analyzing allocation: " << *this;
    LDBG(1) << "  Type: " << getResult().getType();

    const auto shape = getUpperShapeBound();
    if (mlir::failed(shape) ||
        llvm::any_of(*shape, [](int64_t size) { return size < 0; }))
      return llvm::failure();
    if (llvm::is_contained(*shape, int64_t{0})) return size_t{0};
    const auto maybe_offset = getUpperOffsetBound();
    if (llvm::failed(maybe_offset)) {
      return llvm::failure();
    }
    const auto maybe_size =
        tryGetSizeInBytes(getResult().getType().getElementType());
    if (!maybe_size) {
      LDBG() << "unsupported element type";
      return llvm::failure();
    }

    size_t elements, total_bytes;
    if (__builtin_add_overflow(*maybe_offset, size_t{1}, &elements) ||
        __builtin_mul_overflow(elements, *maybe_size, &total_bytes)) {
      LDBG() << "allocation too large";
      return llvm::failure();
    }

    LDBG(1) << "  Total size: " << total_bytes << " bytes (" << *maybe_offset
            << " elements * " << *maybe_size << " bytes/element)";
    return total_bytes;
  }

  /// Gets the alignment requirement, if any.
  [[nodiscard]] auto getAlignment() -> llvm::MaybeAlign {
    if (auto iface =
            llvm::dyn_cast<mlir::AlignmentAttrOpInterface>(getOperation());
        iface) {
      return iface.getMaybeAlign();
    }
    return {};
  }
};

/// Gets the allocations of \p scope that name a memory space, stopping at a
/// nested module so each program is collected on its own.
[[nodiscard]] auto collectAllocations(mlir::ModuleOp scope)
    -> llvm::SmallVector<AllocOp> {
  llvm::SmallVector<AllocOp> allocs;

  scope.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (op != scope.getOperation() && llvm::isa<mlir::ModuleOp>(op)) {
      return mlir::WalkResult::skip();
    }
    if (auto alloc = llvm::dyn_cast<AllocOp>(op)) {
      if (alloc.getType().getMemorySpace()) {
        allocs.push_back(alloc);
      } else {
        LDBG(1) << "Skipping allocation with missing memory space: " << alloc;
      }
    }
    return mlir::WalkResult::advance();
  });

  return allocs;
}

/// Replace memref.alloc with unrealized_conversion_cast using the assigned
/// address, and remove any corresponding memref.dealloc operations.
/// @p assigned_address counts elements of the allocation's element type, as a
/// memory view's offset does.
void materializeAddressAssignment(AllocOp alloc, size_t assigned_address,
                                  mlir::OpBuilder& builder) {
  builder.setInsertionPoint(alloc);

  auto alloc_type = alloc.getType();

  // Create constant index for the assigned address
  mlir::Value addr_value = mlir::arith::ConstantIndexOp::create(
      builder, alloc.getLoc(), assigned_address);

  // Create unrealized_conversion_cast from index to memref
  auto cast = mlir::UnrealizedConversionCastOp::create(builder, alloc.getLoc(),
                                                       alloc_type, addr_value);

  LDBG(1) << "  Materialized address assignment:";
  LDBG(1) << "    Original alloc: " << alloc;
  LDBG(1) << "    Replaced with: " << cast;

  // Collect all dealloc operations that use this allocation
  llvm::SmallVector<mlir::memref::DeallocOp> deallocs_to_remove;
  for (mlir::Operation* user : alloc->getUsers()) {
    if (auto dealloc = mlir::dyn_cast<mlir::memref::DeallocOp>(user)) {
      deallocs_to_remove.push_back(dealloc);
    }
  }

  // Replace all uses of the alloc with the cast
  alloc.replaceAllUsesWith(cast.getResult(0));

  // Remove the dealloc operations
  for (auto dealloc : deallocs_to_remove) {
    LDBG(1) << "    Removing dealloc: " << dealloc;
    dealloc.erase();
  }

  // Erase the original alloc
  alloc.erase();
}

struct AddressAssignmentPass
    : public impl::AddressAssignmentPassBase<AddressAssignmentPass> {
  AddressAssignmentPass() = default;
  explicit AddressAssignmentPass(const SchedulerExtContext&) {}

  void runOnOperation() override {
    if (DisableThisPass || !hasAllocationsWithMemorySpace(getOperation()))
      return;
    auto& analysis = getAnalysis<MemoryTrackerAnalysis>();
    auto plan = planAllocations(getOperation(), analysis);
    if (mlir::failed(plan.diagnose())) {
      ++num_failed;
      signalPassFailure();
      return;
    }
    num_processed += plan.placements.size();
    applyAllocationPlan(plan, analysis);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createAddressAssignmentPass() {
  return std::make_unique<AddressAssignmentPass>();
}

std::unique_ptr<mlir::Pass> scheduler::createAddressAssignmentPass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<AddressAssignmentPass>(scheduler_ctx);
}

bool scheduler::hasAllocationsWithMemorySpace(mlir::ModuleOp module) {
  return module
      .walk([](AllocOp alloc) {
        return alloc.getType().getMemorySpace() ? mlir::WalkResult::interrupt()
                                                : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

mlir::LogicalResult AllocationPlan::diagnose() const {
  if (*this) return mlir::success();
  failed_operation->emitError(error);
  return mlir::failure();
}

AllocationPlan scheduler::planAllocations(
    mlir::ModuleOp module, const MemoryTrackerAnalysis& analysis) {
  AllocationPlan plan(analysis.getTracker());
  const auto& tree = analysis.getMemoryTree();
  auto fail = [&](mlir::Operation* op, llvm::Twine message,
                  bool capacity = false) {
    plan.failed_operation = op;
    plan.error = message.str();
    plan.capacity_exceeded = capacity;
    return mlir::WalkResult::interrupt();
  };
  llvm::DenseMap<ResourceType, size_t> fixed_ends;
  auto reserve_view = [&](auto view) -> mlir::WalkResult {
    auto space = mlir::dyn_cast<ResourceType>(view.getMemorySpace());
    if (!space || !tree.isPerCoreScratchPadMemory(space))
      return mlir::WalkResult::advance();
    llvm::APInt address;
    if (!mlir::matchPattern(view.getOffset(), mlir::m_ConstantInt(&address)))
      return mlir::WalkResult::advance();  // Runtime address: caller-managed.
    int64_t offset = address.getSExtValue();
    auto bytes = tryGetSizeInBytes(view.getType().getElementType());
    size_t end = offset >= 0 ? static_cast<size_t>(offset) : 0;
    bool invalid = offset < 0 || !bytes;
    for (auto [size, stride] :
         llvm::zip_equal(view.getStaticSizes(), view.getStaticStrides())) {
      size_t span;
      if (size <= 0 || stride < 0 ||
          __builtin_mul_overflow(size - 1, stride, &span) ||
          __builtin_add_overflow(end, span, &end))
        invalid = true;
    }
    if (invalid || __builtin_add_overflow(end, size_t{1}, &end) ||
        __builtin_mul_overflow(end, *bytes, &end))
      return fail(view,
                  "fixed scratchpad view requires a bounded nonnegative "
                  "address, shape and strides");
    auto node = tree.getNode(*tree.getNodeIdForMemory(space));
    if (!node->capacity_in_bytes || end > *node->capacity_in_bytes)
      return fail(view, "fixed scratchpad view exceeds memory capacity");
    auto& reserved_end = fixed_ends[node->memory_resource];
    reserved_end = std::max(reserved_end, end);
    return mlir::WalkResult::advance();
  };
  if (module
          .walk([&](mlir::ktdp::ConstructMemoryViewOp view) {
            return reserve_view(view);
          })
          .wasInterrupted() ||
      module
          .walk([&](mlir::ktdp_lowering::ConstructMemoryViewOp view) {
            return reserve_view(view);
          })
          .wasInterrupted())
    return plan;

  llvm::SmallVector<mlir::ModuleOp> scopes{module};
  llvm::append_range(scopes, module.getOps<mlir::ModuleOp>());
  for (auto scope : scopes) {
    auto allocs = collectAllocations(scope);
    llvm::SmallDenseSet<ResourceType> reusable;
    for (auto alloc : allocs) {
      auto space = llvm::cast<ResourceType>(alloc.getType().getMemorySpace());
      auto id = tree.getNodeIdForMemory(space);
      auto node = id ? tree.getNode(*id)
                     : std::optional<arch_view::MemoryTree::MemoryNode>{};
      if (!node || !node->capacity_in_bytes) {
        fail(alloc, "allocation requires a memory resource with capacity");
        return plan;
      }
      if (!tree.isGlobalMemory(space)) reusable.insert(node->memory_resource);
    }
    plan.tracker.reset(llvm::to_vector(reusable));
    for (auto space : reusable) {
      size_t end = fixed_ends.lookup(space);
      size_t next = plan.tracker.getNextAvailableAddress(space);
      if (end <= next) continue;
      auto reserved = plan.tracker.allocate(space, end - next);
      if (!reserved) {
        fail(scope, llvm::toString(reserved.takeError()));
        return plan;
      }
    }
    for (auto alloc : allocs) {
      auto size = alloc.getSize();
      if (mlir::failed(size)) {
        fail(alloc,
             "AddressAssignment: allocation has unknown or overflowing size");
        return plan;
      }
      auto space = llvm::cast<ResourceType>(alloc.getType().getMemorySpace());
      auto resource =
          tree.getNode(*tree.getNodeIdForMemory(space))->memory_resource;
      size_t element_bytes =
          *tryGetSizeInBytes(alloc.getType().getElementType());
      size_t requested = alloc.getAlignment().valueOrOne().value();
      size_t alignment;
      if (__builtin_mul_overflow(requested / std::gcd(requested, element_bytes),
                                 element_bytes, &alignment)) {
        fail(alloc, "allocation alignment overflows");
        return plan;
      }
      auto address = plan.tracker.allocate(resource, *size, alignment);
      if (!address) {
        std::string message;
        llvm::raw_string_ostream os(message);
        os << "Failed to allocate " << space
           << " memory: " << llvm::toString(address.takeError());
        fail(alloc, message, true);
        return plan;
      }
      plan.placements.push_back({alloc, *address / element_bytes});
    }
  }
  return plan;
}

void scheduler::applyAllocationPlan(const AllocationPlan& plan,
                                    MemoryTrackerAnalysis& analysis) {
  assert(plan && "cannot apply an unsuccessful allocation plan");
  for (auto placement : plan.placements) {
    mlir::OpBuilder builder(placement.allocation);
    materializeAddressAssignment(mlir::cast<AllocOp>(placement.allocation),
                                 placement.element_address, builder);
  }
  analysis.getTracker() = plan.tracker;
}

// Made with Bob
