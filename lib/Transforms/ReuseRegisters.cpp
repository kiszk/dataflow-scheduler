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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/CommandLine.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/Passes.h"

namespace scheduler {
#define GEN_PASS_DEF_REUSEREGISTERSPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"

using namespace mlir;

namespace {
llvm::cl::opt<bool> disable_reuse("disable-reuse-registers",
                                  llvm::cl::desc("Disable register reuse"),
                                  llvm::cl::init(false));

// The common block must execute its children sequentially. In particular,
// sibling stages of a pipeline may overlap, including across modulo phases.
bool before(Operation* a, Operation* b) {
  for (Block* block = a->getBlock(); block;
       block = block->getParentOp()->getBlock()) {
    auto* lhs = block->findAncestorOpInBlock(*a);
    auto* rhs = block->findAncestorOpInBlock(*b);
    if (!rhs) continue;
    return lhs != rhs &&
           isa<func::FuncOp, scf::ForOp, scf::IfOp, ktdf::StageOp>(
               block->getParentOp()) &&
           lhs->isBeforeInBlock(rhs);
  }
  return false;
}

// Lexical order alone does not prove initialization: a preceding loop may
// execute zero times, or a branch containing the fill may not be taken.
bool initializesBefore(Operation* fill, Operation* use) {
  for (auto* op = fill; op; op = op->getParentOp()) {
    auto* block = op->getBlock();
    if (!block) return false;
    if (auto* consumer = block->findAncestorOpInBlock(*use))
      return before(op, consumer);
    if (!isa<ktdf::StageOp, ktdf::PipelineOp>(op->getParentOp())) return false;
  }
  return false;
}

Attribute unit(Operation* op) {
  if (auto mapping = op->getAttr("ktdf_arch.maps_to")) return mapping;
  auto stage = op->getParentOfType<ktdf::StageOp>();
  auto units =
      stage ? stage->getAttrOfType<ArrayAttr>("applicable_units") : ArrayAttr{};
  return units && units.size() == 1 ? units[0] : Attribute{};
}

bool isRegisterUse(OpOperand& use) {
  auto opaque = dyn_cast<ktdf::OpaqueOp>(use.getOwner());
  auto names = opaque ? opaque->getAttrOfType<ArrayAttr>(
                            "dataflow_scheduler.register_names")
                      : ArrayAttr{};
  return names && names.size() == opaque->getNumOperands();
}

// Only cross a receive fed directly from memory. Other protocols can couple
// receiving the next packet to sending this one, and need their own proof.
bool readsMemory(ktdf::ReadFromFifoOp read) {
  ktdf::DataTransferOp writer;
  for (auto* user : read.getFifoSlot().getUsers()) {
    if (user == read) continue;
    auto transfer = dyn_cast<ktdf::DataTransferOp>(user);
    if (!transfer || writer ||
        transfer.getDestination() != read.getFifoSlot() ||
        !isa<MemRefType>(transfer.getSource().getType()))
      return false;
    writer = transfer;
  }
  return bool(writer);
}

bool localCompute(linalg::GenericOp generic) {
  if (!generic.hasPureTensorSemantics()) return false;
  return !generic.getBody()
              ->walk([](Operation* op) {
                if (isMemoryEffectFree(op)) return WalkResult::advance();
                if (isa<ktdf::OpaqueOp>(op)) {
                  if (llvm::all_of(op->getOpOperands(), isRegisterUse))
                    return WalkResult::advance();
                } else if (isa<memref::LoadOp, memref::StoreOp>(op)) {
                  Value buffer = isa<memref::LoadOp>(op)
                                     ? cast<memref::LoadOp>(op).getMemRef()
                                     : cast<memref::StoreOp>(op).getMemRef();
                  if (isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(
                          buffer.getDefiningOp()) &&
                      llvm::any_of(buffer.getUses(), isRegisterUse))
                    return WalkResult::advance();
                }
                return WalkResult::interrupt();
              })
              .wasInterrupted();
}

void sendCompletedPackets(func::FuncOp func) {
  DominanceInfo dominance(func);
  SmallVector<ktdf::WriteToFifoOp> writes;
  func.walk([&](ktdf::WriteToFifoOp op) { writes.push_back(op); });
  for (auto write : writes) {
    auto generic = write.getData().getDefiningOp<linalg::GenericOp>();
    if (!generic || generic->getBlock() != write->getBlock() ||
        !isa<ktdf::StageOp>(generic->getParentOp()) ||
        generic.getNumResults() != 1 || !write.getData().hasOneUse() ||
        !dominance.properlyDominates(write.getFifoSlot(), generic))
      continue;
    bool safe = true;
    for (auto* op = generic->getNextNode(); op != write;
         op = op->getNextNode()) {
      if (auto read = dyn_cast<ktdf::ReadFromFifoOp>(op)) {
        safe = readsMemory(read);
      } else if (auto next = dyn_cast<linalg::GenericOp>(op)) {
        safe = unit(next) && unit(next) == unit(generic) && localCompute(next);
      } else {
        safe = isa<memref::AllocOp, memref::AllocaOp>(op) ||
               isMemoryEffectFree(op);
      }
      if (!safe) break;
    }
    if (safe) write->moveAfter(generic);
  }
}

struct Interval {
  Operation* first;
  Operation* last;
};

// A template's loaded result may still name its output register until the
// FIFO send. Keep the whole register set alive through every result's send.
std::optional<Interval> interval(linalg::GenericOp generic) {
  Operation* last = generic;
  for (auto result : generic.getResults()) {
    for (auto* user : result.getUsers()) {
      if (!isa<ktdf::WriteToFifoOp>(user) ||
          user->getBlock() != generic->getBlock())
        return std::nullopt;
      if (last->isBeforeInBlock(user)) last = user;
    }
  }
  return Interval{generic, last};
}

struct Register {
  Operation* alloc;
  Attribute compute;
  linalg::FillOp fill;
  Attribute constant;
  SmallVector<Interval> intervals;
};

std::optional<Register> describe(Operation* alloc) {
  auto value = alloc->getResult(0);
  auto type = cast<MemRefType>(value.getType());
  if (alloc->getNumOperands() || !type.hasStaticShape() ||
      type.getRank() != 1 || !type.getMemorySpace())
    return std::nullopt;
  for (auto* parent = alloc->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<scf::ParallelOp, scf::ForallOp>(parent)) return std::nullopt;
  }

  Register reg{alloc, {}, {}, {}, {}};
  linalg::GenericOp owner;
  bool has_template = false, read_only = true;
  SmallVector<Operation*> readers;
  for (auto& use : value.getUses()) {
    auto* user = use.getOwner();
    if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (reg.fill || fill.getOutputs().front() != value ||
          !matchPattern(fill.getInputs().front(), m_Constant(&reg.constant)))
        return std::nullopt;
      reg.fill = fill;
      continue;
    }
    auto generic = user->getParentOfType<linalg::GenericOp>();
    if (!generic || !unit(generic)) return std::nullopt;
    if (reg.compute && reg.compute != unit(generic)) return std::nullopt;
    reg.compute = unit(generic);
    if (isRegisterUse(use)) {
      has_template = true;
      read_only &= use.getOperandNumber() <
                   cast<ktdf::OpaqueOp>(user).getInputs().size();
    } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (store.getMemRef() != value) return std::nullopt;
      read_only = false;
    } else if (!isa<memref::LoadOp>(user)) {
      // This also excludes aliases, escapes and loop-carried buffers.
      return std::nullopt;
    }
    readers.push_back(generic);
    if (!owner) owner = generic;
  }
  if (!has_template) return std::nullopt;
  if (reg.fill) {
    if (!read_only || unit(reg.fill) != reg.compute ||
        llvm::any_of(readers, [&](Operation* reader) {
          return !initializesBefore(reg.fill, reader);
        }))
      return std::nullopt;
    return reg;
  }
  if (llvm::any_of(readers, [&](Operation* reader) { return reader != owner; }))
    return std::nullopt;
  // Storage outside the generic's block may carry a value between loop
  // iterations even when there is only one static generic using it.
  if (alloc->getBlock() != owner->getBlock()) return std::nullopt;
  auto live = interval(owner);
  if (!live) return std::nullopt;
  reg.intervals.push_back(*live);
  return reg;
}

bool compatible(const Register& a, const Register& b) {
  return a.alloc->getName() == b.alloc->getName() &&
         a.alloc->getResult(0).getType() == b.alloc->getResult(0).getType() &&
         a.alloc->getAttrs() == b.alloc->getAttrs() && a.compute == b.compute &&
         a.constant == b.constant;
}

void reuse(func::FuncOp func) {
  sendCompletedPackets(func);
  SmallVector<Operation*> allocations;
  func.walk([&](Operation* op) {
    if (isa<memref::AllocOp, memref::AllocaOp>(op)) allocations.push_back(op);
  });
  SmallVector<Register> pools;
  for (auto* alloc : allocations) {
    auto reg = describe(alloc);
    if (!reg) continue;
    auto found = llvm::find_if(pools, [&](const Register& pool) {
      if (!compatible(pool, *reg)) return false;
      if (reg->fill) return initializesBefore(pool.fill, reg->fill);
      const auto& live = reg->intervals.front();
      return llvm::all_of(pool.intervals, [&](const Interval& previous) {
        return before(previous.last, live.first) ||
               before(live.last, previous.first);
      });
    });
    if (found == pools.end()) {
      pools.push_back(std::move(*reg));
      continue;
    }
    // Static register storage has one address per program and physical unit.
    // Hoist the shared declaration so it also dominates later loop segments.
    found->alloc->moveBefore(&func.getBody().front(),
                             func.getBody().front().begin());
    if (reg->fill) reg->fill.erase();
    alloc->getResult(0).replaceAllUsesWith(found->alloc->getResult(0));
    llvm::append_range(found->intervals, reg->intervals);
    alloc->erase();
  }
}

struct ReuseRegistersPass
    : scheduler::impl::ReuseRegistersPassBase<ReuseRegistersPass> {
  void runOnOperation() override {
    if (!disable_reuse)
      getOperation()->walk([](func::FuncOp func) { reuse(func); });
  }
};
}  // namespace

}  // namespace scheduler
