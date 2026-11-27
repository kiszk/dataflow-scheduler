//===-- MaterializeRegisters.cpp -------------------------*- c++ -*-===//
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
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/RegionUtils.h>

#include "dataflow-scheduler/Analysis/ArchViews/GroupLocalMemory.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchInterfaces.h"
#include "dataflow-scheduler/Transforms/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Transforms/Utils/Utils.h"

namespace scheduler {
#define GEN_PASS_DEF_MATERIALIZEREGISTERSPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace mlir;

namespace scheduler {
namespace {

/// Recognizes one complete, contiguous row of a square word tile.
template <typename Slice>
auto wordRow(Slice slice, int64_t width) -> std::optional<int64_t> {
  auto offsets = slice.getStaticOffsets();
  if (offsets.size() != 2 || offsets[0] < 0 || offsets[0] >= width ||
      offsets[1] != 0 ||
      slice.getStaticSizes() != ArrayRef<int64_t>{1, width} ||
      slice.getStaticStrides() != ArrayRef<int64_t>{1, 1})
    return std::nullopt;
  return offsets[0];
}

/// Expands the word assembly, transpose and word extraction together. Only
/// words cross FIFO boundaries; the square tensor exists inside this stage.
auto materializeTranspose(linalg::GenericOp generic, int64_t width,
                          ktdf_arch::Mapping& mapping, RewriterBase& rw)
    -> LogicalResult {
  auto type = cast<RankedTensorType>(generic.getResult(0).getType());
  auto element = type.getElementType();
  auto word_type = RankedTensorType::get({width}, element);
  SmallVector<Value> inputs(width);
  Value assembled = generic.getDpsInputs()[0];
  SmallVector<Operation*> dead;
  while (auto insert = assembled.getDefiningOp<tensor::InsertSliceOp>()) {
    auto row = wordRow(insert, width);
    if (!row || inputs[*row] || insert.getSourceType() != word_type)
      return generic.emitError(
          "transpose input must be assembled from complete words");
    inputs[*row] = insert.getSource();
    dead.push_back(insert);
    assembled = insert.getDest();
  }
  if (!llvm::all_of(inputs, [](Value v) { return bool(v); }))
    return generic.emitError(
        "transpose input must be assembled from complete words");

  SmallVector<std::pair<tensor::ExtractSliceOp, int64_t>> outputs;
  for (auto* user : generic.getResult(0).getUsers()) {
    auto slice = dyn_cast<tensor::ExtractSliceOp>(user);
    auto row = slice ? wordRow(slice, width) : std::nullopt;
    if (!row || slice.getType() != word_type)
      return generic.emitError(
          "transpose output must be consumed as complete words");
    outputs.emplace_back(slice, *row);
  }
  auto compute = mapping.getOrMap<ktdf_arch::ExecutionUnitOp>(
      generic, mapping.byKind().getDefaultCompute().getKind());
  auto pipeline = generic->getParentOfType<ktdf::PipelineOp>();
  auto simd = compute ? compute.getFeature<ktdf_arch::feature::SIMD>()
                      : ktdf_arch::feature::SIMD{};
  if (!pipeline || !simd || simd.getLanes(element) != width * 8 ||
      simd.getSubSimdLanes(element) != width)
    return generic.emitError(
        "word transpose requires a mapped compute stage with 128-byte SIMD and "
        "16-byte sub-SIMD lanes");
  arch_view::GroupLocalMemory local(mapping.getDevice());
  auto registers = local.getLocalMemoryKind(compute.getKind());
  if (!registers)
    return generic.emitError("word transpose requires compute-local registers");

  auto loc = generic.getLoc();
  const int64_t lanes = width * 8;
  // Four rows ping-pong between two banks. Eight rows reuse a pair of
  // temporaries, so the 16-bit recipe leaves six of the sixteen registers free.
  const int64_t rows = width == 4 ? 8 : 10;
  ktdf::PrivateBuilder priv(pipeline);
  Value scratch = memref::AllocOp::create(
      priv, loc,
      MemRefType::get({rows, lanes}, element, MemRefLayoutAttrInterface{},
                      registers));
  rw.setInsertionPoint(generic);
  auto vector_type = VectorType::get({lanes}, element);
  auto word_vector = VectorType::get({width}, element);
  Value zero = arith::ConstantIndexOp::create(rw, loc, 0);
  Value zeros = arith::ConstantOp::create(
      rw, loc, DenseElementsAttr::get(vector_type, rw.getZeroAttr(element)));
  auto indices = [&](int64_t row) {
    return SmallVector<Value>{arith::ConstantIndexOp::create(rw, loc, row),
                              zero};
  };
  auto load = [&](int64_t row, VectorType vt) -> Value {
    return vector::LoadOp::create(rw, loc, vt, scratch, indices(row));
  };
  auto store = [&](Value value, int64_t row) {
    vector::StoreOp::create(rw, loc, value, scratch, indices(row));
  };
  // Consume words in the order of the scheduled FIFO reads. A receive is a
  // hardware stream operand, so reordering its first use would reorder the
  // arriving rows even though the SSA references still name the right words.
  SmallVector<int64_t> input_order;
  for (Operation& op : *generic->getBlock())
    for (auto [row, input] : llvm::enumerate(inputs))
      if (input.getDefiningOp() == &op) input_order.push_back(row);
  for (auto [row, input] : llvm::enumerate(inputs))
    if (!input.getDefiningOp() ||
        input.getDefiningOp()->getBlock() != generic->getBlock())
      input_order.push_back(row);
  for (int64_t row : input_order) {
    // MERGE instructions read whole registers. Define padding too, although
    // their group-local masks cannot bring it into the first output word.
    store(zeros, row);
    auto word = vector::TransferReadOp::create(
        rw, loc, word_vector, inputs[row], ValueRange{zero}, std::nullopt,
        rw.getMultiDimIdentityMap(1), ArrayRef<bool>{true});
    store(word, row);
  }
  for (int64_t distance = 1, bank = 0; distance < width; distance *= 2) {
    const int64_t next_bank = width == 4 ? width - bank : 0;
    for (int64_t row = 0; row < width; ++row) {
      if (row & distance) continue;
      // Each step exchanges one bit of the row and column coordinates.
      // In vector.shuffle the second input starts at 'lanes', whereas pack
      // will express the same indices within each word-sized group.
      for (int64_t half = 0; half < 2; ++half) {
        SmallVector<int64_t> mask;
        for (int64_t group = 0; group < lanes; group += width)
          for (int64_t column = 0; column < width; ++column)
            mask.push_back(group + (column & ~distance) + half * distance +
                           ((column & distance) ? lanes : 0));
        Value left = load(bank + row, vector_type);
        Value right = load(bank + row + distance, vector_type);
        Value merged = vector::ShuffleOp::create(rw, loc, left, right, mask);
        store(merged,
              width == 4 ? next_bank + row + half * distance : width + half);
      }
      if (width == 8) {
        // Both halves are ready before either input register is overwritten.
        store(load(width, vector_type), row);
        store(load(width + 1, vector_type), row + distance);
      }
    }
    bank = next_bank;
  }
  SmallVector<Value> words(width);
  for (int64_t row = 0; row < width; ++row) {
    auto empty = tensor::EmptyOp::create(rw, loc, word_type, ValueRange{});
    words[row] = vector::TransferWriteOp::create(
                     rw, loc, load(row, word_vector), empty, ValueRange{zero},
                     rw.getMultiDimIdentityMap(1), ArrayRef<bool>{true})
                     .getResult();
  }
  for (auto [slice, row] : outputs) rw.replaceOp(slice, words[row]);
  dead.push_back(generic.getDpsInits()[0].getDefiningOp());
  rw.eraseOp(generic);
  llvm::erase_if(dead, [](Operation* op) { return !op; });
  eraseDeadAncestorOps(dead);
  return success();
}

/// Gets the mapped constants the body of \p generic reads, in reading order.
///
/// Found by use rather than by where they stand: folding materializes a
/// constant in the entry block of its region, so by now most are outside the
/// body a pattern put them in. The mapping marks one as belonging in a register
/// rather than being an operand of the arithmetic.
auto getMappedConstants(linalg::GenericOp generic)
    -> SmallVector<arith::ConstantOp> {
  // Ordered, because the order they are found in is the operand order that
  // names each of them a register.
  SetVector<Operation*> seen;
  generic.getBody()->walk([&](Operation* op) {
    for (const auto operand : op->getOperands()) {
      const auto constant = operand.getDefiningOp<arith::ConstantOp>();
      if (!constant) continue;
      if (!ktdf_arch::getProperty<ktdf_arch::MapsToAttr>(constant)) continue;
      seen.insert(constant);
    }
  });

  SmallVector<arith::ConstantOp> result;
  for (auto* op : seen) result.push_back(cast<arith::ConstantOp>(op));
  return result;
}

/// Turns \p constant into a register in front of \p generic.
///
/// A template reads a whole vector out of a register, so the register is filled
/// with the value repeated across its lanes, as many as \p compute has. Returns
/// it, for the body to read instead of the constant.
auto materializeRegister(arith::ConstantOp constant, linalg::GenericOp generic,
                         ktdf_arch::ExecutionUnitOp compute,
                         RewriterBase& rewriter) -> Value {
  const auto maps_to = ktdf_arch::getProperty<ktdf_arch::MapsToAttr>(constant);
  const auto element = constant.getType();
  const auto lanes = getVectorLanes(element, compute);

  const auto register_type =
      MemRefType::get({lanes}, element, MemRefLayoutAttrInterface{}, maps_to);

  rewriter.setInsertionPoint(generic);
  auto reg =
      memref::AllocOp::create(rewriter, constant.getLoc(), register_type);
  linalg::FillOp::create(rewriter, constant.getLoc(),
                         ValueRange{constant.getResult()}, ValueRange{reg});
  return reg;
}

/// Gets the registers the body of \p generic allocates for itself, in order.
///
/// A pattern allocates the scratch a template computes in where it matched, so
/// that lands in the body too. Only allocations sized entirely by their type
/// are taken; one the body computes a size for could not move out of it.
auto getBodyAllocations(linalg::GenericOp generic) -> SmallVector<Operation*> {
  SmallVector<Operation*> result;
  generic.getBody()->walk([&](Operation* op) {
    if (!llvm::isa<memref::AllocaOp, memref::AllocOp>(op)) return;
    if (op->getNumOperands() != 0) return;
    result.push_back(op);
  });
  return result;
}

/// Gets how many elements of its tile \p generic covers, zero if it is not
/// static.
///
/// The parallel dimensions only. A register holds one element per lane of what
/// the compute works on at once, and a reduction dimension is iterated rather
/// than held -- counting it would size the register to every step of the
/// reduction instead of the accumulator it carries.
auto getTileSize(linalg::GenericOp generic) -> int64_t {
  const auto iterators = generic.getIteratorTypesArray();

  // Only the parallel dimensions inside the last reduction are lanes of the
  // register. One outside it is a loop around the whole reduction, and every
  // turn of it writes the register's same position.
  const auto ranges = generic.getStaticLoopRanges();
  int64_t result = 1;
  for (int64_t dim = static_cast<int64_t>(ranges.size()) - 1; dim >= 0; --dim) {
    if (iterators[dim] == utils::IteratorType::reduction) break;
    if (ShapedType::isDynamic(ranges[dim])) return 0;
    if (llvm::MulOverflow<int64_t>(result, ranges[dim], result)) return 0;
  }
  return result;
}

/// Sizes \p alloc to the tile \p generic covers and moves it in front of it.
///
/// A pattern says the element a register holds and leaves the count to here,
/// where the tile says it: what the template computes on is the tile, whether
/// that is one register or several. An allocation that already carries a shape
/// is taken at its word and only moved.
///
/// \p zero is the index of the lane the body's own accesses land on, made here
/// on the first allocation that needs one.
auto hoistAllocation(Operation* alloc, linalg::GenericOp generic,
                     ktdf_arch::ExecutionUnitOp compute, Value& zero,
                     RewriterBase& rewriter) -> LogicalResult {
  const auto type = cast<MemRefType>(alloc->getResult(0).getType());
  if (type.getRank() != 0) {
    rewriter.moveOpBefore(alloc, generic);
    return success();
  }

  for (auto* const user : alloc->getUsers()) {
    if (!llvm::isa<memref::StoreOp, memref::LoadOp, ktdf::OpaqueOp>(user)) {
      return alloc->emitError("unable to hoist allocation")
                 .attachNote(user->getLoc())
             << "user can't be vectorized";
    }
  }

  const auto element = type.getElementType();
  const auto tile = getTileSize(generic);
  if (tile == 0) {
    return generic.emitError("the tile is not a static number of elements");
  }

  // A register holds whole lanes of the element, so a tile of more than one has
  // to be a whole number of them: otherwise the template would read a lane it
  // never wrote. Less than one register still gets one -- what the body works
  // on is a position in it.
  const auto lanes = getVectorLanes(element, compute);
  int64_t held = tile;
  if (tile < lanes) {
    held = lanes;
  } else if (tile % lanes != 0) {
    return alloc->emitError("a tile of ")
           << tile << " does not divide into registers of " << lanes << " "
           << element;
  }

  const auto register_type = MemRefType::get(
      {held}, element, MemRefLayoutAttrInterface{}, type.getMemorySpace());

  rewriter.setInsertionPoint(generic);
  // The kind the body asked for is kept: what hoists a register's fill out of a
  // loop hoists an alloc and leaves an alloca alone, so the pattern says which
  // it wants by which one it wrote.
  Value reg =
      llvm::isa<memref::AllocOp>(alloc)
          ? memref::AllocOp::create(rewriter, alloc->getLoc(), register_type)
                .getResult()
          : memref::AllocaOp::create(rewriter, alloc->getLoc(), register_type)
                .getResult();
  if (!zero) {
    zero = arith::ConstantIndexOp::create(rewriter, alloc->getLoc(), 0);
  }

  // What the body writes and reads is one element, and it becomes lane zero of
  // the register. The lowering below turns either into the whole of it.
  for (auto* user : llvm::to_vector(alloc->getUsers())) {
    rewriter.setInsertionPoint(user);
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      // A value from outside the body is the same for every element, so it
      // belongs to the whole register rather than to lane zero. Runtime scalars
      // such as the base and the stride of an address arrive this way, hoisted
      // out of the body before this pass runs.
      const auto stored = store.getValueToStore();
      if (areValuesDefinedAbove(ValueRange{stored}, generic.getBodyRegion())) {
        rewriter.setInsertionPoint(generic);
        linalg::FillOp::create(rewriter, store.getLoc(), ValueRange{stored},
                               ValueRange{reg});
      } else {
        memref::StoreOp::create(rewriter, store.getLoc(), stored, reg,
                                ValueRange{zero});
      }
      rewriter.eraseOp(store);
      continue;
    }
    if (auto load = dyn_cast<memref::LoadOp>(user)) {
      auto value = memref::LoadOp::create(rewriter, load.getLoc(), reg,
                                          ValueRange{zero});
      rewriter.replaceOp(load, value.getResult());
    }
  }

  // Whoever is left takes the register as it stands -- the opaque, which reads
  // and writes the whole of it.
  rewriter.replaceOp(alloc, reg);
  return success();
}

/// Whether \p reader takes a float rather than the register: arithmetic does,
/// a template call does not.
auto wantsTheValue(Operation* reader) -> bool {
  Dialect* const dialect = reader->getDialect();
  return isa_and_nonnull<arith::ArithDialect>(dialect) ||
         isa_and_nonnull<math::MathDialect>(dialect);
}

/// Reads the first lane of \p reg in front of \p reader. The fill put the same
/// value in every lane, so any lane would do.
auto loadFirstLane(Value reg, Operation* reader, RewriterBase& rewriter)
    -> Value {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(reader);
  Value zero = arith::ConstantIndexOp::create(rewriter, reader->getLoc(), 0);
  return memref::LoadOp::create(rewriter, reader->getLoc(), reg,
                                ValueRange{zero});
}

/// The operand a body argument stands for, or the value itself otherwise.
auto getGenericOperand(Value value, linalg::GenericOp generic) -> Value {
  auto argument = dyn_cast<BlockArgument>(value);
  if (argument && argument.getOwner() == generic.getBody())
    return generic.getMatchingOpOperand(argument)->get();
  return value;
}

/// Values that one arithmetic instruction reads through multiple ports.
/// A stream can only be consumed once. Its second port cannot reuse a latch
/// until an earlier instruction has filled that port's latch.
auto getRepeatedOperands(linalg::GenericOp generic) -> SmallVector<Value> {
  SetVector<Value> repeated;
  for (Operation& op : generic.getBody()->without_terminator()) {
    if (!wantsTheValue(&op)) continue;
    SmallVector<Value> seen;
    for (Value value : op.getOperands()) {
      if ((!value.getType().isF16() && !value.getType().isF32()) ||
          matchPattern(getGenericOperand(value, generic), m_Constant()))
        continue;
      if (llvm::is_contained(seen, value))
        repeated.insert(value);
      else
        seen.push_back(value);
    }
  }
  return llvm::to_vector(repeated);
}

/// Gives each use its own register read while consuming each shared value
/// once. The scalar allocations use the same sizing and hoisting as registers
/// introduced by device patterns. Body arguments are stored at their first
/// use; OrderFifoSlotsByUse then orders transfers and reads to match the new
/// consumption order. Computed values are captured at their definition before
/// a later instruction can overwrite the source register.
auto materializeRepeatedOperands(linalg::GenericOp generic,
                                 ArrayRef<Value> repeated,
                                 ktdf_arch::ExecutionUnitOp compute,
                                 ktdf_arch::Mapping& mapping,
                                 SmallVectorImpl<Operation*>& allocations,
                                 RewriterBase& rewriter) -> LogicalResult {
  if (repeated.empty()) return success();
  arch_view::GroupLocalMemory local(mapping.getDevice());
  const auto registers = local.getLocalMemoryKind(compute.getKind());
  if (!registers)
    return generic.emitError(
        "repeated arithmetic operands require compute-local registers");

  for (Value value : repeated) {
    // Rewrite every body use, including later instructions. Leaving another
    // use on the stream would mix its latch lifetime with the register write.
    SmallVector<OpOperand*> uses;
    generic.getBody()->walk([&](Operation* op) {
      for (OpOperand& operand : op->getOpOperands())
        if (operand.get() == value) uses.push_back(&operand);
    });

    if (auto load = value.getDefiningOp<memref::LoadOp>();
        load && load.getMemRefType().getMemorySpace() == registers &&
        load->getNextNode() == uses.front()->getOwner() &&
        llvm::all_of(uses, [&](OpOperand* use) {
          return use->getOwner() == uses.front()->getOwner();
        })) {
      // An immediately consumed register needs no copy. Deferred uses must
      // capture it below: DFIR register loads become physical operand reads,
      // so cloning a load before an overwriting store would not save its value.
      rewriter.setInsertionPointAfter(load);
      for (OpOperand* use : llvm::drop_begin(uses))
        use->set(rewriter.clone(*load)->getResult(0));
      continue;
    }

    if (auto* definition = value.getDefiningOp();
        definition && definition->getBlock() == generic.getBody())
      rewriter.setInsertionPointAfter(definition);
    else
      rewriter.setInsertionPoint(uses.front()->getOwner());
    auto type = MemRefType::get({}, value.getType(),
                                MemRefLayoutAttrInterface{}, registers);
    auto reg = memref::AllocaOp::create(rewriter, generic.getLoc(), type);
    allocations.push_back(reg);
    memref::StoreOp::create(rewriter, generic.getLoc(), value, reg,
                            ValueRange{});
    for (OpOperand* use : uses) {
      rewriter.setInsertionPoint(use->getOwner());
      Value loaded =
          memref::LoadOp::create(rewriter, generic.getLoc(), reg, ValueRange{});
      use->set(loaded);
    }
  }
  return success();
}

/// Makes the registers \p generic uses real in front of its body.
auto materializeRegisters(linalg::GenericOp generic,
                          ktdf_arch::Mapping& mapping, RewriterBase& rewriter)
    -> LogicalResult {
  const auto constants = getMappedConstants(generic);
  auto allocations = getBodyAllocations(generic);
  const auto repeated = getRepeatedOperands(generic);
  if (constants.empty() && allocations.empty() && repeated.empty()) {
    // Nothing to size, so nothing to say about where this one computes.
    return success();
  }

  // We assume the register is for the compute that the generic is mapped to,
  // which is defaulted if it is not yet mapped.
  auto compute = mapping.getOrMap<ktdf_arch::ExecutionUnitOp>(
      generic, mapping.byKind().getDefaultCompute().getKind());
  if (!compute) {
    return generic.emitError("no compute resource is mapped to size registers");
  }

  for (auto constant : constants) {
    // Only one still inside has to move. Moving one that is already out could
    // put it after something else that reads it.
    if (generic->isProperAncestor(constant)) {
      rewriter.moveOpBefore(constant, generic);
    }
    const auto reg = materializeRegister(constant, generic, compute, rewriter);
    if (!reg) return failure();

    // Whatever read the constant reads the register now, where it stands: the
    // body is not isolated from above. Arithmetic reads a lane out of it.
    SmallVector<OpOperand*> takes_the_value;
    for (OpOperand& use : constant.getResult().getUses()) {
      if (generic->isProperAncestor(use.getOwner()) &&
          wantsTheValue(use.getOwner())) {
        takes_the_value.push_back(&use);
      }
    }
    rewriter.replaceUsesWithIf(constant.getResult(), reg, [&](OpOperand& use) {
      return generic->isProperAncestor(use.getOwner()) &&
             !wantsTheValue(use.getOwner());
    });
    for (OpOperand* use : takes_the_value) {
      use->set(loadFirstLane(reg, use->getOwner(), rewriter));
    }
  }

  if (failed(materializeRepeatedOperands(generic, repeated, compute, mapping,
                                         allocations, rewriter)))
    return failure();

  // The scratch moves out sized to the tile, and the body goes on reading it
  // there. Handing it in as an operand is not open: a generic takes either
  // tensors or buffers throughout, and the data here is tensors.
  Value zero;
  for (auto* alloc : allocations) {
    if (failed(hoistAllocation(alloc, generic, compute, zero, rewriter))) {
      return failure();
    }
  }

  return success();
}

struct MaterializeRegistersPass
    : public impl::MaterializeRegistersPassBase<MaterializeRegistersPass> {
  using MaterializeRegistersPassBase::MaterializeRegistersPassBase;

  void runOnOperation() override {
    // Obtain the default device, emitting a diagnostic on failure.
    const auto& default_device = getAnalysis<ktdf_arch::DefaultDevice>();
    if (!default_device) {
      signalPassFailure();
      return;
    }
    ktdf_arch::Mapping mapping(default_device.getRef());
    if (!mapping.byKind().getDefaultCompute()) {
      mapping.getDevice().getDeclaration().emitError(
          "no (unambiguous) default compute resource");
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    // The transpose rewrite also erases sibling slices. Collect the generics
    // first so a walk never advances through an erased sibling.
    SmallVector<linalg::GenericOp> generics;
    getOperation()->walk([&](linalg::GenericOp op) { generics.push_back(op); });
    for (auto generic : generics) {
      if (auto width = getWordTransposeWidth(generic)) {
        if (failed(materializeTranspose(generic, width, mapping, rewriter)))
          return signalPassFailure();
      } else if (failed(materializeRegisters(generic, mapping, rewriter))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace
}  // namespace scheduler
