//===-- RelayoutMaterialization.cpp --------------------------*- c++ -*-===//
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

#include "RelayoutMaterialization.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Analysis/Presburger/PresburgerRelation.h>
#include <mlir/Dialect/Affine/Analysis/AffineStructures.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/PatternMatch.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/PipelineBuilder.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/SpyreOp/SpyreOp.h"

using namespace mlir;
using namespace scheduler;

namespace {
bool sameIndexingMaps(ArrayRef<AffineMap> actual, ArrayRef<AffineMap> expected,
                      ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  // Compare relations over the iteration box, not map spelling. A matching
  // recipe is a total bijection, including all batch/row coordinates.
  presburger::IntegerPolyhedron box(
      presburger::PresburgerSpace::getSetSpace(rank));
  for (auto [d, n] : llvm::enumerate(shape)) {
    box.addBound(presburger::BoundType::LB, d, 0);
    box.addBound(presburger::BoundType::UB, d, n - 1);
  }
  auto relation = [&](AffineMap map) {
    presburger::IntegerRelation graph(
        presburger::PresburgerSpace::getRelationSpace(rank, rank));
    if (map.getNumDims() != rank || map.getNumResults() != rank ||
        map.getNumSymbols() || failed(affine::getRelationFromMap(map, graph)))
      return presburger::PresburgerRelation::getEmpty(graph.getSpace());
    graph.setSpace(presburger::PresburgerSpace::getRelationSpace(
        rank, rank, 0, graph.getNumLocalVars()));
    return presburger::PresburgerRelation(graph).intersectDomain(
        presburger::PresburgerSet(box));
  };
  return llvm::all_of(llvm::zip(actual, expected), [&](auto pair) {
    return relation(std::get<0>(pair)).isEqual(relation(std::get<1>(pair)));
  });
}
}  // namespace

spyreop::NumericCast scheduler::getNumericCast(Operation* op) {
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic || generic.getRegion().empty()) return {};
  auto casts = generic.getBody()->getOps<spyreop::NumericCast>();
  return casts.empty() ? spyreop::NumericCast{} : *casts.begin();
}

FailureOr<bool> scheduler::getNumericCastLayout(linalg::GenericOp generic) {
  auto conversion = getNumericCast(generic);
  if (!conversion || !generic.hasPureTensorSemantics() ||
      generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
      generic.getNumResults() != 1 ||
      !llvm::all_of(
          generic.getIteratorTypesArray(),
          [](auto it) { return it == utils::IteratorType::parallel; }) ||
      std::distance(generic.getBody()->begin(), generic.getBody()->end()) !=
          2 ||
      conversion.getInput() != generic.getBody()->getArgument(0) ||
      generic.getBody()->getTerminator()->getOperand(0) !=
          conversion.getResult()) {
    generic.emitError(
        "numeric cast requires a parallel generic with one input, "
        "one output, and a scalar cast/yield body");
    return failure();
  }
  bool widen = conversion.getKind() == "dl16tofp32";
  auto input = cast<RankedTensorType>(generic.getDpsInputs()[0].getType());
  auto output = cast<RankedTensorType>(generic.getResult(0).getType());
  auto native = widen ? input : output;
  auto wide = widen ? output : input;
  int64_t rank = native.getRank();
  int64_t packet_axis = rank > 2 ? rank - 3 : 0;
  if (rank < 2 || wide.getRank() != rank || generic.getNumLoops() != rank ||
      !native.hasStaticShape() || !wide.hasStaticShape() ||
      native.getEncoding() || wide.getEncoding() ||
      native.getShape().back() != 64 || wide.getShape().back() != 32 ||
      !llvm::all_of(
          native.getShape(), [](int64_t n) { return n > 0; }) ||
      !llvm::all_of(
          llvm::seq<int64_t>(0, rank - 1),
          [&](int64_t d) {
            return d == packet_axis
                       ? wide.getDimSize(d) % 2 == 0 &&
                             wide.getDimSize(d) / 2 == native.getDimSize(d)
                       : wide.getDimSize(d) == native.getDimSize(d);
          })) {
    generic.emitError(
        "numeric cast requires positive static device shapes "
        "[N,64] <-> [2*N,32] or [B...,N,R,64] <-> "
        "[B...,2*N,R,32], without encodings");
    return failure();
  }
  auto* ctx = generic.getContext();
  auto identity = AffineMap::getMultiDimIdentityMap(rank, ctx);
  auto g = getAffineDimExpr(packet_axis, ctx);
  auto l = getAffineDimExpr(rank - 1, ctx);
  for (bool standard : {false, true}) {
    SmallVector<AffineExpr> coords(identity.getResults());
    coords[packet_axis] =
        g * 2 + (standard ? l.floorDiv(32) : (l % 8).floorDiv(4));
    coords.back() = standard ? l % 32 : l.floorDiv(8) * 4 + l % 4;
    auto fp32 = AffineMap::get(rank, 0, coords, ctx);
    SmallVector<AffineMap> maps = widen
                                      ? SmallVector<AffineMap>{identity, fp32}
                                      : SmallVector<AffineMap>{fp32, identity};
    if (sameIndexingMaps(generic.getIndexingMapsArray(), maps,
                         native.getShape()))
      return standard;
  }
  generic.emitError("no numeric cast packet recipe for these indexing_maps");
  return failure();
}

SmallVector<AffineMap> scheduler::getPacketRelayoutMaps(unsigned rank,
                                                        bool to_staggered,
                                                        MLIRContext* ctx) {
  unsigned packet_axis = rank > 2 ? rank - 3 : 0;
  auto k = getAffineDimExpr(packet_axis, ctx);
  auto e = getAffineDimExpr(rank - 1, ctx);
  SmallVector<AffineExpr> coords;
  for (unsigned d = 0; d < rank; ++d)
    coords.push_back(getAffineDimExpr(d, ctx));
  coords[packet_axis] = k.floorDiv(2) * 2 + (e % 8).floorDiv(4);
  coords.back() = (k % 2) * 16 + e.floorDiv(8) * 4 + e % 4;
  auto order = AffineMap::get(rank, 0, coords, ctx);
  auto identity = AffineMap::getMultiDimIdentityMap(rank, ctx);
  return to_staggered ? SmallVector<AffineMap>{identity, order}
                      : SmallVector<AffineMap>{order, identity};
}

std::optional<bool> scheduler::getPacketRelayoutDirection(Operation* op) {
  auto generic = dyn_cast<linalg::GenericOp>(op);
  if (!generic || !generic.hasPureTensorSemantics() ||
      generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
      generic.getDpsInputs()[0].getType() != generic.getResult(0).getType() ||
      !llvm::all_of(generic.getIteratorTypesArray(), [](auto it) {
        return it == utils::IteratorType::parallel;
      }))
    return std::nullopt;
  auto type = cast<RankedTensorType>(generic.getResult(0).getType());
  auto rank = type.getRank();
  auto shape = type.getShape();
  auto packet_axis = rank > 2 ? rank - 3 : 0;
  if (rank < 2 || !type.getElementType().isF32() || type.getEncoding() ||
      !type.hasStaticShape() ||
      !llvm::all_of(
          shape, [](int64_t n) { return n > 0; }) ||
      shape.back() != 32 || shape[packet_axis] % 2)
    return std::nullopt;
  auto& body = generic.getRegion().front();
  if (std::distance(body.begin(), body.end()) != 2) return std::nullopt;
  auto identity = dyn_cast<spyreop::Identity>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!identity || identity.getOperand() != body.getArgument(0) || !yield ||
      yield.getOperand(0) != identity.getResult())
    return std::nullopt;

  auto maps = generic.getIndexingMapsArray();
  for (bool to_staggered : {false, true}) {
    auto expected = getPacketRelayoutMaps(rank, to_staggered, op->getContext());
    if (sameIndexingMaps(maps, expected, shape)) return to_staggered;
  }
  return std::nullopt;
}

namespace {
SmallVector<OpFoldResult> sizes(OpBuilder& b, ArrayRef<int64_t> shape) {
  return llvm::map_to_vector(
      shape, [&](int64_t n) -> OpFoldResult { return b.getIndexAttr(n); });
}

// The target's cast/shuffle/pack lowering requires a direct receive-to-cast
// and cast-to-send packet path. Separate adjacent tensor computes through
// ordinary local-memory transfers, with the surrounding iteration owning all
// phases and their scratch. Scalar address setup stays in the original block.
LogicalResult isolatePacketComputes(func::FuncOp func, IRRewriter& rw,
                                    Attribute local) {
  SmallVector<ktdf_lowering::NumericCastValueOp> casts;
  func.walk([&](ktdf_lowering::NumericCastValueOp op) { casts.push_back(op); });
  for (auto conversion : casts) {
    auto* block = conversion->getBlock();
    SmallVector<Operation*> before, after;
    bool seen_cast = false, has_before = false, has_after = false;
    for (auto& op : block->without_terminator()) {
      if (&op == conversion.getOperation()) {
        seen_cast = true;
        continue;
      }
      bool tensors =
          llvm::any_of(op.getOperandTypes(),
                       [](Type t) { return isa<RankedTensorType>(t); }) ||
          llvm::any_of(op.getResultTypes(),
                       [](Type t) { return isa<RankedTensorType>(t); });
      if (!tensors) continue;
      (seen_cast ? after : before).push_back(&op);
      if (isa<linalg::GenericOp>(op))
        (seen_cast ? has_after : has_before) = true;
    }
    if (!has_before && !has_after) continue;
    if (llvm::any_of(before,
                     [](Operation* op) {
                       return isa<ktdf_lowering::NumericCastValueOp>(op);
                     }) ||
        llvm::any_of(after, [](Operation* op) {
          return isa<ktdf_lowering::NumericCastValueOp>(op);
        }))
      return conversion.emitError(
          "packet phases require one cast per iteration");

    auto loc = conversion.getLoc();
    rw.setInsertionPoint(block, block->without_terminator().end());
    auto pipeline = ktdf::PipelineOp::create(rw, loc);
    ktdf::PrivateBuilder priv(pipeline);
    Value pre_done = has_before ? priv.createToken(loc) : Value{};
    Value cast_done = has_after ? priv.createToken(loc) : Value{};
    rw.setInsertionPointToEnd(pipeline.getBody());
    ktdf::StageOp pre, post;
    if (has_before)
      pre = ktdf::StageOp::create(rw, loc, ValueRange{}, ValueRange{pre_done});
    auto cast_stage = ktdf::StageOp::create(
        rw, loc, pre_done ? ValueRange{pre_done} : ValueRange{},
        cast_done ? ValueRange{cast_done} : ValueRange{});
    if (has_after)
      post =
          ktdf::StageOp::create(rw, loc, ValueRange{cast_done}, ValueRange{});
    auto* first_body = (pre ? pre : cast_stage).getBody();
    auto* last_body = (post ? post : cast_stage).getBody();
    for (auto* op : before) op->moveBefore(first_body, first_body->end());
    conversion->moveBefore(cast_stage.getBody(), cast_stage.getBody()->end());
    for (auto* op : after) op->moveBefore(last_body, last_body->end());

    auto allocate = [&](Value packet) -> Value {
      auto type = cast<RankedTensorType>(packet.getType());
      return memref::AllocOp::create(
          priv, loc,
          MemRefType::get(type.getShape(), type.getElementType(),
                          MemRefLayoutAttrInterface{}, local));
    };
    auto store = [&](Value packet, Value buffer, ktdf::StageOp stage) {
      auto type = cast<RankedTensorType>(packet.getType());
      rw.setInsertionPointToEnd(stage.getBody());
      auto op = ktdp_lowering::StoreOp::create(rw, loc, packet, buffer, {},
                                               type.getShape());
      setThrottle(op, type.getNumElements());
    };
    auto load = [&](Value packet, Value buffer, Operation* before) -> Value {
      auto type = cast<RankedTensorType>(packet.getType());
      rw.setInsertionPoint(before);
      auto op = ktdp_lowering::LoadOp::create(
          rw, loc, buffer, {}, type.getShape(), type.getElementType());
      setThrottle(op, type.getNumElements());
      return op.getResult();
    };
    if (pre) {
      for (auto& operand : conversion->getOpOperands()) {
        Value packet = operand.get(), buffer = allocate(packet);
        store(packet, buffer, pre);
        operand.set(load(packet, buffer, conversion));
      }
    }
    if (post) {
      auto* first = &post.getBody()->front();
      for (Value packet : conversion.getResults()) {
        Value buffer = allocate(packet);
        Value reloaded = load(packet, buffer, first);
        rw.replaceAllUsesWith(packet, reloaded);
        store(packet, buffer, cast_stage);
      }
    }
  }
  return success();
}

void transfer(OpBuilder& b, Location loc, Value src, ArrayRef<Value> si,
              ArrayRef<int64_t> ss, Value dst, ArrayRef<Value> di,
              ArrayRef<int64_t> ds, int64_t throttle) {
  auto id = [&](unsigned rank) {
    return rank ? AffineMap::getMultiDimIdentityMap(rank, b.getContext())
                : AffineMap{};
  };
  auto op = ktdf::DataTransferOp::create(
      b, loc, src, id(si.size()), si, sizes(b, ss), dst, id(di.size()), di,
      sizes(b, ds), ktdf::TransferMode::Copy);
  setThrottle(op, throttle);
}

// Each unit sees the same h/a/p order. Linear offsets avoid an unsupported
// floor/mod address-generator recurrence in the hardware backend.
void copyLoop(OpBuilder& b, Location loc,
              function_ref<void(OpBuilder&, Value, Value, Value)> body) {
  auto c0 = arith::ConstantIndexOp::create(b, loc, 0);
  auto c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto c2 = arith::ConstantIndexOp::create(b, loc, 2);
  auto c4 = arith::ConstantIndexOp::create(b, loc, 4);
  scf::buildLoopNest(b, loc, ValueRange{c0, c0, c0}, ValueRange{c2, c4, c2},
                     ValueRange{c1, c1, c1},
                     [&](OpBuilder& builder, Location, ValueRange ivs) {
                       body(builder, ivs[0], ivs[1], ivs[2]);
                     });
}

bool isNextPacket(OpFoldResult first, OpFoldResult second) {
  auto a = getConstantIntValue(first), b = getConstantIntValue(second);
  if (a && b) return *a % 2 == 0 && *b == *a + 1;
  auto v = dyn_cast<Value>(first), w = dyn_cast<Value>(second);
  auto mul = v ? v.getDefiningOp<arith::MulIOp>() : nullptr;
  auto add = w ? w.getDefiningOp<arith::AddIOp>() : nullptr;
  if (!mul || !add) return false;
  auto even = [](Value n) {
    auto c = getConstantIntValue(n);
    return c && *c % 2 == 0;
  };
  return (even(mul.getLhs()) || even(mul.getRhs())) &&
         ((add.getLhs() == v && getConstantIntValue(add.getRhs()) == 1) ||
          (add.getRhs() == v && getConstantIntValue(add.getLhs()) == 1));
}

// The ordinary route planner recognizes one reader/writer per shared buffer.
// Express adjacent packet copies as one transfer in a two-iteration loop.
FailureOr<bool> coalescePacketCopies(ktdf::StageOp stage, IRRewriter& rw,
                                     Attribute local, Attribute compute,
                                     const AttrMapping& spaces) {
  auto stores =
      llvm::to_vector(stage.getBody()->getOps<ktdp_lowering::StoreOp>());
  if (stores.size() != 2)
    return (stage.emitError("expected two packet stores"), failure());
  SmallVector<ktdp_lowering::LoadOp> loads;
  for (auto store : stores) {
    Value source = store.getSource();
    while (
        llvm::isa_and_present<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
            source.getDefiningOp()))
      source = source.getDefiningOp()->getOperand(0);
    auto load = source.getDefiningOp<ktdp_lowering::LoadOp>();
    if (!load) return (store.emitError("expected a packet copy"), failure());
    loads.push_back(load);
  }
  if (loads[0].getSource() != loads[1].getSource() ||
      stores[0].getDest() != stores[1].getDest())
    return (stage.emitError("packet copies require common memory endpoints"),
            failure());
  auto adjacent = [](OffsetSizeAndStrideOpInterface a,
                     OffsetSizeAndStrideOpInterface b) {
    auto first = a.getMixedOffsets(), second = b.getMixedOffsets();
    if (first.empty() || first.size() != second.size() ||
        a.getStaticSizes() != b.getStaticSizes() ||
        a.getStaticStrides() != b.getStaticStrides() ||
        a.getStaticSizes().back() != 32 ||
        !llvm::all_of(a.getStaticStrides(), [](int64_t n) { return n == 1; }))
      return false;
    unsigned axis = first.size() > 2 ? first.size() - 3 : 0;
    for (unsigned d = 0; d < first.size(); ++d)
      if (d != axis && first[d] != second[d]) return false;
    return isNextPacket(first[axis], second[axis]);
  };
  if (!adjacent(loads[0], loads[1]) || !adjacent(stores[0], stores[1]))
    return (stage.emitError("packet copies must be adjacent complete packets"),
            failure());
  // A same-memory copy needs a compute hop; there is no local-memory DMA
  // unit. Keep an ordinary identity generic so PipelineBuilder supplies that
  // route instead of leaving an unassigned memory-only stage.
  if (spaces.map(getMemorySpace(loads[0].getSource())) == local &&
      spaces.map(getMemorySpace(stores[0].getDest())) == local) {
    for (auto store : stores) {
      rw.setInsertionPoint(store);
      auto type = cast<RankedTensorType>(store.getSource().getType());
      auto identity =
          AffineMap::getMultiDimIdentityMap(type.getRank(), rw.getContext());
      auto init =
          tensor::EmptyOp::create(rw, store.getLoc(), type, ValueRange{});
      auto copy = linalg::GenericOp::create(
          rw, store.getLoc(), TypeRange{type}, ValueRange{store.getSource()},
          ValueRange{init}, ArrayRef{identity, identity},
          SmallVector<utils::IteratorType>(type.getRank(),
                                           utils::IteratorType::parallel),
          [&](OpBuilder& b, Location l, ValueRange args) {
            auto value = spyreop::Identity::create(b, l, args[0]);
            linalg::YieldOp::create(b, l, value.getResult());
          });
      copy->setAttr("ktdf_arch.maps_to", compute);
      setThrottle(copy, 32);
      store.getSourceMutable().assign(copy.getResult(0));
    }
    return true;
  }
  auto loc = stage.getLoc();
  rw.setInsertionPointToEnd(stage.getBody());
  auto zero = arith::ConstantIndexOp::create(rw, loc, 0);
  auto one = arith::ConstantIndexOp::create(rw, loc, 1);
  auto two = arith::ConstantIndexOp::create(rw, loc, 2);
  scf::ForOp::create(
      rw, loc, zero, two, one, ValueRange{},
      [&](OpBuilder& b, Location l, Value packet, ValueRange) {
        auto offsets = [&](OffsetSizeAndStrideOpInterface access) {
          auto off = access.getMixedOffsets();
          unsigned axis = off.size() > 2 ? off.size() - 3 : 0;
          off[axis] =
              arith::AddIOp::create(
                  b, l, getValueOrCreateConstantIndexOp(b, l, off[axis]),
                  packet)
                  .getResult();
          return off;
        };
        auto load = ktdp_lowering::LoadOp::create(
            b, l, loads[0].getSource(), offsets(loads[0]),
            loads[0].getMixedSizes(), loads[0].getMixedStrides(),
            b.getF32Type());
        setThrottle(load, 32);
        auto rank = load.getType().getRank();
        ReassociationIndices dims;
        for (int64_t d = 0; d < rank; ++d) dims.push_back(d);
        Value flat = tensor::CollapseShapeOp::create(
            b, l, RankedTensorType::get({32}, b.getF32Type()), load.getResult(),
            ArrayRef<ReassociationIndices>{dims});
        auto type = cast<RankedTensorType>(stores[0].getSource().getType());
        dims.clear();
        for (int64_t d = 0; d < type.getRank(); ++d) dims.push_back(d);
        auto expanded = tensor::ExpandShapeOp::create(
            b, l, type, flat, ArrayRef<ReassociationIndices>{dims});
        auto store = ktdp_lowering::StoreOp::create(
            b, l, expanded.getResult(), stores[0].getDest(), offsets(stores[0]),
            stores[0].getMixedSizes(), stores[0].getMixedStrides());
        setThrottle(store, 32);
        scf::YieldOp::create(b, l);
      });
  for (auto store : stores) rw.eraseOp(store);
  return false;
}

// Group tiling assembles two physical packets without changing their order.
// Keep the generic until this point; its indexing maps select actual movement.
LogicalResult materialize(linalg::GenericOp generic, bool to_staggered,
                          IRRewriter& rw, Attribute local, Attribute compute,
                          const AttrMapping& spaces) {
  auto type = RankedTensorType::get({2, 32}, rw.getF32Type());
  if (generic.getResult(0).getType() != type)
    return generic.emitError("relayout must be tiled into two FP32 packets");
  SmallVector<Value> packets(2);
  SmallVector<Operation*> assembly;
  Value value = generic.getDpsInputs()[0];
  for (int64_t p = 1; p >= 0; --p) {
    auto insert = value.getDefiningOp<tensor::InsertSliceOp>();
    if (!insert || !insert->hasOneUse() ||
        insert.getStaticOffsets() != ArrayRef<int64_t>{p, 0} ||
        insert.getStaticSizes() != ArrayRef<int64_t>{1, 32} ||
        insert.getStaticStrides() != ArrayRef<int64_t>{1, 1})
      return generic.emitError("expected two complete relayout input packets");
    packets[p] = insert.getSource();
    assembly.push_back(insert);
    value = insert.getDest();
  }
  SmallVector<tensor::ExtractSliceOp> results(2);
  for (auto* user : generic.getResult(0).getUsers()) {
    auto slice = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!slice || slice.getStaticOffsets().size() != 2 ||
        slice.getStaticOffsets()[0] < 0 || slice.getStaticOffsets()[0] > 1 ||
        slice.getStaticOffsets()[1] != 0 ||
        slice.getStaticSizes() != ArrayRef<int64_t>{1, 32} ||
        slice.getStaticStrides() != ArrayRef<int64_t>{1, 1} ||
        results[slice.getStaticOffsets()[0]])
      return generic.emitError("expected two complete relayout output packets");
    results[slice.getStaticOffsets()[0]] = slice;
  }
  if (!results[0] || !results[1])
    return generic.emitError("relayout requires both output packets");

  auto loc = generic.getLoc();
  auto* block = generic->getBlock();
  SmallVector<Operation*> before, after;
  bool seen = false;
  for (auto& op : block->without_terminator()) {
    if (&op == generic.getOperation()) {
      seen = true;
      continue;
    }
    if (llvm::is_contained(assembly, &op) || &op == results[0] ||
        &op == results[1])
      continue;
    if (llvm::any_of(op.getOperandTypes(),
                     [](Type t) { return isa<RankedTensorType>(t); }) ||
        llvm::any_of(op.getResultTypes(),
                     [](Type t) { return isa<RankedTensorType>(t); }))
      (seen ? after : before).push_back(&op);
  }
  rw.setInsertionPoint(block, block->without_terminator().end());
  auto outer = ktdf::PipelineOp::create(rw, loc);
  ktdf::PrivateBuilder priv(outer);
  auto memtype = MemRefType::get({2, 32}, rw.getF32Type(),
                                 MemRefLayoutAttrInterface{}, local);
  Value src = memref::AllocOp::create(priv, loc, memtype);
  Value dst = memref::AllocOp::create(priv, loc, memtype);
  auto ready = priv.createToken(loc), done = priv.createToken(loc);
  rw.setInsertionPointToEnd(outer.getBody());
  auto pre = ktdf::StageOp::create(rw, loc, ValueRange{}, ValueRange{ready});
  auto reorder =
      ktdf::StageOp::create(rw, loc, ValueRange{ready}, ValueRange{done});
  auto post = ktdf::StageOp::create(rw, loc, ValueRange{done}, ValueRange{});
  auto pre_phase = pre, post_phase = post;
  auto has_compute = [](ArrayRef<Operation*> ops) {
    return llvm::any_of(ops, [](Operation* op) {
      return isa<linalg::GenericOp, ktdf_lowering::NumericCastValueOp>(op);
    });
  };
  bool pre_compute = has_compute(before), post_compute = has_compute(after);
  // Every outer phase contains one pipeline, even when packet load/store
  // folding turns its entire body into direct memory-to-memory transfers.
  for (auto* phase : {&pre, &post}) {
    rw.setInsertionPointToEnd(phase->getBody());
    auto nested = ktdf::PipelineOp::create(rw, loc);
    rw.setInsertionPointToEnd(nested.getBody());
    *phase = ktdf::StageOp::create(rw, loc, ValueRange{}, ValueRange{});
  }
  for (auto* op : before) op->moveBefore(pre.getBody(), pre.getBody()->end());
  rw.setInsertionPointToEnd(pre.getBody());
  for (unsigned p = 0; p < 2; ++p) {
    auto packet = tensor::ExpandShapeOp::create(
        rw, loc, RankedTensorType::get({1, 32}, rw.getF32Type()), packets[p],
        ArrayRef<ReassociationIndices>{{0, 1}});
    auto store = ktdp_lowering::StoreOp::create(
        rw, loc, packet.getResult(), src, sizes(rw, {p, 0}), sizes(rw, {1, 32}),
        sizes(rw, {1, 1}));
    setThrottle(store, 32);
  }
  rw.setInsertionPointToEnd(post.getBody());
  for (unsigned p = 0; p < 2; ++p) {
    auto load = ktdp_lowering::LoadOp::create(
        rw, loc, dst, sizes(rw, {p, 0}), sizes(rw, {1, 32}), sizes(rw, {1, 1}),
        rw.getF32Type());
    setThrottle(load, 32);
    Value packet = tensor::CollapseShapeOp::create(
        rw, loc, RankedTensorType::get({32}, rw.getF32Type()), load.getResult(),
        ArrayRef<ReassociationIndices>{{0, 1}});
    rw.replaceOp(results[p], packet);
  }
  for (auto* op : after) op->moveBefore(post.getBody(), post.getBody()->end());
  rw.eraseOp(generic);
  for (auto* op : assembly) rw.eraseOp(op);

  rw.setInsertionPointToEnd(reorder.getBody());
  auto pipe = ktdf::PipelineOp::create(rw, loc);
  ktdf::PrivateBuilder cp(pipe);
  auto in_type = ktdf::FifoSlotType::get(rw.getContext(), local, compute, 4,
                                         rw.getF32Type());
  auto out_type = ktdf::FifoSlotType::get(rw.getContext(), compute, local, 4,
                                          rw.getF32Type());
  Value input =
      ktdf::FifoAllocateOp::create(cp, loc, TypeRange{in_type}, ValueRange{})
          ->getResult(0);
  Value output =
      ktdf::FifoAllocateOp::create(cp, loc, TypeRange{out_type}, ValueRange{})
          ->getResult(0);
  auto loaded = cp.createToken(loc), copied = cp.createToken(loc);
  Value input_ready = pre_compute ? Value{} : cp.createToken(loc);
  Value output_ready = post_compute ? Value{} : cp.createToken(loc);
  rw.setInsertionPointToEnd(pipe.getBody());
  auto position = [&](OpBuilder& b, Value h, Value a, Value p, bool staggered) {
    auto d0 = getAffineDimExpr(0, b.getContext()),
         d1 = getAffineDimExpr(1, b.getContext());
    auto lane = affine::AffineApplyOp::create(
        b, loc,
        AffineMap::get(2, 0, staggered ? d0 * 16 + d1 * 4 : d0 * 8 + d1 * 4),
        staggered ? ValueRange{h, a} : ValueRange{a, p});
    return SmallVector<Value>{staggered ? p : h, lane};
  };
  auto read_stage = ktdf::StageOp::create(
      rw, loc, input_ready ? ValueRange{input_ready} : ValueRange{},
      ValueRange{loaded}, [&](OpBuilder& b, Location l) {
        copyLoop(b, l, [&](OpBuilder& b, Value h, Value a, Value p) {
          transfer(b, l, src, position(b, h, a, p, !to_staggered), {1, 4},
                   input, {}, {4}, 4);
        });
      });
  auto copy = ktdf::StageOp::create(
      rw, loc, ValueRange{loaded}, ValueRange{copied},
      [&](OpBuilder& b, Location l) {
        copyLoop(b, l, [&](OpBuilder& b, Value, Value, Value) {
          auto val = ktdf::ReadFromFifoOp::create(
              b, l, RankedTensorType::get({4}, b.getF32Type()), input);
          ktdf::WriteToFifoOp::create(b, l, val.getResult(), output);
        });
      });
  copy.setApplicableUnitsAttr(rw.getArrayAttr({compute}));
  auto write_stage = ktdf::StageOp::create(
      rw, loc, ValueRange{copied},
      output_ready ? ValueRange{output_ready} : ValueRange{},
      [&](OpBuilder& b, Location l) {
        copyLoop(b, l, [&](OpBuilder& b, Value h, Value a, Value p) {
          transfer(b, l, output, {}, {4}, dst,
                   position(b, h, a, p, to_staggered), {1, 4}, 4);
        });
      });
  if (!pre_compute) {
    auto result = coalescePacketCopies(pre, rw, local, compute, spaces);
    if (failed(result)) return failure();
    pre_compute = *result;
    if (pre_compute) read_stage.setDependsIn({});
  }
  if (!post_compute) {
    auto result = coalescePacketCopies(post, rw, local, compute, spaces);
    if (failed(result)) return failure();
    post_compute = *result;
    if (post_compute) write_stage.setDependsOut({});
  }

  // Memory-only packet fills/drains belong to the same pipeline as the
  // relayout's identity compute, so ordinary path expansion has an anchor.
  if (!pre_compute) {
    pre->moveBefore(read_stage);
    pre.setDependsOut({input_ready});
    reorder.setDependsIn({});
    rw.eraseOp(pre_phase);
  }
  if (!post_compute) {
    post->moveAfter(write_stage);
    post.setDependsIn({output_ready});
    reorder.setDependsOut({});
    rw.eraseOp(post_phase);
  }
  return success();
}
}  // namespace

LogicalResult scheduler::materializeRelayouts(func::FuncOp func,
                                              ktdf_arch::Mapping& mapping,
                                              const AttrMapping& spaces) {
  SmallVector<std::pair<linalg::GenericOp, bool>> requests;
  func.walk([&](linalg::GenericOp generic) {
    if (auto direction = getPacketRelayoutDirection(generic))
      requests.emplace_back(generic, *direction);
  });
  auto local = spaces.map(ktdp::MemorySpaceAttr::get(
      func.getContext(), ktdp::MemorySpaceKind::ct_local, -1));
  IRRewriter rw(func);
  if (!requests.empty()) {
    Attribute compute;
    for (const auto& kind : mapping.byKind()) {
      auto unit = dyn_cast<ktdf_arch::ExecutionUnitOp>(
          kind.getExemplar().getOperation());
      if (unit && unit.getFeature<ktdf_arch::feature::Compute>()) {
        if (compute)
          return func.emitError("relayout requires one default compute kind");
        compute = unit.getKind();
      }
    }
    if (!compute) return func.emitError("relayout requires a compute route");
    for (auto [generic, direction] : requests)
      if (failed(materialize(generic, direction, rw, local, compute, spaces)))
        return failure();
  }
  return isolatePacketComputes(func, rw, local);
}
