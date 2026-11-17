//===-- AccessOrderMaterialization.cpp --------------------------*- c++ -*-===//
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

#include "AccessOrderMaterialization.h"

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Analysis/Presburger/PresburgerRelation.h>
#include <mlir/Dialect/Affine/Analysis/AffineStructures.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
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

using namespace mlir;
using namespace scheduler;

namespace {
// Compare relations over the declared domain, rather than map spelling.
bool hasPacketCopyRecipe(ktdp::ConstructAccessTilesOp tile) {
  const auto rank = tile.getType().getRank();
  auto shape = tile.getType().getShape();
  const auto packet_axis = rank > 2 ? rank - 3 : 0;
  if (rank < 2 || shape.back() != 32 || shape[packet_axis] <= 0 ||
      shape[packet_axis] % 2 ||
      tile.getAccessTileSet().getValue().getNumSymbols())
    return false;
  auto* ctx = tile.getContext();
  auto k = getAffineDimExpr(packet_axis, ctx);
  auto e = getAffineDimExpr(rank - 1, ctx);
  SmallVector<AffineExpr> coords;
  for (unsigned d = 0; d < rank; ++d)
    coords.push_back(getAffineDimExpr(d, ctx));
  coords[packet_axis] = (k.floorDiv(2)) * 2 + (e % 8).floorDiv(4);
  coords.back() = (k % 2) * 16 + e.floorDiv(8) * 4 + e % 4;
  auto expected = AffineMap::get(rank, 0, coords, ctx);
  auto relation = [&](AffineMap map) {
    presburger::IntegerRelation graph(
        presburger::PresburgerSpace::getRelationSpace(rank, rank));
    if (failed(affine::getRelationFromMap(map, graph)))
      return presburger::PresburgerRelation::getEmpty(graph.getSpace());
    graph.setSpace(presburger::PresburgerSpace::getRelationSpace(
        rank, rank, 0, graph.getNumLocalVars()));
    return presburger::PresburgerRelation(graph).intersectDomain(
        presburger::PresburgerSet(affine::FlatAffineValueConstraints(
            tile.getAccessTileSet().getValue())));
  };
  return relation(tile.getAccessTileOrder()).isEqual(relation(expected));
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

struct Request {
  ktdp::ConstructAccessTilesOp tile;
  SmallVector<Operation*> accesses;
  SmallVector<SmallVector<OpFoldResult>> offsets;
  bool load;
};

// A pair must be complete in one iteration before its scratch can be read.
FailureOr<Request> getRequest(ktdp::ConstructAccessTilesOp tile,
                              ArrayRef<Operation*> accesses) {
  if (!hasPacketCopyRecipe(tile)) return failure();
  const auto rank = tile.getType().getRank();
  const auto packet_axis = rank > 2 ? rank - 3 : 0;
  Request r{tile, llvm::to_vector(accesses), {}, false};
  if (r.accesses.size() != 2) return failure();
  auto* block = r.accesses[0]->getBlock();
  if (r.accesses[1]->getBlock() != block ||
      !isa<scf::ForOp, func::FuncOp>(block->getParentOp()))
    return failure();
  if (r.accesses[1]->isBeforeInBlock(r.accesses[0]))
    std::swap(r.accesses[0], r.accesses[1]);
  r.load = isa<ktdp_lowering::LoadOp>(r.accesses[0]);
  for (auto* op : r.accesses) {
    if (r.load ? !isa<ktdp_lowering::LoadOp>(op)
               : !isa<ktdp_lowering::StoreOp>(op))
      return failure();
    auto slice = cast<OffsetSizeAndStrideOpInterface>(op);
    auto sizes = slice.getStaticSizes();
    if (sizes.back() != 32 ||
        !llvm::all_of(sizes.drop_back(), [](int64_t n) { return n == 1; }) ||
        !llvm::all_of(slice.getStaticStrides(),
                      [](int64_t n) { return n == 1; }))
      return failure();
    auto offsets = slice.getMixedOffsets();
    if (getConstantIntValue(offsets.back()) != 0) return failure();
    r.offsets.push_back(offsets);
  }
  if (!isNextPacket(r.offsets[0][packet_axis], r.offsets[1][packet_axis])) {
    if (!isNextPacket(r.offsets[1][packet_axis], r.offsets[0][packet_axis]))
      return failure();
    std::swap(r.offsets[0], r.offsets[1]);
    std::swap(r.accesses[0], r.accesses[1]);
  }
  for (unsigned d = 0; d + 1 < r.offsets[0].size(); ++d)
    if (d != packet_axis && r.offsets[0][d] != r.offsets[1][d])
      return failure();
  for (auto* op : r.accesses) {
    auto type = r.load ? op->getResult(0).getType()
                       : cast<ktdp_lowering::StoreOp>(op).getSource().getType();
    if (!cast<ShapedType>(type).getElementType().isF32()) return failure();
  }
  return r;
}

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

LogicalResult materialize(Request& req, IRRewriter& rw, Attribute local,
                          Attribute compute, const AttrMapping& spaces) {
  auto loc = req.tile.getLoc();
  auto rank = req.tile.getType().getRank();
  const auto packet_axis = rank > 2 ? rank - 3 : 0;
  auto* block = req.accesses[0]->getBlock();
  // The surrounding iteration owns both packets and all phases.
  SmallVector<Operation*> original;
  for (auto& op : block->without_terminator()) {
    bool tensors =
        llvm::any_of(op.getOperandTypes(),
                     [](Type t) { return isa<RankedTensorType>(t); }) ||
        llvm::any_of(op.getResultTypes(),
                     [](Type t) { return isa<RankedTensorType>(t); });
    if (tensors) original.push_back(&op);
  }
  rw.setInsertionPoint(block->getTerminator());
  SmallVector<Value> base_args(req.tile.getIndices());
  llvm::append_range(base_args, req.tile.getSymbolOperands());
  auto base =
      affine::expandAffineMap(rw, loc, req.tile.getBaseMap(), base_args);
  if (!base) return req.tile.emitError("cannot expand access base");
  SmallVector<SmallVector<Value>> external_offsets;
  for (const auto& offsets : req.offsets) {
    SmallVector<Value> coords;
    for (auto [b, o] : llvm::zip(*base, offsets))
      coords.push_back(arith::AddIOp::create(
          rw, loc, b, getValueOrCreateConstantIndexOp(rw, loc, o)));
    external_offsets.push_back(coords);
  }
  Value external = req.tile.getBase();
  auto ext_type = cast<MemRefType>(external.getType());
  auto external_space = spaces.map(getMemorySpace(external));
  const bool needs_dma = external_space != local;
  external = memref::MemorySpaceCastOp::create(
      rw, loc,
      MemRefType::get(ext_type.getShape(), ext_type.getElementType(),
                      ext_type.getLayout(), external_space),
      external);
  auto outer = ktdf::PipelineOp::create(rw, loc);
  ktdf::PrivateBuilder priv(outer);
  SmallVector<int64_t> shape(rank, 1);
  shape[packet_axis] = 2;
  shape.back() = 32;
  auto memtype = MemRefType::get(shape, rw.getF32Type(),
                                 MemRefLayoutAttrInterface{}, local);
  Value s = memref::AllocOp::create(priv, loc, memtype);
  auto done = priv.createToken(loc);
  // PrivateBuilder makes values escape the allocation scope via private_yield.
  rw.setInsertionPointToEnd(outer.getBody());
  auto first = ktdf::StageOp::create(rw, loc, ValueRange{}, ValueRange{done});
  auto second = ktdf::StageOp::create(rw, loc, ValueRange{done}, ValueRange{});
  auto normal = req.load ? second : first;
  auto reorder = req.load ? first : second;
  for (auto* op : original)
    op->moveBefore(normal.getBody(), normal.getBody()->end());
  for (unsigned p = 0; p < 2; ++p) {
    rw.setInsertionPoint(req.accesses[p]);
    SmallVector<OpFoldResult> off(rank, rw.getIndexAttr(0));
    off[packet_axis] = rw.getIndexAttr(p);
    auto packet_shape = shape;
    packet_shape[packet_axis] = 1;
    if (req.load) {
      auto old = cast<ktdp_lowering::LoadOp>(req.accesses[p]);
      auto load = ktdp_lowering::LoadOp::create(
          rw, loc, s, off, sizes(rw, packet_shape),
          sizes(rw, SmallVector<int64_t>(rank, 1)), rw.getF32Type());
      setThrottle(load, 32);
      rw.replaceOp(old, load.getResult());
    } else {
      auto old = cast<ktdp_lowering::StoreOp>(req.accesses[p]);
      auto store = ktdp_lowering::StoreOp::create(
          rw, loc, old.getSource(), s, off, sizes(rw, packet_shape),
          sizes(rw, SmallVector<int64_t>(rank, 1)));
      setThrottle(store, 32);
      rw.eraseOp(old);
    }
  }
  rw.setInsertionPointToEnd(reorder.getBody());
  auto pipe = ktdf::PipelineOp::create(rw, loc);
  ktdf::PrivateBuilder cp(pipe);
  Value n = needs_dma ? memref::AllocOp::create(cp, loc, memtype).getResult()
                      : external;
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
  auto loaded = cp.createToken(loc), copied = cp.createToken(loc),
       stored = cp.createToken(loc);
  Value dma_done = needs_dma && req.load ? cp.createToken(loc) : Value{};
  rw.setInsertionPointToEnd(pipe.getBody());
  // A local Standard view can be accessed by the reorder stages directly.
  // This also preserves row/packet offsets without a same-memory DMA stage.
  auto normalPosition = [&](OpBuilder& b, Location l, SmallVector<Value> pos) {
    if (!needs_dma)
      for (auto [i, offset] : llvm::enumerate(external_offsets[0]))
        pos[i] = arith::AddIOp::create(b, l, offset, pos[i]);
    return pos;
  };
  auto dma = [&](ValueRange ins, ValueRange outs) {
    ktdf::StageOp::create(rw, loc, ins, outs, [&](OpBuilder& b, Location l) {
      auto zero = arith::ConstantIndexOp::create(b, l, 0);
      auto one = arith::ConstantIndexOp::create(b, l, 1);
      auto two = arith::ConstantIndexOp::create(b, l, 2);
      scf::ForOp::create(b, l, zero, two, one, ValueRange{},
                         [&](OpBuilder& b, Location l, Value p, ValueRange) {
                           SmallVector<Value> pos(rank, zero);
                           pos[packet_axis] = p;
                           auto ext = external_offsets[0];
                           ext[packet_axis] =
                               arith::AddIOp::create(b, l, ext[packet_axis], p);
                           auto packet_shape = shape;
                           packet_shape[packet_axis] = 1;
                           if (req.load)
                             transfer(b, l, external, ext, packet_shape, n, pos,
                                      packet_shape, 32);
                           else
                             transfer(b, l, n, pos, packet_shape, external, ext,
                                      packet_shape, 32);
                           scf::YieldOp::create(b, l);
                         });
    });
  };
  if (dma_done) dma({}, ValueRange{dma_done});
  ktdf::StageOp::create(
      rw, loc, dma_done ? ValueRange{dma_done} : ValueRange{},
      ValueRange{loaded}, [&](OpBuilder& b, Location l) {
        copyLoop(b, l, [&](OpBuilder& b, Value h, Value a, Value p) {
          auto z = arith::ConstantIndexOp::create(b, l, 0);
          auto d0 = getAffineDimExpr(0, b.getContext()),
               d1 = getAffineDimExpr(1, b.getContext());
          auto idx = affine::AffineApplyOp::create(
              b, l,
              AffineMap::get(2, 0,
                             req.load ? d0 * 8 + d1 * 4 : d0 * 16 + d1 * 4),
              req.load ? ValueRange{a, p} : ValueRange{h, a});
          SmallVector<Value> pos(rank, z);
          pos[packet_axis] = req.load ? h : p;
          pos.back() = idx;
          if (req.load) pos = normalPosition(b, l, std::move(pos));
          auto piece = shape;
          piece[packet_axis] = 1;
          piece.back() = 4;
          transfer(b, l, req.load ? n : s, pos, piece, input, {},
                   ArrayRef<int64_t>{4}, 4);
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
  ktdf::StageOp::create(
      rw, loc, ValueRange{copied}, ValueRange{stored},
      [&](OpBuilder& b, Location l) {
        copyLoop(b, l, [&](OpBuilder& b, Value h, Value a, Value p) {
          auto z = arith::ConstantIndexOp::create(b, l, 0);
          auto d0 = getAffineDimExpr(0, b.getContext()),
               d1 = getAffineDimExpr(1, b.getContext());
          auto idx = affine::AffineApplyOp::create(
              b, l,
              AffineMap::get(2, 0,
                             req.load ? d0 * 16 + d1 * 4 : d0 * 8 + d1 * 4),
              req.load ? ValueRange{h, a} : ValueRange{a, p});
          SmallVector<Value> pos(rank, z);
          pos[packet_axis] = req.load ? p : h;
          pos.back() = idx;
          if (!req.load) pos = normalPosition(b, l, std::move(pos));
          auto piece = shape;
          piece[packet_axis] = 1;
          piece.back() = 4;
          transfer(b, l, output, {}, ArrayRef<int64_t>{4}, req.load ? s : n,
                   pos, piece, 4);
        });
      });
  if (needs_dma && !req.load) dma(ValueRange{stored}, {});
  return success();
}
}  // namespace

LogicalResult scheduler::materializeAccessOrders(func::FuncOp func,
                                                 ktdf_arch::Mapping& mapping,
                                                 const AttrMapping& spaces) {
  SmallVector<Request> requests;
  func.walk([&](ktdp::ConstructAccessTilesOp tile) {
    if (tile->use_empty() || tile.getAccessTileOrder().isIdentity()) return;
    llvm::MapVector<Block*, SmallVector<Operation*>> by_block;
    for (auto* use : tile->getUsers()) by_block[use->getBlock()].push_back(use);
    for (auto& [block, accesses] : by_block) {
      auto request = getRequest(tile, accesses);
      if (failed(request))
        continue;  // The existing Bufferize guard diagnoses it.
      requests.push_back(std::move(*request));
    }
  });
  auto local = spaces.map(ktdp::MemorySpaceAttr::get(
      func.getContext(), ktdp::MemorySpaceKind::ct_local, -1));
  IRRewriter rw(func);
  if (requests.empty()) return isolatePacketComputes(func, rw, local);
  Attribute compute;
  for (const auto& kind : mapping.byKind()) {
    auto unit =
        dyn_cast<ktdf_arch::ExecutionUnitOp>(kind.getExemplar().getOperation());
    if (unit && unit.getFeature<ktdf_arch::feature::Compute>()) {
      if (compute)
        return func.emitError("access copy requires one default compute kind");
      compute = unit.getKind();
    }
  }
  if (!compute) return func.emitError("access copy requires a compute route");
  for (auto& r : requests) {
    if (failed(materialize(r, rw, local, compute, spaces))) return failure();
    if (r.tile->use_empty()) rw.eraseOp(r.tile);
  }
  return isolatePacketComputes(func, rw, local);
}
