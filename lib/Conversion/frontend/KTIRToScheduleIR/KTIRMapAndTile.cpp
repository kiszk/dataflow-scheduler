//===-- KTIRMapAndTile.cpp --------------------------------------*- c++ -*-===//
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
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Transforms.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
#include <cstdint>
#include <utility>

#include "RelayoutMaterialization.h"
#include "Utils.h"
#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "dataflow-scheduler/Transforms/Utils/LoopSimplification.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/SpyreOp/SpyreOp.h"

#define PASS_NAME "ktir-map-and-tile"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable KTIR Map and Tile pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTIRMAPANDTILEPASS
#define GEN_PASS_DEF_KTIRNUMERICCASTTILINGPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

const auto kSkipRegions = mlir::OpPrintingFlags().skipRegions();

using I64Vec = llvm::SmallVector<int64_t>;

struct KTIRMapAndTilePass
    : public impl::KTIRMapAndTilePassBase<KTIRMapAndTilePass> {
  using KTIRMapAndTilePassBase<KTIRMapAndTilePass>::KTIRMapAndTilePassBase;

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::ktdf::KTDFDialect,
                    mlir::ktdf_lowering::KTDFLoweringDialect,
                    mlir::ktdp_lowering::KTDPLoweringDialect>();
  }

  void runOnOperation() override;
};

struct KTIRNumericCastTilingPass
    : public impl::KTIRNumericCastTilingPassBase<KTIRNumericCastTilingPass> {
  void runOnOperation() override;
};

struct PacketTransform {
  mlir::Operation* op;
  llvm::SmallVector<mlir::Operation*> scalars;
};

std::optional<PacketTransform> packetTransform(mlir::Operation* op,
                                               bool single_use = true) {
  if (op->hasAttr("ktdf_arch.maps_to") || op->getNumResults() != 1 ||
      (single_use && !op->getResult(0).hasOneUse()))
    return std::nullopt;
  if (getPacketRelayoutDirection(op)) return PacketTransform{op, {}};
  if (mlir::isa<mlir::arith::NegFOp, mlir::math::FloorOp>(op))
    return PacketTransform{op, {op}};
  auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
  if (!generic || generic.getNumDpsInputs() > 1 ||
      generic.getNumDpsInits() != 1 || generic.getRegion().empty() ||
      (generic.getNumDpsInputs() == 1 &&
       generic.getDpsInputs()[0].getType() != generic.getResult(0).getType()) ||
      !llvm::all_of(generic.getIndexingMapsArray(),
                    [](mlir::AffineMap map) { return map.isIdentity(); }) ||
      !llvm::all_of(generic.getIteratorTypesArray(), [](auto iterator) {
        return iterator == mlir::utils::IteratorType::parallel;
      }))
    return std::nullopt;
  auto& body = generic.getRegion().front();
  if (body.getNumArguments() != 1 + generic.getNumDpsInputs())
    return std::nullopt;
  // Fusion may put several unary intrinsics in one generic. Keep the same
  // single-input, same-type chain contract after that transformation.
  PacketTransform unary{op, {}};
  mlir::Value value = body.getArgument(0);
  for (auto& scalar : body.without_terminator()) {
    if (scalar.getNumOperands() != 1 || scalar.getNumResults() != 1 ||
        scalar.getNumRegions() || !mlir::isMemoryEffectFree(&scalar) ||
        scalar.getOperand(0) != value ||
        scalar.getResult(0).getType() != value.getType())
      return std::nullopt;
    unary.scalars.push_back(&scalar);
    value = scalar.getResult(0);
  }
  auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
  if (unary.scalars.empty() || !yield || yield.getOperand(0) != value)
    return std::nullopt;
  return unary;
}

mlir::Value packetTransformInput(const PacketTransform& unary) {
  if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(unary.op))
    return generic.getNumDpsInputs() ? generic.getDpsInputs()[0]
                                     : generic.getDpsInits()[0];
  return unary.op->getOperand(0);
}

mlir::Value applyPacketUnaries(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value value,
                               llvm::ArrayRef<PacketTransform> unaries) {
  for (const auto& unary : unaries) {
    auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
    auto init =
        mlir::tensor::EmptyOp::create(builder, loc, type, mlir::ValueRange{});
    auto identity =
        mlir::AffineMap::getMultiDimIdentityMap(1, builder.getContext());
    auto generic = mlir::linalg::GenericOp::create(
        builder, loc, mlir::TypeRange{type}, mlir::ValueRange{value},
        mlir::ValueRange{init}, llvm::ArrayRef{identity, identity},
        llvm::ArrayRef{mlir::utils::IteratorType::parallel},
        [&](mlir::OpBuilder& body_builder, mlir::Location body_loc,
            mlir::ValueRange args) {
          mlir::Value result = args.front();
          for (auto* scalar : unary.scalars) {
            mlir::OperationState state(body_loc,
                                       scalar->getName().getStringRef());
            state.addOperands(result);
            state.addTypes(type.getElementType());
            state.addAttributes(scalar->getAttrDictionary().getValue());
            result = body_builder.create(state)->getResult(0);
          }
          mlir::linalg::YieldOp::create(body_builder, body_loc, result);
        });
    value = generic.getResult(0);
  }
  return value;
}

// Preserve both packets until bufferization; relayout exchanges four-lane
// groups across them, including when introduced by a scalar cast's maps.
llvm::SmallVector<mlir::Value> relayoutPackets(
    mlir::OpBuilder& builder, mlir::Location loc,
    llvm::SmallVector<mlir::Value> values, bool to_staggered) {
  auto type = mlir::RankedTensorType::get({2, 32}, builder.getF32Type());
  auto empty =
      mlir::tensor::EmptyOp::create(builder, loc, type, mlir::ValueRange{});
  mlir::Value pair = empty;
  auto index = [&](int64_t n) -> mlir::OpFoldResult {
    return builder.getIndexAttr(n);
  };
  for (unsigned p = 0; p < 2; ++p)
    pair = mlir::tensor::InsertSliceOp::create(
        builder, loc, values[p], pair,
        llvm::ArrayRef<mlir::OpFoldResult>{index(p), index(0)},
        llvm::ArrayRef<mlir::OpFoldResult>{index(1), index(32)},
        llvm::ArrayRef<mlir::OpFoldResult>{index(1), index(1)});
  auto generic = mlir::linalg::GenericOp::create(
      builder, loc, mlir::TypeRange{type}, mlir::ValueRange{pair},
      mlir::ValueRange{empty},
      getPacketRelayoutMaps(2, to_staggered, builder.getContext()),
      llvm::ArrayRef{mlir::utils::IteratorType::parallel,
                     mlir::utils::IteratorType::parallel},
      [&](mlir::OpBuilder& b, mlir::Location l, mlir::ValueRange args) {
        auto identity = mlir::spyreop::Identity::create(b, l, args.front());
        mlir::linalg::YieldOp::create(b, l, identity.getResult());
      });
  for (unsigned p = 0; p < 2; ++p)
    values[p] = mlir::tensor::ExtractSliceOp::create(
        builder, loc, mlir::RankedTensorType::get({32}, builder.getF32Type()),
        generic.getResult(0),
        llvm::ArrayRef<mlir::OpFoldResult>{index(p), index(0)},
        llvm::ArrayRef<mlir::OpFoldResult>{index(1), index(32)},
        llvm::ArrayRef<mlir::OpFoldResult>{index(1), index(1)});
  return values;
}

llvm::SmallVector<mlir::Value> applyPacketTransforms(
    mlir::OpBuilder& builder, mlir::Location loc,
    llvm::SmallVector<mlir::Value> values,
    llvm::ArrayRef<PacketTransform> transforms) {
  for (const auto& transform : transforms) {
    if (auto direction = getPacketRelayoutDirection(transform.op)) {
      values = relayoutPackets(builder, loc, std::move(values), *direction);
    } else {
      for (auto& value : values)
        value = applyPacketUnaries(builder, loc, value, {transform});
    }
  }
  return values;
}

/// Keep the physical packet axes while tiling a cast by native sticks.  The
/// packet values have separate SSA results so PipelineBuilder assigns each a
/// 32-element FIFO slot instead of flattening them into a 64-element slot.
auto tileNumericCast(mlir::IRRewriter& rewriter, mlir::Operation* root,
                     mlir::ktdf_arch::Mapping& mapping) -> mlir::LogicalResult {
  auto generic = mlir::cast<mlir::linalg::GenericOp>(root);
  auto cast_op = getNumericCast(root);
  if (cast_op && failed(verifyNumericCastGeneric(generic)))
    return mlir::failure();
  auto root_input = generic.getDpsInputs()[0];
  auto input_type = mlir::cast<mlir::RankedTensorType>(root_input.getType());
  auto output_type =
      mlir::cast<mlir::RankedTensorType>(root->getResult(0).getType());
  const bool widening = cast_op && cast_op.getKind() == "dl16tofp32";
  auto native_type = widening ? input_type : output_type;
  const int64_t rank = native_type.getRank();
  const int64_t packet_axis = rank > 2 ? rank - 3 : 0;
  const int64_t groups_per_row =
      native_type.getDimSize(packet_axis) / (cast_op ? 1 : 2);
  const int64_t rows = rank > 2 ? native_type.getDimSize(rank - 2) : 1;
  auto loc = root->getLoc();
  llvm::SmallVector<PacketTransform> before, after;
  mlir::Value input = root_input;
  while (auto* op = input.getDefiningOp()) {
    auto unary = packetTransform(op);
    if (!unary) break;
    before.push_back(*unary);
    input = packetTransformInput(*unary);
  }
  std::reverse(before.begin(), before.end());
  mlir::Value last = root->getResult(0);
  while (!last.use_empty()) {
    auto* user = *last.user_begin();
    if (!llvm::all_of(last.getUsers(),
                      [&](mlir::Operation* other) { return other == user; }))
      break;
    auto unary = packetTransform(user);
    if (!unary || packetTransformInput(*unary) != last) break;
    after.push_back(*unary);
    last = unary->op->getResult(0);
  }
  rewriter.setInsertionPoint(root);
  auto zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto two = mlir::arith::ConstantIndexOp::create(rewriter, loc, 2);
  auto chunks =
      mlir::arith::ConstantIndexOp::create(rewriter, loc, groups_per_row);
  auto row_count = mlir::arith::ConstantIndexOp::create(rewriter, loc, rows);
  auto init = mlir::tensor::EmptyOp::create(rewriter, loc, output_type,
                                            mlir::ValueRange{});
  auto default_compute = getUniqueComputeUnit(mapping.byKind());
  if (!default_compute)
    return root->emitError("numeric cast requires one default compute kind");
  auto compute = mapping.getOrMap<mlir::ktdf_arch::ExecutionUnitOp>(
      root, default_compute.getKind());
  if (!compute || !compute.getFeature<mlir::ktdf_arch::feature::SIMD>())
    return root->emitError("numeric cast requires a SIMD compute unit");

  auto packet_type = mlir::RankedTensorType::get({widening ? 64 : 32},
                                                 input_type.getElementType());
  auto result_packet_type = mlir::RankedTensorType::get(
      {output_type.getElementType().isF32() ? 32 : 64},
      output_type.getElementType());
  llvm::SmallVector<mlir::OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  llvm::SmallVector<mlir::Value> batch_coords(packet_axis, zero);
  auto packet_slice = [&](mlir::Value packet, mlir::Value row, int64_t lanes) {
    llvm::SmallVector<mlir::OpFoldResult> offsets(rank,
                                                  rewriter.getIndexAttr(0));
    for (int64_t d = 0; d < packet_axis; ++d) offsets[d] = batch_coords[d];
    offsets[packet_axis] = packet;
    if (rank > 2) offsets[rank - 2] = row;
    auto sizes = strides;
    sizes.back() = rewriter.getIndexAttr(lanes);
    return std::pair{std::move(offsets), std::move(sizes)};
  };
  auto slice = [&](mlir::OpBuilder& builder, mlir::Value source,
                   mlir::Value packet, mlir::Value row,
                   mlir::RankedTensorType type) -> mlir::Value {
    auto [offsets, sizes] = packet_slice(packet, row, type.getDimSize(0));
    return mlir::tensor::ExtractSliceOp::create(builder, loc, type, source,
                                                offsets, sizes, strides);
  };
  auto insert = [&](mlir::OpBuilder& builder, mlir::Value packet_value,
                    mlir::Value output, mlir::Value packet,
                    mlir::Value row) -> mlir::Value {
    auto [offsets, sizes] =
        packet_slice(packet, row, result_packet_type.getDimSize(0));
    return mlir::tensor::InsertSliceOp::create(builder, loc, packet_value,
                                               output, offsets, sizes, strides);
  };
  auto cast_row = [&](mlir::OpBuilder& builder, mlir::Value chunk,
                      mlir::Value row, mlir::Value output) -> mlir::Value {
    auto first = mlir::arith::MulIOp::create(builder, loc, chunk, two);
    auto second = mlir::arith::AddIOp::create(builder, loc, first, one);
    llvm::SmallVector<mlir::Value> inputs;
    if (widening)
      inputs.push_back(slice(builder, input, chunk, row, packet_type));
    else {
      inputs.push_back(slice(builder, input, first, row, packet_type));
      inputs.push_back(slice(builder, input, second, row, packet_type));
    }
    inputs = applyPacketTransforms(builder, loc, std::move(inputs), before);
    llvm::SmallVector<mlir::Value> outputs;
    if (cast_op) {
      mlir::OperationState state(loc, "ktdf_lowering.numeric_cast_value");
      state.addOperands(inputs);
      state.addTypes(
          llvm::SmallVector<mlir::Type>(widening ? 2 : 1, result_packet_type));
      state.addAttribute("kind", builder.getStringAttr(cast_op.getKind()));
      auto* conversion = builder.create(state);
      std::ignore = mlir::ktdf_arch::Mappable::setMapsTo(
          conversion, mlir::ktdf_arch::Mappable::getMapsTo(root));
      llvm::append_range(outputs, conversion->getResults());
    } else {
      outputs =
          applyPacketTransforms(builder, loc, std::move(inputs), {{root, {}}});
    }
    outputs = applyPacketTransforms(builder, loc, std::move(outputs), after);
    if (outputs.size() == 2) {
      output = insert(builder, outputs[0], output, first, row);
      return insert(builder, outputs[1], output, second, row);
    }
    return insert(builder, outputs[0], output, chunk, row);
  };
  int cores = 1;
  auto func = root->getParentOfType<mlir::func::FuncOp>();
  // A frontend may supply per-core tensors and explicit core-dependent memory
  // offsets. Its work must not be partitioned a second time by this tiling.
  if (!root->hasAttr("dataflow_scheduler.work_divided") &&
      func->hasAttr("grid") && failed(extractGridSize(func, cores)))
    return mlir::failure();
  // Reuse the row/chunk partition for each leading batch coordinate. Keep
  // each head/middle/tail segment as a separate nest, so the ordinary store
  // tiling can sink its updates before pipeline construction.
  auto batches = [&](mlir::OpBuilder& builder, mlir::Value initial, auto emit) {
    auto visit = [&](auto&& self, mlir::OpBuilder& b, int64_t axis,
                     mlir::Value carried) -> mlir::Value {
      if (axis == packet_axis) return emit(b, carried);
      auto bound = mlir::arith::ConstantIndexOp::create(
          b, loc, native_type.getDimSize(axis));
      return mlir::scf::ForOp::create(
                 b, loc, zero, bound, one, mlir::ValueRange{carried},
                 [&](mlir::OpBuilder& inner, mlir::Location, mlir::Value iv,
                     mlir::ValueRange values) {
                   batch_coords[axis] = iv;
                   mlir::scf::YieldOp::create(
                       inner, loc, self(self, inner, axis + 1, values.front()));
                 })
          .getResult(0);
    };
    return visit(visit, builder, 0, initial);
  };
  auto row_loop = [&](mlir::OpBuilder& b, mlir::Value chunk, mlir::Value lb,
                      mlir::Value ub, mlir::Value output) {
    return mlir::scf::ForOp::create(
               b, loc, lb, ub, one, mlir::ValueRange{output},
               [&](mlir::OpBuilder& rb, mlir::Location, mlir::Value row,
                   mlir::ValueRange carried) {
                 mlir::scf::YieldOp::create(
                     rb, loc, cast_row(rb, chunk, row, carried.front()));
               })
        .getResult(0);
  };
  auto chunk_loop = [&](mlir::OpBuilder& b, mlir::Value lb, mlir::Value ub,
                        mlir::Value output) {
    return mlir::scf::ForOp::create(
               b, loc, lb, ub, one, mlir::ValueRange{output},
               [&](mlir::OpBuilder& cb, mlir::Location, mlir::Value chunk,
                   mlir::ValueRange carried) {
                 auto result = rank == 2
                                   ? cast_row(cb, chunk, zero, carried.front())
                                   : row_loop(cb, chunk, zero, row_count,
                                              carried.front());
                 mlir::scf::YieldOp::create(cb, loc, result);
               })
        .getResult(0);
  };
  mlir::Value result;
  auto& builder = rewriter;
  if (cores == 1) {
    result = batches(builder, init,
                     [&](mlir::OpBuilder& builder, mlir::Value batch_init) {
                       return chunk_loop(builder, zero, chunks, batch_init);
                     });
  } else {
    // Own complete 64-element groups, including both packets and every phase.
    // Only the core-uniform bounds use floor/mod. Address-generator loops keep
    // linear row/chunk coordinates, including a partial first and last chunk.
    auto core = mlir::ktdp::GetComputeTileIdOp::create(builder, loc,
                                                       builder.getIndexType())
                    .getResult()[0];
    auto count = mlir::arith::ConstantIndexOp::create(builder, loc, cores);
    auto groups = mlir::arith::ConstantIndexOp::create(builder, loc,
                                                       groups_per_row * rows);
    auto next = mlir::arith::AddIOp::create(builder, loc, core, one);
    auto begin = mlir::arith::DivUIOp::create(
        builder, loc, mlir::arith::MulIOp::create(builder, loc, core, groups),
        count);
    auto end = mlir::arith::DivUIOp::create(
        builder, loc, mlir::arith::MulIOp::create(builder, loc, next, groups),
        count);
    if (rank == 2) {
      result = chunk_loop(builder, begin, end, init);
    } else {
      auto first = mlir::arith::DivUIOp::create(builder, loc, begin, row_count);
      auto last = mlir::arith::DivUIOp::create(builder, loc, end, row_count);
      auto first_row =
          mlir::arith::RemUIOp::create(builder, loc, begin, row_count);
      auto last_row =
          mlir::arith::RemUIOp::create(builder, loc, end, row_count);
      auto first_end = mlir::arith::MinUIOp::create(
          builder, loc, row_count,
          mlir::arith::SubIOp::create(
              builder, loc, end,
              mlir::arith::MulIOp::create(builder, loc, first, row_count)));
      result = batches(builder, init, [&](mlir::OpBuilder& b, mlir::Value out) {
        return row_loop(b, first, first_row, first_end, out);
      });
      auto middle_begin = mlir::arith::AddIOp::create(builder, loc, first, one);
      result =
          batches(builder, result,
                  [&](mlir::OpBuilder& batch_builder, mlir::Value out) {
                    return chunk_loop(batch_builder, middle_begin, last, out);
                  });
      auto has_tail = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::ugt, last, first);
      auto tail_end =
          mlir::arith::SelectOp::create(builder, loc, has_tail, last_row, zero);
      result =
          batches(builder, result, [&](mlir::OpBuilder& b, mlir::Value out) {
            return row_loop(b, last, zero, tail_end, out);
          });
    }
  }
  if (after.empty())
    rewriter.replaceOp(root, result);
  else {
    rewriter.replaceAllUsesWith(last, result);
    for (auto unary : llvm::reverse(after)) rewriter.eraseOp(unary.op);
    rewriter.eraseOp(root);
  }
  for (auto unary : llvm::reverse(before)) rewriter.eraseOp(unary.op);
  return mlir::success();
}

void KTIRNumericCastTilingPass::runOnOperation() {
  auto func = getOperation();
  for (auto generic : func.getOps<mlir::linalg::GenericOp>()) {
    if (!generic.getBody()->getOps<mlir::spyreop::Identity>().empty() &&
        llvm::any_of(generic.getIndexingMapsArray(),
                     [](auto map) {
                       return !map.isProjectedPermutation(
                           /*allowZeroInResults=*/true);
                     }) &&
        !getPacketRelayoutDirection(generic)) {
      generic.emitError(
          "no target packet relayout recipe for these indexing_maps and tensor "
          "shape");
      return signalPassFailure();
    }
  }
  bool has_packet_op =
      llvm::any_of(func.getOps<mlir::linalg::GenericOp>(), [](auto op) {
        return getNumericCast(op) || getPacketRelayoutDirection(op).has_value();
      });
  if (!has_packet_op) return;
  auto& default_device = getAnalysis<mlir::ktdf_arch::DefaultDevice>();
  if (!default_device) return signalPassFailure();
  mlir::ktdf_arch::Mapping mapping(default_device.getRef());
  mlir::IRRewriter rewriter(func);
  // The current FIFO route has one consumer per packet. Specialize pure
  // tensor producers per use before tiling, so independently ordered stores
  // never share a FIFO read or overwrite a sibling use's intermediate value.
  // This intentionally trades recomputation for a valid ordinary pipeline.
  auto eligible = [](mlir::Operation* op) {
    return op->getNumResults() == 1 &&
           mlir::isa<mlir::RankedTensorType>(op->getResult(0).getType()) &&
           (getNumericCast(op) || packetTransform(op, false));
  };
  const bool specialize =
      llvm::any_of(func.getBody().front(), [&](mlir::Operation& op) {
        return eligible(&op) && !op.getResult(0).hasOneUse() &&
               !op.getResult(0).use_empty();
      });
  if (specialize) {
    // Tiling rematerializes loads. Only distinct address arguments provide the
    // scheduler's non-alias guarantee; derived views need a preserved snapshot.
    auto address = [&](mlir::Value access) -> mlir::Value {
      auto tile = access.getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
      auto view = tile ? tile.getBase()
                             .getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>()
                       : nullptr;
      auto arg = view ? mlir::dyn_cast<mlir::BlockArgument>(view.getOffset())
                      : mlir::BlockArgument{};
      return arg && arg.getOwner() == &func.getBody().front() ? arg
                                                              : mlir::Value{};
    };
    llvm::SmallVector<mlir::Value> inputs, outputs;
    bool safe = true;
    for (auto& op : func.getBody().front()) {
      if (auto load = mlir::dyn_cast<mlir::ktdp::LoadOp>(op)) {
        auto addr = address(load.getAccessTile());
        safe &= bool(addr);
        inputs.push_back(addr);
      } else if (auto store = mlir::dyn_cast<mlir::ktdp::StoreOp>(op)) {
        auto addr = address(store.getAccessTile());
        safe &= bool(addr) && !llvm::is_contained(outputs, addr);
        outputs.push_back(addr);
      } else if (!op.hasTrait<mlir::OpTrait::IsTerminator>() &&
                 !mlir::isMemoryEffectFree(&op)) {
        safe = false;
      }
    }
    if (!safe || llvm::any_of(outputs, [&](mlir::Value addr) {
          return llvm::is_contained(inputs, addr);
        })) {
      func.emitError(
          "branched numeric cast requires distinct, unmodified input/output "
          "address arguments");
      return signalPassFailure();
    }
  }
  auto operations =
      llvm::map_to_vector(func.getBody().front().without_terminator(),
                          [](mlir::Operation& op) { return &op; });
  for (auto* op : llvm::reverse(operations)) {
    if (!specialize || !eligible(op)) continue;
    auto uses = llvm::map_to_vector(op->getResult(0).getUses(),
                                    [](mlir::OpOperand& use) { return &use; });
    if (uses.empty()) continue;
    if (llvm::any_of(uses, [&](mlir::OpOperand* use) {
          return use->getOwner()->getBlock() != op->getBlock();
        })) {
      op->emitError("branched numeric cast requires same-block tensor uses");
      return signalPassFailure();
    }
    llvm::sort(uses, [](mlir::OpOperand* lhs, mlir::OpOperand* rhs) {
      return lhs->getOwner() == rhs->getOwner()
                 ? lhs->getOperandNumber() < rhs->getOperandNumber()
                 : lhs->getOwner()->isBeforeInBlock(rhs->getOwner());
    });
    for (auto* use : llvm::drop_begin(uses)) {
      rewriter.setInsertionPoint(use->getOwner());
      use->set(rewriter.clone(*op)->getResult(0));
    }
    // Keep each specialized chain next to its consumer. Once tiled, another
    // chain is effectful and a store cannot sink across its loop nest.
    rewriter.moveOpBefore(op, uses.front()->getOwner());
  }
  auto tile_all = [&](auto matches) -> mlir::LogicalResult {
    while (true) {
      auto generics = func.getOps<mlir::linalg::GenericOp>();
      auto it = llvm::find_if(generics, matches);
      if (it == generics.end()) return mlir::success();
      if (failed(tileNumericCast(rewriter, *it, mapping)))
        return mlir::failure();
    }
  };
  // Tile casts first so they can absorb adjacent transforms. Remaining
  // relayouts use the same group ownership and batch/row tiling.
  if (failed(tile_all([](auto op) { return bool(getNumericCast(op)); })) ||
      failed(tile_all(
          [](auto op) { return getPacketRelayoutDirection(op).has_value(); })))
    signalPassFailure();
}

/// Given a shape and a target vector_length, compute per-dimension tile sizes
/// working rightmost-first until the product of covered dims reaches
/// vector_length.  Uncovered dims (leftmost) get tile size 1 so a loop is
/// generated for them; covered dims get the largest value that divides both
/// the dim size and the remaining vector-length product.
[[nodiscard]] auto determineTileSizes(
    llvm::ArrayRef<int64_t> shape, int64_t vector_length,
    llvm::SmallVectorImpl<int64_t>& tile_sizes) -> llvm::LogicalResult {
  const int64_t rank = shape.size();

  // Find how many rightmost dims are needed to cover vector_length.
  int64_t product = 1;
  unsigned covered_dims = 0;
  while (covered_dims < shape.size()) {
    const auto size = shape[shape.size() - covered_dims - 1];
    if (mlir::ShapedType::isDynamic(size)) {
      // There weren't enough statically known dims to cover the vector size.
      return llvm::failure();
    }

    product *= size;
    covered_dims++;
    if (product >= vector_length) {
      break;
    }
  }

  // Uncovered dims (leftmost) get tile size 1 — a loop is generated that
  // iterates over the full extent one element at a time.
  tile_sizes.resize(rank, 1);

  // Covered dims get a size that divides both the dim and the remaining
  // product.
  int64_t remaining = vector_length;
  for (int64_t i = rank - 1; i >= rank - covered_dims; --i) {
    const auto size = std::min(std::gcd(shape[i], remaining), shape[i]);
    tile_sizes[i] = size;
    remaining = std::max<int64_t>(remaining / size, 1);
  }

  return llvm::success();
}

/// Maps @p op to the default compute resource, unless it already is, and
/// annotates the vector width for @p element_type on it as the throttle.
///
/// @return The vector width.
auto mapToCompute(mlir::linalg::LinalgOp op, mlir::ktdf_arch::Mapping& mapping,
                  mlir::Type element_type) -> int64_t {
  auto compute =
      mapping.getOrMap(op, mapping.byKind().getDefaultCompute().getKind());
  assert(compute && "no default compute resource");

  LDBG() << "  compute: " << mlir::OpWithFlags(compute, kSkipRegions);

  // Determine the desired vector width on the compute resource.
  const auto vector_length = getVectorLanes(
      element_type,
      mlir::cast<mlir::ktdf_arch::ExecutionUnitOp>(compute.getOperation()));

  // Annotate the vector width as the throttle for this operation.
  setThrottle(op, vector_length);
  return vector_length;
}

auto determineTileSizes(mlir::linalg::LinalgOp op,
                        mlir::ktdf_arch::Mapping& mapping,
                        llvm::SmallVectorImpl<int64_t>& tile_sizes)
    -> llvm::LogicalResult {
  if (!op.hasPureTensorSemantics() || op->getNumResults() == 0) {
    return op->emitError("tiling error: expected pure tensor semantics");
  }
  const auto result_type =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!llvm::all_equal(op->getResultTypes())) {
    op.emitWarning("tiling heterogeneous results is not supported");
  }

  LDBG() << "determining tile sizes for "
         << mlir::OpWithFlags(op, kSkipRegions);

  // Map the op to a compute resource, if it isn't already.
  const auto vector_length =
      mapToCompute(op, mapping, result_type.getElementType());

  // The tile-sizes vector must cover every loop dimension (getNumLoops()).
  // Reduction dimensions are skipped (tile size = 0) so the tiling infra does
  // not create a loop for them.  Only parallel dimensions are tiled, using the
  // same rightmost-first shape-based algorithm as before.
  // Collect the loop bounds for each dimension by querying the op directly.
  // getStaticLoopRanges() scans all operand indexing maps so it works
  // regardless of whether any single operand has a full-rank map (e.g. when
  // an input has a broadcast/projection indexing map that skips some dims).
  const auto loop_ranges = op.getStaticLoopRanges();
  I64Vec parallel_loop_indices;
  I64Vec parallel_shape;
  for (auto [index, type] : llvm::enumerate(op.getIteratorTypesArray())) {
    if (type == mlir::utils::IteratorType::parallel) {
      parallel_loop_indices.push_back(index);
      parallel_shape.push_back(loop_ranges[index]);
    }
  }

  LDBG_OS([&](llvm::raw_ostream& os) {
    os << "  parallel_shape: {";
    llvm::interleaveComma(parallel_shape, os);
    os << "}";
  });

  // Compute tile sizes for the parallel dimensions.
  I64Vec parallel_tile_sizes;
  if (failed(::determineTileSizes(parallel_shape, vector_length,
                                  parallel_tile_sizes))) {
    return op->emitError("tiling error: inner dimensions not tileable");
  }

  // Assemble the final vector: 0 for reduction dims, computed size for
  // parallel dims.
  tile_sizes.resize(loop_ranges.size(), 0);
  for (int64_t k = 0; k < static_cast<int64_t>(parallel_loop_indices.size());
       ++k) {
    tile_sizes[parallel_loop_indices[k]] = parallel_tile_sizes[k];
  }

  LDBG_OS([&](llvm::raw_ostream& os) {
    os << "  tile_sizes: {";
    llvm::interleaveComma(tile_sizes, os);
    os << "}";
  });

  return llvm::success();
}

auto compressInput(mlir::RewriterBase& rewriter, mlir::linalg::LinalgOp op,
                   mlir::OpOperand* input) -> llvm::LogicalResult {
  const auto type =
      llvm::dyn_cast<mlir::RankedTensorType>(input->get().getType());
  const auto bcast_dims = op.getMatchingIndexingMap(input).getBroadcastDims();
  if (!type || bcast_dims.empty()) {
    return llvm::failure();
  }

  const auto zero = rewriter.getI64IntegerAttr(0);
  const auto one = rewriter.getI64IntegerAttr(1);

  llvm::SmallVector<mlir::Range> ranges;
  ranges.reserve(type.getRank());
  for (auto dim : llvm::iota_range<int64_t>(0, type.getRank(), false)) {
    const auto size =
        mlir::tensor::getMixedSize(rewriter, op.getLoc(), input->get(), dim);
    ranges.push_back({zero, size, one});
  }

  for (auto dim : bcast_dims) {
    ranges[dim] = {zero, one, zero};
  }

  input->set(mlir::tensor::ExtractSliceOp::create(rewriter, op.getLoc(),
                                                  input->get(), ranges));
  return llvm::success();
}

auto compressInputs(mlir::RewriterBase& rewriter, mlir::linalg::LinalgOp op)
    -> llvm::LogicalResult {
  rewriter.startOpModification(op);
  auto changed = false;

  for (auto* const input : op.getDpsInputOperands()) {
    changed |= succeeded(compressInput(rewriter, op, input));
  }

  if (!changed) {
    rewriter.cancelOpModification(op);
    return llvm::failure();
  }

  LDBG() << "compressed " << mlir::OpWithFlags(op, kSkipRegions);
  rewriter.finalizeOpModification(op);
  return llvm::success();
}

auto tile(mlir::RewriterBase& rewriter, mlir::linalg::LinalgOp& op,
          mlir::ktdf_arch::Mapping& mapping)
    -> llvm::FailureOr<mlir::scf::LoopVector> {
  rewriter.setInsertionPoint(op);

  // Remove needless dependencies on broadcasted dimensions.
  std::ignore = compressInputs(rewriter, op);

  // Determine tile sizes from output operand shape (needed for loop creation).
  I64Vec tile_sizes;
  if (failed(determineTileSizes(op, mapping, tile_sizes))) {
    return llvm::failure();
  }
  if (tile_sizes.empty()) {
    op.emitError("tiling error: no tileable dimensions");
    return llvm::failure();
  }

  // Tile the linalg.generic operation with the computed tile sizes
  mlir::linalg::LinalgTilingOptions tiling_options;
  tiling_options.setTileSizes(tile_sizes);
  tiling_options.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);
  const auto maybe_tiling = tileLinalgOp(rewriter, op, tiling_options);
  if (failed(maybe_tiling)) {
    op->emitError("tiling error: unable to create loops");
    return llvm::failure();
  }

  // Annotate the created loops with their iterator type.
  {
    unsigned loop_index = 0;
    const auto iterator_types = op.getIteratorTypesArray();
    for (auto [dim, tile_size] : llvm::enumerate(tile_sizes)) {
      if (tile_size != 0) {
        // FIXME: Put the attribute name somewhere else.
        maybe_tiling->loops[loop_index++]->setAttr(
            "loop_type",
            getLoopTypeAttr(rewriter.getContext(), iterator_types[dim]));
      }
    }
  }

  // Replace the old with the tiled op.
  maybe_tiling->op->setDiscardableAttrs(op->getRawDictionaryAttrs());
  rewriter.replaceOp(op, maybe_tiling->tensorResults);
  op = maybe_tiling->op;

  return llvm::map_to_vector(maybe_tiling->loops, [](mlir::Operation* op) {
    return llvm::cast<mlir::scf::ForOp>(op);
  });
}

void bubbleUpExtractSlice(mlir::RewriterBase& rewriter,
                          mlir::tensor::ExtractSliceOp op,
                          const AttrMapping& mem_space_map) {
  while (auto via = op.getSource().getDefiningOp<mlir::ktdf::ViaOp>()) {
    rewriter.modifyOpInPlace(op, [&]() { op.setOperand(0, via.getOperand()); });
    rewriter.modifyOpInPlace(via, [&]() {
      via.getResult().setType(op.getType());
      via.setOperand(op);
    });
    rewriter.replaceAllUsesExcept(op, via, via);
    rewriter.moveOpBefore(op, via);
  }

  if (auto load = op.getSource().getDefiningOp<mlir::ktdp_lowering::LoadOp>();
      load) {
    rewriter.modifyOpInPlace(op, [&]() {
      std::ignore = mlir::ktdf_arch::Mappable::setMapsTo(
          op, mlir::ktdf_arch::MapsToAttr::get(
                  rewriter.getContext(),
                  {llvm::cast<mlir::ktdf_arch::ResourceSpecAttr>(
                      mem_space_map.map(getMemorySpace(load.getSource())))}));
    });
  }
}

[[nodiscard]] auto isBroadcast(mlir::Value value, int64_t dim) -> bool {
  while (auto via = value.getDefiningOp<mlir::ktdf::ViaOp>()) {
    value = via.getOperand();
  }
  auto slice = value.getDefiningOp<mlir::OffsetSizeAndStrideOpInterface>();
  if (!slice) {
    return false;
  }
  const auto strides = slice.getMixedStrides();
  if (static_cast<int64_t>(strides.size()) !=
      llvm::cast<mlir::ShapedType>(value.getType()).getRank()) {
    return false;
  }
  return mlir::isZeroInteger(strides[dim]) &&
         !mlir::isConstantIntValue(slice.getMixedSizes()[dim], 1);
}

[[nodiscard]] auto getReassociationIndices(
    mlir::Operation* op,
    llvm::SmallVectorImpl<mlir::ReassociationIndices>& groups,
    bool& is_collapse) -> mlir::LogicalResult {
  if (auto collapse = llvm::dyn_cast<mlir::tensor::CollapseShapeOp>(op);
      collapse) {
    groups = collapse.getReassociationIndices();
    is_collapse = true;
    return llvm::success();
  }

  if (auto expand = llvm::dyn_cast<mlir::tensor::ExpandShapeOp>(op); expand) {
    groups = expand.getReassociationIndices();
    is_collapse = false;
    return llvm::success();
  }

  return llvm::failure();
}

/// Bubbles `tensor.extract_slice` up through reshaping operations.
struct BubbleUpSliceThroughReshape
    : mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::tensor::ExtractSliceOp slice,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto* reshape = slice.getSource().getDefiningOp();
    bool is_collapse;
    llvm::SmallVector<mlir::ReassociationIndices> groups;
    // A reshape of static result has a static source, and one without groups
    // is from or to rank 0, where a sliced dimension has nowhere to go.
    const auto extents = slice.getSourceType().getShape();
    if (!reshape ||
        failed(getReassociationIndices(reshape, groups, is_collapse)) ||
        groups.empty() || !slice.getSizes().empty() ||
        !slice.getStrides().empty() ||
        slice.getResultType().getRank() != slice.getSourceType().getRank() ||
        !slice.getSourceType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(slice, "not a slice of a reshape");
    }

    auto sizes = llvm::to_vector(
        llvm::cast<mlir::ShapedType>(reshape->getOperand(0).getType())
            .getShape());
    llvm::SmallVector<mlir::OpFoldResult> offsets(sizes.size(),
                                                  rewriter.getIndexAttr(0));
    llvm::SmallVector<int64_t> strides(sizes.size(), 1);
    llvm::SmallVector<bool> sliced(sizes.size(), false);
    for (auto [index, group] : llvm::enumerate(groups)) {
      const mlir::ReassociationIndices single{static_cast<int64_t>(index)};
      const auto& target = is_collapse ? group : single;
      for (auto from : is_collapse ? single : group) {
        const auto size = slice.getStaticSizes()[from];
        const auto stride = slice.getStaticStrides()[from];
        if (size == extents[from] && (stride == 1 || extents[from] == 1)) {
          continue;
        }
        const auto to = llvm::find_if(llvm::reverse(target), [&](int64_t dim) {
          return !sliced[dim] && sizes[dim] == extents[from];
        });
        if (to == target.rend()) {
          return rewriter.notifyMatchFailure(slice, "unsupported reshape");
        }
        std::tie(offsets[*to], sizes[*to], strides[*to]) =
            std::tuple(slice.getMixedOffsets()[from], size, stride);
        sliced[*to] = true;
      }
    }

    // Slice the source, then reshape with the same op unless it is shared.
    const auto type = slice.getResultType();
    if (!reshape->hasOneUse()) {
      reshape = rewriter.clone(*reshape);
    }
    rewriter.modifyOpInPlace(slice, [&]() {
      const auto [static_offsets, dynamic_offsets] =
          mlir::decomposeMixedValues(offsets);
      slice.getSourceMutable().assign(reshape->getOperand(0));
      slice.getOffsetsMutable().assign(dynamic_offsets);
      slice.setStaticOffsets(static_offsets);
      slice.setStaticSizes(sizes);
      slice.setStaticStrides(strides);
      slice.getResult().setType(type.clone(sizes));
    });
    rewriter.replaceAllUsesWith(slice.getResult(), reshape->getResult(0));
    rewriter.moveOpAfter(reshape, slice);
    rewriter.modifyOpInPlace(reshape, [&]() {
      reshape->setOperand(0, slice.getResult());
      reshape->getResult(0).setType(type);
      if (auto expanded =
              llvm::dyn_cast<mlir::tensor::ExpandShapeOp>(reshape)) {
        expanded.setStaticOutputShape(type.getShape());
      }
    });
    return llvm::success();
  }
};

/// Index the innermost dimension of @p operand by the innermost loop rather
/// than by the constant that addressed its single element.
///
/// The map carries that constant on entry because it is how the front end
/// expresses a broadcast, and what finds the dimension to widen. It stops being
/// accurate once the operand is widened by a stride-0 slice, which replicates
/// that element across the dimension, so this is only to be called right after
/// such a widening. The innermost loop must also be what the dimension spans: a
/// parallel loop whose extent is the dimension's size, and that no other
/// dimension of the operand is indexed by. Otherwise the constant is kept,
/// which reads the same element.
///
/// @return Whether the indexing map changed.
[[nodiscard]] auto legalizeBroadcastMap(mlir::linalg::GenericOp op,
                                        mlir::OpOperand* operand) -> bool {
  const auto type =
      llvm::dyn_cast<mlir::RankedTensorType>(operand->get().getType());
  const auto map = op.getMatchingIndexingMap(operand);
  if (!type || map.getNumDims() == 0 || map.getNumResults() == 0 ||
      map.getNumResults() != type.getRank() ||
      !llvm::isa<mlir::AffineConstantExpr>(map.getResults().back())) {
    return false;
  }

  const auto loop = map.getNumDims() - 1;
  if (op.getIteratorTypesArray()[loop] != mlir::utils::IteratorType::parallel ||
      op.getStaticLoopRanges()[loop] != type.getShape().back()) {
    return false;
  }
  if (llvm::any_of(map.getResults().drop_back(), [&](mlir::AffineExpr expr) {
        return expr.isFunctionOfDim(loop);
      })) {
    return false;
  }

  auto results = llvm::to_vector(map.getResults());
  results.back() = mlir::getAffineDimExpr(loop, op.getContext());
  auto maps = op.getIndexingMapsArray();
  maps[operand->getOperandNumber()] = mlir::AffineMap::get(
      map.getNumDims(), map.getNumSymbols(), results, op.getContext());
  op.setIndexingMapsAttr(
      mlir::Builder(op.getContext()).getAffineMapArrayAttr(maps));
  return true;
}

struct BroadcastInputs : mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  explicit BroadcastInputs(mlir::MLIRContext* context,
                           const AttrMapping& mem_space_map)
      : OpRewritePattern(context), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(mlir::linalg::GenericOp op,
                       mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult override {
    const auto throttle_attr =
        op->getAttrOfType<mlir::ktdf_arch::I64Attr>(kThrottleAttrName);
    if (!throttle_attr) {
      return rewriter.notifyMatchFailure(op, "missing throttle attribute");
    }
    const auto vector_lanes = throttle_attr.getValue();
    const auto zero = rewriter.getIndexAttr(0);
    const auto one = rewriter.getIndexAttr(1);

    rewriter.startOpModification(op);
    auto changed = false;
    for (auto* const input : op.getDpsInputOperands()) {
      auto type =
          llvm::dyn_cast<mlir::RankedTensorType>(input->get().getType());
      const auto map = op.getMatchingIndexingMap(input);
      if (!type || map.getNumDims() == 0) {
        continue;
      }

      // An operand whose map does not involve the innermost, vectorized loop is
      // broadcast along a unit dimension that is added, unless its innermost
      // dimension already is one. This is a rank-0 operand, or one whose
      // trailing unit dimension was dropped.
      auto bcast_dims = map.getBroadcastDims();
      const auto add_unit_dim =
          !map.isFunctionOfDim(map.getNumDims() - 1) &&
          (bcast_dims.empty() || bcast_dims.back() != type.getRank() - 1);
      if (add_unit_dim) {
        bcast_dims.push_back(type.getRank());
      }
      const auto rank = type.getRank() + (add_unit_dim ? 1 : 0);
      if (bcast_dims.empty() || bcast_dims.back() != rank - 1 ||
          (!add_unit_dim && isBroadcast(input->get(), bcast_dims.back()))) {
        continue;
      }

      int64_t elements = 1;
      for (auto [dim, size] : llvm::enumerate(type.getShape())) {
        if (llvm::is_contained(bcast_dims, dim)) {
          continue;
        }
        if (mlir::ShapedType::isDynamic(size) ||
            llvm::MulOverflow(elements, size, elements) != 0) {
          elements = 0;
          break;
        }
      }

      // The operand fills one vector, so only the lanes the other dimensions
      // leave are spread along the innermost broadcast dimension. A broadcast
      // outside the lanes keeps size 1.
      if (elements <= 0 || vector_lanes % elements != 0) {
        continue;
      }
      const auto fill = vector_lanes / elements;
      if (fill == 1) {
        continue;
      }
      // Already broadcast, possibly by a slice that was bubbled up past a
      // reshape and is therefore not seen by `isBroadcast`.
      if (!add_unit_dim && type.getDimSize(bcast_dims.back()) == fill) {
        continue;
      }

      if (add_unit_dim) {
        llvm::SmallVector<int64_t> shape(type.getShape());
        shape.push_back(1);
        llvm::SmallVector<mlir::ReassociationIndices> reassociation;
        for (auto dim : llvm::seq<int64_t>(type.getRank())) {
          reassociation.push_back({dim});
        }
        if (!reassociation.empty()) {
          reassociation.back().push_back(type.getRank());
        }
        auto expand = mlir::tensor::ExpandShapeOp::create(
            rewriter, op.getLoc(),
            mlir::RankedTensorType::get(shape, type.getElementType()),
            input->get(), reassociation);
        input->set(expand);
        type = expand.getResultType();

        auto maps = op.getIndexingMapsArray();
        auto& expanded = maps[input->getOperandNumber()];
        expanded = expanded.insertResult(rewriter.getAffineConstantExpr(0),
                                         expanded.getNumResults());
        op.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
      }

      auto sizes =
          mlir::tensor::getMixedSizes(rewriter, op.getLoc(), input->get());
      llvm::SmallVector<mlir::OpFoldResult> offsets(type.getRank(), zero);
      llvm::SmallVector<mlir::OpFoldResult> strides(type.getRank(), one);
      for (auto dim : bcast_dims) {
        sizes[dim] = one;
        strides[dim] = zero;
      }
      sizes[bcast_dims.back()] = rewriter.getIndexAttr(fill);

      auto broadcast = mlir::tensor::ExtractSliceOp::create(
          rewriter, op.getLoc(), input->get(), offsets, sizes, strides);
      input->set(broadcast);
      bubbleUpExtractSlice(rewriter, broadcast, mem_space_map_);
      std::ignore = legalizeBroadcastMap(op, input);
      changed = true;
    }

    if (!changed) {
      rewriter.cancelOpModification(op);
      return rewriter.notifyMatchFailure(op, "nothing to broadcast");
    }

    LDBG() << "broadcasted " << mlir::OpWithFlags(op, kSkipRegions);
    rewriter.finalizeOpModification(op);
    return llvm::success();
  }

 private:
  const AttrMapping& mem_space_map_;
};

}  // namespace

void KTIRMapAndTilePass::runOnOperation() {
  if (disable_this_pass) {
    return;
  }

  mlir::func::FuncOp func = getOperation();
  mlir::IRRewriter rewriter(func);

  // Compute operations already nested in loops (e.g. by
  // IndirectAccessLoopMaterialization) are mapped but not tiled.
  auto computes = llvm::to_vector(func.getOps<mlir::linalg::LinalgOp>());
  llvm::SmallVector<mlir::linalg::LinalgOp> nested_computes;
  func.walk([&](mlir::linalg::LinalgOp op) {
    if (op->getParentOp() != func) {
      nested_computes.push_back(op);
    }
  });
  bool has_numeric_cast = false;
  func.walk([&](mlir::ktdf_lowering::NumericCastValueOp) {
    has_numeric_cast = true;
  });
  if (computes.empty() && nested_computes.empty() && !has_numeric_cast) {
    return;
  }

  // Obtain the default device and start a mapping.
  auto& default_device = getAnalysis<mlir::ktdf_arch::DefaultDevice>();
  if (!default_device) {
    signalPassFailure();
    return;
  }
  mlir::ktdf_arch::Mapping mapping(default_device.getRef());

  // This pass supports remapping of the memory space attributes.
  AttrMapping mem_space_map;
  if (auto map = default_device->getAttrOfType<mlir::ktdf_arch::MapAttr>(
          "mem_space_mapping");
      map) {
    mem_space_map.insert_range(map);
  }

  for (auto op : nested_computes) {
    const auto result_type =
        op->getNumResults() == 1
            ? llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType())
            : nullptr;
    if (!op.hasPureTensorSemantics() || !result_type) {
      op->emitError("mapping error: expected pure tensor semantics");
      signalPassFailure();
      return;
    }

    LDBG() << "mapping nested " << mlir::OpWithFlags(op, kSkipRegions);
    mapToCompute(op, mapping, result_type.getElementType());
  }

  // Tile all the compute operations and put them into loop nests that feature
  // loop-carried dependencies.
  for (auto& op : computes) {
    auto maybe_loops = tile(rewriter, op, mapping);
    if (failed(maybe_loops)) {
      signalPassFailure();
      return;
    }
    if (maybe_loops->empty()) {
      LDBG() << "no loops generated for "
             << mlir::OpWithFlags(op, kSkipRegions);
      continue;
    }
  }

  // Merge `tensor.(insert|extract)_slice` ops, tile `ktdp.(load|store)`,
  // broadcast compute operands back to target vector size and remove all the
  // unneeded `iter_args`.
  {
    mlir::RewritePatternSet patterns(&getContext());
    populateKTDPTilingPatterns(patterns, mem_space_map);
    populateDataMovementCoalescingPatterns(patterns);
    patterns.add<BroadcastInputs>(patterns.getContext(), mem_space_map);
    patterns.add<BubbleUpSliceThroughReshape>(patterns.getContext());
    mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    mlir::tensor::populateFoldTensorEmptyPatterns(patterns);
    populateSimplifyLoopsPatterns(patterns);

    if (failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  // Clean up all the unused function-level constants that remain.
  for (auto& empty : llvm::make_early_inc_range(
           llvm::reverse(func.getFunctionBody().front()))) {
    if (mlir::isOpTriviallyDead(&empty)) {
      rewriter.eraseOp(&empty);
    }
  }
}
