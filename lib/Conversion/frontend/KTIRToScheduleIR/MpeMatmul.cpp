//===-- MpeMatmul.cpp --------------------------------------------*- c++
//-*-===//
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

#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/MpeMatmul.h"

#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/ReductionChunkAnalysis.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/ReductionUtils.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

bool scheduler::isMpeMatmul(linalg::LinalgOp op, ktdf_arch::Mapping& mapping) {
  if (!linalg::isaContractionOpInterface(op)) return false;
  auto unit = mapping.resolve<ktdf_arch::ExecutionUnitOp>(op.getOperation());
  return unit && unit->hasAttr("dataflow_scheduler.mpe_matmul");
}

namespace {

ktdp_lowering::LoadOp getLoad(Value tensor) {
  while (auto collapse = tensor.getDefiningOp<tensor::CollapseShapeOp>())
    tensor = collapse.getSrc();
  return tensor.getDefiningOp<ktdp_lowering::LoadOp>();
}

ktdp_lowering::StoreOp getStore(Value tensor,
                                SmallVectorImpl<Operation*>& reshapes) {
  while (tensor.hasOneUse()) {
    auto* user = *tensor.getUsers().begin();
    if (auto store = dyn_cast<ktdp_lowering::StoreOp>(user)) return store;
    auto expand = dyn_cast<tensor::ExpandShapeOp>(user);
    if (!expand) return nullptr;
    reshapes.push_back(expand);
    tensor = expand.getResult();
  }
  return nullptr;
}

void transfer(OpBuilder& b, Location loc, Value src, Value dst,
              OffsetSizeAndStrideOpInterface access, bool load,
              ArrayRef<int64_t> shape) {
  SmallVector<OpFoldResult> sizes;
  for (int64_t size : shape) sizes.push_back(b.getIndexAttr(size));
  auto map = b.getMultiDimIdentityMap(shape.size());
  SmallVector<Value> zeros(shape.size(),
                           arith::ConstantIndexOp::create(b, loc, 0));
  SmallVector<Value> offsets;
  for (auto offset : access.getMixedOffsets())
    offsets.push_back(getValueOrCreateConstantIndexOp(b, loc, offset));
  auto access_map = b.getMultiDimIdentityMap(offsets.size());
  auto access_sizes = access.getMixedSizes();
  auto op = ktdf::DataTransferOp::create(
      b, loc, src, load ? access_map : map, load ? offsets : zeros,
      load ? access_sizes : sizes, dst, load ? map : access_map,
      load ? zeros : offsets, load ? sizes : access_sizes,
      ktdf::TransferMode::Copy);
  scheduler::setThrottle(op, 64);
}

// Reuse a complete, contiguous operand already materialized in local memory.
// The reshape changes only coordinates; the preceding identity generic is
// responsible for putting the elements in that order. L3 DMA cannot copy LX
// to LX, and no copy is needed for such an input or output.
template <typename AccessOp>
Value localOperand(IRRewriter& rw, AccessOp access, Value source,
                   RankedTensorType tensor_type, Attribute local) {
  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type || scheduler::getMemorySpace(source) != local ||
      access.getStaticSizes() != type.getShape() ||
      llvm::any_of(access.getMixedOffsets(), [](OpFoldResult offset) {
        return getConstantIntValue(offset) != 0;
      }))
    return {};
  Value base = source;
  while (auto cast = base.getDefiningOp<memref::MemorySpaceCastOp>())
    base = cast.getSource();
  auto view = base.getDefiningOp<ktdp::ConstructMemoryViewOp>();
  if (!view) return {};
  int64_t stride = 1;
  for (int dim = type.getRank() - 1; dim >= 0; --dim) {
    if (view.getStaticStrides()[dim] != stride) return {};
    stride *= type.getDimSize(dim);
  }
  SmallVector<OpFoldResult> sizes, strides(tensor_type.getRank());
  stride = 1;
  for (int dim = tensor_type.getRank() - 1; dim >= 0; --dim) {
    strides[dim] = rw.getIndexAttr(stride);
    stride *= tensor_type.getDimSize(dim);
  }
  for (int64_t size : tensor_type.getShape())
    sizes.push_back(rw.getIndexAttr(size));
  auto result_type =
      MemRefType::get(tensor_type.getShape(), tensor_type.getElementType(),
                      MemRefLayoutAttrInterface{}, local);
  return memref::ReinterpretCastOp::create(rw, access.getLoc(), result_type,
                                           source, rw.getIndexAttr(0), sizes,
                                           strides);
}

void prepare(linalg::GenericOp op, IRRewriter& rw) {
  // Target validation has already checked the contraction and memory accesses.
  // This only constructs the common transfer/compute pipeline.
  auto a_type = cast<RankedTensorType>(op.getInputs()[0].getType());
  auto b_type = cast<RankedTensorType>(op.getInputs()[1].getType());
  auto c_type = cast<RankedTensorType>(op.getResult(0).getType());
  auto a_load = getLoad(op.getInputs()[0]);
  auto b_load = getLoad(op.getInputs()[1]);
  SmallVector<Operation*> result_reshapes;
  auto store = getStore(op.getResult(0), result_reshapes);
  auto init = ktdf::getReductionInitScalar(op.getOutputs()[0]);

  auto loc = op.getLoc();
  rw.setInsertionPoint(store);
  auto pipeline = ktdf::PipelineOp::create(rw, loc);
  ktdf::PrivateBuilder priv(pipeline);
  auto local = ktdp::MemorySpaceAttr::get(rw.getContext(),
                                          ktdp::MemorySpaceKind::ct_local, -1);
  auto alloc = [&](RankedTensorType type) -> Value {
    return memref::AllocOp::create(
        priv, loc,
        MemRefType::get(type.getShape(), type.getElementType(),
                        MemRefLayoutAttrInterface{}, local));
  };
  rw.setInsertionPoint(pipeline);
  Value local_a = localOperand(rw, a_load, a_load.getSource(), a_type, local);
  Value local_b = localOperand(rw, b_load, b_load.getSource(), b_type, local);
  Value local_c = localOperand(rw, store, store.getDest(), c_type, local);
  Value lx_a = local_a ? local_a : alloc(a_type);
  Value lx_b = local_b ? local_b : alloc(b_type);
  Value lx_c = local_c ? local_c : alloc(c_type);
  auto ready = priv.createToken(loc), done = priv.createToken(loc);
  rw.setInsertionPointToEnd(pipeline.getBody());
  auto load_stage =
      ktdf::StageOp::create(rw, loc, ValueRange{}, ValueRange{ready},
                            [&](OpBuilder& b, Location loc) {
                              if (!local_a)
                                transfer(b, loc, a_load.getSource(), lx_a,
                                         a_load, true, a_type.getShape());
                              if (!local_b)
                                transfer(b, loc, b_load.getSource(), lx_b,
                                         b_load, true, b_type.getShape());
                            });
  load_stage.setApplicableUnitsAttr(
      rw.getArrayAttr({rw.getStringAttr("L3LU")}));
  auto compute = ktdf::StageOp::create(
      rw, loc, ValueRange{ready}, ValueRange{done},
      [&](OpBuilder& b, Location loc) {
        Value seed = arith::ConstantOp::create(b, loc, init);
        auto fill =
            linalg::FillOp::create(b, loc, ValueRange{seed}, ValueRange{lx_c});
        fill->setAttr("dataflow_scheduler.mpe_init", b.getUnitAttr());
        auto buffer_op = ktdf::cloneLinalgGenericAsBufferOp(
            b, op, ValueRange{lx_a, lx_b}, ValueRange{lx_c});
        buffer_op->setAttr("dataflow_scheduler.mpe_matmul", b.getUnitAttr());
        buffer_op->setAttr("dataflow_scheduler.requires_target_lowering",
                           b.getUnitAttr());
      });
  compute.setApplicableUnitsAttr(
      rw.getArrayAttr({ktdf_arch::Mappable::getMapsTo(op)}));
  auto store_stage = ktdf::StageOp::create(
      rw, loc, ValueRange{done}, ValueRange{}, [&](OpBuilder& b, Location loc) {
        if (!local_c)
          transfer(b, loc, lx_c, store.getDest(), store, false,
                   c_type.getShape());
      });
  store_stage.setApplicableUnitsAttr(
      rw.getArrayAttr({rw.getStringAttr("L3SU")}));
  rw.eraseOp(store);
  for (Operation* reshape : llvm::reverse(result_reshapes)) rw.eraseOp(reshape);
  rw.eraseOp(op);
}

}  // namespace

bool scheduler::deferMpeMatmulPipelines(func::FuncOp func,
                                        ktdf_arch::Mapping& mapping) {
  bool deferred = false;
  func.walk([&](linalg::GenericOp op) {
    if (!op.hasPureTensorSemantics() || !isMpeMatmul(op, mapping)) return;
    op->setAttr("dataflow_scheduler.mpe_matmul",
                UnitAttr::get(op.getContext()));
    op->setAttr("dataflow_scheduler.requires_target_lowering",
                UnitAttr::get(op.getContext()));
    deferred = true;
  });
  return deferred;
}

void scheduler::prepareMpeMatmulPipelines(func::FuncOp func) {
  SmallVector<linalg::GenericOp> matmuls;
  func.walk([&](linalg::GenericOp op) {
    if (op->hasAttr("dataflow_scheduler.mpe_matmul") &&
        op.hasPureTensorSemantics())
      matmuls.push_back(op);
  });
  IRRewriter rw(func.getContext());
  for (auto op : matmuls) prepare(op, rw);
  if (!matmuls.empty()) (void)runRegionDCE(rw, func.getBody());
}

namespace scheduler {
#define GEN_PASS_DEF_MPEMATMULCHUNKINGPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

// The only scheduler-side target policy. Capacities, fixed prefixes, source
// strides and allocation alignments are derived from the device and actual IR.
struct MpeMatmulChunkPolicy {
  static constexpr int64_t rows = 8;
  static constexpr int64_t columns = 64;
  static constexpr int64_t k_granularity = 64;
  // Each staging address begins on a complete DL16 stick.
  static constexpr int64_t alignment = k_granularity * 2;
  static constexpr llvm::StringLiteral profile = "dd2_dl16_chunk";
  static constexpr int64_t version = 1;
};

struct MatmulAccess {
  Value buffer;
  ktdf::DataTransferOp transfer;
  bool input;
};

Value allocateChunkBuffer(OpBuilder& b, Location loc, MemRefType type,
                          Value original) {
  int64_t alignment = MpeMatmulChunkPolicy::alignment;
  if (auto priv = original.getDefiningOp<ktdf::PrivateOp>())
    original = priv.getYieldOp().getOperand(
        cast<OpResult>(original).getResultNumber());
  if (auto alloc = original.getDefiningOp<memref::AllocOp>())
    alignment = std::max(
        alignment, static_cast<int64_t>(alloc.getAlignment().value_or(1)));
  auto alloc = memref::AllocOp::create(b, loc, type);
  alloc->setAttr("alignment", b.getI64IntegerAttr(alignment));
  return alloc;
}

// The prepared pipeline has one transfer per staged operand. Resident operands
// already have a legal logical view, and retain its strides when sliced.
FailureOr<MatmulAccess> findAccess(linalg::GenericOp op, Value operand,
                                   bool input) {
  auto pipe = op->getParentOfType<ktdf::PipelineOp>();
  if (!operand.getDefiningOp<ktdf::PrivateOp>())
    return MatmulAccess{operand, {}, input};
  SmallVector<ktdf::DataTransferOp> matches;
  pipe.walk([&](ktdf::DataTransferOp xfer) {
    if ((input ? xfer.getDestination() : xfer.getSource()) == operand)
      matches.push_back(xfer);
  });
  if (matches.size() != 1) return failure();
  auto xfer = matches.front();
  return MatmulAccess{input ? xfer.getSource() : xfer.getDestination(), xfer,
                      input};
}

Value aliasRoot(Value value) {
  while (auto view = value.getDefiningOp<ViewLikeOpInterface>()) {
    if (!isa<MemRefType>(view.getViewSource().getType())) break;
    value = view.getViewSource();
  }
  return value;
}

// Distinct SSA views can name overlapping caller-owned LX storage. Check their
// original spans as well as SSA aliases; a slice alone is not enough because a
// later output group may still read another part of the same input.
bool aliasesInput(Value input, Value output) {
  input = aliasRoot(input);
  output = aliasRoot(output);
  if (input == output) return true;
  if (scheduler::getMemorySpace(input) != scheduler::getMemorySpace(output))
    return false;
  auto a = input.getDefiningOp<ktdp::ConstructMemoryViewOp>();
  auto c = output.getDefiningOp<ktdp::ConstructMemoryViewOp>();
  if (!a || !c) return false;
  auto begin_a = getConstantIntValue(a.getOffset());
  auto begin_c = getConstantIntValue(c.getOffset());
  if (!begin_a || !begin_c) return a.getOffset() == c.getOffset();
  auto end = [](ktdp::ConstructMemoryViewOp view,
                int64_t start) -> std::optional<int64_t> {
    for (auto [size, stride] :
         llvm::zip_equal(view.getStaticSizes(), view.getStaticStrides())) {
      int64_t span;
      if (size <= 0 || stride < 0 ||
          __builtin_mul_overflow(size - 1, stride, &span) ||
          __builtin_add_overflow(start, span, &start))
        return std::nullopt;
    }
    if (__builtin_add_overflow(start, int64_t{1}, &start)) return std::nullopt;
    return start;
  };
  auto end_a = end(a, *begin_a), end_c = end(c, *begin_c);
  return !end_a || !end_c || (*begin_a < *end_c && *begin_c < *end_a);
}

// Reshape a logical rectangular slice into the source access's coordinates.
// Only the reassociation already represented by the prepared full transfer is
// used. No element permutation or invented contiguous row stride is introduced.
FailureOr<SmallVector<int64_t>> reshapeSlice(
    ArrayRef<int64_t> logical_shape, ArrayRef<int64_t> tile,
    ArrayRef<int64_t> physical_shape, ArrayRef<AffineExpr> logical_offsets,
    SmallVectorImpl<AffineExpr>& offsets, MLIRContext* ctx) {
  SmallVector<int64_t> sizes(physical_shape.size(), 1);
  offsets.assign(physical_shape.size(), getAffineConstantExpr(0, ctx));
  unsigned first = 0;
  for (auto [axis, extent] : llvm::enumerate(logical_shape)) {
    unsigned end = first;
    int64_t product = 1;
    do {
      if (end == physical_shape.size() || physical_shape[end] <= 0 ||
          __builtin_mul_overflow(product, physical_shape[end++], &product))
        return failure();
    } while (product < extent);
    if (product != extent || tile[axis] <= 0 || extent % tile[axis])
      return failure();
    // Include trailing unit axes in the final group.
    if (axis + 1 == logical_shape.size())
      while (end < physical_shape.size() && physical_shape[end] == 1) ++end;
    int64_t trailing = extent;
    bool tiled = false;
    for (unsigned d = first; d < end; ++d) {
      trailing /= physical_shape[d];
      auto offset = logical_offsets[axis].floorDiv(trailing);
      if (d != first) offset = offset % physical_shape[d];
      offsets[d] = offset;
      if (tiled) {
        sizes[d] = physical_shape[d];
      } else if (tile[axis] >= trailing && tile[axis] % trailing == 0) {
        sizes[d] = tile[axis] / trailing;
        if (physical_shape[d] % sizes[d]) return failure();
        tiled = true;
      }
    }
    if (!tiled) return failure();
    first = end;
  }
  if (first != physical_shape.size()) return failure();
  return sizes;
}

FailureOr<ktdf::ChunkTransferSlice> sliceAccess(
    OpBuilder& b, Location loc, const MatmulAccess& access,
    ArrayRef<int64_t> logical_shape, ArrayRef<int64_t> tile,
    ArrayRef<Value> logical_offsets) {
  auto transfer = access.transfer;
  auto base_map =
      access.input ? transfer.getSourceMap() : transfer.getDestMap();
  auto original_indices =
      access.input ? transfer.getSourceIndices() : transfer.getDestIndices();
  auto shape = access.input ? transfer.getStaticSourceSizes().value()
                            : transfer.getStaticDestSizes().value();
  SmallVector<Value> indices;
  // Constants in the old transfer stage must not escape when it is erased.
  for (Value index : original_indices) {
    auto constant = getConstantIntValue(index);
    indices.push_back(
        constant ? arith::ConstantIndexOp::create(b, loc, *constant).getResult()
                 : index);
  }
  if (!base_map) base_map = b.getMultiDimIdentityMap(shape.size());
  SmallVector<AffineExpr> logical_exprs;
  for (unsigned i = 0; i < logical_offsets.size(); ++i)
    logical_exprs.push_back(b.getAffineDimExpr(indices.size() + i));
  SmallVector<AffineExpr> physical_offsets;
  auto sizes = reshapeSlice(logical_shape, tile, shape, logical_exprs,
                            physical_offsets, b.getContext());
  if (failed(sizes)) return failure();
  llvm::append_range(indices, logical_offsets);
  for (auto [d, expr] : llvm::enumerate(physical_offsets))
    physical_offsets[d] = base_map->getResult(d) + expr;
  return ktdf::ChunkTransferSlice{
      access.buffer,
      AffineMap::get(indices.size(), 0, physical_offsets, b.getContext()),
      indices, *sizes};
}

ktdf::ChunkTransferSlice compactSlice(OpBuilder& b, Location loc, Value value) {
  auto type = cast<MemRefType>(value.getType());
  return {value, b.getMultiDimIdentityMap(type.getRank()),
          SmallVector<Value>(type.getRank(),
                             arith::ConstantIndexOp::create(b, loc, 0)),
          llvm::to_vector(type.getShape())};
}

Value residentSlice(OpBuilder& b, Location loc, Value source,
                    ArrayRef<int64_t> sizes, ArrayRef<Value> offsets,
                    ArrayRef<int64_t> matrix_shape) {
  SmallVector<OpFoldResult> mixed_offsets(offsets.begin(), offsets.end());
  SmallVector<OpFoldResult> mixed_sizes,
      strides(sizes.size(), b.getIndexAttr(1));
  for (int64_t size : sizes) mixed_sizes.push_back(b.getIndexAttr(size));
  auto type = memref::SubViewOp::inferRankReducedResultType(
      matrix_shape, cast<MemRefType>(source.getType()), mixed_offsets,
      mixed_sizes, strides);
  return memref::SubViewOp::create(b, loc, cast<MemRefType>(type), source,
                                   mixed_offsets, mixed_sizes, strides);
}

// Validate all accesses without changing the prepared pipeline. In particular,
// output/input aliasing is not safe when later chunks still read the input.
FailureOr<ktdf::ContractionChunkDomain> validateChunk(
    linalg::GenericOp op, scheduler::MpeMatmulChunkPlan plan) {
  auto domain = ktdf::analyzeContractionChunkDomain(op);
  if (!op->hasAttr("dataflow_scheduler.mpe_matmul") ||
      !op.hasPureBufferSemantics() || failed(domain) ||
      !cast<MemRefType>(op.getInputs()[0].getType()).getElementType().isF16())
    return failure();
  auto& dims = domain->dimensions;
  auto sizes = domain->loop_sizes;
  if (plan.k <= 0 || plan.k % MpeMatmulChunkPolicy::k_granularity ||
      sizes[dims.k[0]] % plan.k || (plan.buffers != 1 && plan.buffers != 2) ||
      sizes[dims.m[0]] % MpeMatmulChunkPolicy::rows ||
      sizes[dims.n[0]] % MpeMatmulChunkPolicy::columns)
    return failure();
  auto pipe = op->getParentOfType<ktdf::PipelineOp>();
  // Preparation's DCE removes empty load/store stages for resident operands.
  // Their absence does not change the contraction or its chunkability.
  if (!pipe || op->getParentOp() != op->getParentOfType<ktdf::StageOp>() ||
      llvm::range_size(pipe.getStages()) > 3)
    return failure();
  SmallVector<Value> roots;
  for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
    auto access = findAccess(op, operand, i < 2);
    if (failed(access)) return failure();
    roots.push_back(aliasRoot(access->buffer));
    if (!access->transfer) continue;
    SmallVector<int64_t> tile;
    SmallVector<AffineExpr> zero;
    for (auto result : op.getIndexingMapsArray()[i].getResults()) {
      unsigned d = cast<AffineDimExpr>(result).getPosition();
      tile.push_back(d == dims.k[0]   ? plan.k
                     : d == dims.m[0] ? MpeMatmulChunkPolicy::rows
                     : d == dims.n[0] ? MpeMatmulChunkPolicy::columns
                                      : 1);
      zero.push_back(getAffineConstantExpr(0, op.getContext()));
    }
    auto shape = i < 2 ? access->transfer.getStaticSourceSizes()
                       : access->transfer.getStaticDestSizes();
    if (!shape) return failure();
    SmallVector<AffineExpr> offsets;
    if (failed(reshapeSlice(cast<MemRefType>(operand.getType()).getShape(),
                            tile, *shape, zero, offsets, op.getContext())))
      return failure();
  }
  if (aliasesInput(roots[0], roots[2]) || aliasesInput(roots[1], roots[2]))
    return failure();
  return domain;
}

}  // namespace

SmallVector<scheduler::MpeMatmulChunkPlan> scheduler::getMpeMatmulChunkPlans(
    linalg::GenericOp op) {
  SmallVector<MpeMatmulChunkPlan> plans;
  auto domain = ktdf::analyzeContractionChunkDomain(op);
  if (failed(domain)) return plans;
  int64_t k = domain->loop_sizes[domain->dimensions.k[0]];
  bool has_staged_input = false;
  for (Value input : op.getInputs()) {
    auto access = findAccess(op, input, true);
    if (failed(access)) return plans;
    has_staged_input |= static_cast<bool>(access->transfer);
  }
  for (int64_t buffers : {2, 1}) {
    // With resident inputs there is nothing to prefetch. Prefer output tiling
    // with Kc=K over introducing unnecessary partial rounding at K boundaries.
    if (buffers == 2 && !has_staged_input) continue;
    for (int64_t size :
         ktdf::getReductionChunkSizes(k, MpeMatmulChunkPolicy::k_granularity)) {
      if (buffers == 2 && size == k) continue;
      MpeMatmulChunkPlan plan{size, buffers};
      if (succeeded(validateChunk(op, plan))) plans.push_back(plan);
    }
  }
  return plans;
}

LogicalResult scheduler::chunkMpeMatmulPipeline(linalg::GenericOp op,
                                                MpeMatmulChunkPlan plan) {
  auto domain = validateChunk(op, plan);
  if (failed(domain))
    return op.emitError(
        "matmul chunk plan requires static, nonaliasing Standard f16 operands, "
        "legal K slices and a prepared transfer/compute/store pipeline");
  IRRewriter rw(op.getContext());
  auto pipe = op->getParentOfType<ktdf::PipelineOp>();
  auto compute = op->getParentOfType<ktdf::StageOp>();
  SmallVector<MatmulAccess> accesses;
  SmallVector<ArrayRef<int64_t>> full_shapes;
  for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
    accesses.push_back(*findAccess(op, operand, i < 2));
    full_shapes.push_back(cast<MemRefType>(operand.getType()).getShape());
  }
  auto loc = op.getLoc();
  rw.setInsertionPoint(pipe);
  Value zero = arith::ConstantIndexOp::create(rw, loc, 0);
  Value one = arith::ConstantIndexOp::create(rw, loc, 1);
  auto dims = domain->dimensions;
  SmallVector<Value> offsets(domain->loop_sizes.size(), zero);
  SmallVector<int64_t> tile(domain->loop_sizes.size(), 1);
  tile[dims.m[0]] = MpeMatmulChunkPolicy::rows;
  tile[dims.n[0]] = MpeMatmulChunkPolicy::columns;
  tile[dims.k[0]] = plan.k;
  SmallVector<unsigned> outer_dims(dims.batch.begin(), dims.batch.end());
  outer_dims.push_back(dims.m[0]);
  outer_dims.push_back(dims.n[0]);
  // Output groups are sequential; only pre-existing outer core loops retain
  // parallel annotations. Every group drains its final export before reuse.
  for (unsigned d : outer_dims) {
    Value upper =
        arith::ConstantIndexOp::create(rw, loc, domain->loop_sizes[d]);
    Value step = arith::ConstantIndexOp::create(rw, loc, tile[d]);
    auto loop = scf::ForOp::create(rw, loc, zero, upper, step);
    loop->setAttr("dataflow_scheduler.mpe_output_group", rw.getUnitAttr());
    offsets[d] = loop.getInductionVar();
    rw.setInsertionPointToStart(loop.getBody());
  }
  auto group = ktdf::PipelineOp::create(rw, loc);
  group->setAttr("dataflow_scheduler.mpe_chunk_group", rw.getUnitAttr());
  ktdf::PrivateBuilder shared(group);
  auto local = cast<MemRefType>(op.getOutputs()[0].getType()).getMemorySpace();
  auto matrix_type = MemRefType::get(
      {MpeMatmulChunkPolicy::rows, MpeMatmulChunkPolicy::columns},
      rw.getF16Type(), MemRefLayoutAttrInterface{}, local);
  SmallVector<Value> c_offsets;
  SmallVector<int64_t> c_sizes;
  for (auto expr : op.getIndexingMapsArray().back().getResults()) {
    unsigned d = cast<AffineDimExpr>(expr).getPosition();
    c_offsets.push_back(offsets[d]);
    c_sizes.push_back(tile[d]);
  }
  Value partial;
  if (accesses[2].transfer) {
    partial = allocateChunkBuffer(shared, loc, matrix_type, op.getOutputs()[0]);
  } else {
    rw.setInsertionPoint(group);
    partial = residentSlice(rw, loc, accesses[2].buffer, c_sizes, c_offsets,
                            matrix_type.getShape());
  }
  Value temporary =
      allocateChunkBuffer(shared, loc, matrix_type, op.getOutputs()[0]);
  Value output_ready = shared.createToken(loc);
  rw.setInsertionPointToEnd(group.getBody());
  auto calculate =
      ktdf::StageOp::create(rw, loc, ValueRange{}, ValueRange{output_ready});
  rw.setInsertionPointToStart(calculate.getBody());
  Value count = arith::ConstantIndexOp::create(
      rw, loc, domain->loop_sizes[dims.k[0]] / plan.k);
  auto q_loop = scf::ForOp::create(rw, loc, zero, count, one);
  q_loop->setAttr(
      "loop_type",
      ktdf::LoopTypeAttr::get(rw.getContext(), ktdf::LoopType::ReductionLoop));
  q_loop->setAttr("dataflow_scheduler.mpe_chunk_loop", rw.getUnitAttr());
  q_loop->setAttr("dataflow_scheduler.mpe_chunk_buffers",
                  rw.getI64IntegerAttr(plan.buffers));
  rw.setInsertionPointToStart(q_loop.getBody());
  offsets[dims.k[0]] = affine::AffineApplyOp::create(
      rw, loc, AffineMap::get(1, 0, rw.getAffineDimExpr(0) * plan.k),
      q_loop.getInductionVar());
  SmallVector<SmallVector<Value>> operand_offsets;
  SmallVector<SmallVector<int64_t>> operand_sizes;
  for (auto map : op.getIndexingMapsArray()) {
    operand_offsets.emplace_back();
    operand_sizes.emplace_back();
    for (auto result : map.getResults()) {
      unsigned d = cast<AffineDimExpr>(result).getPosition();
      operand_offsets.back().push_back(offsets[d]);
      operand_sizes.back().push_back(tile[d]);
    }
  }
  auto chunk = ktdf::PipelineOp::create(rw, loc);
  chunk->setAttr("dataflow_scheduler.mpe_chunk_buffers",
                 rw.getI64IntegerAttr(plan.buffers));
  ktdf::PrivateBuilder staging(chunk);
  Value ready = staging.createToken(loc);
  SmallVector<Value> inputs;
  for (unsigned i = 0; i < 2; ++i) {
    // Drop singleton batch axes, preserving the operand's matrix axis order.
    SmallVector<int64_t> matrix_shape;
    auto map = op.getIndexingMapsArray()[i];
    for (auto [j, expr] : llvm::enumerate(map.getResults()))
      if (!llvm::is_contained(dims.batch,
                              cast<AffineDimExpr>(expr).getPosition()))
        matrix_shape.push_back(operand_sizes[i][j]);
    if (accesses[i].transfer) {
      inputs.push_back(allocateChunkBuffer(
          staging, loc,
          MemRefType::get(matrix_shape, rw.getF16Type(),
                          MemRefLayoutAttrInterface{}, local),
          op.getInputs()[i]));
    } else {
      rw.setInsertionPoint(chunk);
      inputs.push_back(residentSlice(rw, loc, accesses[i].buffer,
                                     operand_sizes[i], operand_offsets[i],
                                     matrix_shape));
    }
  }
  rw.setInsertionPointToEnd(chunk.getBody());
  auto load_stage =
      ktdf::StageOp::create(rw, loc, ValueRange{}, ValueRange{ready});
  for (const auto& access : ArrayRef<MatmulAccess>(accesses).take_front(2)) {
    if (!access.transfer) continue;
    load_stage.setApplicableUnitsAttr(
        access.transfer->getParentOfType<ktdf::StageOp>()
            .getApplicableUnitsAttr());
    break;
  }
  load_stage->setAttr("dataflow_scheduler.mpe_chunk_load", rw.getUnitAttr());
  {
    OpBuilder b(load_stage.getBody(), load_stage.getBody()->end());
    SmallVector<ktdf::ChunkTransfer, 2> transfers;
    for (unsigned i = 0; i < 2; ++i) {
      if (!accesses[i].transfer) continue;
      auto src = sliceAccess(b, loc, accesses[i], full_shapes[i],
                             operand_sizes[i], operand_offsets[i]);
      assert(succeeded(src) && "prevalidated slice");
      transfers.push_back({*src, compactSlice(b, loc, inputs[i])});
    }
    for (auto xfer : ktdf::StageFactory::buildTransfers(b, loc, transfers))
      setThrottle(xfer, 64);
  }
  auto compute_stage =
      ktdf::StageOp::create(rw, loc, ValueRange{ready}, ValueRange{});
  compute_stage.setApplicableUnitsAttr(compute.getApplicableUnitsAttr());
  compute_stage->setAttr("dataflow_scheduler.mpe_chunk_compute",
                         rw.getUnitAttr());
  {
    OpBuilder b(compute_stage.getBody(), compute_stage.getBody()->end());
    Value seed = arith::ConstantOp::create(b, loc, b.getF16FloatAttr(0));
    auto fill =
        linalg::FillOp::create(b, loc, ValueRange{seed}, ValueRange{temporary});
    fill->setAttr("dataflow_scheduler.mpe_init", b.getUnitAttr());
    SmallVector<AffineMap> maps;
    for (auto map : op.getIndexingMapsArray()) {
      SmallVector<AffineExpr> exprs;
      for (auto expr : map.getResults()) {
        unsigned d = cast<AffineDimExpr>(expr).getPosition();
        if (llvm::is_contained(dims.batch, d)) continue;
        exprs.push_back(b.getAffineDimExpr(d == dims.m[0]   ? 0
                                           : d == dims.n[0] ? 1
                                                            : 2));
      }
      maps.push_back(AffineMap::get(3, 0, exprs, b.getContext()));
    }
    auto product = linalg::GenericOp::create(
        b, loc, TypeRange{}, inputs, ValueRange{temporary}, maps,
        ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                      utils::IteratorType::parallel,
                                      utils::IteratorType::reduction});
    IRMapping mapping;
    op.getRegion().cloneInto(&product.getRegion(), mapping);
    for (auto name : {"dataflow_scheduler.mpe_matmul",
                      "dataflow_scheduler.requires_target_lowering"})
      product->setAttr(name, b.getUnitAttr());
    product->setAttr("dataflow_scheduler.mpe_chunk_profile",
                     b.getStringAttr(MpeMatmulChunkPolicy::profile));
    product->setAttr("dataflow_scheduler.mpe_chunk_version",
                     b.getI64IntegerAttr(MpeMatmulChunkPolicy::version));
    product->setAttr("dataflow_scheduler.mpe_partial_policy",
                     b.getStringAttr("post_add_sdsc"));
    product->setAttr("dataflow_scheduler.mpe_chunk_size",
                     b.getI64IntegerAttr(plan.k));
    product->setAttr("dataflow_scheduler.mpe_chunk_buffers",
                     b.getI64IntegerAttr(plan.buffers));
    OpBuilder condition_builder(product);
    auto first =
        arith::CmpIOp::create(condition_builder, loc, arith::CmpIPredicate::eq,
                              q_loop.getInductionVar(), zero);
    auto select = scf::IfOp::create(b, loc, first, true);
    auto combine = [&](Block* block, bool initial) {
      OpBuilder body(block, block->begin());
      SmallVector<Value> args = initial
                                    ? SmallVector<Value>{temporary}
                                    : SmallVector<Value>{partial, temporary};
      SmallVector<AffineMap> maps(args.size() + 1, b.getMultiDimIdentityMap(2));
      auto generic = linalg::GenericOp::create(
          body, loc, TypeRange{}, args, ValueRange{partial}, maps,
          SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
          [&](OpBuilder& body, Location loc, ValueRange values) {
            Value result;
            if (initial)
              result = spyreop::Identity::create(body, loc, values[0]);
            else
              result = arith::AddFOp::create(body, loc, values[0], values[1]);
            linalg::YieldOp::create(body, loc, result);
          });
      generic->setAttr("dataflow_scheduler.mpe_partial_combine",
                       b.getUnitAttr());
    };
    combine(&select.getThenRegion().front(), true);
    combine(&select.getElseRegion().front(), false);
  }
  if (!accesses[2].transfer) {
    rw.eraseOp(pipe);
    return success();
  }
  rw.setInsertionPointToEnd(group.getBody());
  auto export_phase =
      ktdf::StageOp::create(rw, loc, ValueRange{output_ready}, ValueRange{});
  rw.setInsertionPointToEnd(export_phase.getBody());
  auto export_pipe = ktdf::PipelineOp::create(rw, loc);
  rw.setInsertionPointToEnd(export_pipe.getBody());
  auto export_stage =
      ktdf::StageOp::create(rw, loc, ValueRange{}, ValueRange{});
  export_stage.setApplicableUnitsAttr(
      accesses[2]
          .transfer->getParentOfType<ktdf::StageOp>()
          .getApplicableUnitsAttr());
  OpBuilder out(export_stage.getBody(), export_stage.getBody()->end());
  // Output offsets are defined in the group, not in the completed q loop.
  auto destination =
      *sliceAccess(out, loc, accesses[2], full_shapes[2], c_sizes, c_offsets);
  for (auto xfer : ktdf::StageFactory::buildTransfers(
           out, loc,
           {ktdf::ChunkTransfer{compactSlice(out, loc, partial), destination}}))
    setThrottle(xfer, 64);
  rw.eraseOp(pipe);
  return success();
}

namespace {
struct MpeMatmulChunkingPass
    : scheduler::impl::MpeMatmulChunkingPassBase<MpeMatmulChunkingPass> {
  using MpeMatmulChunkingPassBase::MpeMatmulChunkingPassBase;
  void runOnOperation() override {
    SmallVector<linalg::GenericOp> ops;
    getOperation().walk([&](linalg::GenericOp op) {
      if (op->hasAttr("dataflow_scheduler.mpe_matmul") &&
          !op->hasAttr("dataflow_scheduler.mpe_chunk_profile"))
        ops.push_back(op);
    });
    for (auto op : ops)
      if (failed(scheduler::chunkMpeMatmulPipeline(op, {chunkSize, buffers})))
        return signalPassFailure();
  }
};
}  // namespace
