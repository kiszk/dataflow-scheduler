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

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
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
