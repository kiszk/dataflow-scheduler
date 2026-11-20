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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/NumericCastLowering.h"

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

LogicalResult scheduler::verifyNumericCastUnit(ktdf_arch::ExecutionUnitOp unit,
                                               Operation* diagnostic_op) {
  if (!unit) {
    return diagnostic_op->emitError(
        "numeric_cast requires a resolvable execution unit");
  }
  if (!unit.getFeature<ktdf_arch::feature::Compute>()) {
    return diagnostic_op->emitError(
        "numeric_cast requires a programmable conversion unit");
  }
  auto* context = diagnostic_op->getContext();
  auto native = Float16Type::get(context);
  auto fp32 = Float32Type::get(context);
  auto simd = unit.getFeature<ktdf_arch::feature::SIMD>();
  if (!simd || simd.getLanes(native) != 64 || simd.getLanes(fp32) != 32) {
    return diagnostic_op->emitError(
        "numeric_cast requires 64 f16 lanes and 32 f32 lanes");
  }
  if (simd.getSubSimdLanes(native) != 8 || simd.getSubSimdLanes(fp32) != 4) {
    return diagnostic_op->emitError(
        "numeric_cast requires 8 f16 and 4 f32 sub-SIMD lanes");
  }
  return success();
}

namespace {

void emitNumericCast(PatternRewriter& rewriter, Location loc, bool widening,
                     ktdf_arch::feature::SIMD simd,
                     llvm::function_ref<Value(unsigned, VectorType)> get_input,
                     llvm::function_ref<void(Value)> emit_result) {
  const int64_t native_lanes = simd.getLanes(rewriter.getF16Type());
  const int64_t fp32_lanes = simd.getLanes(rewriter.getF32Type());
  const int64_t native_sub = simd.getSubSimdLanes(rewriter.getF16Type());
  const int64_t fp32_sub = simd.getSubSimdLanes(rewriter.getF32Type());
  const int64_t groups = native_lanes / native_sub;
  auto native_stick = VectorType::get({native_lanes}, rewriter.getF16Type());
  auto fp32_packet = VectorType::get({fp32_lanes}, rewriter.getF32Type());
  if (widening) {
    auto input = get_input(0, native_stick);
    for (int64_t part = 0; part < 2; ++part) {
      // Each FCVT packet is selected from its own cast expression.
      auto converted = vectorchain::CastOp::create(
          rewriter, loc, VectorType::get({native_lanes}, rewriter.getF32Type()),
          input, nullptr, nullptr);
      SmallVector<Attribute> indices;
      for (int64_t lane = 0; lane < fp32_sub; ++lane)
        indices.push_back(rewriter.getI32IntegerAttr(part * fp32_sub + lane));
      auto packet = vectorchain::ShuffleOp::create(
          rewriter, loc, fp32_packet, converted.getResult(), ValueRange{},
          ValueRange{}, nullptr, nullptr, rewriter.getArrayAttr(indices),
          rewriter.getI32IntegerAttr(groups));
      emit_result(packet.getOutput());
    }
  } else {
    SmallVector<Value> converted;
    for (unsigned part = 0; part < 2; ++part)
      converted.push_back(
          vectorchain::CastOp::create(
              rewriter, loc,
              VectorType::get({fp32_lanes}, rewriter.getF16Type()),
              get_input(part, fp32_packet), nullptr, nullptr)
              .getData());
    SmallVector<Attribute> indices;
    for (int64_t lane = 0; lane < native_sub; ++lane)
      indices.push_back(rewriter.getI32IntegerAttr(lane));
    auto packed = vectorchain::PackOp::create(
        rewriter, loc, native_stick, converted[0], converted[1], nullptr,
        nullptr, rewriter.getArrayAttr(indices), rewriter.getIndexAttr(groups));
    emit_result(packed.getData());
  }
}

struct LowerNumericCastPattern
    : OpRewritePattern<ktdf_lowering::NumericCastOp> {
  LowerNumericCastPattern(MLIRContext* context,
                          const scheduler::ResourceToUnits& components,
                          ktdf_arch::Mapping& mapping)
      : OpRewritePattern(context), components_(components), mapping_(mapping) {}

  LogicalResult matchAndRewrite(ktdf_lowering::NumericCastOp op,
                                PatternRewriter& rewriter) const override {
    auto program = op->getParentOfType<dataflow::ProgramUnitOp>();
    if (!program) {
      return op.emitError("numeric_cast must be inside a program_unit");
    }
    auto input = cast<ktdf::FifoSlotType>(op.getInputs().front().getType());
    auto output = cast<ktdf::FifoSlotType>(op.getOutputs().front().getType());
    auto compute = mapping_.resolve<ktdf_arch::ExecutionUnitOp>(program);
    auto compute_kind =
        compute ? dyn_cast<StringAttr>(compute.getKind()) : StringAttr();
    if (!compute_kind) {
      return op.emitError("numeric_cast requires a resolvable execution unit");
    }
    auto input_dest = dyn_cast<StringAttr>(input.getDest());
    if (!input_dest ||
        input_dest.getValue().upper() != compute_kind.getValue()) {
      return op.emitError(
          "numeric_cast FIFO endpoints do not match the executing "
          "program_unit");
    }
    if (failed(scheduler::verifyNumericCastUnit(compute, op))) return failure();
    auto source = scheduler::resolveUnitFromFifoAttr(
        input.getSrc(), components_, rewriter, program, op.getLoc(), op);
    if (failed(source)) return failure();
    auto destination = scheduler::resolveUnitFromFifoAttr(
        output.getDest(), components_, rewriter, program, op.getLoc(), op);
    if (failed(destination)) return failure();

    const auto loc = op.getLoc();
    emitNumericCast(
        rewriter, loc, op.getKind() == "dl16tofp32",
        compute.getFeature<ktdf_arch::feature::SIMD>(),
        [&](unsigned, VectorType type) {
          return dataflow::ReceiveOp::create(rewriter, loc, type, *source,
                                             nullptr)
              .getData();
        },
        [&](Value result) {
          dataflow::SendOp::create(rewriter, loc, *destination, result, nullptr,
                                   nullptr);
        });
    rewriter.eraseOp(op);
    return success();
  }

 private:
  const scheduler::ResourceToUnits& components_;
  ktdf_arch::Mapping& mapping_;
};

struct LowerNumericCastValuePattern
    : OpRewritePattern<ktdf_lowering::NumericCastValueOp> {
  LowerNumericCastValuePattern(MLIRContext* context,
                               ktdf_arch::Mapping& mapping)
      : OpRewritePattern(context), mapping_(mapping) {}

  LogicalResult matchAndRewrite(ktdf_lowering::NumericCastValueOp op,
                                PatternRewriter& rewriter) const override {
    if (!llvm::all_of(op.getInputs(), [](Value input) {
          return isa<VectorType>(input.getType());
        }))
      return failure();
    auto program = op->getParentOfType<dataflow::ProgramUnitOp>();
    if (!program)
      return op.emitError("numeric_cast_value requires a program_unit");
    auto compute = mapping_.resolve<ktdf_arch::ExecutionUnitOp>(program);
    if (failed(scheduler::verifyNumericCastUnit(compute, op))) return failure();
    SmallVector<Value> results;
    emitNumericCast(
        rewriter, op.getLoc(), op.getKind() == "dl16tofp32",
        compute.getFeature<ktdf_arch::feature::SIMD>(),
        [&](unsigned part, VectorType) { return op.getInputs()[part]; },
        [&](Value result) { results.push_back(result); });
    rewriter.replaceOp(op, results);
    return success();
  }

 private:
  ktdf_arch::Mapping& mapping_;
};

}  // namespace

void scheduler::populateNumericCastLoweringPatterns(
    RewritePatternSet& patterns, const ResourceToUnits& components,
    ktdf_arch::Mapping& mapping) {
  patterns.add<LowerNumericCastPattern>(patterns.getContext(), components,
                                        mapping);
  patterns.add<LowerNumericCastValuePattern>(patterns.getContext(), mapping);
}
