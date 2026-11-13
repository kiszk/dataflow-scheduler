//===-- KTDFLoweringOps.cpp -------------------------------------*- c++ -*-===//
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
// This file implements the ktdf_lowering dialect operations.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
// clang-format on

#include <mlir/IR/DialectImplementation.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::ktdf_lowering;

//===----------------------------------------------------------------------===//
// KTDFLoweringDialect
//===----------------------------------------------------------------------===//

void KTDFLoweringDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.cpp.inc"

//===----------------------------------------------------------------------===//
// Custom Assembly Format: ExecuteOnOp
//===----------------------------------------------------------------------===//

ParseResult ExecuteOnOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse operands: (unit operands)
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  llvm::SmallVector<Type, 8> operand_types;

  // If there are operands, parse them
  if (parser.parseOperandList(operands) ||
      parser.resolveOperands(operands, parser.getBuilder().getIndexType(),
                             result.operands)) {
    return failure();
  }

  // Parse the body region
  Region* body_region = result.addRegion();
  if (parser.parseRegion(*body_region, {}, {})) {
    return failure();
  }

  // Parse attributes
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void ExecuteOnOp::print(OpAsmPrinter& printer) {
  // Print operands
  printer << " ";
  printer.printOperands(getUnits());

  // Print body region
  printer << " ";
  printer.printRegion(getBody());

  // Print attributes
  printer.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// Custom Assembly Format: SignalOp
//===----------------------------------------------------------------------===//

ParseResult SignalOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse operands: (unit operands)
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;

  // Parse the operand list
  if (parser.parseOperandList(operands) ||
      parser.resolveOperands(operands, parser.getBuilder().getIndexType(),
                             result.operands)) {
    return failure();
  }

  // Parse attributes
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  return success();
}

void SignalOp::print(OpAsmPrinter& printer) {
  // Print operands
  printer << " ";
  printer.printOperands(getUnits());

  // Print attributes
  printer.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// NativeCastOp
//===----------------------------------------------------------------------===//

LogicalResult NativeCastOp::verify() {
  if (getKind() != "dl16tofp32" && getKind() != "fp32todl16") {
    return emitOpError("requires kind 'dl16tofp32' or 'fp32todl16'");
  }
  const bool widening = getKind() == "dl16tofp32";
  if (getInputs().size() != (widening ? 1u : 2u) ||
      getOutputs().size() != (widening ? 2u : 1u)) {
    return emitOpError(
        widening ? "dl16tofp32 requires one input slot and two output slots"
                 : "fp32todl16 requires two input slots and one output slot");
  }
  llvm::SmallDenseSet<Value, 4> slots;
  for (Value slot : getOperands()) {
    if (!slots.insert(slot).second) {
      return emitOpError("requires distinct input and output FIFO slots");
    }
  }
  auto verify_slots = [&](OperandRange slots, bool fp32) -> LogicalResult {
    auto first = cast<ktdf::FifoSlotType>(slots.front().getType());
    for (Value slot : slots) {
      auto type = cast<ktdf::FifoSlotType>(slot.getType());
      if (type.getNumElements() != (fp32 ? 32 : 64) ||
          (fp32 ? !type.getElementType().isF32()
                : !type.getElementType().isF16())) {
        return emitOpError(fp32
                               ? "requires 32xf32 FIFO slots on the FP32 side"
                               : "requires 64xf16 FIFO slots on the DL16 side");
      }
      if (type.getSrc() != first.getSrc() ||
          type.getDest() != first.getDest()) {
        return emitOpError("requires matching FIFO endpoints within each side");
      }
    }
    return success();
  };
  if (failed(verify_slots(getInputs(), !widening)) ||
      failed(verify_slots(getOutputs(), widening))) {
    return failure();
  }
  auto input = cast<ktdf::FifoSlotType>(getInputs().front().getType());
  auto output = cast<ktdf::FifoSlotType>(getOutputs().front().getType());
  if (input.getDest() != output.getSrc()) {
    return emitOpError(
        "requires the input destination to equal the output source");
  }
  return success();
}

void NativeCastOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects) {
  for (OpOperand& input : getInputsMutable()) {
    effects.emplace_back(MemoryEffects::Read::get(), &input,
                         ktdf::FifoResource::get());
  }
  // Consuming a packet advances the queue shared by its FIFO slots.
  effects.emplace_back(MemoryEffects::Write::get(), ktdf::FifoResource::get());
  for (OpOperand& output : getOutputsMutable()) {
    effects.emplace_back(MemoryEffects::Write::get(), &output,
                         ktdf::FifoResource::get());
  }
}

LogicalResult NativeCastValueOp::verify() {
  const bool widening = getKind() == "dl16tofp32";
  if (!widening && getKind() != "fp32todl16")
    return emitOpError("requires kind 'dl16tofp32' or 'fp32todl16'");
  if (getInputs().size() != (widening ? 1u : 2u) ||
      getOutputs().size() != (widening ? 2u : 1u))
    return emitOpError("requires one native and two FP32 packet values");
  auto valid = [](Type type, bool fp32) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    return tensor && !tensor.getEncoding() && tensor.getRank() == 1 &&
           tensor.getDimSize(0) == (fp32 ? 32 : 64) &&
           (fp32 ? tensor.getElementType().isF32()
                 : tensor.getElementType().isF16());
  };
  for (Value input : getInputs())
    if (!valid(input.getType(), !widening))
      return emitOpError("has an invalid input packet type");
  for (Value output : getOutputs())
    if (!valid(output.getType(), widening))
      return emitOpError("has an invalid output packet type");
  return success();
}
