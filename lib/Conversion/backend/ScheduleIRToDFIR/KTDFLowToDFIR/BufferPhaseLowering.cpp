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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/BufferPhaseLowering.h"

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"
// Marker used only to locate buffer_phase_op inside the cloned loop tree.
// Set before cloning, removed immediately after the cloned op is found.
#define PROCESSING_MARKER_ATTR "__processing_marker__"

using namespace scheduler;

namespace {

/// Controls where the iter-arg toggle (subi) is placed in a double-buffered
/// loop. BufferFirst places it at the yield (most units); BufferLast places it
/// at the loop-body top (MNILU/MNISU).
enum class DoubleBufferingMode { BufferFirst, BufferLast };

[[nodiscard]]
auto getDoubleBufferingMode(mlir::ktdf_arch::Resource resource)
    -> DoubleBufferingMode {
  return resource->getAttr("dataflow_scheduler.double_buffer_last")
             ? DoubleBufferingMode::BufferLast
             : DoubleBufferingMode::BufferFirst;
}

bool areMemoryViewsMatchingExceptOffset(
    mlir::dataflow::GetLogicalMemoryViewOp view_0,
    mlir::dataflow::GetLogicalMemoryViewOp view_1) {
  return view_0.getResult().getType() == view_1.getResult().getType() &&
         view_0.getLayoutMap() == view_1.getLayoutMap() &&
         view_0.getFromUnit() == view_1.getFromUnit();
}

// Add all offsets in one traversal. Cloning once per select would invalidate
// the other selects and lose the shared phase (and existing loop-carried SSA).
mlir::scf::ForOp cloneLoopChainWithDoubleBufferingIterArgs(
    llvm::ArrayRef<mlir::scf::ForOp> loops, mlir::ValueRange initial_offsets,
    mlir::ValueRange offset_sums, DoubleBufferingMode mode) {
  mlir::scf::ForOp inner;
  unsigned count = initial_offsets.size();
  for (auto loop : llvm::reverse(loops)) {
    mlir::IRMapping map;
    auto clone =
        createForOpWithAdditionalIterArgs(loop, count, map, /*delete_op=*/true);
    auto args = clone.getRegionIterArgs().take_back(count);
    auto yield =
        mlir::cast<mlir::scf::YieldOp>(clone.getBody()->getTerminator());
    llvm::SmallVector<mlir::Value> next;
    if (inner) {
      auto child =
          mlir::cast<mlir::scf::ForOp>(map.lookup(inner.getOperation()));
      for (auto [operand, arg] :
           llvm::zip_equal(child.getInitsMutable().take_back(count), args))
        operand.set(arg);
      llvm::append_range(next, child.getResults().take_back(count));
    } else {
      mlir::OpBuilder b(yield);
      if (mode == DoubleBufferingMode::BufferLast)
        b.setInsertionPointToStart(clone.getBody());
      for (auto [sum, arg] : llvm::zip_equal(offset_sums, args))
        next.push_back(
            mlir::arith::SubIOp::create(b, yield.getLoc(), sum, arg));
    }
    yield.getResultsMutable()
        .slice(yield.getNumOperands() - count, count)
        .assign(next);
    inner = clone;
  }
  for (auto [operand, offset] : llvm::zip_equal(
           inner.getInitsMutable().take_back(count), initial_offsets))
    operand.set(offset);
  return inner;
}

mlir::LogicalResult processOneBufferPhasePair(
    mlir::ktdf::BufferPhaseOp phase, mlir::ktdf_arch::Mapping& mapping) {
  if (phase.getNumPhases() != 2)
    return phase.emitError(
        "buffer_phase must have num_phases = 2 for double buffering");
  llvm::SmallVector<mlir::ktdf::SelectMemrefOp> selects;
  for (auto* user : phase->getUsers()) {
    auto select = mlir::dyn_cast<mlir::ktdf::SelectMemrefOp>(user);
    if (!select)
      return phase.emitError(
          "buffer_phase users must be select_memref operations");
    selects.push_back(select);
  }
  if (selects.empty()) {
    phase.erase();
    return mlir::success();
  }
  auto resource = mapping.resolve(phase);
  if (!resource)
    return phase.emitError(
        "lowerDoubleBuffering: could not resolve component type from enclosing "
        "program_unit");
  auto mode = getDoubleBufferingMode(resource);
  llvm::SmallVector<mlir::scf::ForOp> loops;
  for (auto iv : phase.getIvs()) {
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(iv);
    auto loop =
        arg ? mlir::dyn_cast<mlir::scf::ForOp>(arg.getOwner()->getParentOp())
            : mlir::scf::ForOp{};
    if (!loop || iv != loop.getInductionVar())
      return phase.emitError(
          "buffer_phase operands must be scf.for induction variables");
    loops.push_back(loop);
  }
  if (loops.empty())
    return phase.emitError("buffer_phase must have at least one IV");

  // Validate every select before changing any loop or operation.
  llvm::SmallVector<mlir::Value> initial_offsets, sums;
  mlir::OpBuilder b(loops.front());
  for (auto select : selects) {
    if (select.getCandidates().size() != 2)
      return select.emitError(
          "select_memref must have exactly 2 candidate memrefs");
    auto first = select.getCandidates()[0]
                     .getDefiningOp<mlir::dataflow::GetLogicalMemoryViewOp>();
    auto second = select.getCandidates()[1]
                      .getDefiningOp<mlir::dataflow::GetLogicalMemoryViewOp>();
    if (!first || !second)
      return select.emitError(
          "select_memref operands must be get_logical_memory_view operations");
    if (!areMemoryViewsMatchingExceptOffset(first, second))
      return select.emitError(
          "get_logical_memory_view operations must match except for offset");
    auto a = first.getStartAddress(), c = second.getStartAddress();
    initial_offsets.push_back(mode == DoubleBufferingMode::BufferFirst ? a : c);
    sums.push_back(mlir::arith::AddIOp::create(b, phase.getLoc(), a, c));
  }
  phase->setAttr(PROCESSING_MARKER_ATTR, b.getUnitAttr());
  for (auto [i, select] : llvm::enumerate(selects))
    select->setAttr(PROCESSING_MARKER_ATTR, b.getI64IntegerAttr(i));
  auto outer = cloneLoopChainWithDoubleBufferingIterArgs(loops, initial_offsets,
                                                         sums, mode);
  mlir::ktdf::BufferPhaseOp cloned_phase;
  outer.walk([&](mlir::ktdf::BufferPhaseOp op) {
    if (op->hasAttr(PROCESSING_MARKER_ATTR)) cloned_phase = op;
  });
  assert(cloned_phase && "phase must survive cloning");
  auto iv = mlir::cast<mlir::BlockArgument>(cloned_phase.getIvs().back());
  auto inner = mlir::cast<mlir::scf::ForOp>(iv.getOwner()->getParentOp());
  mlir::ValueRange offsets =
      inner.getRegionIterArgs().take_back(selects.size());
  if (mode == DoubleBufferingMode::BufferLast)
    offsets = inner.getBody()->getTerminator()->getOperands().take_back(
        selects.size());
  llvm::SmallVector<mlir::Operation*> users(cloned_phase->getUsers());
  for (auto* user : users) {
    auto select = mlir::cast<mlir::ktdf::SelectMemrefOp>(user);
    auto index =
        select->getAttrOfType<mlir::IntegerAttr>(PROCESSING_MARKER_ATTR)
            .getInt();
    auto view = select.getCandidates()[0]
                    .getDefiningOp<mlir::dataflow::GetLogicalMemoryViewOp>();
    mlir::OpBuilder replace(select);
    auto result = mlir::dataflow::GetLogicalMemoryViewOp::create(
        replace, select.getLoc(), view.getType(), view.getFromUnit(),
        offsets[index], view.getLayoutMapAttr());
    select.getResult().replaceAllUsesWith(result.getResult());
    select.erase();
  }
  cloned_phase.erase();
  return mlir::success();
}

}  // namespace

mlir::LogicalResult scheduler::lowerDoubleBuffering(
    mlir::func::FuncOp func, mlir::ktdf_arch::Mapping& mapping) {
  while (true) {
    mlir::ktdf::BufferPhaseOp found = nullptr;
    func.walk([&](mlir::ktdf::BufferPhaseOp bp) {
      found = bp;
      return mlir::WalkResult::interrupt();
    });
    if (!found) break;
    if (mlir::failed(processOneBufferPhasePair(found, mapping)))
      return mlir::failure();
  }
  return mlir::success();
}
