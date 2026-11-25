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

#include "dataflow-scheduler/Utils/SchedulerExtContext.h"

#include <mlir/IR/Attributes.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

using namespace scheduler;

SchedulerExtContext::SchedulerExtContext() = default;

SchedulerExtContext::~SchedulerExtContext() = default;

const SchedulerExtContext& SchedulerExtContext::dummyContext() {
  // The dummy context is initialized thread-safe and never written to.
  static const DummySchedulerExtContext dummy_ctx;
  return dummy_ctx;
}

namespace {
mlir::LogicalResult rejectTargetOperations(mlir::Operation* root) {
  return root->walk([](mlir::Operation* op) {
               if (!op->hasAttr("dataflow_scheduler.requires_target_lowering"))
                 return mlir::WalkResult::advance();
               op->emitError(
                   "operation requires a target-specific scheduler extension");
               return mlir::WalkResult::interrupt();
             }).wasInterrupted()
             ? mlir::failure()
             : mlir::success();
}
}  // namespace

mlir::LogicalResult SchedulerExtContext::prepareTargetOperations(
    mlir::ModuleOp module) const {
  return rejectTargetOperations(module);
}

mlir::LogicalResult SchedulerExtContext::lowerTargetOperations(
    mlir::func::FuncOp func) const {
  return rejectTargetOperations(func);
}
