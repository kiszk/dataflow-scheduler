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

#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"

namespace scheduler {
#define GEN_PASS_DEF_PREPARETARGETOPERATIONSPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {
struct PrepareTargetOperationsPass
    : scheduler::impl::PrepareTargetOperationsPassBase<
          PrepareTargetOperationsPass> {
  explicit PrepareTargetOperationsPass(
      const scheduler::SchedulerExtContext& ctx)
      : ctx_(ctx) {}

  void runOnOperation() override {
    if (mlir::failed(ctx_.prepareTargetOperations(getOperation())))
      signalPassFailure();
  }

 private:
  const scheduler::SchedulerExtContext& ctx_;
};
}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createPrepareTargetOperationsPass(
    const SchedulerExtContext& ctx) {
  return std::make_unique<PrepareTargetOperationsPass>(ctx);
}

std::unique_ptr<mlir::Pass> scheduler::createPrepareTargetOperationsPass() {
  return createPrepareTargetOperationsPass(SchedulerExtContext::dummyContext());
}
