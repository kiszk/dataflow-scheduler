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

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_ALLOCATIONPLAN_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_ALLOCATIONPLAN_H_

#include "dataflow-scheduler/Analysis/MemoryTrackerAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"

namespace scheduler {

struct AllocationPlacement {
  mlir::Operation* allocation;
  size_t element_address;
};

/// A trial owns its tracker. Failed trials never change IR or live analyses.
struct AllocationPlan {
  explicit AllocationPlan(const MemoryTracker& initial) : tracker(initial) {}
  MemoryTracker tracker;
  llvm::SmallVector<AllocationPlacement> placements;
  mlir::Operation* failed_operation = nullptr;
  std::string error;
  bool capacity_exceeded = false;
  explicit operator bool() const { return !failed_operation; }
  mlir::LogicalResult diagnose() const;
};

bool hasAllocationsWithMemorySpace(mlir::ModuleOp module);
AllocationPlan planAllocations(mlir::ModuleOp module,
                               const MemoryTrackerAnalysis& analysis);
void applyAllocationPlan(const AllocationPlan& plan,
                         MemoryTrackerAnalysis& analysis);

}  // namespace scheduler
#endif
