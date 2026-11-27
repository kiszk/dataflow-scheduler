//===-- MpeMatmul.h ----------------------------------------------*- c++
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

#ifndef DATAFLOW_SCHEDULER_KTIRTOSCHEDULEIR_MPEMATMUL_H_
#define DATAFLOW_SCHEDULER_KTIRTOSCHEDULEIR_MPEMATMUL_H_

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace scheduler {
bool isMpeMatmul(mlir::linalg::LinalgOp op, mlir::ktdf_arch::Mapping& mapping);
/// Retain target contractions until the post-frontend target validation step.
bool deferMpeMatmulPipelines(mlir::func::FuncOp func,
                             mlir::ktdf_arch::Mapping& mapping);

/// Construct the transfer/compute pipelines for validated contractions.
/// The target extension must validate the original tensor operations first.
void prepareMpeMatmulPipelines(mlir::func::FuncOp func);

struct MpeMatmulChunkPlan {
  int64_t k;
  int64_t buffers;
};
/// Plans for a validated, prepared contraction. Full-K is tried by the caller
/// before these candidates; no target queries are needed for enumeration.
llvm::SmallVector<MpeMatmulChunkPlan> getMpeMatmulChunkPlans(
    mlir::linalg::GenericOp op);
mlir::LogicalResult chunkMpeMatmulPipeline(mlir::linalg::GenericOp op,
                                           MpeMatmulChunkPlan plan);
}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_KTIRTOSCHEDULEIR_MPEMATMUL_H_
