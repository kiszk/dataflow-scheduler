//===-- RelayoutMaterialization.h ----------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_FRONTEND_KTIRTOSCHEDULEIR_RELAYOUTMATERIALIZATION_H_
#define DATAFLOW_SCHEDULER_CONVERSION_FRONTEND_KTIRTOSCHEDULEIR_RELAYOUTMATERIALIZATION_H_

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "ktir/Dialect/SpyreOp/SpyreOp.h"

namespace scheduler {
// Inspect the scalar body; the generic owns the tensor shapes and maps.
mlir::spyreop::NumericCast getNumericCast(mlir::Operation* op);
// Validate the native packet domain and return whether FP32 is Standard.
mlir::FailureOr<bool> getNumericCastLayout(mlir::linalg::GenericOp generic);

// A matched recipe is a bijection over the complete static tensor domain.
// true: Standard -> Staggered; false: Staggered -> Standard.
std::optional<bool> getPacketRelayoutDirection(mlir::Operation* op);
llvm::SmallVector<mlir::AffineMap> getPacketRelayoutMaps(
    unsigned rank, bool to_staggered, mlir::MLIRContext* ctx);

// Resolve tiled identity generics into ordinary local-memory transfer stages.
mlir::LogicalResult materializeRelayouts(mlir::func::FuncOp func,
                                         mlir::ktdf_arch::Mapping& mapping,
                                         const AttrMapping& spaces);
}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_FRONTEND_KTIRTOSCHEDULEIR_RELAYOUTMATERIALIZATION_H_
