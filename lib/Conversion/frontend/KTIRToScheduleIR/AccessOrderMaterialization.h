//===-- AccessOrderMaterialization.h ----------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_FRONTEND_KTIRTOSCHEDULEIR_ACCESSORDERMATERIALIZATION_H_
#define DATAFLOW_SCHEDULER_CONVERSION_FRONTEND_KTIRTOSCHEDULEIR_ACCESSORDERMATERIALIZATION_H_

#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"

namespace scheduler {
// Resolve supported access orders into ordinary phased transfers before
// bufferization. Unsupported orders remain for the caller's legality check.
mlir::LogicalResult materializeAccessOrders(mlir::func::FuncOp func,
                                            mlir::ktdf_arch::Mapping& mapping,
                                            const AttrMapping& spaces);
}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_FRONTEND_KTIRTOSCHEDULEIR_ACCESSORDERMATERIALIZATION_H_
