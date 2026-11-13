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

// RUN: sed '/ktdf_arch.feature.compute/d' %S/../../../backend/ScheduleIRToDFIR/NumericCastToDFIR/Inputs/device.mlir > %t.device.mlir
// RUN: awk -v wanted=0 '/^\/\/ -----$/ {section++; next} section == wanted {print}' %S/../../../backend/ScheduleIRToDFIR/NumericCastToDFIR/numeric-cast.mlir | sed 's@Inputs/device.mlir@%t.device.mlir@' > %t.input.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s

// CHECK: error: numeric cast requires one default compute kind
