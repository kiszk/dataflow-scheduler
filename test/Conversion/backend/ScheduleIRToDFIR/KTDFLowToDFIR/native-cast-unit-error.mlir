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

// RUN: sed -n '/^!input =/,$p' %S/native-cast-architecture-errors.mlir | sed -e 's@../NumericCastToDFIR/Inputs/device.mlir@%S/../NumericCastToDFIR/Inputs/device.mlir@' -e 's/@native_cast_capability/@wrong_compute_unit/' -e 's/-> (%compute) : {/-> (%relay) : {/' -e 's/-> (%relay) : {}/-> (%compute) : {}/' > %t.mlir
// RUN: not dataflow-scheduler-opt -ktdflowering-to-dfir %t.mlir 2>&1 | FileCheck %s
// RUN: sed '/^!input\|^!output/s/"SFU"/"MPE_ARRAY"/' %t.mlir > %t.compute.mlir
// RUN: not dataflow-scheduler-opt -ktdflowering-to-dfir %t.compute.mlir 2>&1 | FileCheck %s --check-prefix=COMPUTE

// The slots agree on the compute endpoint, but the operation runs elsewhere.
// CHECK: error: native_cast FIFO endpoints do not match the executing program_unit
// COMPUTE: error: native_cast requires a programmable conversion unit
