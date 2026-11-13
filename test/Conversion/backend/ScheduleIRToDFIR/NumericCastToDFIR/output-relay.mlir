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

// Two FP32 packets pass through the compute relay unchanged.
// RUN: sed -n 's@^// ROUTE: @@p' %s > %t.route.sed
// RUN: sed -f %t.route.sed %S/Inputs/device.mlir > %t.device.mlir
// RUN: awk -v wanted=0 '/^\/\/ -----$/ {section++; next} section == wanted {print}' %S/numeric-cast.mlir | sed 's@Inputs/device.mlir@%t.device.mlir@' > %t.widen.mlir
// RUN: dataflow-scheduler-opt -ktir-to-dfir %t.widen.mlir | FileCheck %s --check-prefix=DFIR --implicit-check-not=ktdf_lowering.native_cast

// ROUTE: s/%l1lu to %mpe_array/%l1lu to %sfu/
// ROUTE: /datapath #L1LU_SUBCORE_FIFO [%]mpe_array to [%]sfu/d
// ROUTE: s/%sfu to %l1su/%sfu to %mpe_array/
// ROUTE: /datapath #L1SU_SUBCORE_FIFO [%]sfu to [%]mpe_array/a\      datapath #L1SU_SUBCORE_FIFO %mpe_array to %l1su : exec_unit, exec_unit

// DFIR: dataflow.get_unit {{.*}}type = "sfu"
// DFIR: %[[RELAY:.*]] = dataflow.get_unit {{.*}}type = "mpe_array"
// DFIR: dataflow.get_unit {{.*}}type = "l1su"
// DFIR: vectorchain.cast {{.*}} : vector<64xf16>, vector<64xf32>
// DFIR: dataflow.send {{.*}} : vector<32xf32>
// DFIR: dataflow.send {{.*}} : vector<32xf32>
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[RELAY]])
// DFIR: dataflow.receive {{.*}} : vector<32xf32>
// DFIR: dataflow.send {{.*}} : vector<32xf32>
// DFIR: dataflow.receive {{.*}} : vector<32xf32>
// DFIR: dataflow.send {{.*}} : vector<32xf32>
