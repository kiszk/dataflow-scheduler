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

// A direct edge bypasses the relay and keeps both FP32 packets.
// RUN: sed '/datapath #L1LU_SUBCORE_FIFO [%]l1lu to [%]mpe_array/a\      datapath #L1LU_SUBCORE_FIFO %l1lu to %%sfu : exec_unit, exec_unit' %S/Inputs/device.mlir > %t.device.mlir
// RUN: awk -v wanted=0 '/^\/\/ -----$/ {section++; next} section == wanted {print}' %S/numeric-cast.mlir | sed 's@Inputs/device.mlir@%t.device.mlir@' > %t.widen.mlir
// RUN: awk -v wanted=1 '/^\/\/ -----$/ {section++; next} section == wanted {print}' %S/numeric-cast.mlir | sed 's@Inputs/device.mlir@%t.device.mlir@' > %t.narrow.mlir
// RUN: dataflow-scheduler-opt -ktir-to-dfir %t.widen.mlir | FileCheck %s --check-prefix=WIDE --implicit-check-not='"mpe_array"' --implicit-check-not=ktdf_lowering.native_cast
// RUN: dataflow-scheduler-opt -ktir-to-dfir %t.narrow.mlir | FileCheck %s --check-prefix=NARROW --implicit-check-not='"mpe_array"' --implicit-check-not=ktdf_lowering.native_cast

// WIDE: dataflow.get_unit {{.*}}type = "l1lu"
// WIDE: dataflow.get_unit {{.*}}type = "sfu"
// WIDE: dataflow.get_unit {{.*}}type = "l1su"
// WIDE: vectorchain.cast {{.*}} : vector<64xf16>, vector<64xf32>
// WIDE: vectorchain.shuffle {{.*}}indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
// WIDE: vectorchain.shuffle {{.*}}indices = [4 : i32, 5 : i32, 6 : i32, 7 : i32]
// WIDE: dataflow.send {{.*}} : vector<32xf32>
// WIDE: dataflow.send {{.*}} : vector<32xf32>
// NARROW: dataflow.get_unit {{.*}}type = "l1lu"
// NARROW: dataflow.get_unit {{.*}}type = "sfu"
// NARROW: dataflow.get_unit {{.*}}type = "l1su"
// NARROW: vectorchain.cast {{.*}} : vector<32xf32>, vector<32xf16>
// NARROW: vectorchain.pack
// NARROW: dataflow.send {{.*}} : vector<64xf16>
