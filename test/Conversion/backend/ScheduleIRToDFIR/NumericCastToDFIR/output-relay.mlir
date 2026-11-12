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

// Move the relay after the cast; each FP32 packet must reach the store unchanged.
// RUN: sed -n 's@^// ROUTE: @@p' %s > %t.route.sed
// RUN: sed -f %t.route.sed %S/Inputs/device.mlir > %t.device.mlir
// RUN: sed 's@Inputs/device.mlir@%t.device.mlir@' %S/widen.mlir > %t.widen.mlir
// RUN: dataflow-scheduler-opt -pipeline-numeric-cast %t.widen.mlir | FileCheck %s --check-prefix=PIPE
// RUN: dataflow-scheduler-opt -mlir-print-local-scope -ktir-to-dfir %t.widen.mlir | FileCheck %s --check-prefix=DFIR --implicit-check-not=ktdf_lowering.native_cast

// ROUTE: s/%l1lu to %mpe_array/%l1lu to %sfu/
// ROUTE: /datapath #L1LU_SUBCORE_FIFO [%]mpe_array to [%]sfu/d
// ROUTE: s/%sfu to %l1su/%sfu to %mpe_array/
// ROUTE: /datapath #L1SU_SUBCORE_FIFO [%]sfu to [%]mpe_array/a\      datapath #L1SU_SUBCORE_FIFO %mpe_array to %l1su : exec_unit, exec_unit

// PIPE-LABEL: func.func @local_schedule_0(
// PIPE: %[[P:.*]]:12 = ktdf.private
// PIPE: ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// PIPE: ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "MPE_ARRAY", 32xf32>, !ktdf.fifo.slot<"SFU" -> "MPE_ARRAY", 32xf32>
// PIPE: ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MPE_ARRAY" -> "L1SU", 32xf32>, !ktdf.fifo.slot<"MPE_ARRAY" -> "L1SU", 32xf32>
// PIPE: ktdf_lowering.native_cast ins(%[[P]]#2 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) outs(%[[P]]#3, %[[P]]#4 : !ktdf.fifo.slot<"SFU" -> "MPE_ARRAY", 32xf32>, !ktdf.fifo.slot<"SFU" -> "MPE_ARRAY", 32xf32>) {kind = "dl16tofp32"}
// PIPE: %[[PACKET0:.*]] = ktdf.read_from_fifo %[[P]]#3 : <"SFU" -> "MPE_ARRAY", 32xf32> -> tensor<32xf32>
// PIPE-NEXT: ktdf.write_to_fifo %[[PACKET0]], %[[P]]#5 : tensor<32xf32>, <"MPE_ARRAY" -> "L1SU", 32xf32>
// PIPE-NEXT: %[[PACKET1:.*]] = ktdf.read_from_fifo %[[P]]#4 : <"SFU" -> "MPE_ARRAY", 32xf32> -> tensor<32xf32>
// PIPE-NEXT: ktdf.write_to_fifo %[[PACKET1]], %[[P]]#6 : tensor<32xf32>, <"MPE_ARRAY" -> "L1SU", 32xf32>
// PIPE-NEXT: } {applicable_units = ["MPE_ARRAY"]}
// PIPE: ktdf.data_transfer from %[[P]]#5 size [32] to %[[P]]#1[0, 0] size [1, 32]
// PIPE-NEXT: ktdf.data_transfer from %[[P]]#6 size [32] to %[[P]]#1[1, 0] size [1, 32]

// DFIR-LABEL: func.func @local_schedule_0_body()
// DFIR-DAG: %[[COMPUTE:.*]] = dataflow.get_unit {{.*}}type = "sfu"
// DFIR-DAG: %[[RELAY:.*]] = dataflow.get_unit {{.*}}type = "mpe_array"
// DFIR-DAG: %[[STORE:.*]] = dataflow.get_unit {{.*}}type = "l1su"
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[COMPUTE]])
// DFIR: uniform.def_immutable_mapping([%[[COMPUTE]] -> %[[RELAY]]])
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[RELAY]])
// DFIR: %[[SOURCE_MAP:.*]] = uniform.def_immutable_mapping([%[[RELAY]] -> %[[COMPUTE]]])
// DFIR-NEXT: %[[SOURCE:.*]] = uniform.query_map(map:%[[SOURCE_MAP]], key:{{.*}})
// DFIR-NEXT: %[[PACKET0:.*]] = dataflow.receive %[[SOURCE]] : vector<32xf32>
// DFIR-NEXT: %[[DEST_MAP:.*]] = uniform.def_immutable_mapping([%[[RELAY]] -> %[[STORE]]])
// DFIR-NEXT: %[[DEST:.*]] = uniform.query_map(map:%[[DEST_MAP]], key:{{.*}})
// DFIR-NEXT: dataflow.send %[[DEST]], %[[PACKET0]] : vector<32xf32>
// DFIR: %[[PACKET1:.*]] = dataflow.receive {{.*}} : vector<32xf32>
// DFIR: dataflow.send {{.*}}, %[[PACKET1]] : vector<32xf32>
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[STORE]])
// DFIR: uniform.def_immutable_mapping([%[[STORE]] -> %[[RELAY]]])
// DFIR: agen.vector_store {{.*}}[0, 0] {{.*}} : memref<2x32xf32>, vector<32xf32>
// DFIR: agen.vector_store {{.*}}[1, 0] {{.*}} : memref<2x32xf32>, vector<32xf32>
