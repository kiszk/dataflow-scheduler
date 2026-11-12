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

// A direct edge must win over the longer route through MPE_ARRAY.
// RUN: sed '/datapath #L1LU_SUBCORE_FIFO [%]l1lu to [%]mpe_array/a\      datapath #L1LU_SUBCORE_FIFO %l1lu to %%sfu : exec_unit, exec_unit' %S/Inputs/device.mlir > %t.device.mlir
// RUN: sed 's@Inputs/device.mlir@%t.device.mlir@' %S/widen.mlir > %t.widen.mlir
// RUN: sed 's@Inputs/device.mlir@%t.device.mlir@' %S/narrow.mlir > %t.narrow.mlir
// RUN: dataflow-scheduler-opt -pipeline-numeric-cast %t.widen.mlir | FileCheck %s --check-prefixes=PIPE,WIDE --implicit-check-not=ktdf.read_from_fifo
// RUN: dataflow-scheduler-opt -pipeline-numeric-cast %t.narrow.mlir | FileCheck %s --check-prefixes=PIPE,NARROW --implicit-check-not=ktdf.read_from_fifo
// RUN: dataflow-scheduler-opt -mlir-print-local-scope -ktir-to-dfir %t.widen.mlir | FileCheck %s --check-prefixes=DFIR,WIDE-DFIR --implicit-check-not='"mpe_array"' --implicit-check-not=ktdf_lowering.native_cast
// RUN: dataflow-scheduler-opt -mlir-print-local-scope -ktir-to-dfir %t.narrow.mlir | FileCheck %s --check-prefixes=DFIR,NARROW-DFIR --implicit-check-not='"mpe_array"' --implicit-check-not=ktdf_lowering.native_cast

// The native cast uses the load-to-compute FIFO directly, with one native
// packet or two FP32 packets. No relay stage or relay FIFO is required.
// PIPE-LABEL: func.func @local_schedule_0(
// PIPE: ktdf.pipeline {
// PIPE: %[[P:.*]]:9 = ktdf.private
// WIDE: ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// NARROW: ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
// PIPE: {applicable_units = ["MNILU"]}
// PIPE: {applicable_units = ["L1LU"]}
// WIDE: ktdf_lowering.native_cast ins(%[[P]]#2 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) outs(%[[P]]#3, %[[P]]#4 : !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {kind = "dl16tofp32"}
// NARROW: ktdf_lowering.native_cast ins(%[[P]]#2, %[[P]]#3 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>) outs(%[[P]]#4 : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>) {kind = "fp32todl16"}
// PIPE: {applicable_units = ["SFU"]}
// PIPE: {applicable_units = ["L1SU"]}
// PIPE: {applicable_units = ["MNISU"]}

// Both DFIR endpoints name the same direct path; casts retain packet order.
// DFIR-LABEL: func.func @local_schedule_0_body()
// DFIR-DAG: %[[LOAD:.*]] = dataflow.get_unit {{.*}}type = "l1lu"
// DFIR-DAG: %[[COMPUTE:.*]] = dataflow.get_unit {{.*}}type = "sfu"
// DFIR-DAG: %[[STORE:.*]] = dataflow.get_unit {{.*}}type = "l1su"
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[LOAD]])
// DFIR: %[[LOAD_MAP:.*]] = uniform.def_immutable_mapping([%[[LOAD]] -> %[[COMPUTE]]])
// DFIR-NEXT: %[[DEST:.*]] = uniform.query_map(map:%[[LOAD_MAP]], key:{{.*}})
// WIDE-DFIR-NEXT: dataflow.send %[[DEST]], {{.*}} : vector<64xf16>
// NARROW-DFIR-NEXT: dataflow.send %[[DEST]], {{.*}} : vector<32xf32>
// NARROW-DFIR: dataflow.send {{.*}} : vector<32xf32>
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[COMPUTE]])
// DFIR: %[[COMPUTE_MAP:.*]] = uniform.def_immutable_mapping([%[[COMPUTE]] -> %[[LOAD]]])
// DFIR-NEXT: %[[SOURCE:.*]] = uniform.query_map(map:%[[COMPUTE_MAP]], key:{{.*}})
// DFIR: %[[STORE_MAP:.*]] = uniform.def_immutable_mapping([%[[COMPUTE]] -> %[[STORE]]])
// DFIR-NEXT: %[[DEST:.*]] = uniform.query_map(map:%[[STORE_MAP]], key:{{.*}})
// WIDE-DFIR: %[[IN:.*]] = dataflow.receive %[[SOURCE]] : vector<64xf16>
// WIDE-DFIR-NEXT: %[[CAST0:.*]] = vectorchain.cast %[[IN]] : vector<64xf16>, vector<64xf32>
// WIDE-DFIR-NEXT: %[[OUT0:.*]] = vectorchain.shuffle input(%[[CAST0]]) {indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32], repetition = 8 : i32} : vector<64xf32>, vector<32xf32>
// WIDE-DFIR-NEXT: dataflow.send %[[DEST]], %[[OUT0]] : vector<32xf32>
// WIDE-DFIR-NEXT: %[[CAST1:.*]] = vectorchain.cast %[[IN]] : vector<64xf16>, vector<64xf32>
// WIDE-DFIR-NEXT: %[[OUT1:.*]] = vectorchain.shuffle input(%[[CAST1]]) {indices = [4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf32>, vector<32xf32>
// WIDE-DFIR-NEXT: dataflow.send %[[DEST]], %[[OUT1]] : vector<32xf32>
// NARROW-DFIR: %[[IN0:.*]] = dataflow.receive %[[SOURCE]] : vector<32xf32>
// NARROW-DFIR-NEXT: %[[CAST0:.*]] = vectorchain.cast %[[IN0]] : vector<32xf32>, vector<32xf16>
// NARROW-DFIR-NEXT: %[[IN1:.*]] = dataflow.receive %[[SOURCE]] : vector<32xf32>
// NARROW-DFIR-NEXT: %[[CAST1:.*]] = vectorchain.cast %[[IN1]] : vector<32xf32>, vector<32xf16>
// NARROW-DFIR-NEXT: %[[OUT:.*]] = vectorchain.pack %[[CAST0]], %[[CAST1]] {indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : index} : vector<32xf16>, vector<32xf16>, vector<64xf16>
// NARROW-DFIR-NEXT: dataflow.send %[[DEST]], %[[OUT]] : vector<64xf16>
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[STORE]])
// WIDE-DFIR: agen.vector_store {{.*}} : memref<2x32xf32>, vector<32xf32>
// WIDE-DFIR: agen.vector_store {{.*}} : memref<2x32xf32>, vector<32xf32>
// NARROW-DFIR: agen.vector_store {{.*}} : memref<1x64xf16>, vector<64xf16>
