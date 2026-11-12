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

// RUN: dataflow-scheduler-opt -mlir-print-local-scope -ktir-to-dfir %s | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=ktdf_lowering.native_cast --implicit-check-not=ktdf.buffer_phase --implicit-check-not=ktdf.select_memref --implicit-check-not=!spyreop.df16

// RUN: dataflow-scheduler-opt -pipeline-numeric-cast %s | FileCheck %s --check-prefix=PIPE
// RUN: dataflow-scheduler-opt -pipeline-numeric-cast -double-buffering %s | FileCheck %s --check-prefix=DB

// Each iteration owns one 64-element cast and all of its memory transfers.
// PIPE-LABEL: func.func @local_schedule_0(
// PIPE: %[[IN:.*]] = ktdp.construct_memory_view {{.*}} : memref<3x64xf16, #ktdp.memory_space<global>>
// PIPE: %[[OUT:.*]] = ktdp.construct_memory_view {{.*}} : memref<6x32xf32, #ktdp.memory_space<global>>
// PIPE: %[[ZERO:.*]] = arith.constant 0 : index
// PIPE: %[[ONE:.*]] = arith.constant 1 : index
// PIPE: %[[THREE:.*]] = arith.constant 3 : index
// PIPE: scf.for %[[T:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] {
// PIPE-NEXT: ktdf.pipeline {
// PIPE-NEXT: %[[P:.*]]:11 = ktdf.private -> (memref<1x64xf16, #ktdp.memory_space<ct_local>>, memref<2x32xf32, #ktdp.memory_space<ct_local>>
// PIPE: memref.alloc() {alignment = 128 : i64} : memref<1x64xf16, #ktdp.memory_space<ct_local>>
// PIPE: memref.alloc() {alignment = 128 : i64} : memref<2x32xf32, #ktdp.memory_space<ct_local>>
// PIPE: ktdf.private_yield
// PIPE-DAG: ktdf.stage depends_in(none) depends_out(%[[P]]#6)
// PIPE-DAG: ktdf.stage depends_in(%[[P]]#6) depends_out(%[[P]]#7)
// PIPE-DAG: ktdf.stage depends_in(%[[P]]#7) depends_out(%[[P]]#8)
// PIPE-DAG: ktdf.stage depends_in(%[[P]]#8) depends_out(%[[P]]#9)
// PIPE-DAG: ktdf.stage depends_in(%[[P]]#9) depends_out(%[[P]]#10)
// PIPE-DAG: ktdf.stage depends_in(%[[P]]#10) depends_out(none)
// PIPE-DAG: {applicable_units = ["MNILU"]}
// PIPE-DAG: {applicable_units = ["L1LU"]}
// PIPE-DAG: {applicable_units = ["MPE_ARRAY"]}
// PIPE-DAG: {applicable_units = ["SFU"]}
// PIPE-DAG: {applicable_units = ["L1SU"]}
// PIPE-DAG: {applicable_units = ["MNISU"]}
// PIPE-DAG: ktdf.data_transfer from %[[IN]][%[[T]], 0] size [1, 64] to %[[P]]#0[0, 0] size [1, 64]
// PIPE-DAG: ktdf.data_transfer from %[[P]]#1[0, 0] size [2, 32] to %[[OUT]][%[[T]] * 2, 0] size [2, 32]
// PIPE-DAG: ktdf.data_transfer from %[[P]]#0[0, 0] size [1, 64] to %[[P]]#2 size [64]
// PIPE-DAG: ktdf_lowering.native_cast ins(%[[P]]#3 : !ktdf.fifo.slot<"MPE_ARRAY" -> "SFU", 64xf16>) outs(%[[P]]#4, %[[P]]#5 : !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {kind = "dl16tofp32"}
// PIPE-DAG: ktdf.data_transfer from %[[P]]#4 size [32] to %[[P]]#1[0, 0] size [1, 32]
// PIPE-DAG: ktdf.data_transfer from %[[P]]#5 size [32] to %[[P]]#1[1, 0] size [1, 32]

// Existing double buffering pairs both scratchpads outside the tile loop.
// DB-LABEL: func.func @local_schedule_0(
// DB: %[[ZERO:.*]] = arith.constant 0 : index
// DB: %[[ONE:.*]] = arith.constant 1 : index
// DB: %[[THREE:.*]] = arith.constant 3 : index
// DB-DAG: %[[IN0:.*]] = memref.alloc() {{.*}} : memref<1x64xf16, #ktdp.memory_space<ct_local>>
// DB-DAG: %[[IN1:.*]] = memref.alloc() {{.*}} : memref<1x64xf16, #ktdp.memory_space<ct_local>>
// DB-DAG: %[[OUT0:.*]] = memref.alloc() {{.*}} : memref<2x32xf32, #ktdp.memory_space<ct_local>>
// DB-DAG: %[[OUT1:.*]] = memref.alloc() {{.*}} : memref<2x32xf32, #ktdp.memory_space<ct_local>>
// DB: scf.for %[[T:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] {
// DB-NEXT: %[[PHASE:.*]] = ktdf.buffer_phase(%[[T]]) {num_phases = 2 : i64} : index
// DB-NEXT: %[[INPUT:.*]] = ktdf.select_memref %[[PHASE]][%[[IN0]], %[[IN1]]]
// DB-NEXT: %[[OUTPUT:.*]] = ktdf.select_memref %[[PHASE]][%[[OUT0]], %[[OUT1]]]
// DB-NEXT: ktdf.pipeline modulo(size : 2) {
// DB-DAG: ktdf.data_transfer from {{.*}}[%[[T]], 0] size [1, 64] to %[[INPUT]][0, 0] size [1, 64]
// DB-DAG: ktdf.data_transfer from %[[INPUT]][0, 0]
// DB-DAG: ktdf.data_transfer from {{.*}} to %[[OUTPUT]][0, 0]
// DB-DAG: ktdf.data_transfer from %[[OUTPUT]][0, 0] size [2, 32] to {{.*}}[%[[T]] * 2, 0] size [2, 32]

// Native f16 storage carries SEN169 bits, not IEEE f16 values.
// CHECK-LABEL: func.func @widen_192(
// CHECK: call @local_schedule_0
// CHECK-LABEL: func.func @local_schedule_0(
// CHECK-SAME: %[[INPUT_ADDR:.*]]: index, %[[OUTPUT_ADDR:.*]]: index)
// CHECK: %[[WORD:.*]] = arith.constant 128 : index
// CHECK: arith.divsi %[[INPUT_ADDR]], %[[WORD]] : index
// CHECK: arith.divsi %[[OUTPUT_ADDR]], %[[WORD]] : index
// CHECK-NOT: arith.divsi
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[THREE:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[IN_SUM:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[OUT_SUM:.*]] = arith.constant 192 : index
// CHECK-DAG: %[[MNILU:.*]] = dataflow.get_unit {{.*}}type = "mnilu"
// CHECK-DAG: %[[L1LU:.*]] = dataflow.get_unit {{.*}}type = "l1lu"
// CHECK-DAG: %[[MPE_ARRAY:.*]] = dataflow.get_unit {{.*}}type = "mpe_array"
// CHECK-DAG: %[[SFU:.*]] = dataflow.get_unit {{.*}}type = "sfu"
// CHECK-DAG: %[[L1SU:.*]] = dataflow.get_unit {{.*}}type = "l1su"
// CHECK-DAG: %[[MNISU:.*]] = dataflow.get_unit {{.*}}type = "mnisu"

// HBM fetch advances by the number of physical packets per cast.
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[MNILU]])
// CHECK: %[[HBM_IN:.*]] = dataflow.get_logical_memory_view {{.*}} {{.*}} : index, index, memref<3x64xf16>
// CHECK: scf.for %[[FETCH:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] iter_args(%[[IN_FETCH:.*]] = %[[ZERO]]) -> (index) {
// CHECK: %[[LOCAL_IN:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[IN_FETCH]] {{.*}} : index, index, memref<1x64xf16>
// CHECK: agen.composite_load_and_store src:%[[HBM_IN]][%[[FETCH]], 0] dst:%[[LOCAL_IN]][0, 0]
// CHECK: load_iv({{.*}}:vector<64xf16>)
// CHECK: time_set = affine_set<(d0) : (d0 == 0)>
// CHECK: dataflow.sync_send
// CHECK: dataflow.sync_recv
// CHECK: %[[NEXT_IN_FETCH:.*]] = arith.subi %[[IN_SUM]], %[[IN_FETCH]] : index
// CHECK-NEXT: scf.yield %[[NEXT_IN_FETCH]] : index

// Local memory loads each packet exactly once before the relay.
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[L1LU]])
// CHECK: scf.for %[[LOAD:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] iter_args(%[[IN_LOAD:.*]] = %[[ZERO]]) -> (index) {
// CHECK: %[[INPUT_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[IN_LOAD]] {{.*}} : index, index, memref<1x64xf16>
// CHECK: dataflow.sync_recv
// CHECK: %[[LOADED0:.*]] = agen.vector_load %[[INPUT_VIEW]][0, 0] {{.*}} : memref<1x64xf16>, vector<64xf16>
// CHECK: dataflow.send {{.*}}, %[[LOADED0]] : vector<64xf16>
// CHECK: %[[NEXT_IN_LOAD:.*]] = arith.subi %[[IN_SUM]], %[[IN_LOAD]] : index
// CHECK-NEXT: scf.yield %[[NEXT_IN_LOAD]] : index
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[MPE_ARRAY]])
// CHECK: scf.for %[[RELAY:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] {
// CHECK: %[[RELAY0:.*]] = dataflow.receive {{.*}} : vector<64xf16>
// CHECK: dataflow.send {{.*}}, %[[RELAY0]] : vector<64xf16>
// CHECK-NOT: dataflow.receive

// The native recipe preserves the staggered four-lane groups.
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[SFU]])
// CHECK: scf.for %[[CAST:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] {
// CHECK: %[[INPUT:.*]] = dataflow.receive {{.*}} : vector<64xf16>
// CHECK-NEXT: %[[CAST0:.*]] = vectorchain.cast %[[INPUT]] : vector<64xf16>, vector<64xf32>
// CHECK-NEXT: %[[PACKET0:.*]] = vectorchain.shuffle input(%[[CAST0]]) {indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32], repetition = 8 : i32} : vector<64xf32>, vector<32xf32>
// CHECK-NEXT: dataflow.send {{.*}}, %[[PACKET0]] : vector<32xf32>
// CHECK-NEXT: %[[CAST1:.*]] = vectorchain.cast %[[INPUT]] : vector<64xf16>, vector<64xf32>
// CHECK-NEXT: %[[PACKET1:.*]] = vectorchain.shuffle input(%[[CAST1]]) {indices = [4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf32>, vector<32xf32>
// CHECK-NEXT: dataflow.send {{.*}}, %[[PACKET1]] : vector<32xf32>
// CHECK-NOT: dataflow.receive

// Local memory and HBM stores use the same selected output scratchpad.
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[L1SU]])
// CHECK: scf.for %[[STORE:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] iter_args(%[[OUT_STORE:.*]] = %[[IN_SUM]]) -> (index) {
// CHECK: %[[OUTPUT_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[OUT_STORE]] {{.*}} : index, index, memref<2x32xf32>
// CHECK: %[[STORED0:.*]] = dataflow.receive {{.*}} : vector<32xf32>
// CHECK-NEXT: agen.vector_store %[[STORED0]], %[[OUTPUT_VIEW]][0, 0] {{.*}} : memref<2x32xf32>, vector<32xf32>
// CHECK: %[[STORED1:.*]] = dataflow.receive {{.*}} : vector<32xf32>
// CHECK-NEXT: agen.vector_store %[[STORED1]], %[[OUTPUT_VIEW]][1, 0] {{.*}} : memref<2x32xf32>, vector<32xf32>
// CHECK: dataflow.sync_send
// CHECK: dataflow.sync_recv
// CHECK: %[[NEXT_OUT_STORE:.*]] = arith.subi %[[OUT_SUM]], %[[OUT_STORE]] : index
// CHECK-NEXT: scf.yield %[[NEXT_OUT_STORE]] : index
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[MNISU]])
// CHECK: %[[HBM_OUT:.*]] = dataflow.get_logical_memory_view {{.*}} {{.*}} : index, index, memref<6x32xf32>
// CHECK: scf.for %[[DRAIN:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] iter_args(%[[OUT_DRAIN:.*]] = %[[IN_SUM]]) -> (index) {
// CHECK: %[[LOCAL_OUT:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[OUT_DRAIN]] {{.*}} : index, index, memref<2x32xf32>
// CHECK: dataflow.sync_recv
// CHECK: agen.composite_load_and_store src:%[[LOCAL_OUT]][0, 0] dst:%[[HBM_OUT]][%[[DRAIN]] * 2, 0]
// CHECK: load_iv({{.*}}:vector<32xf32>)
// CHECK: time_set = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
// CHECK: %[[NEXT_OUT_DRAIN:.*]] = arith.subi %[[OUT_SUM]], %[[OUT_DRAIN]] : index
// CHECK-NEXT: scf.yield %[[NEXT_OUT_DRAIN]] : index
// CHECK: return

#input = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#output = affine_set<(d0, d1) : (d0 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device attributes {} import("Inputs/device.mlir")
  func.func @widen_192(%input: index, %output: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %input_view = ktdp.construct_memory_view %input, sizes: [3, 64], strides: [64, 1] {
      coordinate_set = #input, memory_space = #ktdp.memory_space<global>
    } : memref<3x64xf16>
    %output_view = ktdp.construct_memory_view %output, sizes: [6, 32], strides: [32, 1] {
      coordinate_set = #output, memory_space = #ktdp.memory_space<global>
    } : memref<6x32xf32>
    %input_tile = ktdp.construct_access_tile %input_view[%zero, %zero] {
      access_tile_set = #input, access_tile_order = #identity
    } : memref<3x64xf16> -> !ktdp.access_tile<3x64xindex>
    %output_tile = ktdp.construct_access_tile %output_view[%zero, %zero] {
      access_tile_set = #output, access_tile_order = #identity
    } : memref<6x32xf32> -> !ktdp.access_tile<6x32xindex>
    %value = ktdp.load %input_tile : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>
    %result = spyreop.numeric_cast %value {kind = "dl16tofp32"} : tensor<3x64xf16> -> tensor<6x32xf32>
    ktdp.store %result, %output_tile : tensor<6x32xf32>, !ktdp.access_tile<6x32xindex>
    return
  }
}
