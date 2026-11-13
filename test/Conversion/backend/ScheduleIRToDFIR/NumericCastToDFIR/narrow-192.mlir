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

// RUN: dataflow-scheduler-opt -mlir-print-local-scope -ktir-to-dfir %s | FileCheck %s --check-prefix=GEN --implicit-check-not=spyreop.numeric_cast --implicit-check-not=ktdf_lowering.native_cast --implicit-check-not=ktdf.buffer_phase --implicit-check-not=ktdf.select_memref --implicit-check-not=!spyreop.df16 --implicit-check-not=vectorchain.constant_bitstream --implicit-check-not=pe_lrfreg

// RUN: dataflow-scheduler-opt -pipeline-numeric-cast %s | FileCheck %s --check-prefix=PIPE
// RUN: dataflow-scheduler-opt -pipeline-numeric-cast -double-buffering %s | FileCheck %s --check-prefix=DB

// Each iteration owns one 64-element cast and all of its memory transfers.
// PIPE-LABEL: func.func @local_schedule_0(
// PIPE: %[[IN:.*]] = ktdp.construct_memory_view {{.*}} : memref<6x32xf32, #ktdp.memory_space<global>>
// PIPE: %[[OUT:.*]] = ktdp.construct_memory_view {{.*}} : memref<3x64xf16, #ktdp.memory_space<global>>
// PIPE: %[[ZERO:.*]] = arith.constant 0 : index
// PIPE: %[[ONE:.*]] = arith.constant 1 : index
// PIPE: %[[THREE:.*]] = arith.constant 3 : index
// PIPE: scf.for %[[T:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] {
// PIPE-NEXT: ktdf.pipeline {
// PIPE-NEXT: %[[P:.*]]:12 = ktdf.private -> (memref<2x32xf32, #ktdp.memory_space<ct_local>>, memref<1x64xf16, #ktdp.memory_space<ct_local>>
// PIPE: memref.alloc() {alignment = 128 : i64} : memref<2x32xf32, #ktdp.memory_space<ct_local>>
// PIPE: memref.alloc() {alignment = 128 : i64} : memref<1x64xf16, #ktdp.memory_space<ct_local>>
// PIPE: ktdf.private_yield
// PIPE-DAG: ktdf.stage depends_in(none) depends_out(%[[P]]#7)
// PIPE-DAG: ktdf.stage depends_in(%[[P]]#7) depends_out(%[[P]]#8)
// PIPE-DAG: ktdf.stage depends_in(%[[P]]#8) depends_out(%[[P]]#9)
// PIPE-DAG: ktdf.stage depends_in(%[[P]]#9) depends_out(%[[P]]#10)
// PIPE-DAG: ktdf.stage depends_in(%[[P]]#10) depends_out(%[[P]]#11)
// PIPE-DAG: ktdf.stage depends_in(%[[P]]#11) depends_out(none)
// PIPE-DAG: {applicable_units = ["MNILU"]}
// PIPE-DAG: {applicable_units = ["L1LU"]}
// PIPE-DAG: {applicable_units = ["MPE_ARRAY"]}
// PIPE-DAG: {applicable_units = ["SFU"]}
// PIPE-DAG: {applicable_units = ["L1SU"]}
// PIPE-DAG: {applicable_units = ["MNISU"]}
// PIPE-DAG: ktdf.data_transfer from %[[IN]][%[[T]] * 2, 0] size [2, 32] to %[[P]]#0[0, 0] size [2, 32]
// PIPE-DAG: ktdf.data_transfer from %[[P]]#1[0, 0] size [1, 64] to %[[OUT]][%[[T]], 0] size [1, 64]
// PIPE-DAG: ktdf.data_transfer from %[[P]]#0[0, 0] size [1, 32] to %[[P]]#2 size [32]
// PIPE-DAG: ktdf.data_transfer from %[[P]]#0[1, 0] size [1, 32] to %[[P]]#3 size [32]
// PIPE-DAG: ktdf_lowering.native_cast ins(%[[P]]#4, %[[P]]#5 : !ktdf.fifo.slot<"MPE_ARRAY" -> "SFU", 32xf32>, !ktdf.fifo.slot<"MPE_ARRAY" -> "SFU", 32xf32>) outs(%[[P]]#6 : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>) {kind = "fp32todl16"}
// PIPE-DAG: ktdf.data_transfer from %[[P]]#6 size [64] to %[[P]]#1[0, 0] size [1, 64]

// Existing double buffering pairs both scratchpads outside the tile loop.
// DB-LABEL: func.func @local_schedule_0(
// DB: %[[ZERO:.*]] = arith.constant 0 : index
// DB: %[[ONE:.*]] = arith.constant 1 : index
// DB: %[[THREE:.*]] = arith.constant 3 : index
// DB-DAG: %[[IN0:.*]] = memref.alloc() {{.*}} : memref<2x32xf32, #ktdp.memory_space<ct_local>>
// DB-DAG: %[[IN1:.*]] = memref.alloc() {{.*}} : memref<2x32xf32, #ktdp.memory_space<ct_local>>
// DB-DAG: %[[OUT0:.*]] = memref.alloc() {{.*}} : memref<1x64xf16, #ktdp.memory_space<ct_local>>
// DB-DAG: %[[OUT1:.*]] = memref.alloc() {{.*}} : memref<1x64xf16, #ktdp.memory_space<ct_local>>
// DB: scf.for %[[T:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] {
// DB-NEXT: %[[PHASE:.*]] = ktdf.buffer_phase(%[[T]]) {num_phases = 2 : i64} : index
// DB-NEXT: %[[INPUT:.*]] = ktdf.select_memref %[[PHASE]][%[[IN0]], %[[IN1]]]
// DB-NEXT: %[[OUTPUT:.*]] = ktdf.select_memref %[[PHASE]][%[[OUT0]], %[[OUT1]]]
// DB-NEXT: ktdf.pipeline modulo(size : 2) {
// DB-DAG: ktdf.data_transfer from {{.*}}[%[[T]] * 2, 0] size [2, 32] to %[[INPUT]][0, 0] size [2, 32]
// DB-DAG: ktdf.data_transfer from %[[INPUT]][0, 0]
// DB-DAG: ktdf.data_transfer from {{.*}} to %[[OUTPUT]][0, 0]
// DB-DAG: ktdf.data_transfer from %[[OUTPUT]][0, 0] size [1, 64] to {{.*}}[%[[T]], 0] size [1, 64]

// Native f16 storage carries SEN169 bits, not IEEE f16 values.
// GEN-LABEL: func.func @local_schedule_0_body()
// GEN: memref<6x32xf32>
// GEN: scf.for {{.*}} to %c3 step %c1
// GEN: vectorchain.cast
// GEN: vectorchain.pack {{.*}}repetition = 8
// GEN: memref<3x64xf16>
// GEN: agen.composite_load_and_store

#input = affine_set<(d0, d1) : (d0 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#output = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device attributes {} import("Inputs/device.mlir")
  func.func @narrow_192(%input: index, %output: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %input_view = ktdp.construct_memory_view %input, sizes: [6, 32], strides: [32, 1] {
      coordinate_set = #input, memory_space = #ktdp.memory_space<global>
    } : memref<6x32xf32>
    %output_view = ktdp.construct_memory_view %output, sizes: [3, 64], strides: [64, 1] {
      coordinate_set = #output, memory_space = #ktdp.memory_space<global>
    } : memref<3x64xf16>
    %input_tile = ktdp.construct_access_tile %input_view[%zero, %zero] {
      access_tile_set = #input, access_tile_order = #identity
    } : memref<6x32xf32> -> !ktdp.access_tile<6x32xindex>
    %output_tile = ktdp.construct_access_tile %output_view[%zero, %zero] {
      access_tile_set = #output, access_tile_order = #identity
    } : memref<3x64xf16> -> !ktdp.access_tile<3x64xindex>
    %value = ktdp.load %input_tile : !ktdp.access_tile<6x32xindex> -> tensor<6x32xf32>
    %result = spyreop.numeric_cast %value {kind = "fp32todl16"} : tensor<6x32xf32> -> tensor<3x64xf16>
    ktdp.store %result, %output_tile : tensor<3x64xf16>, !ktdp.access_tile<3x64xindex>
    return
  }
}
