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

// RUN: dataflow-scheduler-opt -ktir-to-schedule-ir %s | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=ktdp.load --implicit-check-not=ktdp.store --implicit-check-not=ktdp.construct_access_tile --implicit-check-not=ktdf.buffer_phase --implicit-check-not=ktdf.select_memref --implicit-check-not=dataflow. --implicit-check-not=vectorchain.

// Exercise the frontend dispatcher, stopping before scheduling and DFIR.
// KTIR shapes describe physical storage: three native sticks and six FP32
// packets. The f16 buffers below carry SEN169 bits for native_cast.
// CHECK-LABEL: func.func @widen_192(
// CHECK-SAME: %[[BASE_IN:.*]]: index, %[[BASE_OUT:.*]]: index)
// CHECK-NEXT: call @local_schedule_0(%[[BASE_IN]], %[[BASE_OUT]]) : (index, index) -> ()
// CHECK-NEXT: return
// CHECK-LABEL: module @local_schedule_0 {
// CHECK: func.func @local_schedule_0(%[[BASE_IN:.*]]: index, %[[BASE_OUT:.*]]: index)
// CHECK-NEXT: %[[IN:.*]] = ktdp.construct_memory_view %[[BASE_IN]], sizes: [3, 64], strides: [64, 1] {{.*}} : memref<3x64xf16, #ktdp.memory_space<global>>
// CHECK-NEXT: %[[OUT:.*]] = ktdp.construct_memory_view %[[BASE_OUT]], sizes: [6, 32], strides: [32, 1] {{.*}} : memref<6x32xf32, #ktdp.memory_space<global>>
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[THREE:.*]] = arith.constant 3 : index
// CHECK-NEXT: scf.for %[[T:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] {
// CHECK-NEXT: ktdf.pipeline {
// CHECK-NEXT: %[[P:.*]]:11 = ktdf.private -> (memref<1x64xf16, #ktdp.memory_space<ct_local>>, memref<2x32xf32, #ktdp.memory_space<ct_local>>
// CHECK-NEXT: %[[INPUT:.*]] = memref.alloc() {alignment = 128 : i64} : memref<1x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT: %[[OUTPUT:.*]] = memref.alloc() {alignment = 128 : i64} : memref<2x32xf32, #ktdp.memory_space<ct_local>>
// CHECK-NEXT: %[[LOAD:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "MPE_ARRAY", 64xf16>
// CHECK-NEXT: %[[RELAY:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MPE_ARRAY" -> "SFU", 64xf16>
// CHECK-NEXT: %[[CAST:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
// CHECK-COUNT-5: = ktdf.create_token : !ktdf.token
// CHECK-NEXT: ktdf.private_yield %[[INPUT]], %[[OUTPUT]], %[[LOAD]], %[[RELAY]], %[[CAST]]#0, %[[CAST]]#1, {{.*}}
// CHECK-NEXT: }
// CHECK-NEXT: ktdf.stage depends_in(none) depends_out(%[[P]]#6) {
// CHECK-NEXT: ktdf.data_transfer from %[[IN]][%[[T]], 0] size [1, 64] to %[[P]]#0[0, 0] size [1, 64] {dataflow_scheduler.throttle = 64 : i64}
// CHECK-NEXT: } {applicable_units = ["MNILU"]}
// CHECK-NEXT: ktdf.stage depends_in(%[[P]]#6) depends_out(%[[P]]#7) {
// CHECK-NEXT: ktdf.data_transfer from %[[P]]#0[0, 0] size [1, 64] to %[[P]]#2 size [64]
// CHECK-NEXT: } {applicable_units = ["L1LU"]}
// CHECK-NEXT: ktdf.stage depends_in(%[[P]]#7) depends_out(%[[P]]#8) {
// CHECK-NEXT: %[[PACKET0:.*]] = ktdf.read_from_fifo %[[P]]#2 : <"L1LU" -> "MPE_ARRAY", 64xf16> -> tensor<64xf16>
// CHECK-NEXT: ktdf.write_to_fifo %[[PACKET0]], %[[P]]#3 : tensor<64xf16>, <"MPE_ARRAY" -> "SFU", 64xf16>
// CHECK-NEXT: } {applicable_units = ["MPE_ARRAY"]}
// CHECK-NEXT: ktdf.stage depends_in(%[[P]]#8) depends_out(%[[P]]#9) {
// CHECK-NEXT: ktdf_lowering.native_cast ins(%[[P]]#3 : !ktdf.fifo.slot<"MPE_ARRAY" -> "SFU", 64xf16>) outs(%[[P]]#4, %[[P]]#5 : !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {kind = "dl16tofp32"}
// CHECK-NEXT: } {applicable_units = ["SFU"]}
// CHECK-NEXT: ktdf.stage depends_in(%[[P]]#9) depends_out(%[[P]]#10) {
// CHECK-NEXT: ktdf.data_transfer from %[[P]]#4 size [32] to %[[P]]#1[0, 0] size [1, 32]
// CHECK-NEXT: ktdf.data_transfer from %[[P]]#5 size [32] to %[[P]]#1[1, 0] size [1, 32]
// CHECK-NEXT: } {applicable_units = ["L1SU"]}
// CHECK-NEXT: ktdf.stage depends_in(%[[P]]#10) depends_out(none) {
// CHECK-NEXT: ktdf.data_transfer from %[[P]]#1[0, 0] size [2, 32] to %[[OUT]][%[[T]] * 2, 0] size [2, 32] {dataflow_scheduler.throttle = 32 : i64}
// CHECK-NEXT: } {applicable_units = ["MNISU"]}
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }

#input = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#output = affine_set<(d0, d1) : (d0 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../backend/ScheduleIRToDFIR/NumericCastToDFIR/Inputs/device.mlir")
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
