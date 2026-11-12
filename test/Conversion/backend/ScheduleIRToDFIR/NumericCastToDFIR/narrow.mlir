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

// RUN: dataflow-scheduler-opt -mlir-print-local-scope -ktir-to-dfir %s | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=ktdf_lowering.native_cast --implicit-check-not=ktdf.buffer_phase --implicit-check-not=ktdf.select_memref --implicit-check-not=!spyreop.df16 --implicit-check-not=vectorchain.constant_bitstream --implicit-check-not=pe_lrfreg

// Native f16 storage carries SEN169 bits, not IEEE f16 values.
// CHECK-LABEL: func.func @narrow(
// CHECK: call @local_schedule_0
// CHECK-LABEL: func.func @local_schedule_0(
// CHECK-SAME: %[[INPUT_ADDR:.*]]: index, %[[OUTPUT_ADDR:.*]]: index)
// CHECK: %[[WORD:.*]] = arith.constant 128 : index
// CHECK: arith.divsi %[[INPUT_ADDR]], %[[WORD]] : index
// CHECK: arith.divsi %[[OUTPUT_ADDR]], %[[WORD]] : index
// CHECK-NOT: arith.divsi
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[OUT_BASE:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[MNILU:.*]] = dataflow.get_unit {{.*}}type = "mnilu"
// CHECK-DAG: %[[L1LU:.*]] = dataflow.get_unit {{.*}}type = "l1lu"
// CHECK-DAG: %[[MPE_ARRAY:.*]] = dataflow.get_unit {{.*}}type = "mpe_array"
// CHECK-DAG: %[[SFU:.*]] = dataflow.get_unit {{.*}}type = "sfu"
// CHECK-DAG: %[[L1SU:.*]] = dataflow.get_unit {{.*}}type = "l1su"
// CHECK-DAG: %[[MNISU:.*]] = dataflow.get_unit {{.*}}type = "mnisu"

// HBM fetch advances by the number of physical packets per cast.
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[MNILU]])
// CHECK: %[[HBM_IN:.*]] = dataflow.get_logical_memory_view {{.*}} {{.*}} : index, index, memref<2x32xf32>
// CHECK: %[[LOCAL_IN:.*]] = dataflow.get_logical_memory_view {{.*}} {{.*}} : index, index, memref<2x32xf32>
// CHECK: agen.composite_load_and_store src:%[[HBM_IN]][%[[ZERO]] * 2, 0] dst:%[[LOCAL_IN]][0, 0]
// CHECK: load_iv({{.*}}:vector<32xf32>)
// CHECK: time_set = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
// CHECK: dataflow.sync_send
// CHECK: dataflow.sync_recv

// Local memory loads each packet exactly once before the relay.
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[L1LU]])
// CHECK: %[[INPUT_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}} {{.*}} : index, index, memref<2x32xf32>
// CHECK: dataflow.sync_recv
// CHECK: %[[LOADED0:.*]] = agen.vector_load %[[INPUT_VIEW]][0, 0] {{.*}} : memref<2x32xf32>, vector<32xf32>
// CHECK: dataflow.send {{.*}}, %[[LOADED0]] : vector<32xf32>
// CHECK: %[[LOADED1:.*]] = agen.vector_load %[[INPUT_VIEW]][1, 0] {{.*}} : memref<2x32xf32>, vector<32xf32>
// CHECK: dataflow.send {{.*}}, %[[LOADED1]] : vector<32xf32>
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[MPE_ARRAY]])
// CHECK: %[[RELAY0:.*]] = dataflow.receive {{.*}} : vector<32xf32>
// CHECK: dataflow.send {{.*}}, %[[RELAY0]] : vector<32xf32>
// CHECK: %[[RELAY1:.*]] = dataflow.receive {{.*}} : vector<32xf32>
// CHECK: dataflow.send {{.*}}, %[[RELAY1]] : vector<32xf32>
// CHECK-NOT: dataflow.receive

// The native recipe preserves the staggered four-lane groups.
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[SFU]])
// CHECK: %[[INPUT0:.*]] = dataflow.receive {{.*}} : vector<32xf32>
// CHECK-NEXT: %[[CAST0:.*]] = vectorchain.cast %[[INPUT0]] : vector<32xf32>, vector<32xf16>
// CHECK-NEXT: %[[INPUT1:.*]] = dataflow.receive {{.*}} : vector<32xf32>
// CHECK-NEXT: %[[CAST1:.*]] = vectorchain.cast %[[INPUT1]] : vector<32xf32>, vector<32xf16>
// CHECK-NEXT: %[[PACKED:.*]] = vectorchain.pack %[[CAST0]], %[[CAST1]] {indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : index} : vector<32xf16>, vector<32xf16>, vector<64xf16>
// CHECK-NEXT: dataflow.send {{.*}}, %[[PACKED]] : vector<64xf16>
// CHECK-NOT: dataflow.receive

// Local memory and HBM stores use the same selected output scratchpad.
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[L1SU]])
// CHECK: %[[OUTPUT_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[OUT_BASE]] {{.*}} : index, index, memref<1x64xf16>
// CHECK: %[[STORED0:.*]] = dataflow.receive {{.*}} : vector<64xf16>
// CHECK-NEXT: agen.vector_store %[[STORED0]], %[[OUTPUT_VIEW]][0, 0] {{.*}} : memref<1x64xf16>, vector<64xf16>
// CHECK: dataflow.sync_send
// CHECK: dataflow.sync_recv
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[MNISU]])
// CHECK: %[[HBM_OUT:.*]] = dataflow.get_logical_memory_view {{.*}} {{.*}} : index, index, memref<1x64xf16>
// CHECK: %[[LOCAL_OUT:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[OUT_BASE]] {{.*}} : index, index, memref<1x64xf16>
// CHECK: dataflow.sync_recv
// CHECK: agen.composite_load_and_store src:%[[LOCAL_OUT]][0, 0] dst:%[[HBM_OUT]][%[[ZERO]], 0]
// CHECK: load_iv({{.*}}:vector<64xf16>)
// CHECK: time_set = affine_set<(d0) : (d0 == 0)>
// CHECK: return

#input = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#output = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device attributes {} import("Inputs/device.mlir")
  func.func @narrow(%input: index, %output: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %input_view = ktdp.construct_memory_view %input, sizes: [2, 32], strides: [32, 1] {
      coordinate_set = #input, memory_space = #ktdp.memory_space<global>
    } : memref<2x32xf32>
    %output_view = ktdp.construct_memory_view %output, sizes: [1, 64], strides: [64, 1] {
      coordinate_set = #output, memory_space = #ktdp.memory_space<global>
    } : memref<1x64xf16>
    %input_tile = ktdp.construct_access_tile %input_view[%zero, %zero] {
      access_tile_set = #input, access_tile_order = #identity
    } : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
    %output_tile = ktdp.construct_access_tile %output_view[%zero, %zero] {
      access_tile_set = #output, access_tile_order = #identity
    } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
    %value = ktdp.load %input_tile : !ktdp.access_tile<2x32xindex> -> tensor<2x32xf32>
    %result = spyreop.numeric_cast %value {kind = "fp32todl16"} : tensor<2x32xf32> -> tensor<1x64xf16>
    ktdp.store %result, %output_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    return
  }
}
