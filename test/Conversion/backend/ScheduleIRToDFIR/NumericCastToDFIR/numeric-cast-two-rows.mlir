//===-- numeric-cast-two-rows.mlir ------------------------------------*- mlir -*-===//
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

// RUN: dataflow-scheduler-opt --split-input-file -ktir-to-dfir %s | FileCheck %s
// RUN: dataflow-scheduler-opt --split-input-file -ktir-to-schedule-ir %s | FileCheck %s --check-prefix=FRONT
// RUN: sed -e 's/grid = \[1\]/grid = [3]/' -e 's@Inputs/device.mlir@%S/Inputs/device.mlir@' %s > %t.multicore.mlir
// RUN: dataflow-scheduler-opt --split-input-file -ktir-to-dfir %t.multicore.mlir | FileCheck %s --check-prefix=MULTI

// MULTI: func.func @local_schedule_0_body() attributes {grid = [3]}
// MULTI: dataflow.get_unit {core = 2 : i32
// MULTI: vectorchain.cast
// MULTI: agen.composite_load_and_store {{.*}}dst:{{.*}}[{{.*}}, {{.*}}, 0]

// FRONT: scf.for %[[CHUNK:.*]] = {{.*}} to %c3 step %c1
// FRONT: scf.for %[[ROW:.*]] = {{.*}} to %c2 step %c1
// FRONT: ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 32xf32>, !ktdf.fifo.slot<"SFU" -> "DDR", 32xf32>
// FRONT: ktdf_lowering.native_cast_value
// FRONT: %[[EVEN:.*]] = arith.muli %[[CHUNK]], %c2 : index
// FRONT: ktdf.data_transfer from {{.*}} size [1, 1, 32] to {{.*}}[%[[EVEN]], %[[ROW]], 0] size [1, 1, 32]
// FRONT: %[[ODD:.*]] = arith.addi %[[EVEN]], %c1 : index
// FRONT: ktdf.data_transfer from {{.*}} size [1, 1, 32] to {{.*}}[%[[ODD]], %[[ROW]], 0] size [1, 1, 32]


// CHECK: scf.for {{.*}} to %c3 step %c1
// CHECK: scf.for {{.*}} to %c2 step %c1
// CHECK: vectorchain.cast
// CHECK: vectorchain.shuffle {{.*}}indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32], repetition = 8
// CHECK: vectorchain.shuffle {{.*}}indices = [4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8
// CHECK: dst:{{.*}}[%{{.*}}, %{{.*}}, 0]
// CHECK: dst:{{.*}}[%{{.*}}, %{{.*}}, 0]

#input = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#output = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 31 >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  ktdf_arch.device @sample_device attributes {} import("Inputs/device.mlir")
  func.func @widen_two_rows(%input: index, %output: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %input_view = ktdp.construct_memory_view %input, sizes: [3, 2, 64], strides: [128, 64, 1] {
      coordinate_set = #input, memory_space = #ktdp.memory_space<global>
    } : memref<3x2x64xf16>
    %output_view = ktdp.construct_memory_view %output, sizes: [6, 2, 32], strides: [64, 32, 1] {
      coordinate_set = #output, memory_space = #ktdp.memory_space<global>
    } : memref<6x2x32xf32>
    %input_tile = ktdp.construct_access_tile %input_view[%zero, %zero, %zero] {
      access_tile_set = #input, access_tile_order = #identity
    } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
    %output_tile = ktdp.construct_access_tile %output_view[%zero, %zero, %zero] {
      access_tile_set = #output, access_tile_order = #identity
    } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
    %value = ktdp.load %input_tile : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>
    %result = spyreop.numeric_cast %value {kind = "dl16tofp32"} : tensor<3x2x64xf16> -> tensor<6x2x32xf32>
    ktdp.store %result, %output_tile : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
    return
  }
}

// -----

// MULTI: func.func @local_schedule_0_body() attributes {grid = [3]}
// MULTI: dataflow.get_unit {core = 2 : i32
// MULTI: vectorchain.cast
// MULTI: vectorchain.pack
// MULTI: agen.composite_load_and_store {{.*}}dst:{{.*}}[{{.*}}, {{.*}}, 0]

// FRONT: scf.for %[[CHUNK:.*]] = {{.*}} to %c3 step %c1
// FRONT: scf.for %[[ROW:.*]] = {{.*}} to %c2 step %c1
// FRONT: ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 32xf32>, !ktdf.fifo.slot<"DDR" -> "SFU", 32xf32>
// FRONT: %[[EVEN:.*]] = arith.muli %[[CHUNK]], %c2 : index
// FRONT: %[[ODD:.*]] = arith.addi %[[EVEN]], %c1 : index
// FRONT: ktdf.data_transfer from {{.*}}[%[[EVEN]], %[[ROW]], 0] size [1, 1, 32]
// FRONT: ktdf.data_transfer from {{.*}}[%[[ODD]], %[[ROW]], 0] size [1, 1, 32]
// FRONT: ktdf_lowering.native_cast_value

// CHECK: scf.for {{.*}} to %c3 step %c1
// CHECK: scf.for {{.*}} to %c2 step %c1
// CHECK: src:{{.*}}[%{{.*}}, %{{.*}}, 0]
// CHECK: src:{{.*}}[%{{.*}}, %{{.*}}, 0]
// CHECK: vectorchain.cast
// CHECK: vectorchain.pack

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("Inputs/device.mlir")
  func.func @narrow_two_rows(%arg0: index, %arg1: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = ktdp.construct_memory_view %arg0, sizes: [6, 2, 32], strides: [64, 32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<6x2x32xf32>
    %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
    %2 = ktdp.construct_memory_view %arg1, sizes: [3, 2, 64], strides: [128, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<3x2x64xf16>
    %3 = ktdp.construct_access_tile %2[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
    %4 = ktdp.load %1 : <6x2x32xindex> -> tensor<6x2x32xf32>
    %5 = spyreop.numeric_cast %4 {kind = "fp32todl16"} : tensor<6x2x32xf32> -> tensor<3x2x64xf16>
    ktdp.store %5, %3 : tensor<3x2x64xf16>, <3x2x64xindex>
    return
  }
}
