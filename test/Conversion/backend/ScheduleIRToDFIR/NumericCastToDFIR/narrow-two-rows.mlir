//===-- narrow-two-rows.mlir ------------------------------------*- mlir -*-===//
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

// RUN: dataflow-scheduler-opt -ktir-to-dfir %s | FileCheck %s
// RUN: dataflow-scheduler-opt -pipeline-numeric-cast %s | FileCheck %s --check-prefix=PIPE

// PIPE: scf.for {{.*}} to %c3 step %c1 {
// PIPE: scf.for {{.*}} to %c2 step %c1 {
// PIPE: ktdf.data_transfer from {{.*}}[%{{.*}} * 4 + %{{.*}}, 0] size [1, 32]
// PIPE: ktdf.data_transfer from {{.*}}[%{{.*}} * 4 + %{{.*}} + 2, 0] size [1, 32]
// CHECK: scf.for {{.*}} to %c3 step %c1
// CHECK: scf.for {{.*}} to %c2 step %c1
// CHECK: src:{{.*}}[%{{.*}} * 4 + %{{.*}}, 0]
// CHECK: src:{{.*}}[%{{.*}} * 4 + %{{.*}} + 2, 0]
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
