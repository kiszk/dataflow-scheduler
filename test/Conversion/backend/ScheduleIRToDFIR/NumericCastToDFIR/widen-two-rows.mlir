//===-- widen-two-rows.mlir ------------------------------------*- mlir -*-===//
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

// PIPE: ktdp.construct_memory_view {{.*}} : memref<6x64xf16, #ktdp.memory_space<global>>
// PIPE: ktdp.construct_memory_view {{.*}} : memref<12x32xf32, #ktdp.memory_space<global>>
// PIPE: scf.for {{.*}} to %c3 step %c1 {
// PIPE: scf.for {{.*}} to %c2 step %c1 {
// PIPE: ktdf.data_transfer from {{.*}}[%{{.*}} * 2 + %{{.*}}, 0] size [1, 64]
// PIPE: size [1, 32] to {{.*}}[%{{.*}} * 4 + %{{.*}}, 0]
// PIPE: size [1, 32] to {{.*}}[%{{.*}} * 4 + %{{.*}} + 2, 0]

// CHECK: memref<6x64xf16>
// CHECK: scf.for {{.*}} to %c3 step %c1
// CHECK: scf.for {{.*}} to %c2 step %c1
// CHECK: vectorchain.cast
// CHECK: vectorchain.shuffle {{.*}}indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32], repetition = 8
// CHECK: vectorchain.shuffle {{.*}}indices = [4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8
// CHECK: memref<12x32xf32>
// CHECK: dst:{{.*}}[%{{.*}} * 4 + %{{.*}}, 0]
// CHECK: dst:{{.*}}[%{{.*}} * 4 + %{{.*}} + 2, 0]

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
