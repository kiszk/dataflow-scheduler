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

// RUN: dataflow-scheduler-opt -mlir-print-local-scope -ktir-to-dfir %s | FileCheck %s --check-prefix=GEN --implicit-check-not=spyreop.numeric_cast --implicit-check-not=ktdf_lowering.native_cast --implicit-check-not=ktdf.buffer_phase --implicit-check-not=ktdf.select_memref --implicit-check-not=!spyreop.df16

// Native f16 storage carries SEN169 bits, not IEEE f16 values.
// GEN-LABEL: func.func @local_schedule_0_body()
// GEN: memref<1x64xf16>
// GEN: vectorchain.cast
// GEN: vectorchain.shuffle {{.*}}indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32], repetition = 8
// GEN: vectorchain.shuffle {{.*}}indices = [4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8
// GEN: memref<2x32xf32>
// GEN: agen.composite_load_and_store

#input = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#output = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device attributes {} import("Inputs/device.mlir")
  func.func @widen(%input: index, %output: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %input_view = ktdp.construct_memory_view %input, sizes: [1, 64], strides: [64, 1] {
      coordinate_set = #input, memory_space = #ktdp.memory_space<global>
    } : memref<1x64xf16>
    %output_view = ktdp.construct_memory_view %output, sizes: [2, 32], strides: [32, 1] {
      coordinate_set = #output, memory_space = #ktdp.memory_space<global>
    } : memref<2x32xf32>
    %input_tile = ktdp.construct_access_tile %input_view[%zero, %zero] {
      access_tile_set = #input, access_tile_order = #identity
    } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
    %output_tile = ktdp.construct_access_tile %output_view[%zero, %zero] {
      access_tile_set = #output, access_tile_order = #identity
    } : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
    %value = ktdp.load %input_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
    %result = spyreop.numeric_cast %value {kind = "dl16tofp32"} : tensor<1x64xf16> -> tensor<2x32xf32>
    ktdp.store %result, %output_tile : tensor<2x32xf32>, !ktdp.access_tile<2x32xindex>
    return
  }
}
