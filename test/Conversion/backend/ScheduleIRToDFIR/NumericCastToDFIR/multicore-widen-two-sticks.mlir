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

// RUN: dataflow-scheduler-opt -pipeline-numeric-cast %s | FileCheck %s --check-prefix=PIPE
// RUN: dataflow-scheduler-opt -ktir-to-dfir %s | FileCheck %s --check-prefix=DFIR --implicit-check-not=spyreop.numeric_cast --implicit-check-not=ktdf_lowering.native_cast

// PIPE-LABEL: func.func @local_schedule_0(
// PIPE-SAME: attributes {grid = [3]}
// PIPE: %[[ONE:.*]] = arith.constant 1 : index
// PIPE: %[[TWO:.*]] = arith.constant 2 : index
// PIPE: scf.for %[[T:.*]] = {{.*}} to %[[TWO]] step
// PIPE: ktdp.get_compute_tile_id : index
// PIPE: arith.addi
// PIPE: ktdp.construct_memory_view
// PIPE: ktdf.data_transfer from
// PIPE: ktdp.get_compute_tile_id : index
// PIPE: arith.addi
// PIPE: ktdp.construct_memory_view
// PIPE: ktdf.data_transfer from
// DFIR-LABEL: func.func @local_schedule_0_body() attributes {grid = [3]}
// DFIR: dataflow.get_unit {core = 0 : i32
// DFIR: dataflow.get_unit {core = 1 : i32
// DFIR: dataflow.get_unit {core = 2 : i32
// DFIR: vectorchain.cast

#input = affine_set<(d0, d1) : (d0 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#output = affine_set<(d0, d1) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device attributes {} import("Inputs/device.mlir")
  func.func @widen_192(%input: index, %output: index) attributes {grid = [3]} {
    %zero = arith.constant 0 : index
    %input_view = ktdp.construct_memory_view %input, sizes: [6, 64], strides: [64, 1] {
      coordinate_set = #input, memory_space = #ktdp.memory_space<global>
    } : memref<6x64xf16>
    %output_view = ktdp.construct_memory_view %output, sizes: [12, 32], strides: [32, 1] {
      coordinate_set = #output, memory_space = #ktdp.memory_space<global>
    } : memref<12x32xf32>
    %input_tile = ktdp.construct_access_tile %input_view[%zero, %zero] {
      access_tile_set = #input, access_tile_order = #identity
    } : memref<6x64xf16> -> !ktdp.access_tile<6x64xindex>
    %output_tile = ktdp.construct_access_tile %output_view[%zero, %zero] {
      access_tile_set = #output, access_tile_order = #identity
    } : memref<12x32xf32> -> !ktdp.access_tile<12x32xindex>
    %value = ktdp.load %input_tile : !ktdp.access_tile<6x64xindex> -> tensor<6x64xf16>
    %result = spyreop.numeric_cast %value {kind = "dl16tofp32"} : tensor<6x64xf16> -> tensor<12x32xf32>
    ktdp.store %result, %output_tile : tensor<12x32xf32>, !ktdp.access_tile<12x32xindex>
    return
  }
}
