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

// RUN: dataflow-scheduler-opt --split-input-file -ktir-to-schedule-ir %s | FileCheck %s --check-prefix=FRONT --implicit-check-not=spyreop.numeric_cast
// FRONT-LABEL: func.func @widen_192
// FRONT-LABEL: func.func @local_schedule_0
// FRONT: scf.for
// FRONT: ktdf.pipeline
// FRONT: ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 32xf32>, !ktdf.fifo.slot<"SFU" -> "DDR", 32xf32>
// FRONT: ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
// FRONT: ktdf.data_transfer from {{.*}} size [1, 64]
// FRONT: ktdf_lowering.native_cast_value ins({{.*}}tensor<64xf16>) outs(tensor<32xf32>, tensor<32xf32>)
// FRONT: ktdf.data_transfer from {{.*}} size [1, 32] to {{.*}} size [1, 32]
// FRONT: ktdf.data_transfer from {{.*}} size [1, 32] to {{.*}} size [1, 32]

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

// -----

// FRONT-LABEL: func.func @narrow_192
// FRONT-LABEL: func.func @local_schedule_0
// FRONT: scf.for
// FRONT: ktdf.pipeline
// FRONT: ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
// FRONT: ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 32xf32>, !ktdf.fifo.slot<"DDR" -> "SFU", 32xf32>
// FRONT: ktdf.data_transfer from {{.*}} size [1, 32] to {{.*}} size [1, 32]
// FRONT: ktdf.data_transfer from {{.*}} size [1, 32] to {{.*}} size [1, 32]
// FRONT: ktdf_lowering.native_cast_value ins({{.*}}tensor<32xf32>, tensor<32xf32>) outs(tensor<64xf16>)
// FRONT: ktdf.data_transfer from {{.*}} size [1, 64] to {{.*}} size [1, 64]

#input = affine_set<(d0, d1) : (d0 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#output = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../backend/ScheduleIRToDFIR/NumericCastToDFIR/Inputs/device.mlir")
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
