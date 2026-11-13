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

// RUN: dataflow-scheduler-opt --split-input-file -mlir-print-local-scope -ktir-to-dfir %s | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=ktdf_lowering.native_cast

// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: %[[BASE:.*]] = vectorchain.neg {{.*}} : vector<64xf16>, vector<64xf16>
// CHECK: %[[CAST0:.*]] = vectorchain.cast %[[BASE]] : vector<64xf16>, vector<64xf32>
// CHECK: %[[PACKET0:.*]] = vectorchain.shuffle input(%[[CAST0]])
// CHECK: %[[CAST1:.*]] = vectorchain.cast %[[BASE]] : vector<64xf16>, vector<64xf32>
// CHECK: %[[PACKET1:.*]] = vectorchain.shuffle input(%[[CAST1]])
// CHECK: %[[FLOOR0:.*]] = vectorchain.floor %[[PACKET0]] : vector<32xf32>, vector<32xf32>
// CHECK: %[[FLOOR1:.*]] = vectorchain.floor %[[PACKET1]] : vector<32xf32>, vector<32xf32>
// CHECK: dataflow.send {{.*}}, %[[FLOOR0]] : vector<32xf32>
// CHECK: dataflow.send {{.*}}, %[[FLOOR1]] : vector<32xf32>

#input = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#output = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device attributes {} import("Inputs/device.mlir")
  func.func @widen_unary(%input: index, %output: index) attributes {grid = [1]} {
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
    %before = arith.negf %value : tensor<1x64xf16>
    %result = spyreop.numeric_cast %before {kind = "dl16tofp32"} : tensor<1x64xf16> -> tensor<2x32xf32>
    %after = math.floor %result : tensor<2x32xf32>
    ktdp.store %after, %output_tile : tensor<2x32xf32>, !ktdp.access_tile<2x32xindex>
    return
  }
}

// -----

// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: %[[FLOOR0:.*]] = vectorchain.floor {{.*}} : vector<32xf32>, vector<32xf32>
// CHECK: %[[FLOOR1:.*]] = vectorchain.floor {{.*}} : vector<32xf32>, vector<32xf32>
// CHECK: %[[CAST0:.*]] = vectorchain.cast %[[FLOOR0]] : vector<32xf32>, vector<32xf16>
// CHECK: %[[CAST1:.*]] = vectorchain.cast %[[FLOOR1]] : vector<32xf32>, vector<32xf16>
// CHECK: %[[PACKED:.*]] = vectorchain.pack %[[CAST0]], %[[CAST1]]
// CHECK: %[[NEG:.*]] = vectorchain.neg %[[PACKED]] : vector<64xf16>, vector<64xf16>
// CHECK: dataflow.send {{.*}}, %[[NEG]] : vector<64xf16>

#input = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#output = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device attributes {} import("Inputs/device.mlir")
  func.func @narrow_unary(%input: index, %output: index) attributes {grid = [1]} {
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
    %before = math.floor %value : tensor<2x32xf32>
    %result = spyreop.numeric_cast %before {kind = "fp32todl16"} : tensor<2x32xf32> -> tensor<1x64xf16>
    %after = arith.negf %result : tensor<1x64xf16>
    ktdp.store %after, %output_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    return
  }
}

// -----

// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: %[[BASE:.*]] = vectorchain.neg {{.*}} : vector<64xf16>, vector<64xf16>
// CHECK: %[[CAST0:.*]] = vectorchain.cast %[[BASE]] : vector<64xf16>, vector<64xf32>
// CHECK: %[[PACKET0:.*]] = vectorchain.shuffle input(%[[CAST0]])
// CHECK: %[[CAST1:.*]] = vectorchain.cast %[[BASE]] : vector<64xf16>, vector<64xf32>
// CHECK: %[[PACKET1:.*]] = vectorchain.shuffle input(%[[CAST1]])
// CHECK: %[[FLOOR0:.*]] = vectorchain.floor %[[PACKET0]] : vector<32xf32>, vector<32xf32>
// CHECK: %[[FLOOR1:.*]] = vectorchain.floor %[[PACKET1]] : vector<32xf32>, vector<32xf32>
// CHECK: dataflow.send {{.*}}, %[[FLOOR0]] : vector<32xf32>
// CHECK: dataflow.send {{.*}}, %[[FLOOR1]] : vector<32xf32>

#input = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#output = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device attributes {} import("Inputs/device.mlir")
  func.func @widen_generic_unary(%input: index, %output: index) attributes {grid = [1]} {
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
    %before_init = tensor.empty() : tensor<1x64xf16>
    %before = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel"]}
      ins(%value : tensor<1x64xf16>) outs(%before_init : tensor<1x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %neg = arith.negf %in : f16
      linalg.yield %neg : f16
    } -> tensor<1x64xf16>
    %result = spyreop.numeric_cast %before {kind = "dl16tofp32"} : tensor<1x64xf16> -> tensor<2x32xf32>
    %after_init = tensor.empty() : tensor<2x32xf32>
    %after = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel"]}
      ins(%result : tensor<2x32xf32>) outs(%after_init : tensor<2x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %floor = math.floor %in : f32
      linalg.yield %floor : f32
    } -> tensor<2x32xf32>
    ktdp.store %after, %output_tile : tensor<2x32xf32>, !ktdp.access_tile<2x32xindex>
    return
  }
}

// -----

// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: %[[FLOOR0:.*]] = vectorchain.floor {{.*}} : vector<32xf32>, vector<32xf32>
// CHECK: %[[FLOOR1:.*]] = vectorchain.floor {{.*}} : vector<32xf32>, vector<32xf32>
// CHECK: %[[CAST0:.*]] = vectorchain.cast %[[FLOOR0]] : vector<32xf32>, vector<32xf16>
// CHECK: %[[CAST1:.*]] = vectorchain.cast %[[FLOOR1]] : vector<32xf32>, vector<32xf16>
// CHECK: %[[PACKED:.*]] = vectorchain.pack %[[CAST0]], %[[CAST1]]
// CHECK: %[[NEG:.*]] = vectorchain.neg %[[PACKED]] : vector<64xf16>, vector<64xf16>
// CHECK: dataflow.send {{.*}}, %[[NEG]] : vector<64xf16>

#input = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#output = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device attributes {} import("Inputs/device.mlir")
  func.func @narrow_generic_unary(%input: index, %output: index) attributes {grid = [1]} {
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
    %before_init = tensor.empty() : tensor<2x32xf32>
    %before = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel"]}
      ins(%value : tensor<2x32xf32>) outs(%before_init : tensor<2x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %floor = math.floor %in : f32
      linalg.yield %floor : f32
    } -> tensor<2x32xf32>
    %result = spyreop.numeric_cast %before {kind = "fp32todl16"} : tensor<2x32xf32> -> tensor<1x64xf16>
    %after_init = tensor.empty() : tensor<1x64xf16>
    %after = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel"]}
      ins(%result : tensor<1x64xf16>) outs(%after_init : tensor<1x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %neg = arith.negf %in : f16
      linalg.yield %neg : f16
    } -> tensor<1x64xf16>
    ktdp.store %after, %output_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    return
  }
}
