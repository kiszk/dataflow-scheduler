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

// RUN: dataflow-scheduler-opt %s --split-input-file --ktir-numeric-cast-tiling | FileCheck %s --check-prefix=TILE
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-input-file --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=access_tile_order --implicit-check-not=ktdf.pipeline
// Packet axis is rank - 3, row is rank - 2, and lanes are last. All leading
// extents are nonunit to expose missing batch loops or incorrect strides.
// Exp on DL16 and floor on FP32 exercise unary computes around each cast.
// Eight cores exceed the six groups per batch, exercising empty owners.

// CHECK-LABEL: func.func @widen_rank4_standard_c1(
// CHECK-DAG: memref<2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @widen_rank4_standard_c1(
// TILE: scf.for %[[BATCH:[^ ]+]] =
// TILE: scf.for %[[GROUP:[^ ]+]] =
// TILE: scf.for %[[ROW:[^ ]+]] =
// TILE: %[[FIRST:.*]] = arith.muli %[[GROUP]],
// TILE: %[[SECOND:.*]] = arith.addi %[[FIRST]],
// TILE: tensor.extract_slice {{.*}}[%[[BATCH]], %[[GROUP]], %[[ROW]], 0] [1, 1, 1, 64] [1, 1, 1, 1] : tensor<2x3x2x64xf16> to tensor<64xf16>
// TILE: %[[CAST:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice %[[CAST]]#0 into {{.*}}[0, 0] [1, 32] [1, 1]
// TILE: tensor.insert_slice %[[CAST]]#1 into {{.*}}[1, 0] [1, 32] [1, 1]
// TILE: %[[RELAYOUT:.*]] = linalg.generic
// TILE: spyreop.identity
// TILE: %[[FIRST_PACKET:.*]] = tensor.extract_slice %[[RELAYOUT]][0, 0] [1, 32] [1, 1]
// TILE: %[[SECOND_PACKET:.*]] = tensor.extract_slice %[[RELAYOUT]][1, 0] [1, 32] [1, 1]
// TILE: tensor.insert_slice %[[FIRST_PACKET]] into {{.*}}[%[[BATCH]], %[[FIRST]], %[[ROW]], 0] [1, 1, 1, 32] [1, 1, 1, 1]
// TILE: tensor.insert_slice %[[SECOND_PACKET]] into {{.*}}[%[[BATCH]], %[[SECOND]], %[[ROW]], 0] [1, 1, 1, 32] [1, 1, 1, 1]
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 2 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 63 - d3 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 5 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 31 - d3 >= 0)>
#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#cast_order = affine_map<(d0, d1, d2, d3) -> (d0, (d1 floordiv 2) * 2 + (d3 mod 8) floordiv 4, d2, (d1 mod 2) * 16 + (d3 floordiv 8) * 4 + d3 mod 4)>

func.func @widen_rank4_standard_c1(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 3, 2, 64], strides: [384, 128, 64, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 6, 2, 32], strides: [384, 64, 32, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x6x2x32xf32>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #src_domain
  } : memref<2x3x2x64xf16> -> !ktdp.access_tile<2x3x2x64xindex>
  %input = ktdp.load %src_tile : !ktdp.access_tile<2x3x2x64xindex> -> tensor<2x3x2x64xf16>
  %cast_init_1 = tensor.empty() : tensor<2x6x2x32xf32>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + (d3 mod 8) floordiv 4, d2, (d3 floordiv 8) * 4 + d3 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x3x2x64xf16>) outs(%cast_init_1 : tensor<2x6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x6x2x32xf32>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #dst_domain
  } : memref<2x6x2x32xf32> -> !ktdp.access_tile<2x6x2x32xindex>
  %relayout_init_1 = tensor.empty() : tensor<2x6x2x32xf32>
  %standard_store_1 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%cast : tensor<2x6x2x32xf32>) outs(%relayout_init_1 : tensor<2x6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x6x2x32xf32>
  ktdp.store %standard_store_1, %dst_tile : tensor<2x6x2x32xf32>, !ktdp.access_tile<2x6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_rank4_standard_c1(
// CHECK-DAG: memref<2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @narrow_rank4_standard_c1(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x6x2x32xf32> to tensor<32xf32>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x3x2x64xf16>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 5 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 31 - d3 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 2 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 63 - d3 >= 0)>
#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#cast_order = affine_map<(d0, d1, d2, d3) -> (d0, (d1 floordiv 2) * 2 + (d3 mod 8) floordiv 4, d2, (d1 mod 2) * 16 + (d3 floordiv 8) * 4 + d3 mod 4)>

func.func @narrow_rank4_standard_c1(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 6, 2, 32], strides: [384, 64, 32, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 3, 2, 64], strides: [384, 128, 64, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x3x2x64xf16>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #src_domain
  } : memref<2x6x2x32xf32> -> !ktdp.access_tile<2x6x2x32xindex>
  %standard_load_1 = ktdp.load %src_tile : !ktdp.access_tile<2x6x2x32xindex> -> tensor<2x6x2x32xf32>
  %relayout_init_1 = tensor.empty() : tensor<2x6x2x32xf32>
  %input = linalg.generic {indexing_maps = [#identity, #cast_order],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%standard_load_1 : tensor<2x6x2x32xf32>) outs(%relayout_init_1 : tensor<2x6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x6x2x32xf32>
  %cast_init_2 = tensor.empty() : tensor<2x3x2x64xf16>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + (d3 mod 8) floordiv 4, d2, (d3 floordiv 8) * 4 + d3 mod 4)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x6x2x32xf32>) outs(%cast_init_2 : tensor<2x3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<2x3x2x64xf16>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #dst_domain
  } : memref<2x3x2x64xf16> -> !ktdp.access_tile<2x3x2x64xindex>
  ktdp.store %cast, %dst_tile : tensor<2x3x2x64xf16>, !ktdp.access_tile<2x3x2x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @widen_rank4_identity_c4_floor(
// CHECK-DAG: memref<2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.floor {{.*}}vector<32xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @widen_rank4_identity_c4_floor(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x3x2x64xf16> to tensor<64xf16>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x6x2x32xf32>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 2 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 63 - d3 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 5 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 31 - d3 >= 0)>
#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @widen_rank4_identity_c4_floor(%src_addr: index, %dst_addr: index) attributes {grid = [4]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 3, 2, 64], strides: [384, 128, 64, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 6, 2, 32], strides: [384, 64, 32, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x6x2x32xf32>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #src_domain
  } : memref<2x3x2x64xf16> -> !ktdp.access_tile<2x3x2x64xindex>
  %input = ktdp.load %src_tile : !ktdp.access_tile<2x3x2x64xindex> -> tensor<2x3x2x64xf16>
  %cast_init_3 = tensor.empty() : tensor<2x6x2x32xf32>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + (d3 mod 8) floordiv 4, d2, (d3 floordiv 8) * 4 + d3 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x3x2x64xf16>) outs(%cast_init_3 : tensor<2x6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x6x2x32xf32>
  %floor_output = math.floor %cast : tensor<2x6x2x32xf32>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #dst_domain
  } : memref<2x6x2x32xf32> -> !ktdp.access_tile<2x6x2x32xindex>
  ktdp.store %floor_output, %dst_tile : tensor<2x6x2x32xf32>, !ktdp.access_tile<2x6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_rank4_identity_c4_floor(
// CHECK-DAG: memref<2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.floor {{.*}}vector<32xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @narrow_rank4_identity_c4_floor(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x6x2x32xf32> to tensor<32xf32>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x3x2x64xf16>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 5 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 31 - d3 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 2 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 63 - d3 >= 0)>
#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @narrow_rank4_identity_c4_floor(%src_addr: index, %dst_addr: index) attributes {grid = [4]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 6, 2, 32], strides: [384, 64, 32, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 3, 2, 64], strides: [384, 128, 64, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x3x2x64xf16>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #src_domain
  } : memref<2x6x2x32xf32> -> !ktdp.access_tile<2x6x2x32xindex>
  %input = ktdp.load %src_tile : !ktdp.access_tile<2x6x2x32xindex> -> tensor<2x6x2x32xf32>
  %floor_input = math.floor %input : tensor<2x6x2x32xf32>
  %cast_init_4 = tensor.empty() : tensor<2x3x2x64xf16>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + (d3 mod 8) floordiv 4, d2, (d3 floordiv 8) * 4 + d3 mod 4)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%floor_input : tensor<2x6x2x32xf32>) outs(%cast_init_4 : tensor<2x3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<2x3x2x64xf16>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #dst_domain
  } : memref<2x3x2x64xf16> -> !ktdp.access_tile<2x3x2x64xindex>
  ktdp.store %cast, %dst_tile : tensor<2x3x2x64xf16>, !ktdp.access_tile<2x3x2x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @widen_rank5_standard_c4(
// CHECK-DAG: memref<2x2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @widen_rank5_standard_c4(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x2x3x2x64xf16> to tensor<64xf16>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x2x6x2x32xf32>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 1 - d1 >= 0, d2 >= 0, 2 - d2 >= 0, d3 >= 0, 1 - d3 >= 0, d4 >= 0, 63 - d4 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 1 - d1 >= 0, d2 >= 0, 5 - d2 >= 0, d3 >= 0, 1 - d3 >= 0, d4 >= 0, 31 - d4 >= 0)>
#identity = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#cast_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, (d2 floordiv 2) * 2 + (d4 mod 8) floordiv 4, d3, (d2 mod 2) * 16 + (d4 floordiv 8) * 4 + d4 mod 4)>

func.func @widen_rank5_standard_c4(%src_addr: index, %dst_addr: index) attributes {grid = [4]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 2, 3, 2, 64], strides: [768, 384, 128, 64, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 2, 6, 2, 32], strides: [768, 384, 64, 32, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x6x2x32xf32>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
    access_tile_set = #src_domain
  } : memref<2x2x3x2x64xf16> -> !ktdp.access_tile<2x2x3x2x64xindex>
  %input = ktdp.load %src_tile : !ktdp.access_tile<2x2x3x2x64xindex> -> tensor<2x2x3x2x64xf16>
  %cast_init_5 = tensor.empty() : tensor<2x2x6x2x32xf32>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 * 2 + (d4 mod 8) floordiv 4, d3, (d4 floordiv 8) * 4 + d4 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x2x3x2x64xf16>) outs(%cast_init_5 : tensor<2x2x6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x2x6x2x32xf32>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
    access_tile_set = #dst_domain
  } : memref<2x2x6x2x32xf32> -> !ktdp.access_tile<2x2x6x2x32xindex>
  %relayout_init_1 = tensor.empty() : tensor<2x2x6x2x32xf32>
  %standard_store_1 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%cast : tensor<2x2x6x2x32xf32>) outs(%relayout_init_1 : tensor<2x2x6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x2x6x2x32xf32>
  ktdp.store %standard_store_1, %dst_tile : tensor<2x2x6x2x32xf32>, !ktdp.access_tile<2x2x6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_rank5_standard_c4(
// CHECK-DAG: memref<2x2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @narrow_rank5_standard_c4(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x2x6x2x32xf32> to tensor<32xf32>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x2x3x2x64xf16>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 1 - d1 >= 0, d2 >= 0, 5 - d2 >= 0, d3 >= 0, 1 - d3 >= 0, d4 >= 0, 31 - d4 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 1 - d1 >= 0, d2 >= 0, 2 - d2 >= 0, d3 >= 0, 1 - d3 >= 0, d4 >= 0, 63 - d4 >= 0)>
#identity = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#cast_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, (d2 floordiv 2) * 2 + (d4 mod 8) floordiv 4, d3, (d2 mod 2) * 16 + (d4 floordiv 8) * 4 + d4 mod 4)>

func.func @narrow_rank5_standard_c4(%src_addr: index, %dst_addr: index) attributes {grid = [4]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 2, 6, 2, 32], strides: [768, 384, 64, 32, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 2, 3, 2, 64], strides: [768, 384, 128, 64, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x3x2x64xf16>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
    access_tile_set = #src_domain
  } : memref<2x2x6x2x32xf32> -> !ktdp.access_tile<2x2x6x2x32xindex>
  %standard_load_1 = ktdp.load %src_tile : !ktdp.access_tile<2x2x6x2x32xindex> -> tensor<2x2x6x2x32xf32>
  %relayout_init_1 = tensor.empty() : tensor<2x2x6x2x32xf32>
  %input = linalg.generic {indexing_maps = [#identity, #cast_order],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%standard_load_1 : tensor<2x2x6x2x32xf32>) outs(%relayout_init_1 : tensor<2x2x6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x2x6x2x32xf32>
  %cast_init_6 = tensor.empty() : tensor<2x2x3x2x64xf16>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 * 2 + (d4 mod 8) floordiv 4, d3, (d4 floordiv 8) * 4 + d4 mod 4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x2x6x2x32xf32>) outs(%cast_init_6 : tensor<2x2x3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<2x2x3x2x64xf16>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
    access_tile_set = #dst_domain
  } : memref<2x2x3x2x64xf16> -> !ktdp.access_tile<2x2x3x2x64xindex>
  ktdp.store %cast, %dst_tile : tensor<2x2x3x2x64xf16>, !ktdp.access_tile<2x2x3x2x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @widen_rank5_identity_c1_exp(
// CHECK-DAG: memref<2x2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_f16_exp"
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @widen_rank5_identity_c1_exp(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x2x3x2x64xf16> to tensor<64xf16>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x2x6x2x32xf32>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 1 - d1 >= 0, d2 >= 0, 2 - d2 >= 0, d3 >= 0, 1 - d3 >= 0, d4 >= 0, 63 - d4 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 1 - d1 >= 0, d2 >= 0, 5 - d2 >= 0, d3 >= 0, 1 - d3 >= 0, d4 >= 0, 31 - d4 >= 0)>
#identity = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @widen_rank5_identity_c1_exp(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 2, 3, 2, 64], strides: [768, 384, 128, 64, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 2, 6, 2, 32], strides: [768, 384, 64, 32, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x6x2x32xf32>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
    access_tile_set = #src_domain
  } : memref<2x2x3x2x64xf16> -> !ktdp.access_tile<2x2x3x2x64xindex>
  %input = ktdp.load %src_tile : !ktdp.access_tile<2x2x3x2x64xindex> -> tensor<2x2x3x2x64xf16>
  %floor_input_init = tensor.empty() : tensor<2x2x3x2x64xf16>
  %floor_input = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x2x3x2x64xf16>) outs(%floor_input_init : tensor<2x2x3x2x64xf16>) {
  ^bb0(%floor_input_value: f16, %floor_input_unused: f16):
    %floor_input_exp = spyreop.exp %floor_input_value : f16
    linalg.yield %floor_input_exp : f16
  } -> tensor<2x2x3x2x64xf16>
  %cast_init_7 = tensor.empty() : tensor<2x2x6x2x32xf32>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 * 2 + (d4 mod 8) floordiv 4, d3, (d4 floordiv 8) * 4 + d4 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%floor_input : tensor<2x2x3x2x64xf16>) outs(%cast_init_7 : tensor<2x2x6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x2x6x2x32xf32>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
    access_tile_set = #dst_domain
  } : memref<2x2x6x2x32xf32> -> !ktdp.access_tile<2x2x6x2x32xindex>
  ktdp.store %cast, %dst_tile : tensor<2x2x6x2x32xf32>, !ktdp.access_tile<2x2x6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_rank5_identity_c1_exp(
// CHECK-DAG: memref<2x2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_f16_exp"
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @narrow_rank5_identity_c1_exp(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x2x6x2x32xf32> to tensor<32xf32>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x2x3x2x64xf16>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 1 - d1 >= 0, d2 >= 0, 5 - d2 >= 0, d3 >= 0, 1 - d3 >= 0, d4 >= 0, 31 - d4 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 1 - d1 >= 0, d2 >= 0, 2 - d2 >= 0, d3 >= 0, 1 - d3 >= 0, d4 >= 0, 63 - d4 >= 0)>
#identity = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @narrow_rank5_identity_c1_exp(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 2, 6, 2, 32], strides: [768, 384, 64, 32, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 2, 3, 2, 64], strides: [768, 384, 128, 64, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x3x2x64xf16>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
    access_tile_set = #src_domain
  } : memref<2x2x6x2x32xf32> -> !ktdp.access_tile<2x2x6x2x32xindex>
  %input = ktdp.load %src_tile : !ktdp.access_tile<2x2x6x2x32xindex> -> tensor<2x2x6x2x32xf32>
  %cast_init_8 = tensor.empty() : tensor<2x2x3x2x64xf16>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 * 2 + (d4 mod 8) floordiv 4, d3, (d4 floordiv 8) * 4 + d4 mod 4)>,
                     affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x2x6x2x32xf32>) outs(%cast_init_8 : tensor<2x2x3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<2x2x3x2x64xf16>
  %floor_output_init = tensor.empty() : tensor<2x2x3x2x64xf16>
  %floor_output = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%cast : tensor<2x2x3x2x64xf16>) outs(%floor_output_init : tensor<2x2x3x2x64xf16>) {
  ^bb0(%floor_output_value: f16, %floor_output_unused: f16):
    %floor_output_exp = spyreop.exp %floor_output_value : f16
    linalg.yield %floor_output_exp : f16
  } -> tensor<2x2x3x2x64xf16>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>,
    access_tile_set = #dst_domain
  } : memref<2x2x3x2x64xf16> -> !ktdp.access_tile<2x2x3x2x64xindex>
  ktdp.store %floor_output, %dst_tile : tensor<2x2x3x2x64xf16>, !ktdp.access_tile<2x2x3x2x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @widen_rank4_standard_c8(
// CHECK-DAG: memref<2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @widen_rank4_standard_c8(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x3x2x64xf16> to tensor<64xf16>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x6x2x32xf32>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 2 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 63 - d3 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 5 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 31 - d3 >= 0)>
#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#cast_order = affine_map<(d0, d1, d2, d3) -> (d0, (d1 floordiv 2) * 2 + (d3 mod 8) floordiv 4, d2, (d1 mod 2) * 16 + (d3 floordiv 8) * 4 + d3 mod 4)>

func.func @widen_rank4_standard_c8(%src_addr: index, %dst_addr: index) attributes {grid = [8]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 3, 2, 64], strides: [384, 128, 64, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 6, 2, 32], strides: [384, 64, 32, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x6x2x32xf32>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #src_domain
  } : memref<2x3x2x64xf16> -> !ktdp.access_tile<2x3x2x64xindex>
  %input = ktdp.load %src_tile : !ktdp.access_tile<2x3x2x64xindex> -> tensor<2x3x2x64xf16>
  %cast_init_9 = tensor.empty() : tensor<2x6x2x32xf32>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + (d3 mod 8) floordiv 4, d2, (d3 floordiv 8) * 4 + d3 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x3x2x64xf16>) outs(%cast_init_9 : tensor<2x6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x6x2x32xf32>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #dst_domain
  } : memref<2x6x2x32xf32> -> !ktdp.access_tile<2x6x2x32xindex>
  %relayout_init_1 = tensor.empty() : tensor<2x6x2x32xf32>
  %standard_store_1 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%cast : tensor<2x6x2x32xf32>) outs(%relayout_init_1 : tensor<2x6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x6x2x32xf32>
  ktdp.store %standard_store_1, %dst_tile : tensor<2x6x2x32xf32>, !ktdp.access_tile<2x6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_rank4_standard_c8(
// CHECK-DAG: memref<2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @narrow_rank4_standard_c8(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x6x2x32xf32> to tensor<32xf32>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x3x2x64xf16>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 5 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 31 - d3 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 2 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 63 - d3 >= 0)>
#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#cast_order = affine_map<(d0, d1, d2, d3) -> (d0, (d1 floordiv 2) * 2 + (d3 mod 8) floordiv 4, d2, (d1 mod 2) * 16 + (d3 floordiv 8) * 4 + d3 mod 4)>

func.func @narrow_rank4_standard_c8(%src_addr: index, %dst_addr: index) attributes {grid = [8]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 6, 2, 32], strides: [384, 64, 32, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 3, 2, 64], strides: [384, 128, 64, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x3x2x64xf16>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #src_domain
  } : memref<2x6x2x32xf32> -> !ktdp.access_tile<2x6x2x32xindex>
  %standard_load_1 = ktdp.load %src_tile : !ktdp.access_tile<2x6x2x32xindex> -> tensor<2x6x2x32xf32>
  %relayout_init_1 = tensor.empty() : tensor<2x6x2x32xf32>
  %input = linalg.generic {indexing_maps = [#identity, #cast_order],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%standard_load_1 : tensor<2x6x2x32xf32>) outs(%relayout_init_1 : tensor<2x6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x6x2x32xf32>
  %cast_init_10 = tensor.empty() : tensor<2x3x2x64xf16>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + (d3 mod 8) floordiv 4, d2, (d3 floordiv 8) * 4 + d3 mod 4)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x6x2x32xf32>) outs(%cast_init_10 : tensor<2x3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<2x3x2x64xf16>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #dst_domain
  } : memref<2x3x2x64xf16> -> !ktdp.access_tile<2x3x2x64xindex>
  ktdp.store %cast, %dst_tile : tensor<2x3x2x64xf16>, !ktdp.access_tile<2x3x2x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @widen_rank6_standard_c1(
// CHECK-DAG: memref<2x2x2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @widen_rank6_standard_c1(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x2x2x3x2x64xf16> to tensor<64xf16>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x2x2x6x2x32xf32>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 1 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 2 - d3 >= 0, d4 >= 0, 1 - d4 >= 0, d5 >= 0, 63 - d5 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 1 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 5 - d3 >= 0, d4 >= 0, 1 - d4 >= 0, d5 >= 0, 31 - d5 >= 0)>
#identity = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#cast_order = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, (d3 floordiv 2) * 2 + (d5 mod 8) floordiv 4, d4, (d3 mod 2) * 16 + (d5 floordiv 8) * 4 + d5 mod 4)>

func.func @widen_rank6_standard_c1(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 2, 2, 3, 2, 64], strides: [1536, 768, 384, 128, 64, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x2x3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 2, 2, 6, 2, 32], strides: [1536, 768, 384, 64, 32, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x2x6x2x32xf32>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
    access_tile_set = #src_domain
  } : memref<2x2x2x3x2x64xf16> -> !ktdp.access_tile<2x2x2x3x2x64xindex>
  %input = ktdp.load %src_tile : !ktdp.access_tile<2x2x2x3x2x64xindex> -> tensor<2x2x2x3x2x64xf16>
  %cast_init_11 = tensor.empty() : tensor<2x2x2x6x2x32xf32>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3 * 2 + (d5 mod 8) floordiv 4, d4, (d5 floordiv 8) * 4 + d5 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x2x2x3x2x64xf16>) outs(%cast_init_11 : tensor<2x2x2x6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x2x2x6x2x32xf32>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
    access_tile_set = #dst_domain
  } : memref<2x2x2x6x2x32xf32> -> !ktdp.access_tile<2x2x2x6x2x32xindex>
  %relayout_init_1 = tensor.empty() : tensor<2x2x2x6x2x32xf32>
  %standard_store_1 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%cast : tensor<2x2x2x6x2x32xf32>) outs(%relayout_init_1 : tensor<2x2x2x6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x2x2x6x2x32xf32>
  ktdp.store %standard_store_1, %dst_tile : tensor<2x2x2x6x2x32xf32>, !ktdp.access_tile<2x2x2x6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_rank6_standard_c1(
// CHECK-DAG: memref<2x2x2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @narrow_rank6_standard_c1(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x2x2x6x2x32xf32> to tensor<32xf32>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x2x2x3x2x64xf16>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 1 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 5 - d3 >= 0, d4 >= 0, 1 - d4 >= 0, d5 >= 0, 31 - d5 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 1 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 2 - d3 >= 0, d4 >= 0, 1 - d4 >= 0, d5 >= 0, 63 - d5 >= 0)>
#identity = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#cast_order = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, (d3 floordiv 2) * 2 + (d5 mod 8) floordiv 4, d4, (d3 mod 2) * 16 + (d5 floordiv 8) * 4 + d5 mod 4)>

func.func @narrow_rank6_standard_c1(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 2, 2, 6, 2, 32], strides: [1536, 768, 384, 64, 32, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x2x6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 2, 2, 3, 2, 64], strides: [1536, 768, 384, 128, 64, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x2x3x2x64xf16>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
    access_tile_set = #src_domain
  } : memref<2x2x2x6x2x32xf32> -> !ktdp.access_tile<2x2x2x6x2x32xindex>
  %standard_load_1 = ktdp.load %src_tile : !ktdp.access_tile<2x2x2x6x2x32xindex> -> tensor<2x2x2x6x2x32xf32>
  %relayout_init_1 = tensor.empty() : tensor<2x2x2x6x2x32xf32>
  %input = linalg.generic {indexing_maps = [#identity, #cast_order],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%standard_load_1 : tensor<2x2x2x6x2x32xf32>) outs(%relayout_init_1 : tensor<2x2x2x6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x2x2x6x2x32xf32>
  %cast_init_12 = tensor.empty() : tensor<2x2x2x3x2x64xf16>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3 * 2 + (d5 mod 8) floordiv 4, d4, (d5 floordiv 8) * 4 + d5 mod 4)>,
                     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x2x2x6x2x32xf32>) outs(%cast_init_12 : tensor<2x2x2x3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<2x2x2x3x2x64xf16>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>,
    access_tile_set = #dst_domain
  } : memref<2x2x2x3x2x64xf16> -> !ktdp.access_tile<2x2x2x3x2x64xindex>
  ktdp.store %cast, %dst_tile : tensor<2x2x2x3x2x64xf16>, !ktdp.access_tile<2x2x2x3x2x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @widen_rank4_standard_c4_floor(
// CHECK-DAG: memref<2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.floor {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @widen_rank4_standard_c4_floor(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x3x2x64xf16> to tensor<64xf16>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x6x2x32xf32>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 2 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 63 - d3 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 5 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 31 - d3 >= 0)>
#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#cast_order = affine_map<(d0, d1, d2, d3) -> (d0, (d1 floordiv 2) * 2 + (d3 mod 8) floordiv 4, d2, (d1 mod 2) * 16 + (d3 floordiv 8) * 4 + d3 mod 4)>

func.func @widen_rank4_standard_c4_floor(%src_addr: index, %dst_addr: index) attributes {grid = [4]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 3, 2, 64], strides: [384, 128, 64, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 6, 2, 32], strides: [384, 64, 32, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x6x2x32xf32>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #src_domain
  } : memref<2x3x2x64xf16> -> !ktdp.access_tile<2x3x2x64xindex>
  %input = ktdp.load %src_tile : !ktdp.access_tile<2x3x2x64xindex> -> tensor<2x3x2x64xf16>
  %cast_init_13 = tensor.empty() : tensor<2x6x2x32xf32>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + (d3 mod 8) floordiv 4, d2, (d3 floordiv 8) * 4 + d3 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x3x2x64xf16>) outs(%cast_init_13 : tensor<2x6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x6x2x32xf32>
  %floor_output = math.floor %cast : tensor<2x6x2x32xf32>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #dst_domain
  } : memref<2x6x2x32xf32> -> !ktdp.access_tile<2x6x2x32xindex>
  %relayout_init_1 = tensor.empty() : tensor<2x6x2x32xf32>
  %standard_store_1 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%floor_output : tensor<2x6x2x32xf32>) outs(%relayout_init_1 : tensor<2x6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x6x2x32xf32>
  ktdp.store %standard_store_1, %dst_tile : tensor<2x6x2x32xf32>, !ktdp.access_tile<2x6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_rank4_standard_c4_floor(
// CHECK-DAG: memref<2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.floor {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @narrow_rank4_standard_c4_floor(
// TILE: scf.for
// TILE: tensor.extract_slice {{.*}} : tensor<2x6x2x32xf32> to tensor<32xf32>
// TILE: ktdf_lowering.numeric_cast_value
// TILE: tensor.insert_slice {{.*}} into tensor<2x3x2x64xf16>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#src_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 5 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 31 - d3 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 2 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 63 - d3 >= 0)>
#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#cast_order = affine_map<(d0, d1, d2, d3) -> (d0, (d1 floordiv 2) * 2 + (d3 mod 8) floordiv 4, d2, (d1 mod 2) * 16 + (d3 floordiv 8) * 4 + d3 mod 4)>

func.func @narrow_rank4_standard_c4_floor(%src_addr: index, %dst_addr: index) attributes {grid = [4]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 6, 2, 32], strides: [384, 64, 32, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 3, 2, 64], strides: [384, 128, 64, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x3x2x64xf16>
  %src_tile = ktdp.construct_access_tile %src[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #src_domain
  } : memref<2x6x2x32xf32> -> !ktdp.access_tile<2x6x2x32xindex>
  %standard_load_1 = ktdp.load %src_tile : !ktdp.access_tile<2x6x2x32xindex> -> tensor<2x6x2x32xf32>
  %relayout_init_1 = tensor.empty() : tensor<2x6x2x32xf32>
  %input = linalg.generic {indexing_maps = [#identity, #cast_order],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%standard_load_1 : tensor<2x6x2x32xf32>) outs(%relayout_init_1 : tensor<2x6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x6x2x32xf32>
  %floor_input = math.floor %input : tensor<2x6x2x32xf32>
  %cast_init_14 = tensor.empty() : tensor<2x3x2x64xf16>
  %cast = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + (d3 mod 8) floordiv 4, d2, (d3 floordiv 8) * 4 + d3 mod 4)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%floor_input : tensor<2x6x2x32xf32>) outs(%cast_init_14 : tensor<2x3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<2x3x2x64xf16>
  %dst_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
    access_tile_set = #dst_domain
  } : memref<2x3x2x64xf16> -> !ktdp.access_tile<2x3x2x64xindex>
  ktdp.store %cast, %dst_tile : tensor<2x3x2x64xf16>, !ktdp.access_tile<2x3x2x64xindex>
  return
}
