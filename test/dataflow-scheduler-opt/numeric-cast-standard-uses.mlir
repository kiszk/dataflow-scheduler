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

// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-input-file --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=access_tile_order --implicit-check-not=ktdf.pipeline
// RUN: dataflow-scheduler-opt %s --ktir-numeric-cast-tiling --ktir-map-and-tile --split-input-file | FileCheck %s --check-prefix=TILE --implicit-check-not=ktdp.load --implicit-check-not=ktdp.store
// Each cast result has two consumers writing distinct allocations. Pure cast
// and unary chains are specialized per use: each consumer gets its own native
// packet pipeline and retains its own explicit output relayout. No branch reloads
// a full tensor from a sibling output. Function address arguments follow the
// scheduler's non-aliasing argument model.

// CHECK-LABEL: func.func @widen_two_stores_identity(
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @widen_two_stores_identity(
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST0:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0:[a-zA-Z0-9_]+]][
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0]][
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST1:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST1:[a-zA-Z0-9_]+]][
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST1]][
// TILE-NOT: ktdf_lowering.numeric_cast_value
// TILE: return
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

func.func @widen_two_stores_identity(%src_addr: index, %dst_addr: index, %branch_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %branch = ktdp.construct_memory_view %branch_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>

  %cast_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%cast_init_1 : tensor<6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %relayout_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %standard_store_1 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%staggered : tensor<6x2x32xf32>) outs(%relayout_init_1 : tensor<6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>
  ktdp.store %standard_store_1, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %branch_tile : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_two_stores_standard(
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @narrow_two_stores_standard(
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST0:.*]] = ktdf_lowering.numeric_cast_value
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0:[a-zA-Z0-9_]+]][
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST1:.*]] = ktdf_lowering.numeric_cast_value
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST1:[a-zA-Z0-9_]+]][
// TILE-NOT: ktdf_lowering.numeric_cast_value
// TILE: return
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

func.func @narrow_two_stores_standard(%src_addr: index, %dst_addr: index, %branch_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>

  %branch = ktdp.construct_memory_view %branch_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %standard_load_1 = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>
  %relayout_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %staggered = linalg.generic {indexing_maps = [#identity, #cast_order],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%standard_load_1 : tensor<6x2x32xf32>) outs(%relayout_init_1 : tensor<6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>

  %cast_init_2 = tensor.empty() : tensor<3x2x64xf16>
  %converted = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%staggered : tensor<6x2x32xf32>) outs(%cast_init_2 : tensor<3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<3x2x64xf16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %converted, %output_tile
      : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %converted, %branch_tile : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @widen_two_stores_standard(
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @widen_two_stores_standard(
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST0:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0:[a-zA-Z0-9_]+]][
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0]][
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST1:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST1:[a-zA-Z0-9_]+]][
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST1]][
// TILE-NOT: ktdf_lowering.numeric_cast_value
// TILE: return
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

func.func @widen_two_stores_standard(%src_addr: index, %dst_addr: index, %branch_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %branch = ktdp.construct_memory_view %branch_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>

  %cast_init_3 = tensor.empty() : tensor<6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%cast_init_3 : tensor<6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %relayout_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %standard_store_1 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%staggered : tensor<6x2x32xf32>) outs(%relayout_init_1 : tensor<6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>
  ktdp.store %standard_store_1, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %relayout_init_2 = tensor.empty() : tensor<6x2x32xf32>
  %standard_store_2 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%staggered : tensor<6x2x32xf32>) outs(%relayout_init_2 : tensor<6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>
  ktdp.store %standard_store_2, %branch_tile : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @widen_neg_branch_identity(
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.neg {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @widen_neg_branch_identity(
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST0:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0:[a-zA-Z0-9_]+]][
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0]][
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST1:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST1:[a-zA-Z0-9_]+]][
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST1]][
// TILE-NOT: ktdf_lowering.numeric_cast_value
// TILE: return
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

func.func @widen_neg_branch_identity(%src_addr: index, %dst_addr: index, %branch_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %branch = ktdp.construct_memory_view %branch_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>

  %cast_init_4 = tensor.empty() : tensor<6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%cast_init_4 : tensor<6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<6x2x32xf32>

  %neg = arith.negf %staggered : tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %relayout_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %standard_store_1 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%neg : tensor<6x2x32xf32>) outs(%relayout_init_1 : tensor<6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>
  ktdp.store %standard_store_1, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %branch_tile : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @widen_neg_branch_standard(
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.neg {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @widen_neg_branch_standard(
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST0:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0:[a-zA-Z0-9_]+]][
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0]][
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST1:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST1:[a-zA-Z0-9_]+]][
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST1]][
// TILE-NOT: ktdf_lowering.numeric_cast_value
// TILE: return
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

func.func @widen_neg_branch_standard(%src_addr: index, %dst_addr: index, %branch_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %branch = ktdp.construct_memory_view %branch_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>

  %cast_init_5 = tensor.empty() : tensor<6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%cast_init_5 : tensor<6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<6x2x32xf32>

  %neg = arith.negf %staggered : tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %relayout_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %standard_store_1 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%neg : tensor<6x2x32xf32>) outs(%relayout_init_1 : tensor<6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>
  ktdp.store %standard_store_1, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %relayout_init_2 = tensor.empty() : tensor<6x2x32xf32>
  %standard_store_2 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%staggered : tensor<6x2x32xf32>) outs(%relayout_init_2 : tensor<6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>
  ktdp.store %standard_store_2, %branch_tile : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_exp_branch_standard(
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_f16_exp"
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @narrow_exp_branch_standard(
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST0:.*]] = ktdf_lowering.numeric_cast_value
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0:[a-zA-Z0-9_]+]][
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST1:.*]] = ktdf_lowering.numeric_cast_value
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST1:[a-zA-Z0-9_]+]][
// TILE-NOT: ktdf_lowering.numeric_cast_value
// TILE: return
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

func.func @narrow_exp_branch_standard(%src_addr: index, %dst_addr: index, %branch_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>

  %branch = ktdp.construct_memory_view %branch_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %standard_load_1 = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>
  %relayout_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %staggered = linalg.generic {indexing_maps = [#identity, #cast_order],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%standard_load_1 : tensor<6x2x32xf32>) outs(%relayout_init_1 : tensor<6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>

  %cast_init_6 = tensor.empty() : tensor<3x2x64xf16>
  %converted = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%staggered : tensor<6x2x32xf32>) outs(%cast_init_6 : tensor<3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<3x2x64xf16>

  %neg_init = tensor.empty() : tensor<3x2x64xf16>
  %neg = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%converted : tensor<3x2x64xf16>) outs(%neg_init : tensor<3x2x64xf16>) {
  ^bb0(%neg_value: f16, %neg_unused: f16):
    %neg_exp = spyreop.exp %neg_value : f16
    linalg.yield %neg_exp : f16
  } -> tensor<3x2x64xf16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %neg, %output_tile
      : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %converted, %branch_tile : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>
  return
}
