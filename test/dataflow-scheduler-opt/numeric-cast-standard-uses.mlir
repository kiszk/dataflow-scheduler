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
// RUN: dataflow-scheduler-opt %s --ktir-numeric-cast-tiling --ktir-map-and-tile --split-input-file | FileCheck %s --check-prefix=TILE --implicit-check-not=ktdp.load --implicit-check-not=ktdp.store --implicit-check-not=tensor.insert_slice
// Each cast result has two consumers writing distinct allocations. Pure cast
// and unary chains are specialized per use: each consumer gets its own native
// packet pipeline and retains its own output access order. No branch reloads
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
// TILE: %[[FIRST0:.*]] = tensor.expand_shape %[[CAST0]]#0
// TILE-NEXT: ktdp_lowering.store %[[FIRST0]] into %[[DEST0:[a-zA-Z0-9_]+]][
// TILE: %[[SECOND0:.*]] = tensor.expand_shape %[[CAST0]]#1
// TILE-NEXT: ktdp_lowering.store %[[SECOND0]] into %[[DEST0]][
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST1:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: %[[FIRST1:.*]] = tensor.expand_shape %[[CAST1]]#0
// TILE-NEXT: ktdp_lowering.store %[[FIRST1]] into %[[DEST1:[a-zA-Z0-9_]+]][
// TILE: %[[SECOND1:.*]] = tensor.expand_shape %[[CAST1]]#1
// TILE-NEXT: ktdp_lowering.store %[[SECOND1]] into %[[DEST1]][
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
  } : memref<3x2x64x!spyreop.df16>
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
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %output_tile
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
// TILE: %[[VALUE0:.*]] = tensor.expand_shape %[[CAST0]]
// TILE-NEXT: ktdp_lowering.store %[[VALUE0]]
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST1:.*]] = ktdf_lowering.numeric_cast_value
// TILE: %[[VALUE1:.*]] = tensor.expand_shape %[[CAST1]]
// TILE-NEXT: ktdp_lowering.store %[[VALUE1]]
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
  } : memref<3x2x64x!spyreop.df16>

  %branch = ktdp.construct_memory_view %branch_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %staggered = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %converted = spyreop.numeric_cast %staggered {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64x!spyreop.df16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %converted, %output_tile
      : tensor<3x2x64x!spyreop.df16>, !ktdp.access_tile<3x2x64xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %converted, %branch_tile : tensor<3x2x64x!spyreop.df16>, !ktdp.access_tile<3x2x64xindex>
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
// TILE: %[[FIRST0:.*]] = tensor.expand_shape %[[CAST0]]#0
// TILE-NEXT: ktdp_lowering.store %[[FIRST0]] into %[[DEST0:[a-zA-Z0-9_]+]][
// TILE: %[[SECOND0:.*]] = tensor.expand_shape %[[CAST0]]#1
// TILE-NEXT: ktdp_lowering.store %[[SECOND0]] into %[[DEST0]][
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST1:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: %[[FIRST1:.*]] = tensor.expand_shape %[[CAST1]]#0
// TILE-NEXT: ktdp_lowering.store %[[FIRST1]] into %[[DEST1:[a-zA-Z0-9_]+]][
// TILE: %[[SECOND1:.*]] = tensor.expand_shape %[[CAST1]]#1
// TILE-NEXT: ktdp_lowering.store %[[SECOND1]] into %[[DEST1]][
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
  } : memref<3x2x64x!spyreop.df16>
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
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %branch_tile : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
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
// TILE: arith.negf {{.*}} : f32
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0:[a-zA-Z0-9_]+]][
// TILE: arith.negf {{.*}} : f32
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0]][
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST1:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: %[[FIRST1:.*]] = tensor.expand_shape %[[CAST1]]#0
// TILE-NEXT: ktdp_lowering.store %[[FIRST1]] into %[[DEST1:[a-zA-Z0-9_]+]][
// TILE: %[[SECOND1:.*]] = tensor.expand_shape %[[CAST1]]#1
// TILE-NEXT: ktdp_lowering.store %[[SECOND1]] into %[[DEST1]][
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
  } : memref<3x2x64x!spyreop.df16>
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
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %neg = arith.negf %staggered : tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %neg, %output_tile
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
// TILE: arith.negf {{.*}} : f32
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0:[a-zA-Z0-9_]+]][
// TILE: arith.negf {{.*}} : f32
// TILE: ktdp_lowering.store %{{.*}} into %[[DEST0]][
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST1:.*]]:2 = ktdf_lowering.numeric_cast_value
// TILE: %[[FIRST1:.*]] = tensor.expand_shape %[[CAST1]]#0
// TILE-NEXT: ktdp_lowering.store %[[FIRST1]] into %[[DEST1:[a-zA-Z0-9_]+]][
// TILE: %[[SECOND1:.*]] = tensor.expand_shape %[[CAST1]]#1
// TILE-NEXT: ktdp_lowering.store %[[SECOND1]] into %[[DEST1]][
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
  } : memref<3x2x64x!spyreop.df16>
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
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %neg = arith.negf %staggered : tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %neg, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %branch_tile : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_exp_branch_standard(
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
// TILE-LABEL: func.func @narrow_exp_branch_standard(
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST0:.*]] = ktdf_lowering.numeric_cast_value
// TILE: spyreop.exp {{.*}} : !spyreop.df16
// TILE: ktdp_lowering.store
// TILE: scf.for
// TILE: scf.for
// TILE: %[[CAST1:.*]] = ktdf_lowering.numeric_cast_value
// TILE: %[[VALUE1:.*]] = tensor.expand_shape %[[CAST1]]
// TILE-NEXT: ktdp_lowering.store %[[VALUE1]]
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
  } : memref<3x2x64x!spyreop.df16>

  %branch = ktdp.construct_memory_view %branch_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %staggered = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %converted = spyreop.numeric_cast %staggered {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64x!spyreop.df16>

  %neg_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %neg = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%converted : tensor<3x2x64x!spyreop.df16>) outs(%neg_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%neg_value: !spyreop.df16, %neg_unused: !spyreop.df16):
    %neg_exp = spyreop.exp %neg_value : !spyreop.df16
    linalg.yield %neg_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %neg, %output_tile
      : tensor<3x2x64x!spyreop.df16>, !ktdp.access_tile<3x2x64xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %converted, %branch_tile : tensor<3x2x64x!spyreop.df16>, !ktdp.access_tile<3x2x64xindex>
  return
}
