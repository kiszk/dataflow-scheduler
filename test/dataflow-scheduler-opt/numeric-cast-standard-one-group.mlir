// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-input-file --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=access_tile_order --implicit-check-not=ktdf.pipeline
// A single group can lose all enclosing iteration loops during canonicalization.
// CHECK-LABEL: func.func @dl16_to_fp32_standard_one_group(
// CHECK-DAG: memref<2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, l) :
    (s >= 0, 0 - s >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, e) :
    (k >= 0, 1 - k >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>

// With k in [0,2), this is equivalent to the full multi-group map.
#cast_order = affine_map<(k, e) -> (
    (e mod 8) floordiv 4,
    k * 16 + (e floordiv 8) * 4 + e mod 4
)>

func.func @dl16_to_fp32_standard_one_group(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [1, 64], strides: [64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<1x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

  %cast_init_1 = tensor.empty() : tensor<2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0 * 2 + (d1 mod 8) floordiv 4, (d1 floordiv 8) * 4 + d1 mod 4)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<1x64xf16>) outs(%cast_init_1 : tensor<2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  %relayout_init_1 = tensor.empty() : tensor<2x32xf32>
  %standard_store_1 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel"]}
      ins(%staggered : tensor<2x32xf32>) outs(%relayout_init_1 : tensor<2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x32xf32>
  ktdp.store %standard_store_1, %output_tile
      : tensor<2x32xf32>, !ktdp.access_tile<2x32xindex>
  return
}
// -----

// CHECK-LABEL: func.func @fp32_to_dl16_standard_one_group(
// CHECK-DAG: memref<2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, l) :
    (s >= 0, 0 - s >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, e) :
    (k >= 0, 1 - k >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>

// With k in [0,2), this is equivalent to the full multi-group map.
#cast_order = affine_map<(k, e) -> (
    (e mod 8) floordiv 4,
    k * 16 + (e floordiv 8) * 4 + e mod 4
)>

func.func @fp32_to_dl16_standard_one_group(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [1, 64], strides: [64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<1x64xf16>

  %input_tile = ktdp.construct_access_tile %src[0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  %standard_load_1 = ktdp.load %input_tile
      : !ktdp.access_tile<2x32xindex> -> tensor<2x32xf32>
  %relayout_init_1 = tensor.empty() : tensor<2x32xf32>
  %staggered = linalg.generic {indexing_maps = [#identity, #cast_order],
      iterator_types = ["parallel", "parallel"]}
      ins(%standard_load_1 : tensor<2x32xf32>) outs(%relayout_init_1 : tensor<2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x32xf32>

  %cast_init_2 = tensor.empty() : tensor<1x64xf16>
  %converted = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0 * 2 + (d1 mod 8) floordiv 4, (d1 floordiv 8) * 4 + d1 mod 4)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%staggered : tensor<2x32xf32>) outs(%cast_init_2 : tensor<1x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<1x64xf16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  ktdp.store %converted, %output_tile
      : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
  return
}
