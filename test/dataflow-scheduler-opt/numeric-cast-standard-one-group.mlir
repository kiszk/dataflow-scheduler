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

#cast_order = affine_map<(k, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
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

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<1x64xf16> -> tensor<2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  ktdp.store %staggered, %output_tile
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

#cast_order = affine_map<(k, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
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
    access_tile_order = #cast_order
  } : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  %staggered = ktdp.load %input_tile
      : !ktdp.access_tile<2x32xindex> -> tensor<2x32xf32>

  %converted = spyreop.numeric_cast %staggered {kind = "fp32todl16"}
      : tensor<2x32xf32> -> tensor<1x64xf16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  ktdp.store %converted, %output_tile
      : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
  return
}
