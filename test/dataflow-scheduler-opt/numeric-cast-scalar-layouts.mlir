// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --split-input-file -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=linalg.generic
// RUN: dataflow-scheduler-opt %s --split-input-file --ktir-numeric-cast-tiling --ktir-map-and-tile --ktir-bufferize --ktir-pipeline | FileCheck %s --check-prefix=PIPE
// Each generic converts directly between Standard layouts; no identity body.

// CHECK-LABEL: func.func @widen_standard_rank2(
// CHECK-DAG: memref<1x64xf16>
// CHECK-DAG: memref<2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// PIPE-LABEL: func.func @widen_standard_rank2(
// PIPE: ktdf.pipeline
// PIPE: ktdf.stage
// PIPE: ktdf_lowering.numeric_cast_value
// PIPE: ktdf.stage
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")
#src_domain = affine_set<(d0, d1) : (d0 >= 0, 0 - d0 >= 0, d1 >= 0, 63 - d1 >= 0)>
#dst_domain = affine_set<(d0, d1) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 31 - d1 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
func.func @widen_standard_rank2(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr, sizes: [1, 64], strides: [64, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<1x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr, sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x32xf32>
  %in_tile = ktdp.construct_access_tile %src[0, 0] {access_tile_set = #src_domain, access_tile_order = #identity}
      : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  %input = ktdp.load %in_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
  %cast_init_1 = tensor.empty() : tensor<2x32xf32>
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0 * 2 + d1 floordiv 32, d1 mod 32)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<1x64xf16>) outs(%cast_init_1 : tensor<2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x32xf32>
  %out_tile = ktdp.construct_access_tile %dst[0, 0] {access_tile_set = #dst_domain, access_tile_order = #identity}
      : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  ktdp.store %output, %out_tile : tensor<2x32xf32>, !ktdp.access_tile<2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_standard_rank2(
// CHECK-DAG: memref<2x32xf32>
// CHECK-DAG: memref<1x64xf16>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// PIPE-LABEL: func.func @narrow_standard_rank2(
// PIPE: ktdf.pipeline
// PIPE: ktdf.stage
// PIPE: ktdf_lowering.numeric_cast_value
// PIPE: ktdf.stage
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")
#src_domain = affine_set<(d0, d1) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 31 - d1 >= 0)>
#dst_domain = affine_set<(d0, d1) : (d0 >= 0, 0 - d0 >= 0, d1 >= 0, 63 - d1 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>
func.func @narrow_standard_rank2(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr, sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr, sizes: [1, 64], strides: [64, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<1x64xf16>
  %in_tile = ktdp.construct_access_tile %src[0, 0] {access_tile_set = #src_domain, access_tile_order = #identity}
      : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  %input = ktdp.load %in_tile : !ktdp.access_tile<2x32xindex> -> tensor<2x32xf32>
  %cast_init_1 = tensor.empty() : tensor<1x64xf16>
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0 * 2 + d1 floordiv 32, d1 mod 32)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<2x32xf32>) outs(%cast_init_1 : tensor<1x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<1x64xf16>
  %out_tile = ktdp.construct_access_tile %dst[0, 0] {access_tile_set = #dst_domain, access_tile_order = #identity}
      : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  ktdp.store %output, %out_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @widen_standard_rank4(
// CHECK-DAG: memref<2x3x2x64xf16>
// CHECK-DAG: memref<2x6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// PIPE-LABEL: func.func @widen_standard_rank4(
// PIPE: ktdf.pipeline
// PIPE: ktdf.stage
// PIPE: ktdf_lowering.numeric_cast_value
// PIPE: ktdf.stage
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")
#src_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 2 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 63 - d3 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 5 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 31 - d3 >= 0)>
#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @widen_standard_rank4(%src_addr: index, %dst_addr: index) attributes {grid = [4]} {
  %src = ktdp.construct_memory_view %src_addr, sizes: [2, 3, 2, 64], strides: [384, 128, 64, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr, sizes: [2, 6, 2, 32], strides: [384, 64, 32, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x6x2x32xf32>
  %in_tile = ktdp.construct_access_tile %src[0, 0, 0, 0] {access_tile_set = #src_domain, access_tile_order = #identity}
      : memref<2x3x2x64xf16> -> !ktdp.access_tile<2x3x2x64xindex>
  %input = ktdp.load %in_tile : !ktdp.access_tile<2x3x2x64xindex> -> tensor<2x3x2x64xf16>
  %cast_init_1 = tensor.empty() : tensor<2x6x2x32xf32>
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + d3 floordiv 32, d2, d3 mod 32)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x3x2x64xf16>) outs(%cast_init_1 : tensor<2x6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x6x2x32xf32>
  %out_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0] {access_tile_set = #dst_domain, access_tile_order = #identity}
      : memref<2x6x2x32xf32> -> !ktdp.access_tile<2x6x2x32xindex>
  ktdp.store %output, %out_tile : tensor<2x6x2x32xf32>, !ktdp.access_tile<2x6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_standard_rank4(
// CHECK-DAG: memref<2x6x2x32xf32>
// CHECK-DAG: memref<2x3x2x64xf16>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// PIPE-LABEL: func.func @narrow_standard_rank4(
// PIPE: ktdf.pipeline
// PIPE: ktdf.stage
// PIPE: ktdf_lowering.numeric_cast_value
// PIPE: ktdf.stage
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")
#src_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 5 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 31 - d3 >= 0)>
#dst_domain = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 1 - d0 >= 0, d1 >= 0, 2 - d1 >= 0, d2 >= 0, 1 - d2 >= 0, d3 >= 0, 63 - d3 >= 0)>
#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @narrow_standard_rank4(%src_addr: index, %dst_addr: index) attributes {grid = [4]} {
  %src = ktdp.construct_memory_view %src_addr, sizes: [2, 6, 2, 32], strides: [384, 64, 32, 1] {
    coordinate_set = #src_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr, sizes: [2, 3, 2, 64], strides: [384, 128, 64, 1] {
    coordinate_set = #dst_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x3x2x64xf16>
  %in_tile = ktdp.construct_access_tile %src[0, 0, 0, 0] {access_tile_set = #src_domain, access_tile_order = #identity}
      : memref<2x6x2x32xf32> -> !ktdp.access_tile<2x6x2x32xindex>
  %input = ktdp.load %in_tile : !ktdp.access_tile<2x6x2x32xindex> -> tensor<2x6x2x32xf32>
  %cast_init_1 = tensor.empty() : tensor<2x3x2x64xf16>
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + d3 floordiv 32, d2, d3 mod 32)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<2x6x2x32xf32>) outs(%cast_init_1 : tensor<2x3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<2x3x2x64xf16>
  %out_tile = ktdp.construct_access_tile %dst[0, 0, 0, 0] {access_tile_set = #dst_domain, access_tile_order = #identity}
      : memref<2x3x2x64xf16> -> !ktdp.access_tile<2x3x2x64xindex>
  ktdp.store %output, %out_tile : tensor<2x3x2x64xf16>, !ktdp.access_tile<2x3x2x64xindex>
  return
}
