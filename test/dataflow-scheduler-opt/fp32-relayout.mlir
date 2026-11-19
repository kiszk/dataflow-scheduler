// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=ktdf.pipeline --implicit-check-not=vectorchain.cast
// CHECK-LABEL: func.func @relayout_roundtrip
// CHECK: memref<6x2x32xf32>
// CHECK: agen.vector_load {{.*}}vector<4xf32>
// CHECK: agen.vector_store {{.*}}vector<4xf32>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

func.func @relayout_roundtrip(%src_addr: index, %dst_addr: index) attributes {grid = [4]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %it = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #fp32_domain
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %ot = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #fp32_domain
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %input = ktdp.load %it : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>
  %init = tensor.empty() : tensor<6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [#identity, #cast_order],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<6x2x32xf32>) outs(%init : tensor<6x2x32xf32>) {
    ^bb0(%x: f32, %unused: f32):
      %y = spyreop.identity %x : f32
      linalg.yield %y : f32
  } -> tensor<6x2x32xf32>
  %standard = linalg.generic {
    indexing_maps = [#cast_order, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%staggered : tensor<6x2x32xf32>) outs(%init : tensor<6x2x32xf32>) {
    ^bb0(%x: f32, %unused: f32):
      %y = spyreop.identity %x : f32
      linalg.yield %y : f32
  } -> tensor<6x2x32xf32>
  ktdp.store %standard, %ot : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}
