// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --fuse-linalg --ktir-map-and-tile --ktir-bufferize --ktir-pipeline | FileCheck %s --implicit-check-not=ktdp_lowering.load --implicit-check-not=ktdp_lowering.store --implicit-check-not=spyreop.identity --implicit-check-not=iter_args

// A reduction followed by an outer-axis permutation must reach the store in
// the same pipeline. Do not leave a full tensor in a separate producer loop.
// CHECK-LABEL: func.func @sum_permuted
// CHECK: ktdf.pipeline
// CHECK: ktdf.data_transfer from
// CHECK-SAME: size [1, 1, 2, 64]
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: ktdf.write_to_fifo
// CHECK: ktdf.data_transfer from
// CHECK-SAME: size [1, 1, 64]

#input_domain = affine_set<(b, r, k, l) :
    (b >= 0, 1 - b >= 0, r >= 0, 2 - r >= 0,
     k >= 0, 1 - k >= 0, l >= 0, 63 - l >= 0)>
#output_domain = affine_set<(r, b, l) :
    (r >= 0, 2 - r >= 0, b >= 0, 1 - b >= 0, l >= 0, 63 - l >= 0)>
#input = affine_map<(b, r, k, l, o) -> (b, r, k, l)>
#reduced = affine_map<(b, r, k, l, o) -> (b, r, o)>
#identity = affine_map<(b, r, l) -> (b, r, l)>
#swap = affine_map<(b, r, l) -> (r, b, l)>
ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
func.func @sum_permuted(%src: index, %dst: index) attributes {grid = [1]} {
  %view = ktdp.construct_memory_view %src,
      sizes: [2, 3, 2, 64], strides: [384, 128, 64, 1] {
    coordinate_set = #input_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x3x2x64xf16>
  %tile = ktdp.construct_access_tile %view[0, 0, 0, 0] {
    access_tile_order = affine_map<(b, r, k, l) -> (b, r, k, l)>,
    access_tile_set = #input_domain
  } : memref<2x3x2x64xf16> -> !ktdp.access_tile<2x3x2x64xindex>
  %input = ktdp.load %tile : <2x3x2x64xindex> -> tensor<2x3x2x64xf16>
  %init = tensor.empty() : tensor<2x3x64xf16>
  %sum = linalg.generic {indexing_maps = [#input, #reduced],
      iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel"]}
      ins(%input : tensor<2x3x2x64xf16>) outs(%init : tensor<2x3x64xf16>) {
    ^bb0(%value: f16, %acc: f16):
      %result = arith.addf %acc, %value : f16
      linalg.yield %result : f16
  } -> tensor<2x3x64xf16>
  %empty = tensor.empty() : tensor<3x2x64xf16>
  %copy = linalg.generic {indexing_maps = [#swap, #identity],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%sum : tensor<2x3x64xf16>) outs(%empty : tensor<3x2x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %result = spyreop.identity %value : f16
      linalg.yield %result : f16
  } -> tensor<3x2x64xf16>
  %out_view = ktdp.construct_memory_view %dst,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #output_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %out_tile = ktdp.construct_access_tile %out_view[0, 0, 0] {
    access_tile_order = #identity, access_tile_set = #output_domain
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %copy, %out_tile : tensor<3x2x64xf16>, <3x2x64xindex>
  return
}
