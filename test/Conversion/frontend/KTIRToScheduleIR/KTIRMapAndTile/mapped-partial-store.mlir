// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-map-and-tile | FileCheck %s --implicit-check-not=tensor.insert_slice --implicit-check-not=iter_args

// A partial scatter already selected rows 1 and 3. Further tiling must compose
// its offset and stride with the compute loop, rather than write rows 0 and 1.

#id = affine_map<(r, l) -> (r, l)>
#range = affine_set<(r, l) : (r >= 0, 3-r >= 0, l >= 0, 63-l >= 0)>
ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK: #[[ROW_MAP:.*]] = affine_map<()[s0] -> (s0 * 2 + 1)>
// CHECK-LABEL: func.func @strided_scatter(
// CHECK: %[[TILE:.*]] = ktdp.construct_access_tile
// CHECK: scf.for %[[ROW:.*]] = {{.*}} {
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK: %[[OFFSET:.*]] = affine.apply #[[ROW_MAP]]()[%[[ROW]]]
// CHECK: ktdp_lowering.store %[[RESULT]] into %[[TILE]][%[[OFFSET]], 0] [1, 64] [2, 1]
// CHECK-NEXT: {{^ *}}}
// CHECK-NEXT: return
func.func @strided_scatter(%addr: index, %input: tensor<1x64xf16>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %view = ktdp.construct_memory_view %addr, sizes: [4, 64], strides: [64, 1] {
    coordinate_set = #range, memory_space = #ktdp.memory_space<global>
  } : memref<4x64xf16>
  %tile = ktdp.construct_access_tile %view[0, 0] {
    access_tile_set = #range, access_tile_order = #id
  } : memref<4x64xf16> -> !ktdp.access_tile<4x64xindex>
  %empty = tensor.empty() : tensor<2x64xf16>
  %result = scf.for %row = %zero to %two step %one iter_args(%out = %empty) -> tensor<2x64xf16> {
    %init = tensor.empty() : tensor<1x64xf16>
    %value = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%input : tensor<1x64xf16>) outs(%init : tensor<1x64xf16>) {
    ^bb0(%x: f16, %unused: f16):
      %neg = arith.negf %x : f16
      linalg.yield %neg : f16
    } -> tensor<1x64xf16>
    %updated = tensor.insert_slice %value into %out[%row, 0] [1, 64] [1, 1]
      : tensor<1x64xf16> into tensor<2x64xf16>
    scf.yield %updated : tensor<2x64xf16>
  }
  ktdp_lowering.store %result into %tile[1, 0] [2, 64] [2, 1]
    : tensor<2x64xf16> into !ktdp.access_tile<4x64xindex>
  return
}
