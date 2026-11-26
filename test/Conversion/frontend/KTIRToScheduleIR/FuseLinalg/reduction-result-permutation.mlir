// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --fuse-linalg | FileCheck %s

// Result permutations preserve the reduction loops and their lane axis. Both
// input and output maps of the identity participate in the composition.
#input = affine_map<(b, r, k, l, o) -> (b, r, k, l)>
#reduced = affine_map<(b, r, k, l, o) -> (b, r, o)>
#identity = affine_map<(b, r, l) -> (b, r, l)>
#swap = affine_map<(b, r, l) -> (r, b, l)>

// CHECK-DAG: #[[INPUT:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[OUTPUT:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d0, d4)>
// CHECK-LABEL: func.func @sum_permute
// CHECK: linalg.generic {indexing_maps = [#[[INPUT]], #[[OUTPUT]]], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel"]}
// CHECK-SAME: outs({{.*}} : tensor<3x2x64xf16>)
// CHECK: arith.addf
// CHECK-NOT: linalg.generic
// CHECK-NOT: spyreop.identity
// CHECK: return
func.func @sum_permute(%input: tensor<2x3x2x64xf16>) -> tensor<3x2x64xf16> {
  %init = tensor.empty() : tensor<2x3x64xf16>
  %sum = linalg.generic {
    indexing_maps = [#input, #reduced],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel"]
  } ins(%input : tensor<2x3x2x64xf16>) outs(%init : tensor<2x3x64xf16>) {
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
  return %copy : tensor<3x2x64xf16>
}

// CHECK-LABEL: func.func @max_permute
// CHECK: linalg.generic {indexing_maps = [#[[INPUT]], #[[OUTPUT]]]
// CHECK-SAME: outs({{.*}} : tensor<3x2x64xf16>)
// CHECK-SAME: ktdf_arch.maps_to = "SFU"
// CHECK: arith.maximumf
// CHECK-NOT: linalg.generic
// CHECK: return
func.func @max_permute(%input: tensor<2x3x2x64xf16>) -> tensor<3x2x64xf16> {
  %init = tensor.empty() : tensor<2x3x64xf16>
  %max = linalg.generic {
    indexing_maps = [#input, #reduced],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel"]
  } ins(%input : tensor<2x3x2x64xf16>) outs(%init : tensor<2x3x64xf16>)
    attrs = {ktdf_arch.maps_to = "SFU"} {
    ^bb0(%value: f16, %acc: f16):
      %result = arith.maximumf %acc, %value : f16
      linalg.yield %result : f16
  } -> tensor<2x3x64xf16>
  %empty = tensor.empty() : tensor<3x2x64xf16>
  %copy = linalg.generic {indexing_maps = [#identity, #swap],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%max : tensor<2x3x64xf16>) outs(%empty : tensor<3x2x64xf16>)
      attrs = {ktdf_arch.maps_to = "SFU"} {
    ^bb0(%value: f16, %unused: f16):
      %result = spyreop.identity %value : f16
      linalg.yield %result : f16
  } -> tensor<3x2x64xf16>
  return %copy : tensor<3x2x64xf16>
}

// A nontrivial accumulator must not be replaced by an empty destination.
// CHECK-LABEL: func.func @preserve_seed
// CHECK: linalg.generic
// CHECK-SAME: outs(%{{.*}} : tensor<2x3x64xf16>)
// CHECK: arith.addf
// CHECK: linalg.generic
// CHECK: spyreop.identity
// CHECK: return
func.func @preserve_seed(%input: tensor<2x3x2x64xf16>,
                        %seed: tensor<2x3x64xf16>) -> tensor<3x2x64xf16> {
  %sum = linalg.generic {
    indexing_maps = [#input, #reduced],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel"]
  } ins(%input : tensor<2x3x2x64xf16>) outs(%seed : tensor<2x3x64xf16>) {
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
  return %copy : tensor<3x2x64xf16>
}

// Moving the physical lane axis needs an actual transpose, not just new stores.
// CHECK-LABEL: func.func @preserve_lane_transpose
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.generic
// CHECK: spyreop.identity
// CHECK: return
func.func @preserve_lane_transpose(%input: tensor<2x64x2x64xf16>)
    -> tensor<2x64x64xf16> {
  %init = tensor.empty() : tensor<2x64x64xf16>
  %sum = linalg.generic {
    indexing_maps = [#input, #reduced],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel"]
  } ins(%input : tensor<2x64x2x64xf16>) outs(%init : tensor<2x64x64xf16>) {
    ^bb0(%value: f16, %acc: f16):
      %result = arith.addf %acc, %value : f16
      linalg.yield %result : f16
  } -> tensor<2x64x64xf16>
  %empty = tensor.empty() : tensor<2x64x64xf16>
  %copy = linalg.generic {
      indexing_maps = [affine_map<(b, r, l) -> (b, l, r)>, #identity],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%sum : tensor<2x64x64xf16>) outs(%empty : tensor<2x64x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %result = spyreop.identity %value : f16
      linalg.yield %result : f16
  } -> tensor<2x64x64xf16>
  return %copy : tensor<2x64x64xf16>
}
