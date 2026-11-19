// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-numeric-cast-tiling --split-input-file --verify-diagnostics -o /dev/null
// Diagnose unsupported/non-bijective recipes before ordinary Linalg tiling.

#bad = affine_map<(k,e) -> (k, e mod 16)>
#identity = affine_map<(k,e) -> (k,e)>
func.func @collision(%src: tensor<2x32xf32>, %init: tensor<2x32xf32>) -> tensor<2x32xf32> {
  // expected-error @+1 {{no target packet relayout recipe for these indexing_maps and tensor shape}}
  %r = linalg.generic {indexing_maps = [#bad, #identity], iterator_types = ["parallel", "parallel"]}
    ins(%src : tensor<2x32xf32>) outs(%init : tensor<2x32xf32>) {
    ^bb0(%x: f32, %y: f32):
      %v = spyreop.identity %x : f32
      linalg.yield %v : f32
  } -> tensor<2x32xf32>
  return %r : tensor<2x32xf32>
}

// -----
#bad = affine_map<(k,e) -> (k, e + 1)>
#identity = affine_map<(k,e) -> (k,e)>
func.func @outside(%src: tensor<2x32xf32>, %init: tensor<2x32xf32>) -> tensor<2x32xf32> {
  // expected-error @+1 {{inferred input/output operand #0 has shape's dimension #1 to be greater than or equal to 33, but found 32}}
  %r = linalg.generic {indexing_maps = [#bad, #identity], iterator_types = ["parallel", "parallel"]}
    ins(%src : tensor<2x32xf32>) outs(%init : tensor<2x32xf32>) {
    ^bb0(%x: f32, %y: f32):
      %v = spyreop.identity %x : f32
      linalg.yield %v : f32
  } -> tensor<2x32xf32>
  return %r : tensor<2x32xf32>
}

// -----
#bad = affine_map<(k,e) -> ((k floordiv 2)*2 + (e mod 8) floordiv 4, (k mod 2)*16+(e floordiv 8)*4+e mod 4)>
#identity = affine_map<(k,e) -> (k,e)>
func.func @incomplete_pair(%src: tensor<3x32xf32>, %init: tensor<3x32xf32>) -> tensor<3x32xf32> {
  // expected-error @+1 {{inferred input/output operand #0 has shape's dimension #0 to be greater than or equal to 4, but found 3}}
  %r = linalg.generic {indexing_maps = [#bad, #identity], iterator_types = ["parallel", "parallel"]}
    ins(%src : tensor<3x32xf32>) outs(%init : tensor<3x32xf32>) {
    ^bb0(%x: f32, %y: f32):
      %v = spyreop.identity %x : f32
      linalg.yield %v : f32
  } -> tensor<3x32xf32>
  return %r : tensor<3x32xf32>
}
