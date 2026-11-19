// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --split-input-file --ktir-numeric-cast-tiling --verify-diagnostics
// The first two cases deliberately use invalid physical shapes and identity
// maps. Other cases reject unsupported maps or bodies, including Standard FP32
// cast maps: relayout must be explicit rather than inserted around the cast.

ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")
func.func @rank1(%input: tensor<64xf16>) -> tensor<64xf32> {
  %cast_init_1 = tensor.empty() : tensor<64xf32>
  // expected-error @below {{numeric cast requires positive static device shapes}}
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%input : tensor<64xf16>) outs(%cast_init_1 : tensor<64xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<64xf32>
  return %output : tensor<64xf32>
}

// -----

ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")
func.func @wrong_stick_size(%input: tensor<3x2x64xf16>) -> tensor<3x2x64xf32> {
  %cast_init_1 = tensor.empty() : tensor<3x2x64xf32>
  // expected-error @below {{numeric cast requires positive static device shapes}}
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%cast_init_1 : tensor<3x2x64xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<3x2x64xf32>
  return %output : tensor<3x2x64xf32>
}

// -----

ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")
func.func @row_permutation(%input: tensor<3x2x64xf16>) -> tensor<6x2x32xf32> {
  %cast_init_1 = tensor.empty() : tensor<6x2x32xf32>
  // expected-error @below {{no numeric cast packet recipe for these indexing_maps}}
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, 1 - d1, (d2 floordiv 8) * 4 + d2 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%cast_init_1 : tensor<6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<6x2x32xf32>
  return %output : tensor<6x2x32xf32>
}

// -----

ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")
func.func @unsupported_body(%input: tensor<3x2x64xf16>) -> tensor<6x2x32xf32> {
  %cast_init_1 = tensor.empty() : tensor<6x2x32xf32>
  // expected-error @below {{numeric cast requires a parallel generic with one input}}
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%cast_init_1 : tensor<6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    %neg = arith.negf %cast : f32
    linalg.yield %neg : f32
  } -> tensor<6x2x32xf32>
  return %output : tensor<6x2x32xf32>
}

// -----

ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")
func.func @standard_widen_requires_explicit_relayout(%input: tensor<3x2x64xf16>) -> tensor<6x2x32xf32> {
  %init = tensor.empty() : tensor<6x2x32xf32>
  // expected-error @below {{no numeric cast packet recipe for these indexing_maps}}
  %output = linalg.generic {
    indexing_maps = [affine_map<(g, r, l) -> (g, r, l)>,
                     affine_map<(g, r, l) -> (2 * g + l floordiv 32, r, l mod 32)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%init : tensor<6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<6x2x32xf32>
  return %output : tensor<6x2x32xf32>
}

// -----

ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")
func.func @standard_narrow_requires_explicit_relayout(%input: tensor<6x2x32xf32>) -> tensor<3x2x64xf16> {
  %init = tensor.empty() : tensor<3x2x64xf16>
  // expected-error @below {{no numeric cast packet recipe for these indexing_maps}}
  %output = linalg.generic {
    indexing_maps = [affine_map<(g, r, l) -> (2 * g + l floordiv 32, r, l mod 32)>,
                     affine_map<(g, r, l) -> (g, r, l)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<6x2x32xf32>) outs(%init : tensor<3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<3x2x64xf16>
  return %output : tensor<3x2x64xf16>
}
