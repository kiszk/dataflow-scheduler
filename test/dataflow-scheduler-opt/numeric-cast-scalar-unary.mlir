// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --split-input-file -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=linalg.generic --implicit-check-not=ktdf_lowering.numeric_cast
// RUN: dataflow-scheduler-opt %s --split-input-file --fuse-linalg --ktir-numeric-cast-tiling | FileCheck %s --check-prefix=TILE
// Scalar casts use the generic maps for Standard memory in both directions.
// FP32 exp uses a twelve-register test template in a sixteen-register file;
// floor uses the ordinary math lowering. Each packet must be sent before the
// next exp overwrites the reused output register.
// RUN: dataflow-scheduler-opt %s --split-input-file -kEmitDFIR --split-dfir-output-dir=%t-regs | FileCheck %s --check-prefix=REG
// RUN: not dataflow-scheduler-opt %s --split-input-file -disable-reuse-registers -kEmitDFIR --split-dfir-output-dir=%t-no-reuse 2>&1 | FileCheck %s --check-prefix=CAPACITY
// CAPACITY: exceeds "SFU_REG" capacity

// REG-LABEL: func.func @standard_dl16_to_standard_fp32(
// REG: func_name = "fake_f32_exp", parameter_dictionary = {{.*}}, read_only_register_dictionary = [[RO:{[^}]+}]], read_write_register_dictionary = [[RW:{[^}]+}]]
// REG: dataflow.send
// REG: func_name = "fake_f32_exp", parameter_dictionary = {{.*}}, read_only_register_dictionary = [[RO]], read_write_register_dictionary = [[RW]]
// CHECK-LABEL: func.func @standard_dl16_to_standard_fp32(
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.floor {{.*}}vector<64xf16>
// CHECK-DAG: func_name = "fake_f32_exp"
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// TILE-LABEL: func.func @standard_dl16_to_standard_fp32(
// TILE: ktdf_lowering.numeric_cast_value
// TILE-NOT: spyreop.numeric_cast
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(g, r, l) :
    (g >= 0, 2 - g >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// Standard DL16 iteration coordinates -> Staggered FP32 coordinates.
// g: 64-element group (0..2); r: row (0..1); l: DL16 lane (0..63).
// Each group of eight DL16 lanes contributes four lanes to each packet.
#dl16_to_staggered_fp32 = affine_map<(g, r, l) -> (
    g * 2 + (l mod 8) floordiv 4, r, (l floordiv 8) * 4 + l mod 4)>

// Standard FP32 coordinates -> Staggered FP32 coordinates.
#cast_order = affine_map<(d0, d1, d2) -> (
    (d0 floordiv 2) * 2 + (d2 mod 8) floordiv 4,
    d1,
    (d0 mod 2) * 16 + (d2 floordiv 8) * 4 + d2 mod 4)>

func.func @standard_dl16_to_standard_fp32(
    %src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>

  %floor_init = tensor.empty() : tensor<3x2x64xf16>
  %rounded = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%floor_init : tensor<3x2x64xf16>) {
  ^bb0(%value: f16, %unused: f16):
    %floor = math.floor %value : f16
    linalg.yield %floor : f16
  } -> tensor<3x2x64xf16>
  // staggered[S(g,r,l)] = cast(rounded[g,r,l]).
  %init = tensor.empty() : tensor<6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [#identity, #dl16_to_staggered_fp32],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%rounded : tensor<3x2x64xf16>) outs(%init : tensor<6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<6x2x32xf32>

  %relayout_init = tensor.empty() : tensor<6x2x32xf32>
  %standard = linalg.generic {
    indexing_maps = [#cast_order, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%staggered : tensor<6x2x32xf32>) outs(%relayout_init : tensor<6x2x32xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %copy = spyreop.identity %value : f32
    linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>

  %exp_init = tensor.empty() : tensor<6x2x32xf32>
  %output = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%standard : tensor<6x2x32xf32>) outs(%exp_init : tensor<6x2x32xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %exp = spyreop.exp %value : f32
    linalg.yield %exp : f32
  } -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %output, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// REG-LABEL: func.func @standard_fp32_to_standard_dl16(
// REG: func_name = "fake_f32_exp", parameter_dictionary = {{.*}}, read_only_register_dictionary = [[RO:{[^}]+}]], read_write_register_dictionary = [[RW:{[^}]+}]]
// REG: dataflow.send
// REG: func_name = "fake_f32_exp", parameter_dictionary = {{.*}}, read_only_register_dictionary = [[RO]], read_write_register_dictionary = [[RW]]
// CHECK-LABEL: func.func @standard_fp32_to_standard_dl16(
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.floor {{.*}}vector<64xf16>
// CHECK-DAG: func_name = "fake_f32_exp"
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// TILE-LABEL: func.func @standard_fp32_to_standard_dl16(
// TILE: ktdf_lowering.numeric_cast_value
// TILE-NOT: spyreop.numeric_cast
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(g, r, l) :
    (g >= 0, 2 - g >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// Standard DL16 iteration coordinates -> Staggered FP32 coordinates.
// g: 64-element group (0..2); r: row (0..1); l: DL16 lane (0..63).
// The cast uses this map on its Staggered FP32 input.
#dl16_to_staggered_fp32 = affine_map<(g, r, l) -> (
    g * 2 + (l mod 8) floordiv 4, r, (l floordiv 8) * 4 + l mod 4)>

// Standard FP32 coordinates -> Staggered FP32 coordinates.
#cast_order = affine_map<(d0, d1, d2) -> (
    (d0 floordiv 2) * 2 + (d2 mod 8) floordiv 4,
    d1,
    (d0 mod 2) * 16 + (d2 floordiv 8) * 4 + d2 mod 4)>

func.func @standard_fp32_to_standard_dl16(
    %src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %standard = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %exp_init = tensor.empty() : tensor<6x2x32xf32>
  %exponential = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%standard : tensor<6x2x32xf32>) outs(%exp_init : tensor<6x2x32xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %exp = spyreop.exp %value : f32
    linalg.yield %exp : f32
  } -> tensor<6x2x32xf32>

  %relayout_init = tensor.empty() : tensor<6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [#identity, #cast_order],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%exponential : tensor<6x2x32xf32>) outs(%relayout_init : tensor<6x2x32xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %copy = spyreop.identity %value : f32
    linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>

  // dl16[g,r,l] = cast(staggered[S(g,r,l)]).
  %init = tensor.empty() : tensor<3x2x64xf16>
  %dl16 = linalg.generic {
    indexing_maps = [#dl16_to_staggered_fp32, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%staggered : tensor<6x2x32xf32>) outs(%init : tensor<3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<3x2x64xf16>

  %floor_init = tensor.empty() : tensor<3x2x64xf16>
  %output = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%dl16 : tensor<3x2x64xf16>) outs(%floor_init : tensor<3x2x64xf16>) {
  ^bb0(%value: f16, %unused: f16):
    %floor = math.floor %value : f16
    linalg.yield %floor : f16
  } -> tensor<3x2x64xf16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %output, %output_tile
      : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>
  return
}
