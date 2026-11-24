// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --split-input-file -allow-unregistered-dialect -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=ktdf.opaque --implicit-check-not=vectorchain.scan_with_gap

// Accumulate two sticks per row, then reduce the lanes with a precision-specific
// SUM test op. Only lane 0 of each output stick is the logical row sum.
// Both FP16 and FP32 must take the same complete KTIR -> KTDF -> DFIR pipeline.

// CHECK-LABEL: func.func @sum_f16(
// CHECK: call @local_schedule_0
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: dataflow.receive {{.*}}vector<64xf16>
// CHECK: vectorchain.binary {{.*}}binary_operator add{{.*}}vector<64xf16>
// CHECK: "test.simd_reduction_sum"
// CHECK: dataflow.send {{.*}}vector<64xf16>

ktdf_arch.device @sample_device attributes {
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../Dialect/KTDFArch/sample_device.mlir")

#input = affine_set<(k, r, l) :
    (k >= 0, 1 - k >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#output = affine_set<(r, l) :
    (r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>

func.func @sum_f16(%src: index, %dst: index) attributes {grid = [1]} {
  %input = ktdp.construct_memory_view %src,
      sizes: [2, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #input, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x64xf16>
  %output = ktdp.construct_memory_view %dst,
      sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #output, memory_space = #ktdp.memory_space<global>
  } : memref<2x64xf16>
  %tile = ktdp.construct_access_tile %input[0, 0, 0] {
    access_tile_set = #input,
    access_tile_order = affine_map<(k, r, l) -> (k, r, l)>
  } : memref<2x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
  %values = ktdp.load %tile : !ktdp.access_tile<2x2x64xindex> -> tensor<2x2x64xf16>
  %init = tensor.empty() : tensor<2x64xf16>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(k, r, l, o) -> (k, r, l)>,
                     affine_map<(k, r, l, o) -> (r, o)>],
    iterator_types = ["reduction", "parallel", "reduction", "parallel"]
  } ins(%values : tensor<2x2x64xf16>) outs(%init : tensor<2x64xf16>) {
  ^bb0(%value: f16, %acc: f16):
    %next = arith.addf %value, %acc : f16
    linalg.yield %next : f16
  } -> tensor<2x64xf16>
  %out_tile = ktdp.construct_access_tile %output[0, 0] {
    access_tile_set = #output,
    access_tile_order = affine_map<(r, l) -> (r, l)>
  } : memref<2x64xf16> -> !ktdp.access_tile<2x64xindex>
  ktdp.store %sum, %out_tile : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @sum_f32(
// CHECK: call @local_schedule_0
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: dataflow.receive {{.*}}vector<32xf32>
// CHECK: vectorchain.binary {{.*}}binary_operator add{{.*}}vector<32xf32>
// CHECK: "test.simd_reduction_sum_fp32"
// CHECK: dataflow.send {{.*}}vector<32xf32>

ktdf_arch.device @sample_device attributes {
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../Dialect/KTDFArch/sample_device.mlir")

#input = affine_set<(k, r, l) :
    (k >= 0, 1 - k >= 0, r >= 0, 1 - r >= 0, l >= 0, 31 - l >= 0)>
#output = affine_set<(r, l) :
    (r >= 0, 1 - r >= 0, l >= 0, 31 - l >= 0)>

func.func @sum_f32(%src: index, %dst: index) attributes {grid = [1]} {
  %input = ktdp.construct_memory_view %src,
      sizes: [2, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #input, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x32xf32>
  %output = ktdp.construct_memory_view %dst,
      sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #output, memory_space = #ktdp.memory_space<global>
  } : memref<2x32xf32>
  %tile = ktdp.construct_access_tile %input[0, 0, 0] {
    access_tile_set = #input,
    access_tile_order = affine_map<(k, r, l) -> (k, r, l)>
  } : memref<2x2x32xf32> -> !ktdp.access_tile<2x2x32xindex>
  %values = ktdp.load %tile : !ktdp.access_tile<2x2x32xindex> -> tensor<2x2x32xf32>
  %init = tensor.empty() : tensor<2x32xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(k, r, l, o) -> (k, r, l)>,
                     affine_map<(k, r, l, o) -> (r, o)>],
    iterator_types = ["reduction", "parallel", "reduction", "parallel"]
  } ins(%values : tensor<2x2x32xf32>) outs(%init : tensor<2x32xf32>) {
  ^bb0(%value: f32, %acc: f32):
    %next = arith.addf %value, %acc : f32
    linalg.yield %next : f32
  } -> tensor<2x32xf32>
  %out_tile = ktdp.construct_access_tile %output[0, 0] {
    access_tile_set = #output,
    access_tile_order = affine_map<(r, l) -> (r, l)>
  } : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  ktdp.store %sum, %out_tile : tensor<2x32xf32>, !ktdp.access_tile<2x32xindex>
  return
}
