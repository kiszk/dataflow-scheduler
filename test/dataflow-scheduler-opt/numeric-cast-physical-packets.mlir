// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-input-file --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=ktdf_lowering.numeric_cast
// RUN: dataflow-scheduler-opt %s --split-input-file --ktir-numeric-cast-tiling --ktir-map-and-tile --ktir-bufferize --ktir-pipeline | FileCheck %s --check-prefix=PACKETS --implicit-check-not=9223372036854775807
// The ordinary pipeline tiles a native stick into two physical FP32 packets.
// FP32 memory here is staggered (identity access order). This test does not
// cover materialization of the generalized order for Standard FP32 memory.

// PACKETS-LABEL: func.func @widen_physical_packets(
// PACKETS-DAG: ktdf.data_transfer {{.*}}size [1, 1, 64]{{.*}}dataflow_scheduler.throttle = 64 : i64
// PACKETS-DAG: ktdf.data_transfer {{.*}}size [1, 1, 32]{{.*}}dataflow_scheduler.throttle = 32 : i64
// CHECK-LABEL: func.func @widen_physical_packets(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.shuffle
#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")

func.func @widen_physical_packets(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// PACKETS-LABEL: func.func @narrow_physical_packets(
// PACKETS-DAG: ktdf.data_transfer {{.*}}size [1, 1, 32]{{.*}}dataflow_scheduler.throttle = 32 : i64
// PACKETS-DAG: ktdf.data_transfer {{.*}}size [1, 1, 64]{{.*}}dataflow_scheduler.throttle = 64 : i64
// CHECK-LABEL: func.func @narrow_physical_packets(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.pack
#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")

func.func @narrow_physical_packets(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %staggered = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %converted = spyreop.numeric_cast %staggered {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64x!spyreop.df16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %converted, %output_tile
      : tensor<3x2x64x!spyreop.df16>, !ktdp.access_tile<3x2x64xindex>
  return
}
