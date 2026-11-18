// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=ktdf.pipeline --implicit-check-not=access_tile_order

// The caller supplies a disjoint local-memory offset for the Standard
// intermediate. Both store and load must reorder it through the local units.
// CHECK-LABEL: func.func @standard_ct_local_roundtrip(
// CHECK: call @local_schedule_0
// CHECK-NEXT: call @local_schedule_1
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: vectorchain.cast {{.*}}vector<64xf16>, vector<64xf32>
// CHECK: agen.vector_store {{.*}}memref<6x2x32xf32>, vector<4xf32>
// CHECK-LABEL: func.func @local_schedule_1_body()
// CHECK: agen.vector_load {{.*}}memref<6x2x32xf32>, vector<4xf32>
// CHECK: %[[LOW:.*]] = vectorchain.cast {{.*}}vector<32xf32>, vector<32xf16>
// CHECK: %[[HIGH:.*]] = vectorchain.cast {{.*}}vector<32xf32>, vector<32xf16>
// CHECK: vectorchain.pack %[[LOW]], %[[HIGH]] {{.*}}vector<64xf16>

ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#identity = affine_map<(k, r, e) -> (k, r, e)>
// Standard memory coordinates -> Staggered cast-result coordinates.
#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r, (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4)>

func.func @standard_ct_local_roundtrip(
    %src_addr: index, %dst_addr: index, %tmp_offset: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %tmp = ktdp.construct_memory_view %tmp_offset,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<ct_local>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>
  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>
  %tmp_store = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %tmp_store
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>

  %tmp_load = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %reloaded = ktdp.load %tmp_load
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>
  %narrow = spyreop.numeric_cast %reloaded {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64x!spyreop.df16>
  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %narrow, %output_tile
      : tensor<3x2x64x!spyreop.df16>, !ktdp.access_tile<3x2x64xindex>
  return
}
