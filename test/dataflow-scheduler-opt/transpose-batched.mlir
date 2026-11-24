// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-numeric-cast-tiling --ktir-map-and-tile --ktir-bufferize --ktir-pipeline --materialize-registers | FileCheck %s --check-prefix=KTDF
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR --implicit-check-not=ktdf_lowering.transpose_packets --implicit-check-not=vector.shuffle

// Two independent batches per core. The Standard rank-4 access tile selects
// the core's slab; only the generic's maps transpose its last two axes.
// KTDF-LABEL: func.func @transpose_batched
// KTDF: ktdp.get_compute_tile_id
// KTDF: memref.alloc() : memref<8x32xf32, "SFU_REG">
// KTDF: vector.shuffle
// KTDF-LABEL: func.func @transpose_batched_f16
// KTDF: ktdp.get_compute_tile_id
// KTDF: memref.alloc() : memref<10x64xf16, "SFU_REG">
// KTDF: vector.shuffle
// KTDF-LABEL: func.func @transpose_batched_i32
// KTDF: ktdp.get_compute_tile_id
// KTDF: memref.alloc() : memref<8x32xi32, "SFU_REG">
// KTDF: vector.shuffle
// DFIR-LABEL: func.func @transpose_batched(
// DFIR-SAME: grid = [2]
// DFIR-LABEL: func.func @transpose_batched_f16(
// DFIR-SAME: grid = [2]
// DFIR-LABEL: func.func @transpose_batched_i32(
// DFIR-SAME: grid = [2]
// DFIR-LABEL: func.func @local_schedule_0_body
// DFIR: dataflow.get_unit {core = 1 : i32
// DFIR: vectorchain.pack {{.*}}vector<32xf32>
// DFIR: agen.vector_store {{.*}}vector<4xf32>
// DFIR-LABEL: func.func @local_schedule_1_body
// DFIR: dataflow.get_unit {core = 1 : i32
// DFIR: vectorchain.pack {{.*}}vector<64xf16>
// DFIR: agen.vector_store {{.*}}vector<8xf16>
// DFIR-LABEL: func.func @local_schedule_2_body
// DFIR: dataflow.get_unit {core = 1 : i32
// DFIR: vectorchain.pack {{.*}}vector<32xi32>
// DFIR: agen.vector_store {{.*}}vector<4xi32>

#id = affine_map<(b, k, r, l) -> (b, k, r, l)>
#transpose = affine_map<(b, k, r, l) -> (b, k, l, r)>
#view = affine_set<(b, k, r, l) : (b >= 0, 1-b >= 0, k >= 0, 1-k >= 0, r >= 0, 31-r >= 0, l >= 0, 31-l >= 0)>
#tile = affine_set<(b, k, r, l) : (b == 0, k >= 0, 1-k >= 0, r >= 0, 31-r >= 0, l >= 0, 31-l >= 0)>
#view16 = affine_set<(b, k, r, l) : (b >= 0, 1-b >= 0, k >= 0, 1-k >= 0, r >= 0, 63-r >= 0, l >= 0, 63-l >= 0)>
#tile16 = affine_set<(b, k, r, l) : (b == 0, k >= 0, 1-k >= 0, r >= 0, 63-r >= 0, l >= 0, 63-l >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @transpose_batched(%src_addr: index, %dst_addr: index) attributes {grid = [2]} {
    %core = ktdp.get_compute_tile_id : index
    %src = ktdp.construct_memory_view %src_addr, sizes: [2, 2, 32, 32], strides: [2048, 1024, 32, 1] {coordinate_set = #view, memory_space = #ktdp.memory_space<global>} : memref<2x2x32x32xf32>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [2, 2, 32, 32], strides: [2048, 1024, 32, 1] {coordinate_set = #view, memory_space = #ktdp.memory_space<global>} : memref<2x2x32x32xf32>
    %a = ktdp.construct_access_tile %src[%core, 0, 0, 0] {access_tile_order = #id, access_tile_set = #tile} : memref<2x2x32x32xf32> -> !ktdp.access_tile<1x2x32x32xindex>
    %x = ktdp.load %a : !ktdp.access_tile<1x2x32x32xindex> -> tensor<1x2x32x32xf32>
    %init = tensor.empty() : tensor<1x2x32x32xf32>
    %y = linalg.generic {indexing_maps = [#transpose, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x : tensor<1x2x32x32xf32>) outs(%init : tensor<1x2x32x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
    } -> tensor<1x2x32x32xf32>
    %b = ktdp.construct_access_tile %dst[%core, 0, 0, 0] {access_tile_order = #id, access_tile_set = #tile} : memref<2x2x32x32xf32> -> !ktdp.access_tile<1x2x32x32xindex>
    ktdp.store %y, %b : tensor<1x2x32x32xf32>, !ktdp.access_tile<1x2x32x32xindex>
    return
  }

  func.func @transpose_batched_f16(%src_addr: index, %dst_addr: index) attributes {grid = [2]} {
    %core = ktdp.get_compute_tile_id : index
    %src = ktdp.construct_memory_view %src_addr, sizes: [2, 2, 64, 64], strides: [8192, 4096, 64, 1] {coordinate_set = #view16, memory_space = #ktdp.memory_space<global>} : memref<2x2x64x64xf16>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [2, 2, 64, 64], strides: [8192, 4096, 64, 1] {coordinate_set = #view16, memory_space = #ktdp.memory_space<global>} : memref<2x2x64x64xf16>
    %a = ktdp.construct_access_tile %src[%core, 0, 0, 0] {access_tile_order = #id, access_tile_set = #tile16} : memref<2x2x64x64xf16> -> !ktdp.access_tile<1x2x64x64xindex>
    %x = ktdp.load %a : !ktdp.access_tile<1x2x64x64xindex> -> tensor<1x2x64x64xf16>
    %init = tensor.empty() : tensor<1x2x64x64xf16>
    %y = linalg.generic {indexing_maps = [#transpose, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x : tensor<1x2x64x64xf16>) outs(%init : tensor<1x2x64x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %copy = spyreop.identity %value : f16
      linalg.yield %copy : f16
    } -> tensor<1x2x64x64xf16>
    %b = ktdp.construct_access_tile %dst[%core, 0, 0, 0] {access_tile_order = #id, access_tile_set = #tile16} : memref<2x2x64x64xf16> -> !ktdp.access_tile<1x2x64x64xindex>
    ktdp.store %y, %b : tensor<1x2x64x64xf16>, !ktdp.access_tile<1x2x64x64xindex>
    return
  }

  func.func @transpose_batched_i32(%src_addr: index, %dst_addr: index) attributes {grid = [2]} {
    %core = ktdp.get_compute_tile_id : index
    %src = ktdp.construct_memory_view %src_addr, sizes: [2, 2, 32, 32], strides: [2048, 1024, 32, 1] {coordinate_set = #view, memory_space = #ktdp.memory_space<global>} : memref<2x2x32x32xi32>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [2, 2, 32, 32], strides: [2048, 1024, 32, 1] {coordinate_set = #view, memory_space = #ktdp.memory_space<global>} : memref<2x2x32x32xi32>
    %a = ktdp.construct_access_tile %src[%core, 0, 0, 0] {access_tile_order = #id, access_tile_set = #tile} : memref<2x2x32x32xi32> -> !ktdp.access_tile<1x2x32x32xindex>
    %x = ktdp.load %a : !ktdp.access_tile<1x2x32x32xindex> -> tensor<1x2x32x32xi32>
    %init = tensor.empty() : tensor<1x2x32x32xi32>
    %y = linalg.generic {indexing_maps = [#transpose, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x : tensor<1x2x32x32xi32>) outs(%init : tensor<1x2x32x32xi32>) {
    ^bb0(%value: i32, %unused: i32):
      %copy = spyreop.identity %value : i32
      linalg.yield %copy : i32
    } -> tensor<1x2x32x32xi32>
    %b = ktdp.construct_access_tile %dst[%core, 0, 0, 0] {access_tile_order = #id, access_tile_set = #tile} : memref<2x2x32x32xi32> -> !ktdp.access_tile<1x2x32x32xindex>
    ktdp.store %y, %b : tensor<1x2x32x32xi32>, !ktdp.access_tile<1x2x32x32xindex>
    return
  }
}
