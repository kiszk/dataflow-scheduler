// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=ktdf_lowering.transpose_packets --implicit-check-not=floordiv --implicit-check-not="{{ mod }}"

// Exchange the lane and row axes across four source sticks, and duplicate
// the 64 features (the cat(freqs, freqs) in Rotary Embedding). Explicit chunk
// loops must remove floor/mod address expressions before target codegen.
// CHECK: agen.vector_load {{.*}} : memref<1x4x64x32xf32>, vector<4xf32>
// CHECK: vectorchain.pack
// CHECK: agen.vector_store {{.*}} : memref<128x2x2x32xf32>, vector<4xf32>

#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#restickify = affine_map<(d0, d1, d2, d3) -> (0, d0 floordiv 32, d2 * 32 + d3, d0 mod 32)>
#src_set = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 >= 0, 3-d1 >= 0, d2 >= 0, 63-d2 >= 0, d3 >= 0, 31-d3 >= 0)>
#dst_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, 127-d0 >= 0, d1 >= 0, 1-d1 >= 0, d2 >= 0, 1-d2 >= 0, d3 >= 0, 31-d3 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @restickify(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
    %src = ktdp.construct_memory_view %src_addr, sizes: [1, 4, 64, 32], strides: [8192, 2048, 32, 1] {coordinate_set = #src_set, memory_space = #ktdp.memory_space<global>} : memref<1x4x64x32xf32>
    %a = ktdp.construct_access_tile %src[0, 0, 0, 0] {access_tile_order = #id, access_tile_set = #src_set} : memref<1x4x64x32xf32> -> !ktdp.access_tile<1x4x64x32xindex>
    %x = ktdp.load %a : !ktdp.access_tile<1x4x64x32xindex> -> tensor<1x4x64x32xf32>
    %init = tensor.empty() : tensor<128x2x2x32xf32>
    %y = linalg.generic {indexing_maps = [#restickify, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%x : tensor<1x4x64x32xf32>) outs(%init : tensor<128x2x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
    } -> tensor<128x2x2x32xf32>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [128, 2, 2, 32], strides: [128, 64, 32, 1] {coordinate_set = #dst_set, memory_space = #ktdp.memory_space<global>} : memref<128x2x2x32xf32>
    %b = ktdp.construct_access_tile %dst[0, 0, 0, 0] {access_tile_order = #id, access_tile_set = #dst_set} : memref<128x2x2x32xf32> -> !ktdp.access_tile<128x2x2x32xindex>
    ktdp.store %y, %b : tensor<128x2x2x32xf32>, !ktdp.access_tile<128x2x2x32xindex>
    return
  }
}
