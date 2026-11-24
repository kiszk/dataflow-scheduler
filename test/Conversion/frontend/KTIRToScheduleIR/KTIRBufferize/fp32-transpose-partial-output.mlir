// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-numeric-cast-tiling --ktir-map-and-tile --ktir-bufferize -verify-diagnostics -o /dev/null

// Staging must not overwrite the untouched half of the global destination
// with uninitialized local memory. A partial destination needs a separate
// packet-preserving update recipe.
#id = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
#tile = affine_set<(d0, d1) : (d0 >= 0, 31-d0 >= 0, d1 >= 0, 31-d1 >= 0)>
#view = affine_set<(d0, d1) : (d0 >= 0, 63-d0 >= 0, d1 >= 0, 31-d1 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @partial_output(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
    %src = ktdp.construct_memory_view %src_addr, sizes: [32, 32], strides: [32, 1] {coordinate_set = #tile, memory_space = #ktdp.memory_space<global>} : memref<32x32xf32>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [64, 32], strides: [32, 1] {coordinate_set = #view, memory_space = #ktdp.memory_space<global>} : memref<64x32xf32>
    %a = ktdp.construct_access_tile %src[0, 0] {access_tile_order = #id, access_tile_set = #tile} : memref<32x32xf32> -> !ktdp.access_tile<32x32xindex>
    %x = ktdp.load %a : !ktdp.access_tile<32x32xindex> -> tensor<32x32xf32>
    %init = tensor.empty() : tensor<32x32xf32>
    // expected-error @+1 {{transpose staging requires a complete destination view}}
    %y = linalg.generic {indexing_maps = [#transpose, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<32x32xf32>) outs(%init : tensor<32x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
    } -> tensor<32x32xf32>
    %b = ktdp.construct_access_tile %dst[0, 0] {access_tile_order = #id, access_tile_set = #tile} : memref<64x32xf32> -> !ktdp.access_tile<32x32xindex>
    ktdp.store %y, %b : tensor<32x32xf32>, !ktdp.access_tile<32x32xindex>
    return
  }
}
