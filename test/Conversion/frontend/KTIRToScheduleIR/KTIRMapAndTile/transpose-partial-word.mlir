// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-numeric-cast-tiling -verify-diagnostics -o /dev/null

// Do not pass a lane transpose with a partial output word to elementwise lowering.
#id = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
#src_set = affine_set<(d0, d1) : (d0 >= 0, 30-d0 >= 0, d1 >= 0, 31-d1 >= 0)>
#dst_set = affine_set<(d0, d1) : (d0 >= 0, 31-d0 >= 0, d1 >= 0, 30-d1 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @partial_word(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
    %src = ktdp.construct_memory_view %src_addr, sizes: [31, 32], strides: [32, 1] {coordinate_set = #src_set, memory_space = #ktdp.memory_space<global>} : memref<31x32xf32>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [32, 31], strides: [31, 1] {coordinate_set = #dst_set, memory_space = #ktdp.memory_space<global>} : memref<32x31xf32>
    %a = ktdp.construct_access_tile %src[0, 0] {access_tile_order = #id, access_tile_set = #src_set} : memref<31x32xf32> -> !ktdp.access_tile<31x32xindex>
    %x = ktdp.load %a : !ktdp.access_tile<31x32xindex> -> tensor<31x32xf32>
    %init = tensor.empty() : tensor<32x31xf32>
    // expected-error @+1 {{Standard lane transpose requires full SIMD input sticks and complete output words}}
    %y = linalg.generic {indexing_maps = [#transpose, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<31x32xf32>) outs(%init : tensor<32x31xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
    } -> tensor<32x31xf32>
    %b = ktdp.construct_access_tile %dst[0, 0] {access_tile_order = #id, access_tile_set = #dst_set} : memref<32x31xf32> -> !ktdp.access_tile<32x31xindex>
    ktdp.store %y, %b : tensor<32x31xf32>, !ktdp.access_tile<32x31xindex>
    return
  }
}
