// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=arith.sitofp

// Position indices use signed i32 storage and convert to IEEE FP32 before
// the Rotary Embedding products. This is distinct from a DL16 numeric cast.
// CHECK: agen.vector_load {{.*}}vector<32xi32>
// CHECK: vectorchain.cast {{.*}} : vector<32xi32>, vector<32xf32>
// CHECK: agen.vector_store {{.*}}vector<32xf32>

#id = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, 3-d0 >= 0, d1 >= 0, 31-d1 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @i32_to_fp32(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
    %src = ktdp.construct_memory_view %src_addr, sizes: [4, 32], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x32xi32>
    %a = ktdp.construct_access_tile %src[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<4x32xi32> -> !ktdp.access_tile<4x32xindex>
    %x = ktdp.load %a : !ktdp.access_tile<4x32xindex> -> tensor<4x32xi32>
    %init = tensor.empty() : tensor<4x32xf32>
    %y = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<4x32xi32>) outs(%init : tensor<4x32xf32>) {
    ^bb0(%value: i32, %unused: f32):
      %converted = arith.sitofp %value : i32 to f32
      linalg.yield %converted : f32
    } -> tensor<4x32xf32>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [4, 32], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x32xf32>
    %b = ktdp.construct_access_tile %dst[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<4x32xf32> -> !ktdp.access_tile<4x32xindex>
    ktdp.store %y, %b : tensor<4x32xf32>, !ktdp.access_tile<4x32xindex>
    return
  }
}
