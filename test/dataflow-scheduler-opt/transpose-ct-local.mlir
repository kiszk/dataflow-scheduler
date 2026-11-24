// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=ktdf_lowering.transpose_packets --implicit-check-not=" vector.shuffle "

// Floor -> ct_local -> transpose -> ct_local -> floor. The explicit local
// views hold full Standard matrices; transpose word traffic stays on LX.
// CHECK-LABEL: func.func @transpose_local_chain
// CHECK: call @local_schedule_0
// CHECK: call @local_schedule_1
// CHECK: call @local_schedule_2
// CHECK-LABEL: func.func @local_schedule_0_body
// CHECK: vectorchain.floor {{.*}}vector<32xf32>
// CHECK-LABEL: func.func @local_schedule_1_body
// CHECK: agen.vector_load {{.*}}vector<4xf32>
// CHECK: vectorchain.pack
// CHECK: agen.vector_store {{.*}}vector<4xf32>
// CHECK-LABEL: func.func @local_schedule_2_body
// CHECK: vectorchain.floor {{.*}}vector<32xf32>

#id = affine_map<(r, l) -> (r, l)>
#transpose = affine_map<(r, l) -> (l, r)>
#set = affine_set<(r, l) : (r >= 0, 31-r >= 0, l >= 0, 31-l >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @transpose_local_chain(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %second = arith.constant 4096 : index
    %src = ktdp.construct_memory_view %src_addr, sizes: [32, 32], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<32x32xf32>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [32, 32], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<32x32xf32>
    %local_a = ktdp.construct_memory_view %zero, sizes: [32, 32], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<32x32xf32>
    %local_b = ktdp.construct_memory_view %second, sizes: [32, 32], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<32x32xf32>
    %input_tile = ktdp.construct_access_tile %src[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<32x32xf32> -> !ktdp.access_tile<32x32xindex>
    %input = ktdp.load %input_tile : !ktdp.access_tile<32x32xindex> -> tensor<32x32xf32>
    %floored_init = tensor.empty() : tensor<32x32xf32>
    %floored = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%input : tensor<32x32xf32>) outs(%floored_init : tensor<32x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %rounded = math.floor %value : f32
      linalg.yield %rounded : f32
    } -> tensor<32x32xf32>
    %tile_a = ktdp.construct_access_tile %local_a[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<32x32xf32> -> !ktdp.access_tile<32x32xindex>
    ktdp.store %floored, %tile_a : tensor<32x32xf32>, !ktdp.access_tile<32x32xindex>
    %reload_a = ktdp.construct_access_tile %local_a[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<32x32xf32> -> !ktdp.access_tile<32x32xindex>
    %x = ktdp.load %reload_a : !ktdp.access_tile<32x32xindex> -> tensor<32x32xf32>
    %init = tensor.empty() : tensor<32x32xf32>
    %y = linalg.generic {indexing_maps = [#transpose, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<32x32xf32>) outs(%init : tensor<32x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
    } -> tensor<32x32xf32>
    %tile_b = ktdp.construct_access_tile %local_b[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<32x32xf32> -> !ktdp.access_tile<32x32xindex>
    ktdp.store %y, %tile_b : tensor<32x32xf32>, !ktdp.access_tile<32x32xindex>
    %reload_b = ktdp.construct_access_tile %local_b[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<32x32xf32> -> !ktdp.access_tile<32x32xindex>
    %reloaded = ktdp.load %reload_b : !ktdp.access_tile<32x32xindex> -> tensor<32x32xf32>
    %result_init = tensor.empty() : tensor<32x32xf32>
    %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%reloaded : tensor<32x32xf32>) outs(%result_init : tensor<32x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %rounded = math.floor %value : f32
      linalg.yield %rounded : f32
    } -> tensor<32x32xf32>
    %output_tile = ktdp.construct_access_tile %dst[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<32x32xf32> -> !ktdp.access_tile<32x32xindex>
    ktdp.store %result, %output_tile : tensor<32x32xf32>, !ktdp.access_tile<32x32xindex>
    return
  }
}
