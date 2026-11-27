// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s

// Rotate two sticks: out[r, 0, l] = -in[r, 1, l],
//                    out[r, 1, l] =  in[r, 0, l].
// Views cover the complete physical buffers; access tiles select each core's
// row shard. Slicing and concatenation use only the identity indexing maps.
// Each store preserves the other stick and lowers to only the updated slice.
// Load/store origins are independently computed to cover frontend CSE as well.

#id = affine_map<(r, l) -> (r, l)>
#first = affine_map<(r, l) -> (r, 0, l)>
#second = affine_map<(r, l) -> (r, 1, l)>
#physical = affine_map<(r, s, l) -> (r, s, l)>
#range = affine_set<(r, s, l) : (r >= 0, 1-r >= 0, s >= 0, 1-s >= 0, l >= 0, 63-l >= 0)>
#all_rows = affine_set<(r, s, l) : (r >= 0, 3-r >= 0, s >= 0, 1-s >= 0, l >= 0, 63-l >= 0)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")

  // CHECK-LABEL: func.func @local_schedule_0_body
  // CHECK: agen.composite_load_and_store src:%{{.*}}[%{{.*}}, 1, 0]
  // CHECK: %[[NEG:.*]] = vectorchain.neg
  // CHECK-NEXT: vectorchain.multiply_and_accumulate %[[NEG]],
  // CHECK-NOT: vectorchain.binary
  // CHECK: agen.composite_load_and_store src:%{{.*}}[%{{.*}}, 0, 0, 0] dst:%{{.*}}[%{{.*}}, 0, {{.*}}]
  // CHECK-LABEL: func.func @local_schedule_1_body
  // CHECK: agen.composite_load_and_store src:%{{.*}}[%{{.*}}, 0, 0]
  // CHECK: agen.composite_load_and_store src:%{{.*}}[%{{.*}}, 0, 0, 0] dst:%{{.*}}[%{{.*}}, 1, {{.*}}]
  func.func @rotate_half(%src_addr: index, %dst_addr: index) attributes {grid = [2]} {
    %two = arith.constant 2 : index
    %core = ktdp.get_compute_tile_id : index
    %row = arith.muli %core, %two : index
    %src = ktdp.construct_memory_view %src_addr, sizes: [4, 2, 64], strides: [128, 64, 1] {coordinate_set = #all_rows, memory_space = #ktdp.memory_space<global>} : memref<4x2x64xf16>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [4, 2, 64], strides: [128, 64, 1] {coordinate_set = #all_rows, memory_space = #ktdp.memory_space<global>} : memref<4x2x64xf16>
    %src_tile = ktdp.construct_access_tile %src[%row, 0, 0] {access_tile_order = #physical, access_tile_set = #range} : memref<4x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
    %dst_tile = ktdp.construct_access_tile %dst[%row, 0, 0] {access_tile_order = #physical, access_tile_set = #range} : memref<4x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
    %input = ktdp.load %src_tile : !ktdp.access_tile<2x2x64xindex> -> tensor<2x2x64xf16>
    %empty = tensor.empty() : tensor<2x64xf16>
    %upper = linalg.generic {indexing_maps = [#second, #id], iterator_types = ["parallel", "parallel"]} ins(%input : tensor<2x2x64xf16>) outs(%empty : tensor<2x64xf16>) {
    ^bb0(%x: f16, %unused: f16):
      %copy = spyreop.identity %x : f16
      linalg.yield %copy : f16
    } -> tensor<2x64xf16>
    %negative = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%upper : tensor<2x64xf16>) outs(%empty : tensor<2x64xf16>) {
    ^bb0(%x: f16, %unused: f16):
      %neg = arith.negf %x : f16
      linalg.yield %neg : f16
    } -> tensor<2x64xf16>
    %initial = ktdp.load %dst_tile : !ktdp.access_tile<2x2x64xindex> -> tensor<2x2x64xf16>
    %first_write = linalg.generic {indexing_maps = [#id, #first], iterator_types = ["parallel", "parallel"]} ins(%negative : tensor<2x64xf16>) outs(%initial : tensor<2x2x64xf16>) {
    ^bb0(%x: f16, %unused: f16):
      %copy = spyreop.identity %x : f16
      linalg.yield %copy : f16
    } -> tensor<2x2x64xf16>
    %store_row = arith.muli %core, %two : index
    %store_tile = ktdp.construct_access_tile %dst[%store_row, 0, 0] {access_tile_order = #physical, access_tile_set = #range} : memref<4x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
    ktdp.store %first_write, %store_tile : tensor<2x2x64xf16>, !ktdp.access_tile<2x2x64xindex>
    // Each stage declares its own views, as in the frontend's emitted KTIR.
    %src1 = ktdp.construct_memory_view %src_addr, sizes: [4, 2, 64], strides: [128, 64, 1] {coordinate_set = #all_rows, memory_space = #ktdp.memory_space<global>} : memref<4x2x64xf16>
    %dst1 = ktdp.construct_memory_view %dst_addr, sizes: [4, 2, 64], strides: [128, 64, 1] {coordinate_set = #all_rows, memory_space = #ktdp.memory_space<global>} : memref<4x2x64xf16>
    %src_tile1 = ktdp.construct_access_tile %src1[%row, 0, 0] {access_tile_order = #physical, access_tile_set = #range} : memref<4x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
    %dst_tile1 = ktdp.construct_access_tile %dst1[%row, 0, 0] {access_tile_order = #physical, access_tile_set = #range} : memref<4x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
    %again = ktdp.load %src_tile1 : !ktdp.access_tile<2x2x64xindex> -> tensor<2x2x64xf16>
    %empty1 = tensor.empty() : tensor<2x64xf16>
    %lower = linalg.generic {indexing_maps = [#first, #id], iterator_types = ["parallel", "parallel"]} ins(%again : tensor<2x2x64xf16>) outs(%empty1 : tensor<2x64xf16>) {
    ^bb0(%x: f16, %unused: f16):
      %copy = spyreop.identity %x : f16
      linalg.yield %copy : f16
    } -> tensor<2x64xf16>
    %previous = ktdp.load %dst_tile1 : !ktdp.access_tile<2x2x64xindex> -> tensor<2x2x64xf16>
    %second_write = linalg.generic {indexing_maps = [#id, #second], iterator_types = ["parallel", "parallel"]} ins(%lower : tensor<2x64xf16>) outs(%previous : tensor<2x2x64xf16>) {
    ^bb0(%x: f16, %unused: f16):
      %copy = spyreop.identity %x : f16
      linalg.yield %copy : f16
    } -> tensor<2x2x64xf16>
    %store_row1 = arith.muli %core, %two : index
    %store_tile1 = ktdp.construct_access_tile %dst1[%store_row1, 0, 0] {access_tile_order = #physical, access_tile_set = #range} : memref<4x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
    ktdp.store %second_write, %store_tile1 : tensor<2x2x64xf16>, !ktdp.access_tile<2x2x64xindex>
    return
  }
}
