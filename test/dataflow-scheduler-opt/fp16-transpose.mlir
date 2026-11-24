// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-numeric-cast-tiling --ktir-map-and-tile --ktir-bufferize --ktir-pipeline --materialize-registers | FileCheck %s --check-prefix=VECTOR
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR --implicit-check-not=ktdf_lowering.transpose_packets --implicit-check-not=vector.shuffle

// Standard f16: 8x8 tiles exchange lane/row bits in three stages. Eight data
// registers plus two temporaries fit in 1280 bytes; no numeric cast is needed.
// VECTOR-LABEL: func.func @transpose
// VECTOR: memref.alloc() : memref<10x64xf16, "SFU_REG">
// VECTOR: vector.transfer_read {{.*}} : tensor<8xf16>, vector<8xf16>
// VECTOR: vector.shuffle {{.*}}[0, 64, 2, 66, 4, 68, 6, 70,
// VECTOR: vector.shuffle {{.*}}[1, 65, 3, 67, 5, 69, 7, 71,
// VECTOR: vector.shuffle {{.*}}[0, 1, 64, 65, 4, 5, 68, 69,
// VECTOR: vector.shuffle {{.*}}[2, 3, 66, 67, 6, 7, 70, 71,
// VECTOR: vector.shuffle {{.*}}[0, 1, 2, 3, 64, 65, 66, 67,
// VECTOR: vector.shuffle {{.*}}[4, 5, 6, 7, 68, 69, 70, 71,
// DFIR-LABEL: func.func @transpose
// DFIR: vectorchain.pack {{.*}}indices = [0 : i32, 8 : i32, 2 : i32, 10 : i32, 4 : i32, 12 : i32, 6 : i32, 14 : i32]
// DFIR: vectorchain.pack {{.*}}indices = [1 : i32, 9 : i32, 3 : i32, 11 : i32, 5 : i32, 13 : i32, 7 : i32, 15 : i32]
// DFIR: vectorchain.pack {{.*}}indices = [0 : i32, 1 : i32, 8 : i32, 9 : i32, 4 : i32, 5 : i32, 12 : i32, 13 : i32]
// DFIR: vectorchain.pack {{.*}}indices = [2 : i32, 3 : i32, 10 : i32, 11 : i32, 6 : i32, 7 : i32, 14 : i32, 15 : i32]
// DFIR: vectorchain.pack {{.*}}indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32]
// DFIR: vectorchain.pack {{.*}}indices = [4 : i32, 5 : i32, 6 : i32, 7 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32]
// DFIR: agen.vector_store {{.*}} : memref<64x64xf16>, vector<8xf16>

#id = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, 63-d0 >= 0, d1 >= 0, 63-d1 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @transpose(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
    %src = ktdp.construct_memory_view %src_addr, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [64, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<64x64xf16>
    %a = ktdp.construct_access_tile %src[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>
    %x = ktdp.load %a : !ktdp.access_tile<64x64xindex> -> tensor<64x64xf16>
    %init = tensor.empty() : tensor<64x64xf16>
    %y = linalg.generic {indexing_maps = [#transpose, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<64x64xf16>) outs(%init : tensor<64x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %copy = spyreop.identity %value : f16
      linalg.yield %copy : f16
    } -> tensor<64x64xf16>
    %b = ktdp.construct_access_tile %dst[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<64x64xf16> -> !ktdp.access_tile<64x64xindex>
    ktdp.store %y, %b : tensor<64x64xf16>, !ktdp.access_tile<64x64xindex>
    return
  }
}
