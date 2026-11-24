// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-numeric-cast-tiling --ktir-map-and-tile --ktir-bufferize --ktir-pipeline | FileCheck %s --check-prefix=KTDF
// RUN: dataflow-scheduler-opt %s --ktir-numeric-cast-tiling --ktir-map-and-tile --ktir-bufferize --ktir-pipeline --materialize-registers | FileCheck %s --check-prefix=VECTOR
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR --implicit-check-not=ktdf_lowering.transpose_packets

// The input maps, rather than the memory views or access tiles, transpose the
// 32x32 matrix. Four-element words are transposed in registers; the complete
// global input/output packets are staged through local memory.
// KTDF-LABEL: func.func @transpose
// KTDF: ktdf.read_from_fifo {{.*}} -> tensor<4xi32>
// KTDF: tensor.insert_slice
// KTDF: linalg.generic {{.*}}ins({{.*}}tensor<4x4xi32>)
// KTDF-SAME: dataflow_scheduler.throttle = 4
// KTDF: tensor.extract_slice
// VECTOR-LABEL: func.func @transpose
// VECTOR: memref.alloc() : memref<8x32xi32, "SFU_REG">
// VECTOR: vector.transfer_read {{.*}} : tensor<4xi32>, vector<4xi32>
// VECTOR: vector.store {{.*}} : memref<8x32xi32, "SFU_REG">, vector<4xi32>
// VECTOR: vector.shuffle {{.*}}[0, 32, 2, 34,
// VECTOR: vector.shuffle {{.*}}[1, 33, 3, 35,
// VECTOR: vector.shuffle {{.*}}[0, 1, 32, 33,
// VECTOR: vector.shuffle {{.*}}[2, 3, 34, 35,
// VECTOR: vector.transfer_write {{.*}} : vector<4xi32>, tensor<4xi32>
// DFIR-LABEL: func.func @transpose
// DFIR: vectorchain.pack {{.*}}indices = [0 : i32, 4 : i32, 2 : i32, 6 : i32]
// DFIR: vectorchain.pack {{.*}}indices = [1 : i32, 5 : i32, 3 : i32, 7 : i32]
// DFIR: vectorchain.pack {{.*}}indices = [0 : i32, 1 : i32, 4 : i32, 5 : i32]
// DFIR: vectorchain.pack {{.*}}indices = [2 : i32, 3 : i32, 6 : i32, 7 : i32]
// DFIR: agen.vector_store {{.*}} : memref<32x32xi32>, vector<4xi32>

#id = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, 31-d0 >= 0, d1 >= 0, 31-d1 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @transpose(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
    %src = ktdp.construct_memory_view %src_addr, sizes: [32, 32], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<32x32xi32>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [32, 32], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<32x32xi32>
    %a = ktdp.construct_access_tile %src[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<32x32xi32> -> !ktdp.access_tile<32x32xindex>
    %x = ktdp.load %a : !ktdp.access_tile<32x32xindex> -> tensor<32x32xi32>
    %init = tensor.empty() : tensor<32x32xi32>
    %y = linalg.generic {indexing_maps = [#id, #transpose], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<32x32xi32>) outs(%init : tensor<32x32xi32>) {
    ^bb0(%value: i32, %unused: i32):
      %copy = spyreop.identity %value : i32
      linalg.yield %copy : i32
    } -> tensor<32x32xi32>
    %b = ktdp.construct_access_tile %dst[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<32x32xi32> -> !ktdp.access_tile<32x32xindex>
    ktdp.store %y, %b : tensor<32x32xi32>, !ktdp.access_tile<32x32xindex>
    return
  }
}
