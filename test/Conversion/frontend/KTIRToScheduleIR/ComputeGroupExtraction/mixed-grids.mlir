// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --compute-group-extraction | FileCheck %s --implicit-check-not=scf.execute_region
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR --implicit-check-not=ktdp.load --implicit-check-not=ktdp.store

// A producer and consumer can use different grids when their boundary is HBM.
// The grid on execute_region belongs to that compute scope. After extraction,
// the caller retains sequential calls, and each callee owns its original grid.
// CHECK-LABEL: func.func @mixed_grids
// CHECK: call @local_schedule_0
// CHECK-NEXT: call @local_schedule_1
// CHECK-LABEL: module @local_schedule_0
// CHECK: func.func @local_schedule_0
// CHECK-SAME: grid = [2]
// CHECK: ktdp.get_compute_tile_id
// CHECK: tensor<4x64xf16>
// CHECK-LABEL: module @local_schedule_1
// CHECK: func.func @local_schedule_1
// CHECK-SAME: grid = [4]
// CHECK: ktdp.get_compute_tile_id
// CHECK: tensor<2x64xf16>
// DFIR-LABEL: func.func @mixed_grids
// DFIR: call @local_schedule_0
// DFIR-NEXT: call @local_schedule_1
// DFIR-LABEL: func.func @local_schedule_0_body
// DFIR: agen.vector_load
// DFIR: agen.vector_store
// DFIR-LABEL: func.func @local_schedule_1_body
// DFIR: agen.vector_load
// DFIR: agen.vector_store

#domain = affine_set<(r, l) : (r >= 0, 7 - r >= 0, l >= 0, 63 - l >= 0)>
#two_core = affine_set<(r, l) : (r >= 0, 3 - r >= 0, l >= 0, 63 - l >= 0)>
#four_core = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#id = affine_map<(r, l) -> (r, l)>
ktdf_arch.device @sample_device attributes {
  mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../../../../Dialect/KTDFArch/sample_device.mlir")
func.func @mixed_grids(%src: index, %tmp: index, %dst: index) attributes {grid = [4]} {
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  scf.execute_region {
    %core = ktdp.get_compute_tile_id : index
    %row = arith.muli %core, %c4 : index
    %in_view = ktdp.construct_memory_view %src, sizes: [8, 64], strides: [64, 1] {
      coordinate_set = #domain, memory_space = #ktdp.memory_space<global>
    } : memref<8x64xf16>
    %in_tile = ktdp.construct_access_tile %in_view[%row, 0] {
      access_tile_order = #id, access_tile_set = #two_core
    } : memref<8x64xf16> -> !ktdp.access_tile<4x64xindex>
    %in = ktdp.load %in_tile : <4x64xindex> -> tensor<4x64xf16>
    %empty = tensor.empty() : tensor<4x64xf16>
    %out = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%in : tensor<4x64xf16>) outs(%empty : tensor<4x64xf16>) {
      ^bb0(%value: f16, %unused: f16):
        %copy = spyreop.identity %value : f16
        linalg.yield %copy : f16
    } -> tensor<4x64xf16>
    %out_view = ktdp.construct_memory_view %tmp, sizes: [8, 64], strides: [64, 1] {
      coordinate_set = #domain, memory_space = #ktdp.memory_space<global>
    } : memref<8x64xf16>
    %out_tile = ktdp.construct_access_tile %out_view[%row, 0] {
      access_tile_order = #id, access_tile_set = #two_core
    } : memref<8x64xf16> -> !ktdp.access_tile<4x64xindex>
    ktdp.store %out, %out_tile : tensor<4x64xf16>, <4x64xindex>
    scf.yield
  } {grid = [2]}
  scf.execute_region {
    %core = ktdp.get_compute_tile_id : index
    %row = arith.muli %core, %c2 : index
    %in_view = ktdp.construct_memory_view %tmp, sizes: [8, 64], strides: [64, 1] {
      coordinate_set = #domain, memory_space = #ktdp.memory_space<global>
    } : memref<8x64xf16>
    %in_tile = ktdp.construct_access_tile %in_view[%row, 0] {
      access_tile_order = #id, access_tile_set = #four_core
    } : memref<8x64xf16> -> !ktdp.access_tile<2x64xindex>
    %in = ktdp.load %in_tile : <2x64xindex> -> tensor<2x64xf16>
    %empty = tensor.empty() : tensor<2x64xf16>
    %out = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%in : tensor<2x64xf16>) outs(%empty : tensor<2x64xf16>) {
      ^bb0(%value: f16, %unused: f16):
        %copy = spyreop.identity %value : f16
        linalg.yield %copy : f16
    } -> tensor<2x64xf16>
    %out_view = ktdp.construct_memory_view %dst, sizes: [8, 64], strides: [64, 1] {
      coordinate_set = #domain, memory_space = #ktdp.memory_space<global>
    } : memref<8x64xf16>
    %out_tile = ktdp.construct_access_tile %out_view[%row, 0] {
      access_tile_order = #id, access_tile_set = #four_core
    } : memref<8x64xf16> -> !ktdp.access_tile<2x64xindex>
    ktdp.store %out, %out_tile : tensor<2x64xf16>, <2x64xindex>
    scf.yield
  } {grid = [4]}
  return
}
