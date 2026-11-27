// RUN: dataflow-scheduler-opt %s --compute-group-extraction --ktir-distributed-memory | FileCheck %s --check-prefix=EXPAND --implicit-check-not=_gather --implicit-check-not=_scatter --implicit-check-not='memref<2x32xf32>'
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR --implicit-check-not=construct_distributed_memory_view

// A two-core source with reversed owners is copied to two complete destination
// views. The single logical copy scope drives all four fixed physical owners.
// EXPAND-LABEL: func.func @replicate()
// EXPAND-NEXT: call @local_schedule_0_relayout()
// EXPAND-NEXT: call @local_schedule_0_relayout_0()
// EXPAND-NEXT: call @local_schedule_0_relayout_1()
// EXPAND-NEXT: call @local_schedule_0_relayout_2()
// EXPAND-NEXT: return
// DFIR-LABEL: func.func @replicate()
// DFIR-NEXT: call @local_schedule_0_relayout_batch()
// DFIR-NEXT: return
// DFIR-LABEL: func.func @local_schedule_0_relayout_batch_body()
// DFIR-DAG: dataflow.get_unit {core = 0 : i32, {{.*}}type = "mnilu"}
// DFIR-DAG: dataflow.get_unit {core = 1 : i32, {{.*}}type = "mnilu"}
// DFIR-DAG: dataflow.get_unit {core = 2 : i32, {{.*}}type = "mnilu"}
// DFIR-DAG: dataflow.get_unit {core = 3 : i32, {{.*}}type = "mnilu"}
// DFIR: dataflow.send
// DFIR: dataflow.receive

// Copyright 2026 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

ktdf_arch.device @sample_device attributes {
  dataflow_scheduler.core_transfer = {
    send_unit = "MNISU", receive_unit = "MNILU", bytes = 128 : i64
  },
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../../../../Dialect/KTDFArch/sample_device.mlir")

#row0 = affine_set<(r, l) : (r == 0, l >= 0, 31 - l >= 0)>
#row1 = affine_set<(r, l) : (r - 1 == 0, l >= 0, 31 - l >= 0)>
#all = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 31 - l >= 0)>
#id = affine_map<(r, l) -> (r, l)>

func.func @replicate() attributes {grid = [1]} {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %s0 = ktdp.construct_memory_view %c0, sizes: [1, 32], strides: [32, 1] {coordinate_set = #row0, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>} : memref<1x32xf32>
  %s1 = ktdp.construct_memory_view %c0, sizes: [1, 32], strides: [32, 1] {coordinate_set = #row1, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>} : memref<1x32xf32>
  %src = ktdp.construct_distributed_memory_view (%s0, %s1 : memref<1x32xf32>, memref<1x32xf32>) : memref<2x32xf32>
  %d0 = ktdp.construct_memory_view %c256, sizes: [1, 32], strides: [32, 1] {coordinate_set = #row0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>} : memref<1x32xf32>
  %d1 = ktdp.construct_memory_view %c256, sizes: [1, 32], strides: [32, 1] {coordinate_set = #row1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>} : memref<1x32xf32>
  %d2 = ktdp.construct_memory_view %c256, sizes: [1, 32], strides: [32, 1] {coordinate_set = #row0, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>} : memref<1x32xf32>
  %d3 = ktdp.construct_memory_view %c256, sizes: [1, 32], strides: [32, 1] {coordinate_set = #row1, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>} : memref<1x32xf32>
  %dst0 = ktdp.construct_distributed_memory_view (%d0, %d1 : memref<1x32xf32>, memref<1x32xf32>) : memref<2x32xf32>
  %dst1 = ktdp.construct_distributed_memory_view (%d2, %d3 : memref<1x32xf32>, memref<1x32xf32>) : memref<2x32xf32>
  %read = ktdp.construct_access_tile %src[%c0, %c0] {access_tile_order = #id, access_tile_set = #all} : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  %write0 = ktdp.construct_access_tile %dst0[%c0, %c0] {access_tile_order = #id, access_tile_set = #all} : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  %write1 = ktdp.construct_access_tile %dst1[%c0, %c0] {access_tile_order = #id, access_tile_set = #all} : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  %x = ktdp.load %read : !ktdp.access_tile<2x32xindex> -> tensor<2x32xf32>
  %empty0 = tensor.empty() : tensor<2x32xf32>
  %copy0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<2x32xf32>) outs(%empty0 : tensor<2x32xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %id0 = spyreop.identity %v : f32
      linalg.yield %id0 : f32
  } -> tensor<2x32xf32>
  ktdp.store %copy0, %write0 : tensor<2x32xf32>, !ktdp.access_tile<2x32xindex>
  %empty1 = tensor.empty() : tensor<2x32xf32>
  %copy1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<2x32xf32>) outs(%empty1 : tensor<2x32xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %id1 = spyreop.identity %v : f32
      linalg.yield %id1 : f32
  } -> tensor<2x32xf32>
  ktdp.store %copy1, %write1 : tensor<2x32xf32>, !ktdp.access_tile<2x32xindex>
  return
}
