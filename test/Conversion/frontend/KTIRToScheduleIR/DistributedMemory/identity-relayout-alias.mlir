// RUN: dataflow-scheduler-opt %s --compute-group-extraction --ktir-distributed-memory | FileCheck %s --implicit-check-not=_relayout_batch

// A source slice must not be overwritten before its remote consumer reads it.
// CHECK-LABEL: func.func @aliased_relayout()
// CHECK-NEXT: call @local_schedule_0_gather()
// CHECK: call @local_schedule_0()
// CHECK: call @local_schedule_0_scatter()
// CHECK: return
// CHECK: sizes: [2, 2, 64], strides: [128, 64, 1]

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

// Design example: the producer has already populated each source LX shard.
// This function shows only the relayout stage, not a standalone host workload.
// s = stick-column index, r = row, l = element within a 128-byte DL16 stick.
// Offsets and strides of construct_memory_view are measured in f16 elements.
// Source and destination alias at byte 0 on both cores. A snapshot is required.

ktdf_arch.device @sample_device attributes {
  dataflow_scheduler.core_transfer = {
    send_unit = "MNISU", receive_unit = "MNILU", bytes = 128 : i64
  },
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../../../../Dialect/KTDFArch/sample_device.mlir")

#row0 = affine_set<(s, r, l) : (s >= 0, 1 - s >= 0, r == 0, l >= 0, 63 - l >= 0)>
#row1 = affine_set<(s, r, l) : (s >= 0, 1 - s >= 0, r - 1 == 0, l >= 0, 63 - l >= 0)>
#stick0 = affine_set<(s, r, l) : (s == 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#stick1 = affine_set<(s, r, l) : (s - 1 == 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#tile = affine_set<(s, r, l) : (s == 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#id = affine_map<(s, r, l) -> (s, r, l)>

func.func @aliased_relayout() attributes {grid = [2]} {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 0 : index
  %core = ktdp.get_compute_tile_id : index

  %src0 = ktdp.construct_memory_view %c0, sizes: [2, 1, 64], strides: [64, 64, 1] {
    coordinate_set = #row0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
  } : memref<2x1x64xf16>
  %src1 = ktdp.construct_memory_view %c0, sizes: [2, 1, 64], strides: [64, 64, 1] {
    coordinate_set = #row1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>
  } : memref<2x1x64xf16>
  %src = ktdp.construct_distributed_memory_view (%src0, %src1 : memref<2x1x64xf16>, memref<2x1x64xf16>) : memref<2x2x64xf16>

  %dst0 = ktdp.construct_memory_view %c512, sizes: [1, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #stick0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
  } : memref<1x2x64xf16>
  %dst1 = ktdp.construct_memory_view %c512, sizes: [1, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #stick1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>
  } : memref<1x2x64xf16>
  %dst = ktdp.construct_distributed_memory_view (%dst0, %dst1 : memref<1x2x64xf16>, memref<1x2x64xf16>) : memref<2x2x64xf16>

  %read = ktdp.construct_access_tile %src[%core, %c0, %c0] {
    access_tile_order = #id, access_tile_set = #tile
  } : memref<2x2x64xf16> -> !ktdp.access_tile<1x2x64xindex>
  %write = ktdp.construct_access_tile %dst[%core, %c0, %c0] {
    access_tile_order = #id, access_tile_set = #tile
  } : memref<2x2x64xf16> -> !ktdp.access_tile<1x2x64xindex>
  %x = ktdp.load %read : !ktdp.access_tile<1x2x64xindex> -> tensor<1x2x64xf16>
  %empty = tensor.empty() : tensor<1x2x64xf16>
  %y = linalg.generic {
    indexing_maps = [#id, #id],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%x : tensor<1x2x64xf16>) outs(%empty : tensor<1x2x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %copy = spyreop.identity %value : f16
      linalg.yield %copy : f16
  } -> tensor<1x2x64xf16>
  ktdp.store %y, %write : tensor<1x2x64xf16>, !ktdp.access_tile<1x2x64xindex>
  return
}
