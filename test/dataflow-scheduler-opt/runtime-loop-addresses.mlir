//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
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
//
//===----------------------------------------------------------------------===//

// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s

// A symbolic HBM base and two enclosing-loop IVs form one runtime address.
// The compiled bodies must have no call arguments. The wrapper declares their
// expressions for the runtime correction; only symbols occur in their DFIR.
// CHECK-LABEL: func.func @loop_correction(
// CHECK: scf.for
// CHECK: scf.for
// CHECK: call @local_schedule_0(
// CHECK: call @local_schedule_1(
// CHECK-LABEL: module @local_schedule_0
// CHECK: func.func @local_schedule_0(
// CHECK: symbol.create_id
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: symbol.create_id
// CHECK: func.func @local_schedule_0_body()
// CHECK: symbol.create_symbol
// CHECK: dataflow.get_logical_memory_view
// CHECK-LABEL: module @local_schedule_1
// CHECK: func.func @local_schedule_1(
// CHECK: symbol.create_id
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: symbol.create_id
// CHECK: func.func @local_schedule_1_body()
// CHECK: symbol.create_symbol
// CHECK: dataflow.get_logical_memory_view

ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#id = affine_map<(s,r,l) -> (s,r,l)>
#all = affine_set<(s,r,l) : (s >= 0, 3-s >= 0, r >= 0, 1-r >= 0, l >= 0, 63-l >= 0)>
#tile = affine_set<(s,r,l) : (s == 0, r == 0, l >= 0, 63-l >= 0)>
  func.func @loop_correction(%input: index, %output: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    scf.for %s = %c0 to %c4 step %c2 {
      scf.for %r = %c0 to %c2 step %c1 {
        %in = ktdp.construct_memory_view %input, sizes: [4, 2, 64], strides: [128, 64, 1] {coordinate_set = #all, memory_space = #ktdp.memory_space<global>} : memref<4x2x64xf16>
        %out = ktdp.construct_memory_view %output, sizes: [4, 2, 64], strides: [128, 64, 1] {coordinate_set = #all, memory_space = #ktdp.memory_space<global>} : memref<4x2x64xf16>
        %it = ktdp.construct_access_tile %in[%s, %r, %c0] {access_tile_order = #id, access_tile_set = #tile} : memref<4x2x64xf16> -> !ktdp.access_tile<1x1x64xindex>
        %ot = ktdp.construct_access_tile %out[%s, %r, %c0] {access_tile_order = #id, access_tile_set = #tile} : memref<4x2x64xf16> -> !ktdp.access_tile<1x1x64xindex>
        %x = ktdp.load %it : <1x1x64xindex> -> tensor<1x1x64xf16>
        %empty = tensor.empty() : tensor<1x1x64xf16>
        %y = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%x : tensor<1x1x64xf16>) outs(%empty : tensor<1x1x64xf16>) {
        ^bb0(%a: f16, %b: f16):
          %v = spyreop.identity %a : f16
          linalg.yield %v : f16
        } -> tensor<1x1x64xf16>
        ktdp.store %y, %ot : tensor<1x1x64xf16>, <1x1x64xindex>
      }
    }
    scf.for %s = %c1 to %c4 step %c2 {
      scf.for %r = %c0 to %c2 step %c1 {
        %in = ktdp.construct_memory_view %input, sizes: [4, 2, 64], strides: [128, 64, 1] {coordinate_set = #all, memory_space = #ktdp.memory_space<global>} : memref<4x2x64xf16>
        %out = ktdp.construct_memory_view %output, sizes: [4, 2, 64], strides: [128, 64, 1] {coordinate_set = #all, memory_space = #ktdp.memory_space<global>} : memref<4x2x64xf16>
        %it = ktdp.construct_access_tile %in[%s, %r, %c0] {access_tile_order = #id, access_tile_set = #tile} : memref<4x2x64xf16> -> !ktdp.access_tile<1x1x64xindex>
        %ot = ktdp.construct_access_tile %out[%s, %r, %c0] {access_tile_order = #id, access_tile_set = #tile} : memref<4x2x64xf16> -> !ktdp.access_tile<1x1x64xindex>
        %x = ktdp.load %it : <1x1x64xindex> -> tensor<1x1x64xf16>
        %empty = tensor.empty() : tensor<1x1x64xf16>
        %y = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%x : tensor<1x1x64xf16>) outs(%empty : tensor<1x1x64xf16>) {
        ^bb0(%a: f16, %b: f16):
          %v = spyreop.identity %a : f16
          linalg.yield %v : f16
        } -> tensor<1x1x64xf16>
        ktdp.store %y, %ot : tensor<1x1x64xf16>, <1x1x64xindex>
      }
    }
    return
  }
