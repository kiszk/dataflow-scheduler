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

// RUN: dataflow-scheduler-opt %s --compute-group-extraction --split-input-file | FileCheck %s --check-prefix=GROUP
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-input-file --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=access_tile_order --implicit-check-not=ktdf.pipeline --implicit-check-not=ktdf.data_transfer --implicit-check-not=linalg.generic

// Each case stores the intermediate once and reloads it through global.
// The four cases cover DL16 exp and FP32 negation around each cast direction.
// Independent unary computes use linalg.generic with an explicit input and init.
// All memory is Standard; the FP32 relayout generic uses #cast_order.
// The caller supplies a disjoint intermediate global address.

// GROUP-LABEL: func.func @dl16_exp_then_widen(
// GROUP-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// GROUP-NEXT: call @local_schedule_0(%[[SRC]], %[[TMP]])
// GROUP-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// GROUP-LABEL: func.func @local_schedule_0(
// GROUP-SAME: %{{.*}}: index, %[[STORE_ADDR:.*]]: index)
// GROUP: %[[STORE_VIEW:.*]] = ktdp.construct_memory_view %[[STORE_ADDR]], {{.*}}memory_space = #ktdp.memory_space<global>{{.*}}memref<3x2x64xf16>
// GROUP: %[[RESULT:.*]] = linalg.generic
// GROUP: %[[STORE_TILE:.*]] = ktdp.construct_access_tile %[[STORE_VIEW]][0, 0, 0]
// GROUP-NEXT: ktdp.store %[[RESULT]], %[[STORE_TILE]]
// GROUP-LABEL: func.func @local_schedule_1(
// GROUP-SAME: %[[LOAD_ADDR:.*]]: index,
// GROUP: %[[LOAD_VIEW:.*]] = ktdp.construct_memory_view %[[LOAD_ADDR]], {{.*}}memory_space = #ktdp.memory_space<global>{{.*}}memref<3x2x64xf16>
// GROUP-NEXT: %[[LOAD_TILE:.*]] = ktdp.construct_access_tile %[[LOAD_VIEW]][0, 0, 0]
// GROUP-NEXT: %[[RELOADED:.*]] = ktdp.load %[[LOAD_TILE]]
// GROUP: linalg.generic {{.*}}ins(%[[RELOADED]] :
// GROUP: spyreop.numeric_cast
// CHECK-LABEL: func.func @dl16_exp_then_widen(
// CHECK-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// CHECK: call @local_schedule_0(%[[SRC]], %[[TMP]])
// CHECK-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: dataflow.opaque {{.*}}func_name = "fake_f16_exp"
// CHECK: %[[STORED:.*]] = dataflow.get_logical_memory_view {{.*}}memref<3x2x64xf16>
// CHECK: agen.composite_load_and_store src:{{.*}} dst:%[[STORED]][
// CHECK-LABEL: func.func @local_schedule_1_body()
// CHECK: %[[RELOAD_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}memref<3x2x64xf16>
// CHECK: agen.composite_load_and_store src:%[[RELOAD_VIEW]][
// CHECK: vectorchain.cast
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// Standard tensor coordinates -> Staggered cast-side tensor coordinates.
#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r, (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4)>

func.func @dl16_exp_then_widen(
    %src_addr: index, %dst_addr: index, %tmp_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %tmp = ktdp.construct_memory_view %tmp_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #dl16_domain
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>

  %init = tensor.empty() : tensor<3x2x64xf16>
  %intermediate = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%init : tensor<3x2x64xf16>) {
  ^bb0(%value: f16, %unused: f16):
    %negated = spyreop.exp %value : f16
    linalg.yield %negated : f16
  } -> tensor<3x2x64xf16>
  %tmp_store = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #dl16_domain
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %intermediate, %tmp_store
      : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>

  %tmp_load = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #dl16_domain
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %reloaded = ktdp.load %tmp_load
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>
  %cast_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%reloaded : tensor<3x2x64xf16>) outs(%cast_init_1 : tensor<6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #fp32_domain
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %relayout_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %standard_store_1 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%output : tensor<6x2x32xf32>) outs(%relayout_init_1 : tensor<6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>
  ktdp.store %standard_store_1, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// GROUP-LABEL: func.func @widen_then_fp32_neg(
// GROUP-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// GROUP-NEXT: call @local_schedule_0(%[[SRC]], %[[TMP]])
// GROUP-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// GROUP-LABEL: func.func @local_schedule_0(
// GROUP-SAME: %{{.*}}: index, %[[STORE_ADDR:.*]]: index)
// GROUP: %[[STORE_VIEW:.*]] = ktdp.construct_memory_view %[[STORE_ADDR]], {{.*}}memory_space = #ktdp.memory_space<global>{{.*}}memref<6x2x32xf32>
// GROUP: %[[CAST:.*]] = linalg.generic
// GROUP: spyreop.numeric_cast
// GROUP: %[[RESULT:.*]] = linalg.generic {{.*}}ins(%[[CAST]] :
// GROUP: spyreop.identity
// GROUP: %[[STORE_TILE:.*]] = ktdp.construct_access_tile %[[STORE_VIEW]][0, 0, 0]
// GROUP-NEXT: ktdp.store %[[RESULT]], %[[STORE_TILE]]
// GROUP-LABEL: func.func @local_schedule_1(
// GROUP-SAME: %[[LOAD_ADDR:.*]]: index,
// GROUP: %[[LOAD_VIEW:.*]] = ktdp.construct_memory_view %[[LOAD_ADDR]], {{.*}}memory_space = #ktdp.memory_space<global>{{.*}}memref<6x2x32xf32>
// GROUP-NEXT: %[[LOAD_TILE:.*]] = ktdp.construct_access_tile %[[LOAD_VIEW]][0, 0, 0]
// GROUP-NEXT: %[[RELOADED:.*]] = ktdp.load %[[LOAD_TILE]]
// GROUP: linalg.generic {{.*}}ins(%[[RELOADED]] :
// CHECK-LABEL: func.func @widen_then_fp32_neg(
// CHECK-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// CHECK: call @local_schedule_0(%[[SRC]], %[[TMP]])
// CHECK-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: vectorchain.cast
// CHECK: %[[STORED:.*]] = dataflow.get_logical_memory_view {{.*}}memref<6x2x32xf32>
// CHECK: agen.composite_load_and_store src:{{.*}} dst:%[[STORED]][
// CHECK-LABEL: func.func @local_schedule_1_body()
// CHECK: %[[RELOAD_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}memref<6x2x32xf32>
// CHECK: agen.composite_load_and_store src:%[[RELOAD_VIEW]][
// CHECK: vectorchain.neg {{.*}}vector<32xf32>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// Standard tensor coordinates -> Staggered cast-side tensor coordinates.
#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r, (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4)>

func.func @widen_then_fp32_neg(
    %src_addr: index, %dst_addr: index, %tmp_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %tmp = ktdp.construct_memory_view %tmp_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #dl16_domain
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>

  %cast_init_2 = tensor.empty() : tensor<6x2x32xf32>
  %intermediate = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%cast_init_2 : tensor<6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<6x2x32xf32>
  %tmp_store = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #fp32_domain
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %relayout_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %standard_store_1 = linalg.generic {indexing_maps = [#cast_order, #identity],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%intermediate : tensor<6x2x32xf32>) outs(%relayout_init_1 : tensor<6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>
  ktdp.store %standard_store_1, %tmp_store
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>

  %tmp_load = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #fp32_domain
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %reloaded = ktdp.load %tmp_load
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>
  %init = tensor.empty() : tensor<6x2x32xf32>
  %output = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%reloaded : tensor<6x2x32xf32>) outs(%init : tensor<6x2x32xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %negated = arith.negf %value : f32
    linalg.yield %negated : f32
  } -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #fp32_domain
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %output, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// GROUP-LABEL: func.func @fp32_neg_then_narrow(
// GROUP-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// GROUP-NEXT: call @local_schedule_0(%[[SRC]], %[[TMP]])
// GROUP-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// GROUP-LABEL: func.func @local_schedule_0(
// GROUP-SAME: %{{.*}}: index, %[[STORE_ADDR:.*]]: index)
// GROUP: %[[STORE_VIEW:.*]] = ktdp.construct_memory_view %[[STORE_ADDR]], {{.*}}memory_space = #ktdp.memory_space<global>{{.*}}memref<6x2x32xf32>
// GROUP: %[[RESULT:.*]] = linalg.generic
// GROUP: %[[STORE_TILE:.*]] = ktdp.construct_access_tile %[[STORE_VIEW]][0, 0, 0]
// GROUP-NEXT: ktdp.store %[[RESULT]], %[[STORE_TILE]]
// GROUP-LABEL: func.func @local_schedule_1(
// GROUP-SAME: %[[LOAD_ADDR:.*]]: index,
// GROUP: %[[LOAD_VIEW:.*]] = ktdp.construct_memory_view %[[LOAD_ADDR]], {{.*}}memory_space = #ktdp.memory_space<global>{{.*}}memref<6x2x32xf32>
// GROUP-NEXT: %[[LOAD_TILE:.*]] = ktdp.construct_access_tile %[[LOAD_VIEW]][0, 0, 0]
// GROUP-NEXT: %[[RELOADED:.*]] = ktdp.load %[[LOAD_TILE]]
// GROUP: %[[STAGGERED:.*]] = linalg.generic {{.*}}ins(%[[RELOADED]] :
// GROUP: spyreop.identity
// GROUP: linalg.generic {{.*}}ins(%[[STAGGERED]] :
// GROUP: spyreop.numeric_cast
// CHECK-LABEL: func.func @fp32_neg_then_narrow(
// CHECK-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// CHECK: call @local_schedule_0(%[[SRC]], %[[TMP]])
// CHECK-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: vectorchain.neg {{.*}}vector<32xf32>
// CHECK: %[[STORED:.*]] = dataflow.get_logical_memory_view {{.*}}memref<6x2x32xf32>
// CHECK: agen.composite_load_and_store src:{{.*}} dst:%[[STORED]][
// CHECK-LABEL: func.func @local_schedule_1_body()
// CHECK: %[[RELOAD_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}memref<6x2x32xf32>
// CHECK: agen.composite_load_and_store src:%[[RELOAD_VIEW]][
// CHECK: vectorchain.cast
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// Standard tensor coordinates -> Staggered cast-side tensor coordinates.
#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r, (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4)>

func.func @fp32_neg_then_narrow(
    %src_addr: index, %dst_addr: index, %tmp_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %tmp = ktdp.construct_memory_view %tmp_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #fp32_domain
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %init = tensor.empty() : tensor<6x2x32xf32>
  %intermediate = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<6x2x32xf32>) outs(%init : tensor<6x2x32xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %negated = arith.negf %value : f32
    linalg.yield %negated : f32
  } -> tensor<6x2x32xf32>
  %tmp_store = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #fp32_domain
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %intermediate, %tmp_store
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>

  %tmp_load = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #fp32_domain
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %standard_load_1 = ktdp.load %tmp_load
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>
  %relayout_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %reloaded = linalg.generic {indexing_maps = [#identity, #cast_order],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%standard_load_1 : tensor<6x2x32xf32>) outs(%relayout_init_1 : tensor<6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>
  %cast_init_3 = tensor.empty() : tensor<3x2x64xf16>
  %output = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%reloaded : tensor<6x2x32xf32>) outs(%cast_init_3 : tensor<3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<3x2x64xf16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #dl16_domain
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %output, %output_tile
      : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>
  return
}

// -----

// GROUP-LABEL: func.func @narrow_then_dl16_exp(
// GROUP-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// GROUP-NEXT: call @local_schedule_0(%[[SRC]], %[[TMP]])
// GROUP-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// GROUP-LABEL: func.func @local_schedule_0(
// GROUP-SAME: %{{.*}}: index, %[[STORE_ADDR:.*]]: index)
// GROUP: %[[STORE_VIEW:.*]] = ktdp.construct_memory_view %[[STORE_ADDR]], {{.*}}memory_space = #ktdp.memory_space<global>{{.*}}memref<3x2x64xf16>
// GROUP: %[[RESULT:.*]] = linalg.generic
// GROUP: spyreop.numeric_cast
// GROUP: %[[STORE_TILE:.*]] = ktdp.construct_access_tile %[[STORE_VIEW]][0, 0, 0]
// GROUP-NEXT: ktdp.store %[[RESULT]], %[[STORE_TILE]]
// GROUP-LABEL: func.func @local_schedule_1(
// GROUP-SAME: %[[LOAD_ADDR:.*]]: index,
// GROUP: %[[LOAD_VIEW:.*]] = ktdp.construct_memory_view %[[LOAD_ADDR]], {{.*}}memory_space = #ktdp.memory_space<global>{{.*}}memref<3x2x64xf16>
// GROUP-NEXT: %[[LOAD_TILE:.*]] = ktdp.construct_access_tile %[[LOAD_VIEW]][0, 0, 0]
// GROUP-NEXT: %[[RELOADED:.*]] = ktdp.load %[[LOAD_TILE]]
// GROUP: linalg.generic {{.*}}ins(%[[RELOADED]] :
// CHECK-LABEL: func.func @narrow_then_dl16_exp(
// CHECK-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// CHECK: call @local_schedule_0(%[[SRC]], %[[TMP]])
// CHECK-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: vectorchain.cast
// CHECK: %[[STORED:.*]] = dataflow.get_logical_memory_view {{.*}}memref<3x2x64xf16>
// CHECK: agen.composite_load_and_store src:{{.*}} dst:%[[STORED]][
// CHECK-LABEL: func.func @local_schedule_1_body()
// CHECK: %[[RELOAD_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}memref<3x2x64xf16>
// CHECK: agen.composite_load_and_store src:%[[RELOAD_VIEW]][
// CHECK: dataflow.opaque {{.*}}func_name = "fake_f16_exp"
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// Standard tensor coordinates -> Staggered cast-side tensor coordinates.
#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r, (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4)>

func.func @narrow_then_dl16_exp(
    %src_addr: index, %dst_addr: index, %tmp_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %tmp = ktdp.construct_memory_view %tmp_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #fp32_domain
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %standard_load_1 = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>
  %relayout_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %input = linalg.generic {indexing_maps = [#identity, #cast_order],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%standard_load_1 : tensor<6x2x32xf32>) outs(%relayout_init_1 : tensor<6x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<6x2x32xf32>

  %cast_init_4 = tensor.empty() : tensor<3x2x64xf16>
  %intermediate = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<6x2x32xf32>) outs(%cast_init_4 : tensor<3x2x64xf16>) {
  ^bb0(%value: f32, %unused: f16):
    %cast = spyreop.numeric_cast %value {kind = "fp32todl16"} : f32 -> f16
    linalg.yield %cast : f16
  } -> tensor<3x2x64xf16>
  %tmp_store = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #dl16_domain
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %intermediate, %tmp_store
      : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>

  %tmp_load = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #dl16_domain
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %reloaded = ktdp.load %tmp_load
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>
  %init = tensor.empty() : tensor<3x2x64xf16>
  %output = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%reloaded : tensor<3x2x64xf16>) outs(%init : tensor<3x2x64xf16>) {
  ^bb0(%value: f16, %unused: f16):
    %negated = spyreop.exp %value : f16
    linalg.yield %negated : f16
  } -> tensor<3x2x64xf16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
    access_tile_set = #dl16_domain
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %output, %output_tile
      : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>
  return
}
