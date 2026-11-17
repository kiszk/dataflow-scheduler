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

// Each case stores the intermediate once and reloads it through ct_local.
// The four cases cover negation before and after each cast direction.
// Independent unary computes use linalg.generic with an explicit input and init.
// All memory is Standard; only the FP32 cast-side access uses #cast_order.
// The caller supplies a disjoint intermediate local offset (outside compiler scratch).

// GROUP-LABEL: func.func @dl16_neg_then_widen(
// GROUP-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// GROUP-NEXT: call @local_schedule_0(%[[SRC]], %[[TMP]])
// GROUP-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// GROUP-LABEL: func.func @local_schedule_0(
// GROUP-SAME: %{{.*}}: index, %[[STORE_ADDR:.*]]: index)
// GROUP: %[[STORE_VIEW:.*]] = ktdp.construct_memory_view %[[STORE_ADDR]], {{.*}}memory_space = #ktdp.memory_space<ct_local>{{.*}}memref<3x2x64xf16>
// GROUP: %[[RESULT:.*]] = linalg.generic
// GROUP: %[[STORE_TILE:.*]] = ktdp.construct_access_tile %[[STORE_VIEW]][0, 0, 0]
// GROUP-NEXT: ktdp.store %[[RESULT]], %[[STORE_TILE]]
// GROUP-LABEL: func.func @local_schedule_1(
// GROUP-SAME: %[[LOAD_ADDR:.*]]: index,
// GROUP: %[[LOAD_VIEW:.*]] = ktdp.construct_memory_view %[[LOAD_ADDR]], {{.*}}memory_space = #ktdp.memory_space<ct_local>{{.*}}memref<3x2x64xf16>
// GROUP-NEXT: %[[LOAD_TILE:.*]] = ktdp.construct_access_tile %[[LOAD_VIEW]][0, 0, 0]
// GROUP-NEXT: %[[RELOADED:.*]] = ktdp.load %[[LOAD_TILE]]
// GROUP: spyreop.numeric_cast %[[RELOADED]]
// CHECK-LABEL: func.func @dl16_neg_then_widen(
// CHECK-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// CHECK: call @local_schedule_0(%[[SRC]], %[[TMP]])
// CHECK-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: vectorchain.neg {{.*}}vector<64xf16>
// CHECK: %[[STORED:.*]] = dataflow.get_logical_memory_view {{.*}}memref<3x2x64xf16>
// CHECK: agen.vector_store {{.*}}, %[[STORED]][{{.*}}memref<3x2x64xf16>, vector<64xf16>
// CHECK-LABEL: func.func @local_schedule_1_body()
// CHECK: %[[RELOAD_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}memref<3x2x64xf16>
// CHECK: agen.vector_load %[[RELOAD_VIEW]][{{.*}}memref<3x2x64xf16>, vector<64xf16>
// CHECK: vectorchain.cast
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// Standard memory coordinates -> Staggered cast-side tensor coordinates.
#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r, (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4)>

func.func @dl16_neg_then_widen(
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
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<ct_local>
  } : memref<3x2x64xf16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>

  %init = tensor.empty() : tensor<3x2x64xf16>
  %intermediate = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%init : tensor<3x2x64xf16>) {
  ^bb0(%value: f16, %unused: f16):
    %negated = arith.negf %value : f16
    linalg.yield %negated : f16
  } -> tensor<3x2x64xf16>
  %tmp_store = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %intermediate, %tmp_store
      : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>

  %tmp_load = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %reloaded = ktdp.load %tmp_load
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>
  %output = spyreop.numeric_cast %reloaded {kind = "dl16tofp32"}
      : tensor<3x2x64xf16> -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %output, %output_tile
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
// GROUP: %[[STORE_VIEW:.*]] = ktdp.construct_memory_view %[[STORE_ADDR]], {{.*}}memory_space = #ktdp.memory_space<ct_local>{{.*}}memref<6x2x32xf32>
// GROUP: %[[RESULT:.*]] = spyreop.numeric_cast
// GROUP: %[[STORE_TILE:.*]] = ktdp.construct_access_tile %[[STORE_VIEW]][0, 0, 0]
// GROUP-NEXT: ktdp.store %[[RESULT]], %[[STORE_TILE]]
// GROUP-LABEL: func.func @local_schedule_1(
// GROUP-SAME: %[[LOAD_ADDR:.*]]: index,
// GROUP: %[[LOAD_VIEW:.*]] = ktdp.construct_memory_view %[[LOAD_ADDR]], {{.*}}memory_space = #ktdp.memory_space<ct_local>{{.*}}memref<6x2x32xf32>
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
// CHECK: agen.vector_store {{.*}}, %[[STORED]][{{.*}}memref<6x2x32xf32>, vector<4xf32>
// CHECK-LABEL: func.func @local_schedule_1_body()
// CHECK: %[[RELOAD_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}memref<6x2x32xf32>
// CHECK: agen.vector_load %[[RELOAD_VIEW]][{{.*}}memref<6x2x32xf32>, vector<32xf32>
// CHECK: vectorchain.neg {{.*}}vector<32xf32>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// Standard memory coordinates -> Staggered cast-side tensor coordinates.
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
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<ct_local>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>

  %intermediate = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64xf16> -> tensor<6x2x32xf32>
  %tmp_store = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %intermediate, %tmp_store
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>

  %tmp_load = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #identity
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
    access_tile_set = #fp32_domain, access_tile_order = #identity
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
// GROUP: %[[STORE_VIEW:.*]] = ktdp.construct_memory_view %[[STORE_ADDR]], {{.*}}memory_space = #ktdp.memory_space<ct_local>{{.*}}memref<6x2x32xf32>
// GROUP: %[[RESULT:.*]] = linalg.generic
// GROUP: %[[STORE_TILE:.*]] = ktdp.construct_access_tile %[[STORE_VIEW]][0, 0, 0]
// GROUP-NEXT: ktdp.store %[[RESULT]], %[[STORE_TILE]]
// GROUP-LABEL: func.func @local_schedule_1(
// GROUP-SAME: %[[LOAD_ADDR:.*]]: index,
// GROUP: %[[LOAD_VIEW:.*]] = ktdp.construct_memory_view %[[LOAD_ADDR]], {{.*}}memory_space = #ktdp.memory_space<ct_local>{{.*}}memref<6x2x32xf32>
// GROUP-NEXT: %[[LOAD_TILE:.*]] = ktdp.construct_access_tile %[[LOAD_VIEW]][0, 0, 0]
// GROUP-NEXT: %[[RELOADED:.*]] = ktdp.load %[[LOAD_TILE]]
// GROUP: spyreop.numeric_cast %[[RELOADED]]
// CHECK-LABEL: func.func @fp32_neg_then_narrow(
// CHECK-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// CHECK: call @local_schedule_0(%[[SRC]], %[[TMP]])
// CHECK-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: vectorchain.neg {{.*}}vector<32xf32>
// CHECK: %[[STORED:.*]] = dataflow.get_logical_memory_view {{.*}}memref<6x2x32xf32>
// CHECK: agen.vector_store {{.*}}, %[[STORED]][{{.*}}memref<6x2x32xf32>, vector<32xf32>
// CHECK-LABEL: func.func @local_schedule_1_body()
// CHECK: %[[RELOAD_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}memref<6x2x32xf32>
// CHECK: agen.vector_load %[[RELOAD_VIEW]][{{.*}}memref<6x2x32xf32>, vector<4xf32>
// CHECK: vectorchain.cast
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// Standard memory coordinates -> Staggered cast-side tensor coordinates.
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
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<ct_local>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #identity
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
    access_tile_set = #fp32_domain, access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %intermediate, %tmp_store
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>

  %tmp_load = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %reloaded = ktdp.load %tmp_load
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>
  %output = spyreop.numeric_cast %reloaded {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64xf16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %output, %output_tile
      : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>
  return
}

// -----

// GROUP-LABEL: func.func @narrow_then_dl16_neg(
// GROUP-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// GROUP-NEXT: call @local_schedule_0(%[[SRC]], %[[TMP]])
// GROUP-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// GROUP-LABEL: func.func @local_schedule_0(
// GROUP-SAME: %{{.*}}: index, %[[STORE_ADDR:.*]]: index)
// GROUP: %[[STORE_VIEW:.*]] = ktdp.construct_memory_view %[[STORE_ADDR]], {{.*}}memory_space = #ktdp.memory_space<ct_local>{{.*}}memref<3x2x64xf16>
// GROUP: %[[RESULT:.*]] = spyreop.numeric_cast
// GROUP: %[[STORE_TILE:.*]] = ktdp.construct_access_tile %[[STORE_VIEW]][0, 0, 0]
// GROUP-NEXT: ktdp.store %[[RESULT]], %[[STORE_TILE]]
// GROUP-LABEL: func.func @local_schedule_1(
// GROUP-SAME: %[[LOAD_ADDR:.*]]: index,
// GROUP: %[[LOAD_VIEW:.*]] = ktdp.construct_memory_view %[[LOAD_ADDR]], {{.*}}memory_space = #ktdp.memory_space<ct_local>{{.*}}memref<3x2x64xf16>
// GROUP-NEXT: %[[LOAD_TILE:.*]] = ktdp.construct_access_tile %[[LOAD_VIEW]][0, 0, 0]
// GROUP-NEXT: %[[RELOADED:.*]] = ktdp.load %[[LOAD_TILE]]
// GROUP: linalg.generic {{.*}}ins(%[[RELOADED]] :
// CHECK-LABEL: func.func @narrow_then_dl16_neg(
// CHECK-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// CHECK: call @local_schedule_0(%[[SRC]], %[[TMP]])
// CHECK-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: vectorchain.cast
// CHECK: %[[STORED:.*]] = dataflow.get_logical_memory_view {{.*}}memref<3x2x64xf16>
// CHECK: agen.vector_store {{.*}}, %[[STORED]][{{.*}}memref<3x2x64xf16>, vector<64xf16>
// CHECK-LABEL: func.func @local_schedule_1_body()
// CHECK: %[[RELOAD_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}memref<3x2x64xf16>
// CHECK: agen.vector_load %[[RELOAD_VIEW]][{{.*}}memref<3x2x64xf16>, vector<64xf16>
// CHECK: vectorchain.neg {{.*}}vector<64xf16>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// Standard memory coordinates -> Staggered cast-side tensor coordinates.
#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r, (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4)>

func.func @narrow_then_dl16_neg(
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
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<ct_local>
  } : memref<3x2x64xf16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %intermediate = spyreop.numeric_cast %input {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64xf16>
  %tmp_store = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %intermediate, %tmp_store
      : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>

  %tmp_load = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %reloaded = ktdp.load %tmp_load
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>
  %init = tensor.empty() : tensor<3x2x64xf16>
  %output = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%reloaded : tensor<3x2x64xf16>) outs(%init : tensor<3x2x64xf16>) {
  ^bb0(%value: f16, %unused: f16):
    %negated = arith.negf %value : f16
    linalg.yield %negated : f16
  } -> tensor<3x2x64xf16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %output, %output_tile
      : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>
  return
}
