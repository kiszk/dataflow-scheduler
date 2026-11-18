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

// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-input-file --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=access_tile_order --implicit-check-not=ktdf.pipeline
// RUN: dataflow-scheduler-opt %s --split-input-file --ktir-numeric-cast-tiling --ktir-map-and-tile --ktir-bufferize --ktir-pipeline | FileCheck %s --check-prefix=PHASE
// The first six cases use Standard external memory around unary/cast chains.
// The FP32 access order maps it to the Staggered layout at the cast boundary.
// The six explicit *_identity_order cases instead keep FP32 memory Staggered,
// so load/store need no access-order conversion. Both sets test each unary
// placement separately, then chains on both sides of each cast.
// PHASE checks that cast inputs are FIFO reads and outputs go directly to FIFO
// writes, with transfers separating the unary compute from the cast.

// CHECK-LABEL: func.func @dl16_exp_then_widen(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

// PHASE-LABEL: func.func @dl16_exp_then_widen(
// PHASE: linalg.generic
// PHASE: ktdf.data_transfer
// PHASE: %[[INPUT0:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<64x!spyreop.df16>
// PHASE-NEXT: %[[CAST:.*]]:2 = ktdf_lowering.numeric_cast_value ins(%[[INPUT0]] :
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]]#0,
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]]#1,
// PHASE: return
func.func @dl16_exp_then_widen(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %before0_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %before0 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64x!spyreop.df16>) outs(%before0_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%before0_value: !spyreop.df16, %before0_unused: !spyreop.df16):
    %before0_exp = spyreop.exp %before0_value : !spyreop.df16
    linalg.yield %before0_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>
  %staggered = spyreop.numeric_cast %before0 {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @widen_then_fp32_neg(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.neg {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

// PHASE-LABEL: func.func @widen_then_fp32_neg(
// PHASE: %[[INPUT0:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<64x!spyreop.df16>
// PHASE-NEXT: %[[CAST:.*]]:2 = ktdf_lowering.numeric_cast_value ins(%[[INPUT0]] :
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]]#0,
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]]#1,
// PHASE: ktdf.data_transfer
// PHASE: linalg.generic
// PHASE: return
func.func @widen_then_fp32_neg(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %after0 = arith.negf %staggered : tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %after0, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @fp32_neg_then_narrow(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.neg {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

// PHASE-LABEL: func.func @fp32_neg_then_narrow(
// PHASE: linalg.generic
// PHASE: ktdf.data_transfer
// PHASE: %[[INPUT0:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<32xf32>
// PHASE-NEXT: %[[INPUT1:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<32xf32>
// PHASE-NEXT: %[[CAST:.*]] = ktdf_lowering.numeric_cast_value ins(%[[INPUT0]], %[[INPUT1]] :
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]],
// PHASE: return
func.func @fp32_neg_then_narrow(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %staggered = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %before0 = arith.negf %staggered : tensor<6x2x32xf32>
  %converted = spyreop.numeric_cast %before0 {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64x!spyreop.df16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %converted, %output_tile
      : tensor<3x2x64x!spyreop.df16>, !ktdp.access_tile<3x2x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_then_dl16_exp(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

// PHASE-LABEL: func.func @narrow_then_dl16_exp(
// PHASE: %[[INPUT0:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<32xf32>
// PHASE-NEXT: %[[INPUT1:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<32xf32>
// PHASE-NEXT: %[[CAST:.*]] = ktdf_lowering.numeric_cast_value ins(%[[INPUT0]], %[[INPUT1]] :
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]],
// PHASE: ktdf.data_transfer
// PHASE: linalg.generic
// PHASE: return
func.func @narrow_then_dl16_exp(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %staggered = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %converted = spyreop.numeric_cast %staggered {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64x!spyreop.df16>

  %after0_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %after0 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%converted : tensor<3x2x64x!spyreop.df16>) outs(%after0_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%after0_value: !spyreop.df16, %after0_unused: !spyreop.df16):
    %after0_exp = spyreop.exp %after0_value : !spyreop.df16
    linalg.yield %after0_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %after0, %output_tile
      : tensor<3x2x64x!spyreop.df16>, !ktdp.access_tile<3x2x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @dl16_chain_widen_fp32_chain(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: vectorchain.floor {{.*}}vector<32xf32>
// CHECK-DAG: vectorchain.neg {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

// PHASE-LABEL: func.func @dl16_chain_widen_fp32_chain(
// PHASE: linalg.generic
// PHASE: ktdf.data_transfer
// PHASE: %[[INPUT0:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<64x!spyreop.df16>
// PHASE-NEXT: %[[CAST:.*]]:2 = ktdf_lowering.numeric_cast_value ins(%[[INPUT0]] :
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]]#0,
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]]#1,
// PHASE: ktdf.data_transfer
// PHASE: linalg.generic
// PHASE: return
func.func @dl16_chain_widen_fp32_chain(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %before0_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %before0 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64x!spyreop.df16>) outs(%before0_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%before0_value: !spyreop.df16, %before0_unused: !spyreop.df16):
    %before0_exp = spyreop.exp %before0_value : !spyreop.df16
    linalg.yield %before0_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>
  %before1_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %before1 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%before0 : tensor<3x2x64x!spyreop.df16>) outs(%before1_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%before1_value: !spyreop.df16, %before1_unused: !spyreop.df16):
    %before1_exp = spyreop.exp %before1_value : !spyreop.df16
    linalg.yield %before1_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>
  %staggered = spyreop.numeric_cast %before1 {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %after0 = math.floor %staggered : tensor<6x2x32xf32>
  %after1 = arith.negf %after0 : tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %after1, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @fp32_chain_narrow_dl16_chain(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.neg {{.*}}vector<32xf32>
// CHECK-DAG: vectorchain.floor {{.*}}vector<32xf32>
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: agen.vector_load {{.*}}vector<4xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<4xf32>
// CHECK: dataflow.sync_send
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

// PHASE-LABEL: func.func @fp32_chain_narrow_dl16_chain(
// PHASE: linalg.generic
// PHASE: ktdf.data_transfer
// PHASE: %[[INPUT0:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<32xf32>
// PHASE-NEXT: %[[INPUT1:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<32xf32>
// PHASE-NEXT: %[[CAST:.*]] = ktdf_lowering.numeric_cast_value ins(%[[INPUT0]], %[[INPUT1]] :
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]],
// PHASE: ktdf.data_transfer
// PHASE: linalg.generic
// PHASE: return
func.func @fp32_chain_narrow_dl16_chain(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %staggered = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %before0 = arith.negf %staggered : tensor<6x2x32xf32>
  %before1 = math.floor %before0 : tensor<6x2x32xf32>
  %converted = spyreop.numeric_cast %before1 {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64x!spyreop.df16>

  %after0_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %after0 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%converted : tensor<3x2x64x!spyreop.df16>) outs(%after0_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%after0_value: !spyreop.df16, %after0_unused: !spyreop.df16):
    %after0_exp = spyreop.exp %after0_value : !spyreop.df16
    linalg.yield %after0_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>
  %after1_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %after1 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%after0 : tensor<3x2x64x!spyreop.df16>) outs(%after1_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%after1_value: !spyreop.df16, %after1_unused: !spyreop.df16):
    %after1_exp = spyreop.exp %after1_value : !spyreop.df16
    linalg.yield %after1_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %after1, %output_tile
      : tensor<3x2x64x!spyreop.df16>, !ktdp.access_tile<3x2x64xindex>
  return
}

// -----

// Identity access orders: FP32 external memory is already Staggered.

// CHECK-LABEL: func.func @dl16_exp_then_widen_identity_order(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: agen.vector_load {{.*}}vector<64xf16>
// CHECK-DAG: agen.vector_store {{.*}}vector<32xf32>
// CHECK: dataflow.sync_send
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// PHASE-LABEL: func.func @dl16_exp_then_widen_identity_order(
// PHASE: linalg.generic
// PHASE: ktdf.data_transfer
// PHASE: %[[INPUT0:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<64x!spyreop.df16>
// PHASE-NEXT: %[[CAST:.*]]:2 = ktdf_lowering.numeric_cast_value ins(%[[INPUT0]] :
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]]#0,
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]]#1,
// PHASE: return
func.func @dl16_exp_then_widen_identity_order(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %before0_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %before0 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64x!spyreop.df16>) outs(%before0_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%before0_value: !spyreop.df16, %before0_unused: !spyreop.df16):
    %before0_exp = spyreop.exp %before0_value : !spyreop.df16
    linalg.yield %before0_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>
  %staggered = spyreop.numeric_cast %before0 {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @widen_then_fp32_neg_identity_order(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.neg {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_load {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<32xf32>
// CHECK: dataflow.sync_send
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// PHASE-LABEL: func.func @widen_then_fp32_neg_identity_order(
// PHASE: %[[INPUT0:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<64x!spyreop.df16>
// PHASE-NEXT: %[[CAST:.*]]:2 = ktdf_lowering.numeric_cast_value ins(%[[INPUT0]] :
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]]#0,
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]]#1,
// PHASE: ktdf.data_transfer
// PHASE: linalg.generic
// PHASE: return
func.func @widen_then_fp32_neg_identity_order(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %after0 = arith.negf %staggered : tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %after0, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @fp32_neg_then_narrow_identity_order(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.neg {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_load {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<32xf32>
// CHECK: dataflow.sync_send
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// PHASE-LABEL: func.func @fp32_neg_then_narrow_identity_order(
// PHASE: linalg.generic
// PHASE: ktdf.data_transfer
// PHASE: %[[INPUT0:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<32xf32>
// PHASE-NEXT: %[[INPUT1:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<32xf32>
// PHASE-NEXT: %[[CAST:.*]] = ktdf_lowering.numeric_cast_value ins(%[[INPUT0]], %[[INPUT1]] :
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]],
// PHASE: return
func.func @fp32_neg_then_narrow_identity_order(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %staggered = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %before0 = arith.negf %staggered : tensor<6x2x32xf32>
  %converted = spyreop.numeric_cast %before0 {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64x!spyreop.df16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %converted, %output_tile
      : tensor<3x2x64x!spyreop.df16>, !ktdp.access_tile<3x2x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @narrow_then_dl16_exp_identity_order(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: agen.vector_load {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<64xf16>
// CHECK: dataflow.sync_send
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// PHASE-LABEL: func.func @narrow_then_dl16_exp_identity_order(
// PHASE: %[[INPUT0:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<32xf32>
// PHASE-NEXT: %[[INPUT1:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<32xf32>
// PHASE-NEXT: %[[CAST:.*]] = ktdf_lowering.numeric_cast_value ins(%[[INPUT0]], %[[INPUT1]] :
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]],
// PHASE: ktdf.data_transfer
// PHASE: linalg.generic
// PHASE: return
func.func @narrow_then_dl16_exp_identity_order(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %staggered = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %converted = spyreop.numeric_cast %staggered {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64x!spyreop.df16>

  %after0_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %after0 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%converted : tensor<3x2x64x!spyreop.df16>) outs(%after0_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%after0_value: !spyreop.df16, %after0_unused: !spyreop.df16):
    %after0_exp = spyreop.exp %after0_value : !spyreop.df16
    linalg.yield %after0_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %after0, %output_tile
      : tensor<3x2x64x!spyreop.df16>, !ktdp.access_tile<3x2x64xindex>
  return
}

// -----

// CHECK-LABEL: func.func @dl16_chain_widen_fp32_chain_identity_order(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: vectorchain.floor {{.*}}vector<32xf32>
// CHECK-DAG: vectorchain.neg {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_load {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<32xf32>
// CHECK: dataflow.sync_send
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// PHASE-LABEL: func.func @dl16_chain_widen_fp32_chain_identity_order(
// PHASE: linalg.generic
// PHASE: ktdf.data_transfer
// PHASE: %[[INPUT0:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<64x!spyreop.df16>
// PHASE-NEXT: %[[CAST:.*]]:2 = ktdf_lowering.numeric_cast_value ins(%[[INPUT0]] :
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]]#0,
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]]#1,
// PHASE: ktdf.data_transfer
// PHASE: linalg.generic
// PHASE: return
func.func @dl16_chain_widen_fp32_chain_identity_order(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %before0_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %before0 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64x!spyreop.df16>) outs(%before0_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%before0_value: !spyreop.df16, %before0_unused: !spyreop.df16):
    %before0_exp = spyreop.exp %before0_value : !spyreop.df16
    linalg.yield %before0_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>
  %before1_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %before1 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%before0 : tensor<3x2x64x!spyreop.df16>) outs(%before1_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%before1_value: !spyreop.df16, %before1_unused: !spyreop.df16):
    %before1_exp = spyreop.exp %before1_value : !spyreop.df16
    linalg.yield %before1_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>
  %staggered = spyreop.numeric_cast %before1 {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %after0 = math.floor %staggered : tensor<6x2x32xf32>
  %after1 = arith.negf %after0 : tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %after1, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

// CHECK-LABEL: func.func @fp32_chain_narrow_dl16_chain_identity_order(
// CHECK-DAG: memref<6x2x32xf32>
// CHECK-DAG: vectorchain.cast
// CHECK-DAG: vectorchain.neg {{.*}}vector<32xf32>
// CHECK-DAG: vectorchain.floor {{.*}}vector<32xf32>
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_df16_exp"
// CHECK-DAG: agen.vector_load {{.*}}vector<32xf32>
// CHECK-DAG: agen.vector_store {{.*}}vector<32xf32>
// CHECK: dataflow.sync_send
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// PHASE-LABEL: func.func @fp32_chain_narrow_dl16_chain_identity_order(
// PHASE: linalg.generic
// PHASE: ktdf.data_transfer
// PHASE: %[[INPUT0:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<32xf32>
// PHASE-NEXT: %[[INPUT1:.*]] = ktdf.read_from_fifo {{.*}} -> tensor<32xf32>
// PHASE-NEXT: %[[CAST:.*]] = ktdf_lowering.numeric_cast_value ins(%[[INPUT0]], %[[INPUT1]] :
// PHASE-NEXT: ktdf.write_to_fifo %[[CAST]],
// PHASE: ktdf.data_transfer
// PHASE: linalg.generic
// PHASE: return
func.func @fp32_chain_narrow_dl16_chain_identity_order(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %staggered = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %before0 = arith.negf %staggered : tensor<6x2x32xf32>
  %before1 = math.floor %before0 : tensor<6x2x32xf32>
  %converted = spyreop.numeric_cast %before1 {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64x!spyreop.df16>

  %after0_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %after0 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%converted : tensor<3x2x64x!spyreop.df16>) outs(%after0_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%after0_value: !spyreop.df16, %after0_unused: !spyreop.df16):
    %after0_exp = spyreop.exp %after0_value : !spyreop.df16
    linalg.yield %after0_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>
  %after1_init = tensor.empty() : tensor<3x2x64x!spyreop.df16>
  %after1 = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%after0 : tensor<3x2x64x!spyreop.df16>) outs(%after1_init : tensor<3x2x64x!spyreop.df16>) {
  ^bb0(%after1_value: !spyreop.df16, %after1_unused: !spyreop.df16):
    %after1_exp = spyreop.exp %after1_value : !spyreop.df16
    linalg.yield %after1_exp : !spyreop.df16
  } -> tensor<3x2x64x!spyreop.df16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %after1, %output_tile
      : tensor<3x2x64x!spyreop.df16>, !ktdp.access_tile<3x2x64xindex>
  return
}
