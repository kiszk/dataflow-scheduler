// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --compute-group-extraction | FileCheck %s --check-prefix=KTIR
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=access_tile_order --implicit-check-not=ktdf.pipeline --implicit-check-not=linalg.generic

// PyTorch: x.to(torch.float32).sum(dim=-1, keepdim=True), x: float16[2,192].
// The logical output is [2,1]. Physical shapes are DL16 [3,2,64],
// FP32 [6,2,32], and output [2,32], with each row's sum in stick lane 0.
// Materialize Standard FP32 in a distinct HBM allocation before the reduction;
// ComputeGroupExtraction must preserve the producer/consumer call order.
// This tests two compute groups, not a direct tensor-SSA cast/reduction fusion.
// Reduce both the six FP32 packets and their 32 lanes, keeping the two rows.
// The output lane axis is retained for the physical whole-stick store contract.

// KTIR-LABEL: func.func @dl16_to_fp32_row_sum(
// KTIR: call @local_schedule_0
// KTIR-NEXT: call @local_schedule_1
// KTIR-LABEL: module @local_schedule_0
// KTIR: linalg.generic
// KTIR-SAME: ins({{.*}} : tensor<3x2x64xf16>) outs({{.*}} : tensor<6x2x32xf32>)
// KTIR: spyreop.numeric_cast {{.*}}kind = "dl16tofp32"} : f16 -> f32
// KTIR-LABEL: module @local_schedule_1
// KTIR: linalg.generic
// KTIR-SAME: iterator_types = ["reduction", "parallel", "reduction", "parallel"]
// KTIR-SAME: ins({{.*}} : tensor<6x2x32xf32>) outs({{.*}} : tensor<2x32xf32>)
// KTIR: arith.addf {{.*}} : f32
// KTIR: ktdp.store {{.*}} : tensor<2x32xf32>, <2x32xindex>

// CHECK-LABEL: func.func @dl16_to_fp32_row_sum(
// CHECK-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// CHECK: call @local_schedule_0(%[[SRC]], %[[TMP]])
// CHECK-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: agen.vector_load {{.*}}vector<4xf32>
// CHECK: vectorchain.cast {{.*}}vector<64xf16>, vector<64xf32>
// CHECK: agen.vector_store {{.*}}vector<4xf32>
// CHECK: memref<6x2x32xf32>
// CHECK-LABEL: func.func @local_schedule_1_body()
// CHECK-DAG: %[[SIX:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: vectorchain.constant_bitstream {value = [0x0]} : vector<1xf32>
// CHECK: scf.for {{.*}} = %[[ZERO]] to %[[TWO]] step %[[ONE]]
// CHECK: scf.for {{.*}} = %[[ZERO]] to %[[SIX]] step %[[ONE]]
// CHECK: %[[INPUT:.*]] = dataflow.receive {{.*}} : vector<32xf32>
// CHECK: %[[ACC:.*]] = agen.vector_load {{.*}}vector<32xf32>
// CHECK: %[[SUM:.*]] = vectorchain.binary %[[INPUT]], %[[ACC]] {binary_op = #vectorchain<binary_operator add>{{.*}} : vector<32xf32>, vector<32xf32>, vector<32xf32>
// CHECK-NEXT: agen.vector_store %[[SUM]], {{.*}}vector<32xf32>
// CHECK: dataflow.opaque {{.*}}func_name = "simdreduction_sum_fp32"{{.*}}parameter_dictionary = {unroll = "1"}
// CHECK: memref<6x2x32xf32>
// CHECK: memref<2x32xf32>

ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#output_domain = affine_set<(r, o) :
    (r >= 0, 1 - r >= 0, o >= 0, 31 - o >= 0)>
#identity = affine_map<(k, r, e) -> (k, r, e)>
// Standard tensor coordinates -> Staggered cast-result coordinates.
#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r, (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4)>

func.func @dl16_to_fp32_row_sum(
    %src_addr: index, %dst_addr: index, %tmp_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #output_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x32xf32>
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
  %cast_init_1 = tensor.empty() : tensor<6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%cast_init_1 : tensor<6x2x32xf32>) {
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
      ins(%staggered : tensor<6x2x32xf32>) outs(%relayout_init_1 : tensor<6x2x32xf32>) {
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
  %standard = ktdp.load %tmp_load
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>
  // MapReductionPartials materializes the additive identity for this destination.
  %init = tensor.empty() : tensor<2x32xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(k, r, e, o) -> (k, r, e)>,
                     affine_map<(k, r, e, o) -> (r, o)>],
    iterator_types = ["reduction", "parallel", "reduction", "parallel"]
  } ins(%standard : tensor<6x2x32xf32>) outs(%init : tensor<2x32xf32>) {
  ^bb0(%value: f32, %acc: f32):
    %next = arith.addf %value, %acc : f32
    linalg.yield %next : f32
  } -> tensor<2x32xf32>
  %output_tile = ktdp.construct_access_tile %dst[0, 0] {
    access_tile_set = #output_domain,
    access_tile_order = affine_map<(r, o) -> (r, o)>
  } : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  ktdp.store %sum, %output_tile
      : tensor<2x32xf32>, !ktdp.access_tile<2x32xindex>
  return
}
