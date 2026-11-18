// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --compute-group-extraction | FileCheck %s --check-prefix=KTIR
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=spyreop.numeric_cast --implicit-check-not=access_tile_order --implicit-check-not=ktdf.pipeline --implicit-check-not=linalg.generic

// PyTorch: x.to(torch.float32).sum(dim=-1, keepdim=True), float16[2,192] -> f32[2,1].
// The Standard FP32 intermediate lives in caller-owned ct_local memory;
// tmp_offset counts f32 elements and must be disjoint from compiler scratch.
// Both compute groups use the same offset on the same core. Physical shapes
// are [3,2,64] -> [6,2,32] -> [2,32]; only output lane 0 is logically valid.
// The reorder writes local memory directly, and reduction reads six packets
// per row directly from it. HBM is used only for the input and final output.

// KTIR-LABEL: func.func @dl16_to_fp32_row_sum_ct_local(
// KTIR: call @local_schedule_0
// KTIR-NEXT: call @local_schedule_1
// KTIR-LABEL: module @local_schedule_0
// KTIR: ktdp.construct_memory_view {{.*}}sizes: [6, 2, 32]
// KTIR-SAME: memory_space = #ktdp.memory_space<ct_local>
// KTIR: spyreop.numeric_cast {{.*}}kind = "dl16tofp32"
// KTIR-LABEL: module @local_schedule_1
// KTIR: ktdp.construct_memory_view {{.*}}sizes: [6, 2, 32]
// KTIR-SAME: memory_space = #ktdp.memory_space<ct_local>
// KTIR: linalg.generic
// KTIR-SAME: iterator_types = ["reduction", "parallel", "reduction", "parallel"]
// KTIR: ktdp.store {{.*}} : tensor<2x32xf32>, <2x32xindex>

// CHECK-DAG: #[[STANDARD:.*]] = affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 32 + d2)>
// CHECK-LABEL: func.func @dl16_to_fp32_row_sum_ct_local(
// CHECK-SAME: %[[SRC:.*]]: index, %[[DST:.*]]: index, %[[TMP:.*]]: index)
// CHECK: call @local_schedule_0(%[[SRC]], %[[TMP]])
// CHECK-NEXT: call @local_schedule_1(%[[TMP]], %[[DST]])
// CHECK-LABEL: func.func @local_schedule_0_body()
// CHECK: %[[STORE:.*]] = dataflow.get_unit {{.*}}type = "l1su"
// CHECK: %[[L1:.*]] = dataflow.get_unit {{.*}}type = "l1"
// CHECK: vectorchain.cast {{.*}}vector<64xf16>, vector<64xf32>
// CHECK: dataflow.program_unit iter_arg : %[[STORE_KEY:.*]] -> (%[[STORE]])
// CHECK: %[[STORE_MAP:.*]] = uniform.def_immutable_mapping([%[[STORE]] -> %[[L1]]])
// CHECK: %[[STORE_MEM:.*]] = uniform.query_map(map:%[[STORE_MAP]], key:%[[STORE_KEY]])
// CHECK: %[[STORE_VIEW:.*]] = dataflow.get_logical_memory_view %[[STORE_MEM]], {{.*}}layout_map = #[[STANDARD]]{{.*}}memref<6x2x32xf32>
// CHECK: agen.vector_store {{.*}}, %[[STORE_VIEW]][{{.*}}] {{.*}}memref<6x2x32xf32>, vector<4xf32>
// CHECK-LABEL: func.func @local_schedule_1_body()
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[SIX:.*]] = arith.constant 6 : index
// CHECK: %[[LOAD:.*]] = dataflow.get_unit {{.*}}type = "l1lu"
// CHECK: %[[L1:.*]] = dataflow.get_unit {{.*}}type = "l1"
// CHECK: scf.for {{.*}} to %[[TWO]]
// CHECK: scf.for {{.*}} to %[[SIX]]
// CHECK: dataflow.receive {{.*}}vector<32xf32>
// CHECK: vectorchain.binary {{.*}}binary_op = #vectorchain<binary_operator add>{{.*}}vector<32xf32>
// CHECK: dataflow.opaque {{.*}}func_name = "simdreduction_sum_fp32"{{.*}}parameter_dictionary = {unroll = "1"}
// CHECK: dataflow.program_unit iter_arg : %[[LOAD_KEY:.*]] -> (%[[LOAD]])
// CHECK: %[[LOAD_MAP:.*]] = uniform.def_immutable_mapping([%[[LOAD]] -> %[[L1]]])
// CHECK: %[[LOAD_MEM:.*]] = uniform.query_map(map:%[[LOAD_MAP]], key:%[[LOAD_KEY]])
// CHECK: %[[LOAD_VIEW:.*]] = dataflow.get_logical_memory_view %[[LOAD_MEM]], {{.*}}layout_map = #[[STANDARD]]{{.*}}memref<6x2x32xf32>
// CHECK: scf.for %[[ROW:.*]] = {{.*}} to %[[TWO]]
// CHECK: scf.for %[[PACKET:.*]] = {{.*}} to %[[SIX]]
// CHECK: agen.vector_load %[[LOAD_VIEW]][%[[PACKET]], %[[ROW]], 0] {{.*}}memref<6x2x32xf32>, vector<32xf32>
// CHECK: memref<2x32xf32>

ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#output_domain = affine_set<(r, o) :
    (r >= 0, 1 - r >= 0, o >= 0, 31 - o >= 0)>
#identity = affine_map<(k, r, e) -> (k, r, e)>
// Standard memory coordinates -> Staggered cast-result coordinates.
#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r, (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4)>

func.func @dl16_to_fp32_row_sum_ct_local(
    %src_addr: index, %dst_addr: index, %tmp_offset: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #output_domain, memory_space = #ktdp.memory_space<global>
  } : memref<2x32xf32>
  %tmp = ktdp.construct_memory_view %tmp_offset,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<ct_local>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>
  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>
  %tmp_store = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %tmp_store
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>

  %tmp_load = ktdp.construct_access_tile %tmp[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #identity
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
