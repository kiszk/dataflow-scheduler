// RUN: dataflow-scheduler-opt %s --compute-group-extraction --ktir-distributed-memory | FileCheck %s --check-prefix=EXPAND
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR --implicit-check-not=ktdp.construct_distributed_memory_view --implicit-check-not=requires_program_barrier

// Rank-3 partitions split the middle axis. Global [b,1,l] on owner 1 is
// local [b,0,l], with a 64-element batch stride rather than the snapshot's
// 128-element stride. Transpose the first two axes in place. All readers
// must be snapshotted before any core writes: core 1 needs an original value
// that core 0 overwrites. The permutation belongs on linalg.generic.
// EXPAND-LABEL: func.func @distributed_in_place_transpose(
// EXPAND: call @local_schedule_1_gather{{.*}}(
// EXPAND-NEXT: call @local_schedule_1_gather{{.*}}(
// EXPAND-NEXT: call @local_schedule_1_gather{{.*}}(
// EXPAND-NEXT: call @local_schedule_1_gather{{.*}}(
// EXPAND-NEXT: call @local_schedule_1(
// EXPAND-NEXT: call @local_schedule_1_scatter{{.*}}(
// EXPAND-NEXT: call @local_schedule_1_scatter{{.*}}(
// EXPAND-NEXT: call @local_schedule_1_scatter{{.*}}(
// EXPAND-NEXT: call @local_schedule_1_scatter{{.*}}(
// EXPAND-NEXT: call @local_schedule_2_gather{{.*}}(
// EXPAND: ktdp.construct_memory_view {{.*}}sizes: [2, 1, 64], strides: [64, 64, 1]
// EXPAND: ktdp.construct_memory_view {{.*}}sizes: [2, 2, 64], strides: [128, 64, 1]
// The gather/compute/scatter boundaries protect the old values. Autopilot
// supplies their completion ordering without a separate host launch per call.
// DFIR-LABEL: func.func @distributed_in_place_transpose(
// DFIR: call @local_schedule_1_gather_batch(
// DFIR-NEXT: call @local_schedule_1(
// DFIR-NEXT: call @local_schedule_1_scatter(
// DFIR-NEXT: call @local_schedule_1_scatter_6(
// DFIR-NEXT: call @local_schedule_1_scatter_7(
// DFIR-NEXT: call @local_schedule_1_scatter_8(
// DFIR-NEXT: call @local_schedule_2_gather_batch(
// DFIR-NEXT: call @local_schedule_2(
// DFIR-NEXT: return
// DFIR: dataflow.get_unit {core = 1 : i32, {{.*}}type = "mnisu"}
// DFIR: agen.vector_load {{.*}} : memref<2x1x64xf16>, vector<64xf16>
// DFIR: agen.vector_store {{.*}}, %{{.*}} + 1, {{.*}}] {{.*}} : memref<2x2x64xf16>, vector<64xf16>

ktdf_arch.device @sample_device attributes {
  dataflow_scheduler.core_transfer = {
    send_unit = "MNISU", receive_unit = "MNILU", bytes = 128 : i64
  },
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../Dialect/KTDFArch/sample_device.mlir")

#all = affine_set<(b, r, l) : (b >= 0, 1 - b >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#first = affine_set<(b, r, l) : (b >= 0, 1 - b >= 0, r == 0, l >= 0, 63 - l >= 0)>
#second = affine_set<(b, r, l) : (b >= 0, 1 - b >= 0, r - 1 == 0, l >= 0, 63 - l >= 0)>
#id = affine_map<(b, r, l) -> (b, r, l)>
#transpose = affine_map<(b, r, l) -> (r, b, l)>

func.func @distributed_in_place_transpose(%src_addr: index, %dst_addr: index) attributes {grid = [2]} {
  %c0 = arith.constant 0 : index
  %src = ktdp.construct_memory_view %src_addr, sizes: [2, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr, sizes: [2, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<global>
  } : memref<2x2x64xf16>
  %first = ktdp.construct_memory_view %c0, sizes: [2, 1, 64], strides: [64, 64, 1] {
    coordinate_set = #first, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
  } : memref<2x1x64xf16>
  %second = ktdp.construct_memory_view %c0, sizes: [2, 1, 64], strides: [64, 64, 1] {
    coordinate_set = #second, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>
  } : memref<2x1x64xf16>
  %distributed = ktdp.construct_distributed_memory_view (%first, %second : memref<2x1x64xf16>, memref<2x1x64xf16>) : memref<2x2x64xf16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_order = #id, access_tile_set = #all
  } : memref<2x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
  %init_tile = ktdp.construct_access_tile %distributed[0, 0, 0] {
    access_tile_order = #id, access_tile_set = #all
  } : memref<2x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
  %input = ktdp.load %input_tile : !ktdp.access_tile<2x2x64xindex> -> tensor<2x2x64xf16>
  %empty0 = tensor.empty() : tensor<2x2x64xf16>
  %copy0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : tensor<2x2x64xf16>) outs(%empty0 : tensor<2x2x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %copy = spyreop.identity %value : f16
      linalg.yield %copy : f16
  } -> tensor<2x2x64xf16>
  ktdp.store %copy0, %init_tile : tensor<2x2x64xf16>, !ktdp.access_tile<2x2x64xindex>

  %read_tile = ktdp.construct_access_tile %distributed[0, 0, 0] {
    access_tile_order = #id, access_tile_set = #all
  } : memref<2x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
  %write_tile = ktdp.construct_access_tile %distributed[0, 0, 0] {
    access_tile_order = #id, access_tile_set = #all
  } : memref<2x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
  %before = ktdp.load %read_tile : !ktdp.access_tile<2x2x64xindex> -> tensor<2x2x64xf16>
  %empty1 = tensor.empty() : tensor<2x2x64xf16>
  %transposed = linalg.generic {indexing_maps = [#transpose, #id], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%before : tensor<2x2x64xf16>) outs(%empty1 : tensor<2x2x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %copy = spyreop.identity %value : f16
      linalg.yield %copy : f16
  } -> tensor<2x2x64xf16>
  ktdp.store %transposed, %write_tile : tensor<2x2x64xf16>, !ktdp.access_tile<2x2x64xindex>

  %load_tile = ktdp.construct_access_tile %distributed[0, 0, 0] {
    access_tile_order = #id, access_tile_set = #all
  } : memref<2x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_order = #id, access_tile_set = #all
  } : memref<2x2x64xf16> -> !ktdp.access_tile<2x2x64xindex>
  %loaded = ktdp.load %load_tile : !ktdp.access_tile<2x2x64xindex> -> tensor<2x2x64xf16>
  %empty2 = tensor.empty() : tensor<2x2x64xf16>
  %copy2 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%loaded : tensor<2x2x64xf16>) outs(%empty2 : tensor<2x2x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %copy = spyreop.identity %value : f16
      linalg.yield %copy : f16
  } -> tensor<2x2x64xf16>
  ktdp.store %copy2, %output_tile : tensor<2x2x64xf16>, !ktdp.access_tile<2x2x64xindex>
  return
}
