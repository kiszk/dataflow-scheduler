// RUN: dataflow-scheduler-opt %s --compute-group-extraction --ktir-distributed-memory | FileCheck %s --check-prefix=EXPAND --implicit-check-not=_gather --implicit-check-not=local_schedule_0_scatter
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR --implicit-check-not=ktdp.construct_distributed_memory_view --implicit-check-not=requires_program_barrier

// Initialize four FP32 rows on two owners, replace global rows 1 and 2, then
// read the whole view. The partial store straddles the partition boundary;
// rows 0 and 3 must retain their previous values. Each core updates one row
// on the other owner. Initialize only the two retained rows. Readback uses
// owner-local storage directly; both remote updates share one program.
// EXPAND-LABEL: func.func @distributed_partial_store(
// EXPAND: call @local_schedule_0_initialize(
// EXPAND-NEXT: call @local_schedule_0_initialize_0(
// EXPAND-NEXT: call @local_schedule_1(
// EXPAND-NEXT: call @local_schedule_1_scatter(
// EXPAND-NEXT: call @local_schedule_1_scatter_1(
// EXPAND-NEXT: call @local_schedule_2(
// EXPAND-NEXT: return
// EXPAND-LABEL: func.func @local_schedule_1(
// EXPAND-SAME: attributes {grid = [2]}
// EXPAND: ktdp.construct_memory_view {{.*}}sizes: [4, 32], strides: [32, 1]
// EXPAND: ktdp.store {{.*}}tensor<1x32xf32>
// EXPAND-LABEL: func.func @local_schedule_2(
// EXPAND: ktdp.construct_memory_view {{.*}}sizes: [2, 32], strides: [32, 1]
// EXPAND-SAME: memory_space = #ktdp.memory_space<ct_local>
// EXPAND-LABEL: func.func @local_schedule_0_initialize(
// EXPAND-SAME: dataflow_scheduler.physical_core = 0 : i32
// EXPAND: %[[ZERO:.*]] = arith.constant 0 : index
// EXPAND-NEXT: %[[ZERO_LANE:.*]] = arith.constant 0 : index
// EXPAND: ktdp.construct_access_tile {{.*}}[%[[ZERO]], %[[ZERO_LANE]]]
// EXPAND: ktdp.load {{.*}} -> tensor<1x32xf32>
// EXPAND: ktdp.store {{.*}}tensor<1x32xf32>
// EXPAND-LABEL: func.func @local_schedule_0_initialize_0(
// EXPAND-SAME: dataflow_scheduler.physical_core = 1 : i32
// EXPAND: %[[THREE:.*]] = arith.constant 3 : index
// EXPAND: ktdp.construct_access_tile {{.*}}[%[[THREE]], {{.*}}!ktdp.access_tile<1x32xindex>
// EXPAND: %[[ONE:.*]] = arith.constant 1 : index
// EXPAND: ktdp.construct_access_tile {{.*}}[%[[ONE]], {{.*}}!ktdp.access_tile<1x32xindex>
// EXPAND: ktdp.store {{.*}}tensor<1x32xf32>
// DFIR-LABEL: func.func @distributed_partial_store(
// DFIR: call @local_schedule_0_initialize_batch(
// DFIR-NEXT: call @local_schedule_1(
// DFIR-NEXT: call @local_schedule_1_scatter_batch(
// DFIR-NEXT: call @local_schedule_2(
// DFIR-NEXT: return
// Core 0 writes global row 2, which is owner 1's local row 0.
// DFIR-LABEL: func.func @local_schedule_1_scatter_batch_body()
// DFIR-DAG: %[[SEND0:.*]] = dataflow.get_unit {core = 0 : i32, {{.*}}type = "mnisu"}
// DFIR-DAG: %[[RECV1:.*]] = dataflow.get_unit {core = 1 : i32, {{.*}}type = "mnilu"}
// DFIR-DAG: %[[SEND1:.*]] = dataflow.get_unit {core = 1 : i32, {{.*}}type = "mnisu"}
// DFIR-DAG: %[[RECV0:.*]] = dataflow.get_unit {core = 0 : i32, {{.*}}type = "mnilu"}
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[SEND0]])
// DFIR: agen.vector_load {{.*}} + 2, {{.*}}vector<32xf32>
// DFIR: dataflow.send %[[RECV1]], {{.*}} : vector<32xf32>
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[RECV1]])
// DFIR: dataflow.receive %[[SEND0]] : vector<32xf32>
// DFIR: agen.vector_store {{.*}}vector<32xf32>
// Core 1 writes global row 1, which is owner 0's local row 1.
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[SEND1]])
// DFIR: agen.vector_load {{.*}} + 1, {{.*}}vector<32xf32>
// DFIR: dataflow.send %[[RECV0]], {{.*}} : vector<32xf32>
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[RECV0]])
// DFIR: dataflow.receive %[[SEND1]] : vector<32xf32>
// DFIR: agen.vector_store {{.*}} + 1, {{.*}}vector<32xf32>

ktdf_arch.device @sample_device attributes {
  dataflow_scheduler.core_transfer = {
    send_unit = "MNISU", receive_unit = "MNILU", bytes = 128 : i64
  },
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../Dialect/KTDFArch/sample_device.mlir")

#all = affine_set<(r, l) : (r >= 0, 3 - r >= 0, l >= 0, 31 - l >= 0)>
#first = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 31 - l >= 0)>
#second = affine_set<(r, l) : (r - 2 >= 0, 3 - r >= 0, l >= 0, 31 - l >= 0)>
#one = affine_set<(r, l) : (r == 0, l >= 0, 31 - l >= 0)>
#id = affine_map<(r, l) -> (r, l)>

func.func @distributed_partial_store(%src_addr: index, %update_addr: index, %dst_addr: index) attributes {grid = [2]} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %core = ktdp.get_compute_tile_id : index
  %row = arith.muli %core, %c2 : index
  %update_row = arith.subi %c2, %core : index
  %src = ktdp.construct_memory_view %src_addr, sizes: [4, 32], strides: [32, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<global>
  } : memref<4x32xf32>
  %update = ktdp.construct_memory_view %update_addr, sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #first, memory_space = #ktdp.memory_space<global>
  } : memref<2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr, sizes: [4, 32], strides: [32, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<global>
  } : memref<4x32xf32>
  %first = ktdp.construct_memory_view %c0, sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #first, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
  } : memref<2x32xf32>
  %second = ktdp.construct_memory_view %c0, sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #second, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>
  } : memref<2x32xf32>
  %distributed = ktdp.construct_distributed_memory_view (%first, %second : memref<2x32xf32>, memref<2x32xf32>) : memref<4x32xf32>

  %input_tile = ktdp.construct_access_tile %src[%row, 0] {
    access_tile_order = #id, access_tile_set = #first
  } : memref<4x32xf32> -> !ktdp.access_tile<2x32xindex>
  %init_tile = ktdp.construct_access_tile %distributed[%row, 0] {
    access_tile_order = #id, access_tile_set = #first
  } : memref<4x32xf32> -> !ktdp.access_tile<2x32xindex>
  %input = ktdp.load %input_tile : !ktdp.access_tile<2x32xindex> -> tensor<2x32xf32>
  %empty0 = tensor.empty() : tensor<2x32xf32>
  %copy0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<2x32xf32>) outs(%empty0 : tensor<2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x32xf32>
  ktdp.store %copy0, %init_tile : tensor<2x32xf32>, !ktdp.access_tile<2x32xindex>

  %update_tile = ktdp.construct_access_tile %update[%core, 0] {
    access_tile_order = #id, access_tile_set = #one
  } : memref<2x32xf32> -> !ktdp.access_tile<1x32xindex>
  %partial_tile = ktdp.construct_access_tile %distributed[%update_row, 0] {
    access_tile_order = #id, access_tile_set = #one
  } : memref<4x32xf32> -> !ktdp.access_tile<1x32xindex>
  %replacement = ktdp.load %update_tile : !ktdp.access_tile<1x32xindex> -> tensor<1x32xf32>
  %empty1 = tensor.empty() : tensor<1x32xf32>
  %copy1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%replacement : tensor<1x32xf32>) outs(%empty1 : tensor<1x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<1x32xf32>
  ktdp.store %copy1, %partial_tile : tensor<1x32xf32>, !ktdp.access_tile<1x32xindex>

  %load_tile = ktdp.construct_access_tile %distributed[%row, 0] {
    access_tile_order = #id, access_tile_set = #first
  } : memref<4x32xf32> -> !ktdp.access_tile<2x32xindex>
  %output_tile = ktdp.construct_access_tile %dst[%row, 0] {
    access_tile_order = #id, access_tile_set = #first
  } : memref<4x32xf32> -> !ktdp.access_tile<2x32xindex>
  %loaded = ktdp.load %load_tile : !ktdp.access_tile<2x32xindex> -> tensor<2x32xf32>
  %empty2 = tensor.empty() : tensor<2x32xf32>
  %copy2 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%loaded : tensor<2x32xf32>) outs(%empty2 : tensor<2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
  } -> tensor<2x32xf32>
  ktdp.store %copy2, %output_tile : tensor<2x32xf32>, !ktdp.access_tile<2x32xindex>
  return
}
