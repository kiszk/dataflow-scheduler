// RUN: dataflow-scheduler-opt %s --compute-group-extraction --ktir-distributed-memory | FileCheck %s --check-prefix=EXPAND --implicit-check-not=requires_program_barrier
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR --implicit-check-not=ktdp.construct_distributed_memory_view --implicit-check-not=ktdf_lowering.execute_on --implicit-check-not=requires_program_barrier

// A store is scattered to both owners before the next group gathers it.
// The full load crosses the partition boundary. The two owners deliberately
// use the same LX address, so treating ct_id as the executing core is wrong.
// EXPAND-LABEL: func.func @distributed_roundtrip(
// EXPAND: call @local_schedule_0(
// EXPAND-NEXT: call @local_schedule_0_scatter(
// EXPAND-NEXT: call @local_schedule_0_scatter_0(
// EXPAND-NEXT: call @local_schedule_0_scatter_1(
// EXPAND-NEXT: call @local_schedule_0_scatter_2(
// EXPAND-NEXT: call @local_schedule_1_gather(
// EXPAND-NEXT: call @local_schedule_1_gather_3(
// EXPAND-NEXT: call @local_schedule_1_gather_4(
// EXPAND-NEXT: call @local_schedule_1_gather_5(
// EXPAND-NEXT: call @local_schedule_1(
// EXPAND-NEXT: return
// EXPAND: func.func @local_schedule_0({{.*}} attributes {grid = [2]}
// EXPAND: func.func @local_schedule_1({{.*}} attributes {grid = [2]}
// EXPAND: attributes {dataflow_scheduler.physical_core = 1 : i32, grid = [1]}
// Overlapping writers keep their order; independent gathers share a program.
// DFIR-LABEL: func.func @distributed_roundtrip(
// DFIR: call @local_schedule_0(
// DFIR-NEXT: call @local_schedule_0_scatter(
// DFIR-NEXT: call @local_schedule_0_scatter_0(
// DFIR-NEXT: call @local_schedule_0_scatter_1(
// DFIR-NEXT: call @local_schedule_0_scatter_2(
// DFIR-NEXT: call @local_schedule_1_gather_batch(
// DFIR-NEXT: call @local_schedule_1(
// DFIR-NEXT: return
// Both compute cores are present in one program.
// DFIR-LABEL: func.func @local_schedule_0_body()
// DFIR-DAG: dataflow.get_unit {core = 0 : i32, {{.*}}type = "l1su"}
// DFIR-DAG: dataflow.get_unit {core = 1 : i32, {{.*}}type = "l1su"}
// DFIR: agen.vector_store
// Core 0's written snapshot rows go to owner 1's local row zero.
// DFIR-LABEL: func.func @local_schedule_0_scatter_0_body()
// DFIR-DAG: %[[S:.*]] = dataflow.get_unit {core = 0 : i32, {{.*}}type = "mnisu"}
// DFIR-DAG: %[[R:.*]] = dataflow.get_unit {core = 1 : i32, {{.*}}type = "mnilu"}
// DFIR: agen.vector_load {{.*}}[{{.*}} + 2, {{.*}}]
// DFIR-NEXT: dataflow.send %[[R]]
// DFIR: dataflow.receive %[[S]]
// DFIR-NEXT: agen.vector_store {{.*}}[%{{[a-zA-Z0-9_]+}}, {{.*}}]
// Owner 1 sends its local rows [0,2); core 0 writes them at global rows [2,4).
// DFIR-LABEL: func.func @local_schedule_1_gather_batch_body()
// DFIR-DAG: %[[SEND:.*]] = dataflow.get_unit {core = 1 : i32, {{.*}}type = "mnisu"}
// DFIR-DAG: %[[RECV:.*]] = dataflow.get_unit {core = 0 : i32, {{.*}}type = "mnilu"}
// DFIR-DAG: %[[LX0:.*]] = dataflow.get_unit {core = 0 : i32, {{.*}}type = "l1"}
// DFIR-DAG: %[[LX1:.*]] = dataflow.get_unit {core = 1 : i32, {{.*}}type = "l1"}
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[SEND]])
// DFIR: uniform.def_immutable_mapping([%[[SEND]] -> %[[LX1]]])
// DFIR: agen.vector_load
// DFIR-NEXT: dataflow.send %[[RECV]]
// DFIR: dataflow.program_unit iter_arg : {{.*}} -> (%[[RECV]])
// DFIR: uniform.def_immutable_mapping([%[[RECV]] -> %[[LX0]]])
// DFIR: dataflow.receive %[[SEND]]
// DFIR-NEXT: agen.vector_store {{.*}}[{{.*}} + 2, {{.*}}]

ktdf_arch.device @sample_device attributes {
  dataflow_scheduler.core_transfer = {
    send_unit = "MNISU", receive_unit = "MNILU", bytes = 128 : i64
  },
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../Dialect/KTDFArch/sample_device.mlir")

#all = affine_set<(r, l) : (r >= 0, 3 - r >= 0, l >= 0, 63 - l >= 0)>
#first = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#second = affine_set<(r, l) : (r - 2 >= 0, 3 - r >= 0, l >= 0, 63 - l >= 0)>
#id = affine_map<(r, l) -> (r, l)>

func.func @distributed_roundtrip(%src_addr: index, %dst_addr: index) attributes {grid = [2]} {
  %c0 = arith.constant 0 : index
  %src = ktdp.construct_memory_view %src_addr, sizes: [4, 64], strides: [64, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<global>
  } : memref<4x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr, sizes: [4, 64], strides: [64, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<global>
  } : memref<4x64xf16>
  %first = ktdp.construct_memory_view %c0, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #first, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
  } : memref<2x64xf16>
  %second = ktdp.construct_memory_view %c0, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #second, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>
  } : memref<2x64xf16>
  %distributed = ktdp.construct_distributed_memory_view (%first, %second : memref<2x64xf16>, memref<2x64xf16>) : memref<4x64xf16>

  %input_tile = ktdp.construct_access_tile %src[0, 0] {
    access_tile_order = #id, access_tile_set = #all
  } : memref<4x64xf16> -> !ktdp.access_tile<4x64xindex>
  %store_tile = ktdp.construct_access_tile %distributed[0, 0] {
    access_tile_order = #id, access_tile_set = #all
  } : memref<4x64xf16> -> !ktdp.access_tile<4x64xindex>
  %input = ktdp.load %input_tile : !ktdp.access_tile<4x64xindex> -> tensor<4x64xf16>
  %empty0 = tensor.empty() : tensor<4x64xf16>
  %copy0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<4x64xf16>) outs(%empty0 : tensor<4x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %copy = spyreop.identity %value : f16
      linalg.yield %copy : f16
  } -> tensor<4x64xf16>
  ktdp.store %copy0, %store_tile : tensor<4x64xf16>, !ktdp.access_tile<4x64xindex>

  %load_tile = ktdp.construct_access_tile %distributed[0, 0] {
    access_tile_order = #id, access_tile_set = #all
  } : memref<4x64xf16> -> !ktdp.access_tile<4x64xindex>
  %output_tile = ktdp.construct_access_tile %dst[0, 0] {
    access_tile_order = #id, access_tile_set = #all
  } : memref<4x64xf16> -> !ktdp.access_tile<4x64xindex>
  %loaded = ktdp.load %load_tile : !ktdp.access_tile<4x64xindex> -> tensor<4x64xf16>
  %empty1 = tensor.empty() : tensor<4x64xf16>
  %copy1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%loaded : tensor<4x64xf16>) outs(%empty1 : tensor<4x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %copy = spyreop.identity %value : f16
      linalg.yield %copy : f16
  } -> tensor<4x64xf16>
  ktdp.store %copy1, %output_tile : tensor<4x64xf16>, !ktdp.access_tile<4x64xindex>
  return
}
