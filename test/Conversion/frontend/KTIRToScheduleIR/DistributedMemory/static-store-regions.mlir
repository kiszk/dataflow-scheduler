// RUN: dataflow-scheduler-opt %s --compute-group-extraction --ktir-distributed-memory | FileCheck %s --check-prefix=EXPAND --implicit-check-not=_gather --implicit-check-not=selected_core
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR

// A two-dimensional logical grid writes four disjoint rows on owner 3.
// Fold both get_compute_tile_id results and affine.apply when determining
// each footprint. The compute keeps the original grid and runs only once.
// EXPAND-LABEL: func.func @distributed_grid_store(
// EXPAND: call @local_schedule_0(
// EXPAND-NEXT: call @local_schedule_0_scatter(
// EXPAND-NEXT: call @local_schedule_0_scatter_0(
// EXPAND-NEXT: call @local_schedule_0_scatter_1(
// EXPAND-NEXT: call @local_schedule_0_scatter_2(
// EXPAND-NEXT: return
// EXPAND-LABEL: func.func @local_schedule_0(
// EXPAND-SAME: attributes {grid = [2, 2]}
// EXPAND-LABEL: func.func @local_schedule_0_scatter_1(
// EXPAND: dataflow.get_unit {core = 2 : i32, {{.*}}type = "mnisu"}
// EXPAND: agen.vector_store {{.*}} + 2, {{.*}}vector<32xf32>
// EXPAND-LABEL: func.func @local_schedule_0_scatter_2(
// EXPAND-SAME: attributes {dataflow_scheduler.distributed_transfer_batch = "local_schedule_0_scatter", dataflow_scheduler.physical_core = 3 : i32, grid = [1]}
// EXPAND: %[[ROW:.*]] = arith.constant 3 : index
// EXPAND: ktdp.construct_access_tile {{.*}}[%[[ROW]], {{.*}}!ktdp.access_tile<1x32xindex>
// EXPAND: ktdp.store {{.*}}tensor<1x32xf32>
// DFIR-LABEL: func.func @local_schedule_0_body()
// DFIR-DAG: dataflow.get_unit {core = 0 : i32, {{.*}}type = "l1su"}
// DFIR-DAG: dataflow.get_unit {core = 3 : i32, {{.*}}type = "l1su"}
// DFIR: agen.vector_store

ktdf_arch.device @sample_device attributes {
  dataflow_scheduler.core_transfer = {
    send_unit = "MNISU", receive_unit = "MNILU", bytes = 128 : i64
  },
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../../../../Dialect/KTDFArch/sample_device.mlir")

#all = affine_set<(r, l) : (r >= 0, 3 - r >= 0, l >= 0, 31 - l >= 0)>
#row = affine_set<(r, l) : (r == 0, l >= 0, 31 - l >= 0)>
#id = affine_map<(r, l) -> (r, l)>

func.func @distributed_grid_store(%address: index) attributes {grid = [2, 2]} {
  %c0 = arith.constant 0 : index
  %x, %y = ktdp.get_compute_tile_id : index, index
  %row = affine.apply affine_map<(x, y) -> (x * 2 + y)>(%x, %y)
  %src = ktdp.construct_memory_view %address, sizes: [4, 32], strides: [32, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<global>
  } : memref<4x32xf32>
  %owner = ktdp.construct_memory_view %c0, sizes: [4, 32], strides: [32, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>
  } : memref<4x32xf32>
  %distributed = ktdp.construct_distributed_memory_view (%owner : memref<4x32xf32>) : memref<4x32xf32>
  %read = ktdp.construct_access_tile %src[0, 0] {
    access_tile_order = #id, access_tile_set = #row
  } : memref<4x32xf32> -> !ktdp.access_tile<1x32xindex>
  %write = ktdp.construct_access_tile %distributed[%row, 0] {
    access_tile_order = #id, access_tile_set = #row
  } : memref<4x32xf32> -> !ktdp.access_tile<1x32xindex>
  %value = ktdp.load %read : !ktdp.access_tile<1x32xindex> -> tensor<1x32xf32>
  %empty = tensor.empty() : tensor<1x32xf32>
  %copy = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%value : tensor<1x32xf32>) outs(%empty : tensor<1x32xf32>) {
    ^bb0(%element: f32, %unused: f32):
      %result = spyreop.identity %element : f32
      linalg.yield %result : f32
  } -> tensor<1x32xf32>
  ktdp.store %copy, %write : tensor<1x32xf32>, !ktdp.access_tile<1x32xindex>
  return
}
