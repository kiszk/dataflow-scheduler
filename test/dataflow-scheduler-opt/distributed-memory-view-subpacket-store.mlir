// RUN: dataflow-scheduler-opt %s --compute-group-extraction --ktir-distributed-memory | FileCheck %s --check-prefix=EXPAND --implicit-check-not=_gather --implicit-check-not=selected_core
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR --implicit-check-not=ktdp.construct_distributed_memory_view

// Two cores update different four-element slices of the same owner-1 packet.
// Core 0 sends a packet to a separate owner-local relay, then the owner copies
// only lanes [4,8). Core 1's lanes [8,12) come from its own write staging.
// Neither the relay nor the short copy may overwrite the other core's values.
// EXPAND-LABEL: func.func @distributed_subpacket_store(
// EXPAND: call @local_schedule_0(
// EXPAND-NEXT: call @local_schedule_0_scatter(
// EXPAND-NEXT: call @local_schedule_0_scatter_0(
// EXPAND-NEXT: call @local_schedule_0_scatter_1(
// EXPAND-NEXT: return
// EXPAND-LABEL: func.func @local_schedule_0_scatter_0(
// EXPAND-SAME: attributes {dataflow_scheduler.physical_core = 1 : i32, grid = [1]}
// EXPAND: %[[OWNER0:.*]] = ktdp.construct_memory_view
// EXPAND: ktdp.construct_access_tile
// EXPAND: %[[FOUR:.*]] = arith.constant 4 : index
// EXPAND-NEXT: %[[SLICE0:.*]] = ktdp.construct_access_tile %[[OWNER0]][{{.*}}, %[[FOUR]]]
// EXPAND: ktdp.store {{.*}}, %[[SLICE0]] : tensor<1x4xf32>
// EXPAND-LABEL: func.func @local_schedule_0_scatter_1(
// EXPAND-SAME: attributes {dataflow_scheduler.physical_core = 1 : i32, grid = [1]}
// EXPAND: %[[OWNER1:.*]] = ktdp.construct_memory_view
// EXPAND: ktdp.construct_access_tile
// EXPAND: %[[EIGHT:.*]] = arith.constant 8 : index
// EXPAND-NEXT: %[[SLICE1:.*]] = ktdp.construct_access_tile %[[OWNER1]][{{.*}}, %[[EIGHT]]]
// EXPAND: ktdp.store {{.*}}, %[[SLICE1]] : tensor<1x4xf32>
// DFIR-LABEL: func.func @local_schedule_0_scatter_body()
// DFIR: dataflow.get_unit {core = 0 : i32, {{.*}}type = "mnisu"}
// DFIR: dataflow.get_unit {core = 1 : i32, {{.*}}type = "mnilu"}
// DFIR: dataflow.send {{.*}}vector<32xf32>
// DFIR: dataflow.receive {{.*}}vector<32xf32>
// DFIR-LABEL: func.func @local_schedule_0_scatter_0_body()
// DFIR: agen.vector_store {{.*}}vector<4xf32>
// DFIR-LABEL: func.func @local_schedule_0_scatter_1_body()
// DFIR: agen.vector_store {{.*}}vector<4xf32>

ktdf_arch.device @sample_device attributes {
  dataflow_scheduler.core_transfer = {
    send_unit = "MNISU", receive_unit = "MNILU", bytes = 128 : i64
  },
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../Dialect/KTDFArch/sample_device.mlir")

#input = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 31 - l >= 0)>
#packet = affine_set<(r, l) : (r == 0, l >= 0, 31 - l >= 0)>
#slice = affine_set<(r, l) : (r == 0, l >= 0, 3 - l >= 0)>
#id = affine_map<(r, l) -> (r, l)>

func.func @distributed_subpacket_store(%address: index) attributes {grid = [2]} {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %core = ktdp.get_compute_tile_id : index
  %offset = arith.muli %core, %c4 : index
  %lane = arith.addi %offset, %c4 : index
  %src = ktdp.construct_memory_view %address, sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #input, memory_space = #ktdp.memory_space<global>
  } : memref<2x32xf32>
  %owner = ktdp.construct_memory_view %c0, sizes: [1, 32], strides: [32, 1] {
    coordinate_set = #packet, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>
  } : memref<1x32xf32>
  %distributed = ktdp.construct_distributed_memory_view (%owner : memref<1x32xf32>) : memref<1x32xf32>
  %read = ktdp.construct_access_tile %src[%core, 0] {
    access_tile_order = #id, access_tile_set = #slice
  } : memref<2x32xf32> -> !ktdp.access_tile<1x4xindex>
  %write = ktdp.construct_access_tile %distributed[0, %lane] {
    access_tile_order = #id, access_tile_set = #slice
  } : memref<1x32xf32> -> !ktdp.access_tile<1x4xindex>
  %value = ktdp.load %read : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf32>
  %empty = tensor.empty() : tensor<1x4xf32>
  %copy = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%value : tensor<1x4xf32>) outs(%empty : tensor<1x4xf32>) {
    ^bb0(%element: f32, %unused: f32):
      %result = spyreop.identity %element : f32
      linalg.yield %result : f32
  } -> tensor<1x4xf32>
  ktdp.store %copy, %write : tensor<1x4xf32>, !ktdp.access_tile<1x4xindex>
  return
}
