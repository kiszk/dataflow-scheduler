// RUN: dataflow-scheduler-opt %s --compute-group-extraction --ktir-distributed-memory | FileCheck %s --implicit-check-not=_initialize --implicit-check-not=_gather

// The middle group observes every initialized element before the partial
// update. Do not remove the initial writes to the subsequently updated rows.
// CHECK-LABEL: func.func @distributed_observed_initialization(
// CHECK: call @local_schedule_0(
// CHECK-NEXT: call @local_schedule_1(
// CHECK-NEXT: call @local_schedule_2(
// CHECK-NEXT: call @local_schedule_2_scatter(
// CHECK-NEXT: call @local_schedule_2_scatter_0(
// CHECK-NEXT: return
// CHECK-LABEL: func.func @local_schedule_0(
// CHECK: ktdp.load {{.*}} -> tensor<2x32xf32>
// CHECK: ktdp.store {{.*}} : tensor<2x32xf32>
// CHECK-LABEL: func.func @local_schedule_1(
// CHECK: ktdp.load {{.*}} -> tensor<2x32xf32>
// CHECK: ktdp.store {{.*}} : tensor<2x32xf32>
// CHECK-LABEL: func.func @local_schedule_2(
// CHECK: ktdp.store {{.*}} : tensor<1x32xf32>

ktdf_arch.device @sample_device attributes {
  dataflow_scheduler.core_transfer = {
    send_unit = "MNISU", receive_unit = "MNILU", bytes = 128 : i64
  },
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../../../../Dialect/KTDFArch/sample_device.mlir")

#all = affine_set<(r, l) : (r >= 0, 3 - r >= 0, l >= 0, 31 - l >= 0)>
#first = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 31 - l >= 0)>
#second = affine_set<(r, l) : (r - 2 >= 0, 3 - r >= 0, l >= 0, 31 - l >= 0)>
#one = affine_set<(r, l) : (r == 0, l >= 0, 31 - l >= 0)>
#id = affine_map<(r, l) -> (r, l)>

func.func @distributed_observed_initialization(%src_addr: index, %update_addr: index, %dst_addr: index) attributes {grid = [2]} {
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

  return
}
