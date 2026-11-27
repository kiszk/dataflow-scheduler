// RUN: dataflow-scheduler-opt %s --compute-group-extraction --ktir-distributed-memory | FileCheck %s

// An ordinary ct_local store can alias a distributed output. Retain staging
// and scatter after the compute group rather than racing the two stores.
// CHECK-LABEL: func.func @distributed_local_write_alias(
// CHECK: call @local_schedule_0(
// CHECK-NEXT: call @local_schedule_0_scatter(
// CHECK-NEXT: return
// CHECK-LABEL: func.func @local_schedule_0(
// CHECK: ktdp.store
// CHECK: ktdp.store
// CHECK-LABEL: func.func @local_schedule_0_scatter(
// CHECK: ktdp.load
// CHECK: ktdp.store

ktdf_arch.device @sample_device attributes {
  dataflow_scheduler.core_transfer = {
    send_unit = "MNISU", receive_unit = "MNILU", bytes = 128 : i64
  },
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../../../../Dialect/KTDFArch/sample_device.mlir")

#all = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#id = affine_map<(r, l) -> (r, l)>

func.func @distributed_local_write_alias(%src_addr: index) attributes {grid = [1]} {
  %c0 = arith.constant 0 : index
  %src = ktdp.construct_memory_view %src_addr, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<global>
  } : memref<2x64xf16>
  %owner = ktdp.construct_memory_view %c0, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
  } : memref<2x64xf16>
  %alias = ktdp.construct_memory_view %c0, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<ct_local>
  } : memref<2x64xf16>
  %distributed = ktdp.construct_distributed_memory_view (%owner : memref<2x64xf16>) : memref<2x64xf16>
  %read = ktdp.construct_access_tile %src[0, 0] {access_tile_order = #id, access_tile_set = #all} : memref<2x64xf16> -> !ktdp.access_tile<2x64xindex>
  %write_alias = ktdp.construct_access_tile %alias[0, 0] {access_tile_order = #id, access_tile_set = #all} : memref<2x64xf16> -> !ktdp.access_tile<2x64xindex>
  %write = ktdp.construct_access_tile %distributed[0, 0] {access_tile_order = #id, access_tile_set = #all} : memref<2x64xf16> -> !ktdp.access_tile<2x64xindex>
  %value = ktdp.load %read : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>
  %empty = tensor.empty() : tensor<2x64xf16>
  %copy = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%value : tensor<2x64xf16>) outs(%empty : tensor<2x64xf16>) {
    ^bb0(%element: f16, %unused: f16):
      %result = spyreop.identity %element : f16
      linalg.yield %result : f16
  } -> tensor<2x64xf16>
  ktdp.store %copy, %write_alias : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>
  ktdp.store %copy, %write : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>
  return
}
