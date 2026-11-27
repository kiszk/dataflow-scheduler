// RUN: dataflow-scheduler-opt %s --ktir-legality-check --compute-group-extraction --ktir-distributed-memory --split-input-file --verify-diagnostics

// No architecture capability: do not silently issue a remote local load.
// expected-error @+1 {{remote distributed access requires dataflow_scheduler.core_transfer}}
ktdf_arch.device @sample_device attributes {
  mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../../../../Dialect/KTDFArch/sample_device.mlir")
#set = affine_set<(r,l) : (r == 0, l >= 0, 63 - l >= 0)>
#id = affine_map<(r,l) -> (r,l)>
func.func @missing_transport(%out: index) attributes {grid = [1]} {
  %c0 = arith.constant 0 : index
  %part = ktdp.construct_memory_view %c0, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>} : memref<1x64xf16>
  %dist = ktdp.construct_distributed_memory_view (%part : memref<1x64xf16>) : memref<1x64xf16>
  %dst = ktdp.construct_memory_view %out, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
  %read = ktdp.construct_access_tile %dist[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  %write = ktdp.construct_access_tile %dst[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  %value = ktdp.load %read : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
  ktdp.store %value, %write : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
  return
}

// -----

ktdf_arch.device @sample_device attributes {
  dataflow_scheduler.core_transfer = {send_unit = "MNISU", receive_unit = "MNILU", bytes = 128 : i64},
  mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../../../../Dialect/KTDFArch/sample_device.mlir")
#set = affine_set<(r,l) : (r == 0, l >= 0, 3 - l >= 0)>
#id = affine_map<(r,l) -> (r,l)>
func.func @partial_packet(%out: index) attributes {grid = [1]} {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{remote distributed partition requires aligned whole transfer packets}}
  %part = ktdp.construct_memory_view %c0, sizes: [1, 4], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>} : memref<1x4xf32>
  %dist = ktdp.construct_distributed_memory_view (%part : memref<1x4xf32>) : memref<1x4xf32>
  %dst = ktdp.construct_memory_view %out, sizes: [1, 4], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x4xf32>
  %read = ktdp.construct_access_tile %dist[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<1x4xf32> -> !ktdp.access_tile<1x4xindex>
  %write = ktdp.construct_access_tile %dst[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<1x4xf32> -> !ktdp.access_tile<1x4xindex>
  %value = ktdp.load %read : !ktdp.access_tile<1x4xindex> -> tensor<1x4xf32>
  ktdp.store %value, %write : tensor<1x4xf32>, !ktdp.access_tile<1x4xindex>
  return
}

// -----

ktdf_arch.device @sample_device attributes {
  mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../../../../Dialect/KTDFArch/sample_device.mlir")
#set = affine_set<(r,l) : (r == 0, l >= 0, 63 - l >= 0)>
#id = affine_map<(r,l) -> (r,l)>
func.func @unbounded_local_address(%local: index, %out: index) attributes {grid = [1]} {
  // expected-error @+1 {{distributed snapshots require constant, bounded ct_local view addresses}}
  %part = ktdp.construct_memory_view %local, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>} : memref<1x64xf16>
  %dist = ktdp.construct_distributed_memory_view (%part : memref<1x64xf16>) : memref<1x64xf16>
  %dst = ktdp.construct_memory_view %out, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
  %read = ktdp.construct_access_tile %dist[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  %write = ktdp.construct_access_tile %dst[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  %value = ktdp.load %read : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
  ktdp.store %value, %write : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
  return
}

// -----

#first = affine_set<(r,l) : (r == 0, l >= 0, 63 - l >= 0)>
#second = affine_set<(r,l) : (r - 1 == 0, l >= 0, 63 - l >= 0)>
#all = affine_set<(r,l) : (r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#id = affine_map<(r,l) -> (r,l)>
func.func @physical_alias(%in: index) attributes {grid = [1]} {
  %c0 = arith.constant 0 : index
  %first = ktdp.construct_memory_view %c0, sizes: [1, 64], strides: [64, 1] {coordinate_set = #first, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>} : memref<1x64xf16>
  // Logical rectangles are disjoint, but their physical storage overlaps.
  // expected-error @+1 {{distributed stores require provably disjoint physical partition storage}}
  %second = ktdp.construct_memory_view %c0, sizes: [1, 64], strides: [64, 1] {coordinate_set = #second, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>} : memref<1x64xf16>
  %dist = ktdp.construct_distributed_memory_view (%first, %second : memref<1x64xf16>, memref<1x64xf16>) : memref<2x64xf16>
  %src = ktdp.construct_memory_view %in, sizes: [2, 64], strides: [64, 1] {coordinate_set = #all, memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
  %read = ktdp.construct_access_tile %src[0, 0] {access_tile_order = #id, access_tile_set = #all} : memref<2x64xf16> -> !ktdp.access_tile<2x64xindex>
  %write = ktdp.construct_access_tile %dist[0, 0] {access_tile_order = #id, access_tile_set = #all} : memref<2x64xf16> -> !ktdp.access_tile<2x64xindex>
  %value = ktdp.load %read : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>
  ktdp.store %value, %write : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>
  return
}

// -----

#set = affine_set<(r,l) : (r == 0, l >= 0, 63 - l >= 0)>
#id = affine_map<(r,l) -> (r,l)>
func.func @store_then_load_same_group(%in: index, %out: index) attributes {grid = [1]} {
  %c0 = arith.constant 0 : index
  %part = ktdp.construct_memory_view %c0, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>} : memref<1x64xf16>
  %dist = ktdp.construct_distributed_memory_view (%part : memref<1x64xf16>) : memref<1x64xf16>
  %src = ktdp.construct_memory_view %in, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
  %dst = ktdp.construct_memory_view %out, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
  %input = ktdp.construct_access_tile %src[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  %access = ktdp.construct_access_tile %dist[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  %output = ktdp.construct_access_tile %dst[0, 0] {access_tile_order = #id, access_tile_set = #set} : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
  %value = ktdp.load %input : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
  ktdp.store %value, %access : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
  // Reusing the access tile merges both sides into one compute group. Do not
  // silently move this load before the preceding store.
  // expected-error @+1 {{distributed loads must precede stores within a compute group}}
  %after = ktdp.load %access : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
  ktdp.store %after, %output : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
  return
}
