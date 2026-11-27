// RUN: dataflow-scheduler-opt %s --compute-group-extraction --ktir-distributed-memory --split-input-file --verify-diagnostics

// Runtime offsets cannot be routed to fixed owners at compile time. Do not
// fall back to writing a whole partition, including uninitialized elements.
#all = affine_set<(r, l) : (r >= 0, 3 - r >= 0, l >= 0, 31 - l >= 0)>
#row = affine_set<(r, l) : (r == 0, l >= 0, 31 - l >= 0)>
#id = affine_map<(r, l) -> (r, l)>
func.func @dynamic_store(%input: index, %row: index) attributes {grid = [1]} {
  %c0 = arith.constant 0 : index
  %src = ktdp.construct_memory_view %input, sizes: [1, 32], strides: [32, 1] {
    coordinate_set = #row, memory_space = #ktdp.memory_space<global>
  } : memref<1x32xf32>
  %owner = ktdp.construct_memory_view %c0, sizes: [4, 32], strides: [32, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
  } : memref<4x32xf32>
  %dist = ktdp.construct_distributed_memory_view (%owner : memref<4x32xf32>) : memref<4x32xf32>
  %read = ktdp.construct_access_tile %src[0, 0] {access_tile_order = #id, access_tile_set = #row} : memref<1x32xf32> -> !ktdp.access_tile<1x32xindex>
  // expected-error @+1 {{distributed stores require a zero-based rectangular tile, identity access_tile_order and offsets constant for each core}}
  %write = ktdp.construct_access_tile %dist[%row, 0] {access_tile_order = #id, access_tile_set = #row} : memref<4x32xf32> -> !ktdp.access_tile<1x32xindex>
  %value = ktdp.load %read : !ktdp.access_tile<1x32xindex> -> tensor<1x32xf32>
  ktdp.store %value, %write : tensor<1x32xf32>, !ktdp.access_tile<1x32xindex>
  return
}

// -----

// Validate every core, including a core whose offset leaves the view.
#all = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 31 - l >= 0)>
#row = affine_set<(r, l) : (r == 0, l >= 0, 31 - l >= 0)>
#id = affine_map<(r, l) -> (r, l)>
func.func @out_of_bounds_store(%input: index) attributes {grid = [2]} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %core = ktdp.get_compute_tile_id : index
  %row = arith.muli %core, %c2 : index
  %src = ktdp.construct_memory_view %input, sizes: [1, 32], strides: [32, 1] {
    coordinate_set = #row, memory_space = #ktdp.memory_space<global>
  } : memref<1x32xf32>
  %owner = ktdp.construct_memory_view %c0, sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
  } : memref<2x32xf32>
  %dist = ktdp.construct_distributed_memory_view (%owner : memref<2x32xf32>) : memref<2x32xf32>
  %read = ktdp.construct_access_tile %src[0, 0] {access_tile_order = #id, access_tile_set = #row} : memref<1x32xf32> -> !ktdp.access_tile<1x32xindex>
  // expected-error @+1 {{distributed access is outside its view on core 1}}
  %write = ktdp.construct_access_tile %dist[%row, 0] {access_tile_order = #id, access_tile_set = #row} : memref<2x32xf32> -> !ktdp.access_tile<1x32xindex>
  %value = ktdp.load %read : !ktdp.access_tile<1x32xindex> -> tensor<1x32xf32>
  ktdp.store %value, %write : tensor<1x32xf32>, !ktdp.access_tile<1x32xindex>
  return
}

// -----

// A bounding box must not turn a triangular footprint into a dense store.
#all = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 31 - l >= 0)>
#triangle = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l - r >= 0, 31 - l >= 0)>
#id = affine_map<(r, l) -> (r, l)>
func.func @nonrectangular_store(%input: index) attributes {grid = [1]} {
  %c0 = arith.constant 0 : index
  %src = ktdp.construct_memory_view %input, sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<global>
  } : memref<2x32xf32>
  %owner = ktdp.construct_memory_view %c0, sizes: [2, 32], strides: [32, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
  } : memref<2x32xf32>
  %dist = ktdp.construct_distributed_memory_view (%owner : memref<2x32xf32>) : memref<2x32xf32>
  %read = ktdp.construct_access_tile %src[0, 0] {access_tile_order = #id, access_tile_set = #all} : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  // expected-error @+1 {{distributed stores require a zero-based rectangular tile, identity access_tile_order and offsets constant for each core}}
  %write = ktdp.construct_access_tile %dist[0, 0] {access_tile_order = #id, access_tile_set = #triangle} : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
  %value = ktdp.load %read : !ktdp.access_tile<2x32xindex> -> tensor<2x32xf32>
  ktdp.store %value, %write : tensor<2x32xf32>, !ktdp.access_tile<2x32xindex>
  return
}
