// RUN: dataflow-scheduler-opt --ktir-legality-check --split-input-file --verify-diagnostics %s

#set = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
func.func @missing_owner() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{ct_local distributed partition requires an explicit ct_id}}
  %part = ktdp.construct_memory_view %c0, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>
  } : memref<2x64xf16>
  %distributed = ktdp.construct_distributed_memory_view (%part : memref<2x64xf16>) : memref<2x64xf16>
  return
}

// -----

#triangle = affine_set<(r, l) : (r >= 0, 63 - r >= 0, l >= 0, r - l >= 0)>
func.func @nonrectangular() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{distributed partition coordinate_set must be rectangular}}
  %part = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1] {
    coordinate_set = #triangle, memory_space = #ktdp.memory_space<global>
  } : memref<64x64xf16>
  %distributed = ktdp.construct_distributed_memory_view (%part : memref<64x64xf16>) : memref<64x64xf16>
  return
}

// -----

#set = affine_set<(r, l) : (r - 2 >= 0, 3 - r >= 0, l >= 0, 63 - l >= 0)>
func.func @local_shape_mismatch() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{distributed partition extents must match its local memref shape}}
  %part = ktdp.construct_memory_view %c0, sizes: [4, 64], strides: [64, 1] {
    coordinate_set = #set, memory_space = #ktdp.memory_space<global>
  } : memref<4x64xf16>
  %distributed = ktdp.construct_distributed_memory_view (%part : memref<4x64xf16>) : memref<4x64xf16>
  return
}

// -----

#set = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
func.func @hole() {
  %c0 = arith.constant 0 : index
  %part = ktdp.construct_memory_view %c0, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #set, memory_space = #ktdp.memory_space<global>
  } : memref<2x64xf16>
  // expected-error @+1 {{distributed lowering currently requires partitions to cover the entire distributed view shape}}
  %distributed = ktdp.construct_distributed_memory_view (%part : memref<2x64xf16>) : memref<4x64xf16>
  return
}

// -----

#set = affine_set<(r, l) : (r - 2 >= 0, 3 - r >= 0, l >= 0, 63 - l >= 0)>
func.func @outside_shape() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{distributed partition coordinate_set is outside the distributed view shape}}
  %part = ktdp.construct_memory_view %c0, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #set, memory_space = #ktdp.memory_space<global>
  } : memref<2x64xf16>
  %distributed = ktdp.construct_distributed_memory_view (%part : memref<2x64xf16>) : memref<2x64xf16>
  return
}

// -----

func.func @unknown_partition(%part: memref<2x64xf16>) {
  // expected-error @+1 {{distributed partitions must be defined by ktdp.construct_memory_view}}
  %distributed = ktdp.construct_distributed_memory_view (%part : memref<2x64xf16>) : memref<2x64xf16>
  return
}
