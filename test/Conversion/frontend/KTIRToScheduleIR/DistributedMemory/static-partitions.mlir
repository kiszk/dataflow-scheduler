// RUN: dataflow-scheduler-opt --ktir-legality-check --verify-diagnostics %s

// Partition lower bounds, rather than the raw allocation address, define the
// global-to-local translation. Both owners may use the same local address.
#first = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#second = affine_set<(r, l) : (r - 2 >= 0, 3 - r >= 0, l >= 0, 63 - l >= 0)>
func.func @fixed_owners() {
  %c0 = arith.constant 0 : index
  %first = ktdp.construct_memory_view %c0, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #first, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
  } : memref<2x64xf16>
  %second = ktdp.construct_memory_view %c0, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #second, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>
  } : memref<2x64xf16>
  %distributed = ktdp.construct_distributed_memory_view (%first, %second :
    memref<2x64xf16>, memref<2x64xf16>) : memref<4x64xf16>
  return
}

// Global partitions need no owner. The second partition has a padded row
// stride; the logical origin must not be folded into that physical stride.
func.func @global_partitions(%address0: index, %address1: index) {
  %first = ktdp.construct_memory_view %address0, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #first, memory_space = #ktdp.memory_space<global>
  } : memref<2x64xf32>
  %second = ktdp.construct_memory_view %address1, sizes: [2, 64], strides: [128, 1] {
    coordinate_set = #second, memory_space = #ktdp.memory_space<global>
  } : memref<2x64xf32>
  %distributed = ktdp.construct_distributed_memory_view (%first, %second :
    memref<2x64xf32>, memref<2x64xf32>) : memref<4x64xf32>
  return
}
