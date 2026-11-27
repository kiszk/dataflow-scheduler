// RUN: dataflow-scheduler-opt --ktir-legality-check --verify-diagnostics %s

#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>

func.func @distributed_view() {
  %A_start_address = arith.constant 1024 : index

  %A_view = ktdp.construct_memory_view %A_start_address, sizes: [32, 64], strides: [64, 1] {
      coordinate_set = #set, memory_space = #ktdp.memory_space<global>
  } : memref<32x64xf16>

  // expected-error @+1 {{distributed partition coordinate_sets must not overlap}}
  %A_dist = ktdp.construct_distributed_memory_view (%A_view, %A_view : memref<32x64xf16>, memref<32x64xf16>) : memref<64x64xf16>

  return
}
