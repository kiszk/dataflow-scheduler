// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --address-assignment -verify-diagnostics

ktdf_arch.device @sample_device attributes {
  mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<ct_local> = "L1">
} import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @too_large() {
  %base = arith.constant 262144 : index
  // expected-error @below {{fixed scratchpad view exceeds memory capacity}}
  %view = ktdp.construct_memory_view %base, sizes: [32], strides: [1] {
    memory_space = #ktdp.memory_space<ct_local>,
    coordinate_set = affine_set<(l) : (l >= 0, 31 - l >= 0)>
  } : memref<32xf32>
  %scratch = memref.alloc() : memref<32xf32, "L1">
  return
}
