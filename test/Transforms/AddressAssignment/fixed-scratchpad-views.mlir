// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --address-assignment | FileCheck %s

// A strided caller-owned view ends at (256 + 64 + 31 + 1) * 4 = 1408
// bytes. Its reservation survives the allocator reset between programs,
// including a program with no direct use of that view. Allocation addresses
// count their own element types, and the architectural aliases share memory.
// CHECK-LABEL: module @producer
// CHECK: arith.constant 256 : index
// CHECK: ktdp.construct_memory_view
// CHECK: %[[HALF:.*]] = arith.constant 704 : index
// CHECK: builtin.unrealized_conversion_cast %[[HALF]] : index to memref<64xf16, "L1">
// CHECK-LABEL: module @between
// CHECK: %[[FLOAT:.*]] = arith.constant 352 : index
// CHECK: builtin.unrealized_conversion_cast %[[FLOAT]] : index to memref<32xf32, "L1">
// CHECK: %[[SECOND:.*]] = arith.constant 384 : index
// CHECK: builtin.unrealized_conversion_cast %[[SECOND]] : index to memref<32xf32, "L1">

ktdf_arch.device @sample_device attributes {
  mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../../Dialect/KTDFArch/sample_device.mlir")

module @producer {
  func.func @producer() {
    %base = arith.constant 256 : index
    %view = ktdp.construct_memory_view %base, sizes: [2, 32], strides: [64, 1] {
      memory_space = #ktdp.memory_space<ct_local>,
      coordinate_set = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 31 - l >= 0)>
    } : memref<2x32xf32>
    %scratch = memref.alloc() : memref<64xf16, "L1">
    return
  }
}
module @between {
  func.func @between() {
    %scratch = memref.alloc() : memref<32xf32, "L1">
    %other = memref.alloc() : memref<32xf32, "L1">
    return
  }
}
