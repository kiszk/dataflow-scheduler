// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --address-assignment | FileCheck %s

// The lowering dialect's constant offset is also in elements, not bytes.
// (640 + 64) * 2 bytes are reserved, so the FP32 scratch starts at 352.
// CHECK: ktdp_lowering.construct_memory_view
// CHECK: %[[BASE:.*]] = arith.constant 352 : index
// CHECK: builtin.unrealized_conversion_cast %[[BASE]] : index to memref<32xf32, "L1">

ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @lowering_view() {
  %base = arith.constant 640 : index
  %view = ktdp_lowering.construct_memory_view %base, sizes: [64], strides: [1] {
    memory_space = "L1",
    coordinate_set = affine_set<(l) : (l >= 0, 63 - l >= 0)>
  } : memref<64xf16, "L1">
  %scratch = memref.alloc() : memref<32xf32, "L1">
  return
}
