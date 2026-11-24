// SPDX-License-Identifier: Apache-2.0
// Six constants and six working registers per invocation: the unreused pair
// needs 24 registers, while the device has 16. Both sends must consume the
// output register before the next invocation overwrites it.
// RUN: dataflow-scheduler-opt %s -reuse-registers | FileCheck %s
// RUN: dataflow-scheduler-opt %s -reuse-registers -address-assignment | FileCheck %s --check-prefix=ADDR
// RUN: not dataflow-scheduler-opt %s -address-assignment 2>&1 | FileCheck %s --check-prefix=CAPACITY
// CHECK-LABEL: func.func @two_packets(
// CHECK-COUNT-12: memref.alloc
// CHECK-NOT: memref.alloc
// CHECK: ktdf.opaque "test_exp"
// CHECK-NEXT: ins([[INPUTS:.*]])
// CHECK-NEXT: outs([[OUTPUTS:.*]])
// CHECK: } -> tensor<32xf32>
// CHECK-NEXT: ktdf.write_to_fifo
// CHECK: linalg.generic
// CHECK: ktdf.opaque "test_exp"
// CHECK-NEXT: ins([[INPUTS]])
// CHECK-NEXT: outs([[OUTPUTS]])
// CHECK: } -> tensor<32xf32>
// CHECK-NEXT: ktdf.write_to_fifo
// CHECK-NOT: memref.alloc
// CHECK: return
// ADDR-LABEL: func.func @two_packets(
// ADDR-COUNT-12: builtin.unrealized_conversion_cast {{.*}} to memref<32xf32, "SFU_REG">
// ADDR-NOT: builtin.unrealized_conversion_cast
// ADDR: return
// CAPACITY: exceeds "SFU_REG" capacity

ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
#id = affine_map<(i) -> (i)>
func.func @two_packets(%x0: tensor<32xf32>, %x1: tensor<32xf32>, %out0: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, %out1: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  %zero = arith.constant 0 : index
  %init = tensor.empty() : tensor<32xf32>
  %c0 = arith.constant 1.0 : f32
  %c1 = arith.constant 2.0 : f32
  %c2 = arith.constant 3.0 : f32
  %c3 = arith.constant 4.0 : f32
  %c4 = arith.constant 5.0 : f32
  %c5 = arith.constant 6.0 : f32
  %c0_0 = memref.alloc() : memref<32xf32, "SFU_REG">
  %c0_1 = memref.alloc() : memref<32xf32, "SFU_REG">
  %c0_2 = memref.alloc() : memref<32xf32, "SFU_REG">
  %c0_3 = memref.alloc() : memref<32xf32, "SFU_REG">
  %c0_4 = memref.alloc() : memref<32xf32, "SFU_REG">
  %c0_5 = memref.alloc() : memref<32xf32, "SFU_REG">
  %c1_0 = memref.alloc() : memref<32xf32, "SFU_REG">
  %c1_1 = memref.alloc() : memref<32xf32, "SFU_REG">
  %c1_2 = memref.alloc() : memref<32xf32, "SFU_REG">
  %c1_3 = memref.alloc() : memref<32xf32, "SFU_REG">
  %c1_4 = memref.alloc() : memref<32xf32, "SFU_REG">
  %c1_5 = memref.alloc() : memref<32xf32, "SFU_REG">
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      linalg.fill ins(%c0 : f32) outs(%c0_0 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%c1 : f32) outs(%c0_1 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%c2 : f32) outs(%c0_2 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%c3 : f32) outs(%c0_3 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%c4 : f32) outs(%c0_4 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%c5 : f32) outs(%c0_5 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%c0 : f32) outs(%c1_0 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%c1 : f32) outs(%c1_1 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%c2 : f32) outs(%c1_2 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%c3 : f32) outs(%c1_3 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%c4 : f32) outs(%c1_4 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%c5 : f32) outs(%c1_5 : memref<32xf32, "SFU_REG">)
    } {applicable_units = ["SFU"]}
  }
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      %r0_0 = memref.alloca() : memref<32xf32, "SFU_REG">
      %r0_1 = memref.alloca() : memref<32xf32, "SFU_REG">
      %r0_2 = memref.alloca() : memref<32xf32, "SFU_REG">
      %r0_3 = memref.alloca() : memref<32xf32, "SFU_REG">
      %r0_4 = memref.alloca() : memref<32xf32, "SFU_REG">
      %r0_5 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x0 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        memref.store %x, %r0_0[%zero] : memref<32xf32, "SFU_REG">
        ktdf.opaque "test_exp" {dataflow_scheduler.register_names = ["c0", "c1", "c2", "c3", "c4", "c5", "in", "out", "t0", "t1", "t2", "t3"]}
          ins(%c0_0, %c0_1, %c0_2, %c0_3, %c0_4, %c0_5, %r0_0: memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">)
          outs(%r0_1, %r0_2, %r0_3, %r0_4, %r0_5: memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">)
        %result = memref.load %r0_1[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      %r1_0 = memref.alloca() : memref<32xf32, "SFU_REG">
      %r1_1 = memref.alloca() : memref<32xf32, "SFU_REG">
      %r1_2 = memref.alloca() : memref<32xf32, "SFU_REG">
      %r1_3 = memref.alloca() : memref<32xf32, "SFU_REG">
      %r1_4 = memref.alloca() : memref<32xf32, "SFU_REG">
      %r1_5 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x1 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        memref.store %x, %r1_0[%zero] : memref<32xf32, "SFU_REG">
        ktdf.opaque "test_exp" {dataflow_scheduler.register_names = ["c0", "c1", "c2", "c3", "c4", "c5", "in", "out", "t0", "t1", "t2", "t3"]}
          ins(%c1_0, %c1_1, %c1_2, %c1_3, %c1_4, %c1_5, %r1_0: memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">)
          outs(%r1_1, %r1_2, %r1_3, %r1_4, %r1_5: memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">)
        %result = memref.load %r1_1[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y0, %out0 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      ktdf.write_to_fifo %y1, %out1 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}
