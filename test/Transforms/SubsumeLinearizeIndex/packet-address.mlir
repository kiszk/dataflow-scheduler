// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --subsume-linearize-index | FileCheck %s
// Two loop indices and a uniform packet base must become a dims-only map.
// CHECK-LABEL: func.func @packet_address(
// CHECK-SAME: %[[CHUNK:arg[0-9]+]]: index)
// CHECK: scf.for %[[ROW:.*]] =
// CHECK: scf.for %[[PACKET:.*]] =
// CHECK: ktdf.data_transfer from %{{.*}} size [1, 1, 32] to %{{.*}}[%[[PACKET]] + %[[CHUNK]] * 2, %[[ROW]], 0] size [1, 1, 32]
func.func @packet_address(%src: memref<2x1x32xf32, "L1">, %dst: memref<6x2x32xf32, "DDR">, %chunk: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %base = arith.muli %chunk, %c2 : index
  scf.for %row = %c0 to %c2 step %c1 {
    scf.for %packet = %c0 to %c2 step %c1 {
      %dst_packet = arith.addi %base, %packet : index
      ktdf.data_transfer from %src[%packet, %c0, %c0] size [1, 1, 32] to %dst[%dst_packet, %row, %c0] size [1, 1, 32] : memref<2x1x32xf32, "L1">, memref<6x2x32xf32, "DDR">
    }
  }
  return
}
