// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --normalize-scf-for-loops --canonicalize | FileCheck %s

// Normalizing a partial-row loop composes its uniform lower bound and chunk
// with the row IV. Both transfer maps must remain dims-only after composition.
// CHECK-LABEL: func.func @partial_row(
// CHECK-SAME: %[[SRC:.*]]: memref<3x2x64xf16>, %[[DST:.*]]: memref<3x2x64xf16>, %[[CHUNK:.*]]: index, %[[BEGIN:.*]]: index,
// CHECK: scf.for %[[ROW:.*]] =
// CHECK: ktdf.data_transfer from %[[SRC]][%[[CHUNK]], %[[ROW]] + %[[BEGIN]], 0] size [1, 1, 64] to %[[DST]][%[[CHUNK]], %[[ROW]] + %[[BEGIN]], 0] size [1, 1, 64]
func.func @partial_row(%src: memref<3x2x64xf16>, %dst: memref<3x2x64xf16>,
    %chunk: index, %begin: index, %end: index) {
  %one = arith.constant 1 : index
  scf.for %row = %begin to %end step %one {
    ktdf.data_transfer from %src[%chunk, %row, 0] size [1, 1, 64]
        to %dst[%chunk, %row, 0] size [1, 1, 64]
        : memref<3x2x64xf16>, memref<3x2x64xf16>
  }
  return
}
