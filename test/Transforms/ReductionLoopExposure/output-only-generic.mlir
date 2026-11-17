// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt --reduction-loop-exposure %s | FileCheck %s

// Output-only pointwise bodies have no input from which to discover a reduction.
// CHECK-LABEL: func.func @output_only
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK-NOT: scf.for
// CHECK: arith.negf
// CHECK: ktdf.write_to_fifo
func.func @output_only() {
  ktdf.pipeline {
    %slots:2 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
      %in = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      %out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      ktdf.private_yield %in, %out : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %value = ktdf.read_from_fifo %slots#0 : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      %result = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%value : tensor<32xf32>) {
      ^bb0(%x: f32):
        %neg = arith.negf %x : f32
        linalg.yield %neg : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %result, %slots#1 : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}
