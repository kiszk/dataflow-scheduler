// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --mpe-matmul-chunking="chunk-size=64 buffers=1" --verify-diagnostics

func.func @overflowing_shape(%a: memref<8x1152921504606846976xf16>, %b: memref<1152921504606846976x64xf16>, %c: memref<8x64xf16>) {
  // expected-error @+1 {{matmul chunk plan requires static, nonaliasing Standard f16 operands, legal K slices}}
  linalg.generic {indexing_maps = [affine_map<(m,n,k) -> (m,k)>, affine_map<(m,n,k) -> (k,n)>, affine_map<(m,n,k) -> (m,n)>], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : memref<8x1152921504606846976xf16>, memref<1152921504606846976x64xf16>) outs(%c : memref<8x64xf16>) attrs = {dataflow_scheduler.mpe_matmul} {
    ^bb0(%lhs: f16, %rhs: f16, %old: f16):
      %product = arith.mulf %lhs, %rhs : f16
      %sum = arith.addf %product, %old : f16
      linalg.yield %sum : f16
  }
  return
}
