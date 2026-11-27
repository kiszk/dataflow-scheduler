// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --schedule-ktdf | FileCheck %s

// K=64 cannot be split. Tiling batch/M/N still makes the prepared matmul fit
// into the remaining 16 KiB. Canonicalization removes the Q=1 loop and keeps
// the first-chunk identity; no previous partial is read.
// CHECK-LABEL: func.func @batch_output_tiles
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: dataflow_scheduler.mpe_chunk_buffers = 1 : i64
// CHECK: dataflow_scheduler.mpe_chunk_size = 64 : i64
// CHECK: dataflow_scheduler.mpe_partial_combine
// CHECK: spyreop.identity
// CHECK-NOT: dataflow_scheduler.mpe_chunk_loop
// CHECK: return

module {
  ktdf_arch.device @chunk_test attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("Inputs/device.mlir")
  // A live caller-owned view in another function also reserves this LX prefix.
  func.func @caller_storage() -> memref<516096xf16, #ktdp.memory_space<ct_local>> {
    %base = arith.constant 0 : index
    %view = ktdp.construct_memory_view %base, sizes: [516096], strides: [1] {memory_space = #ktdp.memory_space<ct_local>, coordinate_set = affine_set<(i) : (i >= 0, 516095-i >= 0)>} : memref<516096xf16, #ktdp.memory_space<ct_local>>
    return %view : memref<516096xf16, #ktdp.memory_space<ct_local>>
  }
  module @batch_output_tiles {
    func.func @batch_output_tiles(%a: memref<2x16x64xf16, #ktdp.memory_space<global>>, %b: memref<2x64x128xf16, #ktdp.memory_space<global>>, %c: memref<2x16x128xf16, #ktdp.memory_space<global>>) attributes {grid = [1]} {
      %z = arith.constant 0 : index
      ktdf.pipeline {
        %p:5 = ktdf.private -> (memref<2x16x64xf16, #ktdp.memory_space<ct_local>>, memref<2x64x128xf16, #ktdp.memory_space<ct_local>>, memref<2x16x128xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
          %v0 = memref.alloc() : memref<2x16x64xf16, #ktdp.memory_space<ct_local>>
          %v1 = memref.alloc() : memref<2x64x128xf16, #ktdp.memory_space<ct_local>>
          %v2 = memref.alloc() : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>
          %ready = ktdf.create_token : !ktdf.token
          %done = ktdf.create_token : !ktdf.token
          ktdf.private_yield %v0, %v1, %v2, %ready, %done : memref<2x16x64xf16, #ktdp.memory_space<ct_local>>, memref<2x64x128xf16, #ktdp.memory_space<ct_local>>, memref<2x16x128xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#3) {
          ktdf.data_transfer from %a[%z, %z, %z] size [2, 16, 64] to %p#0[%z, %z, %z] size [2, 16, 64] : memref<2x16x64xf16, #ktdp.memory_space<global>>, memref<2x16x64xf16, #ktdp.memory_space<ct_local>>
          ktdf.data_transfer from %b[%z, %z, %z] size [2, 64, 128] to %p#1[%z, %z, %z] size [2, 64, 128] : memref<2x64x128xf16, #ktdp.memory_space<global>>, memref<2x64x128xf16, #ktdp.memory_space<ct_local>>
        } {applicable_units = ["LOAD"]}
        ktdf.stage depends_in(%p#3) depends_out(%p#4) {
          %zero = arith.constant 0.0 : f16
          linalg.fill {dataflow_scheduler.mpe_init} ins(%zero : f16) outs(%p#2 : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>)
          linalg.generic {indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>, affine_map<(b, m, n, k) -> (b, k, n)>, affine_map<(b, m, n, k) -> (b, m, n)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
            ins(%p#0, %p#1 : memref<2x16x64xf16, #ktdp.memory_space<ct_local>>, memref<2x64x128xf16, #ktdp.memory_space<ct_local>>) outs(%p#2 : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>)
            attrs = {dataflow_scheduler.mpe_matmul, dataflow_scheduler.requires_target_lowering} {
          ^bb0(%lhs: f16, %rhs: f16, %old: f16):
            %mul = arith.mulf %lhs, %rhs : f16
            %add = arith.addf %mul, %old : f16
            linalg.yield %add : f16
          }
        } {applicable_units = ["MATRIX"]}
        ktdf.stage depends_in(%p#4) depends_out(none) {
          ktdf.data_transfer from %p#2[%z, %z, %z] size [2, 16, 128] to %c[%z, %z, %z] size [2, 16, 128] : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>, memref<2x16x128xf16, #ktdp.memory_space<global>>
        } {applicable_units = ["STORE"]}
      }
      return
    }
  }
}
