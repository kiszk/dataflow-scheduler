// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --schedule-ktdf | FileCheck %s

// CHECK-LABEL: func.func @capacity
// CHECK-NOT: ktdf.buffer_phase
// CHECK: dataflow_scheduler.mpe_chunk_buffers = 1 : i64
// CHECK: dataflow_scheduler.mpe_chunk_profile = "dd2_dl16_chunk"
// CHECK: dataflow_scheduler.mpe_chunk_size = 64 : i64
// CHECK-NOT: ktdf.buffer_phase

module {
  ktdf_arch.device @chunk_test attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("Inputs/device.mlir")
  // With 24 KiB free, S=2 would fit by payload alone, but A requires 4096-byte alignment.
  // A live caller-owned view in another function also reserves this LX prefix.
  func.func @caller_storage() -> memref<512000xf16, #ktdp.memory_space<ct_local>> {
    %base = arith.constant 0 : index
    %view = ktdp.construct_memory_view %base, sizes: [512000], strides: [1] {memory_space = #ktdp.memory_space<ct_local>, coordinate_set = affine_set<(i) : (i >= 0, 511999-i >= 0)>} : memref<512000xf16, #ktdp.memory_space<ct_local>>
    return %view : memref<512000xf16, #ktdp.memory_space<ct_local>>
  }
  module @capacity {
    func.func @capacity(%a: memref<8x192xf16, #ktdp.memory_space<global>>, %b: memref<192x64xf16, #ktdp.memory_space<global>>, %c: memref<8x64xf16, #ktdp.memory_space<global>>) attributes {grid = [1]} {
      %z = arith.constant 0 : index
      ktdf.pipeline {
        %p:5 = ktdf.private -> (memref<8x192xf16, #ktdp.memory_space<ct_local>>, memref<192x64xf16, #ktdp.memory_space<ct_local>>, memref<8x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
          %v0 = memref.alloc() {alignment = 4096 : i64} : memref<8x192xf16, #ktdp.memory_space<ct_local>>
          %v1 = memref.alloc() : memref<192x64xf16, #ktdp.memory_space<ct_local>>
          %v2 = memref.alloc() : memref<8x64xf16, #ktdp.memory_space<ct_local>>
          %ready = ktdf.create_token : !ktdf.token
          %done = ktdf.create_token : !ktdf.token
          ktdf.private_yield %v0, %v1, %v2, %ready, %done : memref<8x192xf16, #ktdp.memory_space<ct_local>>, memref<192x64xf16, #ktdp.memory_space<ct_local>>, memref<8x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#3) {
          ktdf.data_transfer from %a[%z, %z] size [8, 192] to %p#0[%z, %z] size [8, 192] : memref<8x192xf16, #ktdp.memory_space<global>>, memref<8x192xf16, #ktdp.memory_space<ct_local>>
          ktdf.data_transfer from %b[%z, %z] size [192, 64] to %p#1[%z, %z] size [192, 64] : memref<192x64xf16, #ktdp.memory_space<global>>, memref<192x64xf16, #ktdp.memory_space<ct_local>>
        } {applicable_units = ["LOAD"]}
        ktdf.stage depends_in(%p#3) depends_out(%p#4) {
          %zero = arith.constant 0.0 : f16
          linalg.fill {dataflow_scheduler.mpe_init} ins(%zero : f16) outs(%p#2 : memref<8x64xf16, #ktdp.memory_space<ct_local>>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]}
            ins(%p#0, %p#1 : memref<8x192xf16, #ktdp.memory_space<ct_local>>, memref<192x64xf16, #ktdp.memory_space<ct_local>>) outs(%p#2 : memref<8x64xf16, #ktdp.memory_space<ct_local>>)
            attrs = {dataflow_scheduler.mpe_matmul, dataflow_scheduler.requires_target_lowering} {
          ^bb0(%lhs: f16, %rhs: f16, %old: f16):
            %mul = arith.mulf %lhs, %rhs : f16
            %add = arith.addf %mul, %old : f16
            linalg.yield %add : f16
          }
        } {applicable_units = ["MATRIX"]}
        ktdf.stage depends_in(%p#4) depends_out(none) {
          ktdf.data_transfer from %p#2[%z, %z] size [8, 64] to %c[%z, %z] size [8, 64] : memref<8x64xf16, #ktdp.memory_space<ct_local>>, memref<8x64xf16, #ktdp.memory_space<global>>
        } {applicable_units = ["STORE"]}
      }
      return
    }
  }
}
