// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --schedule-ktdf --ktdf-to-ktdflowering | FileCheck %s --implicit-check-not=dataflow_scheduler.mpe_input_ready --implicit-check-not=dataflow_scheduler.mpe_input_credit --implicit-check-not=ktdf.buffer_phase

// Both inputs are resident. Only P and T need allocation, so splitting the
// output makes this BMM fit in 4 KiB while retaining the full K. There is
// nothing to prefetch: do not introduce K chunks just to select two buffers.
// There must be no DMA staging or exchange with an empty load stage.
// CHECK-LABEL: func.func @resident_inputs
// CHECK-NOT: dataflow_scheduler.mpe_input_ready
// CHECK-NOT: dataflow_scheduler.mpe_input_credit
// CHECK-NOT: ktdf.buffer_phase
// CHECK: memref.subview
// CHECK-SAME: to memref<8x192xf16, strided<[192, 1], offset: ?>
// CHECK: memref.subview
// CHECK-SAME: to memref<192x64xf16, strided<[128, 1], offset: ?>
// CHECK: dataflow_scheduler.mpe_chunk_buffers = 1 : i64
// CHECK: dataflow_scheduler.mpe_chunk_size = 192 : i64
// CHECK: dataflow_scheduler.mpe_partial_combine
// CHECK: ktdf.data_transfer
// CHECK-NOT: ktdf.data_transfer
// CHECK: return

module {
  ktdf_arch.device @chunk_test attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("Inputs/device.mlir")
  func.func @caller_storage() -> memref<522240xf16, #ktdp.memory_space<ct_local>> {
    %base = arith.constant 0 : index
    %view = ktdp.construct_memory_view %base, sizes: [522240], strides: [1] {memory_space = #ktdp.memory_space<ct_local>, coordinate_set = affine_set<(i) : (i >= 0, 522239-i >= 0)>} : memref<522240xf16, #ktdp.memory_space<ct_local>>
    return %view : memref<522240xf16, #ktdp.memory_space<ct_local>>
  }
  module @resident_inputs {
    func.func @resident_inputs(%c: memref<2x16x128xf16, #ktdp.memory_space<global>>) attributes {grid = [1]} {
      %z = arith.constant 0 : index
      %b_addr = arith.constant 6144 : index
      %a = ktdp.construct_memory_view %z, sizes: [2,16,192], strides: [3072,192,1] {memory_space = #ktdp.memory_space<ct_local>, coordinate_set = affine_set<(b,m,k) : (b >= 0, 1-b >= 0, m >= 0, 15-m >= 0, k >= 0, 191-k >= 0)>} : memref<2x16x192xf16, #ktdp.memory_space<ct_local>>
      %b = ktdp.construct_memory_view %b_addr, sizes: [2,192,128], strides: [24576,128,1] {memory_space = #ktdp.memory_space<ct_local>, coordinate_set = affine_set<(b,k,n) : (b >= 0, 1-b >= 0, k >= 0, 191-k >= 0, n >= 0, 127-n >= 0)>} : memref<2x192x128xf16, #ktdp.memory_space<ct_local>>
      ktdf.pipeline {
        %p:3 = ktdf.private -> (memref<2x16x128xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
          %v = memref.alloc() : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>
          %ready = ktdf.create_token : !ktdf.token
          %done = ktdf.create_token : !ktdf.token
          ktdf.private_yield %v, %ready, %done : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(%p#1) depends_out(%p#2) {
          %zero = arith.constant 0.0 : f16
          linalg.fill {dataflow_scheduler.mpe_init} ins(%zero : f16) outs(%p#0 : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>)
          linalg.generic {indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>, affine_map<(b, m, n, k) -> (b, k, n)>, affine_map<(b, m, n, k) -> (b, m, n)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
            ins(%a, %b : memref<2x16x192xf16, #ktdp.memory_space<ct_local>>, memref<2x192x128xf16, #ktdp.memory_space<ct_local>>) outs(%p#0 : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>)
            attrs = {dataflow_scheduler.mpe_matmul, dataflow_scheduler.requires_target_lowering} {
          ^bb0(%lhs: f16, %rhs: f16, %old: f16):
            %mul = arith.mulf %lhs, %rhs : f16
            %add = arith.addf %mul, %old : f16
            linalg.yield %add : f16
          }
        } {applicable_units = ["MATRIX"]}
        ktdf.stage depends_in(%p#2) depends_out(none) {
          ktdf.data_transfer from %p#0[%z, %z, %z] size [2, 16, 128] to %c[%z, %z, %z] size [2, 16, 128] : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>, memref<2x16x128xf16, #ktdp.memory_space<global>>
        } {applicable_units = ["STORE"]}
      }
      return
    }
  }
}
