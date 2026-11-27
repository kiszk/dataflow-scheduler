// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --mpe-matmul-chunking="chunk-size=64 buffers=2" --double-buffering --canonicalize | FileCheck %s
// P is a slice of the final resident output. There is no LX-to-LX export.
// CHECK-LABEL: func.func @resident_bmm
// CHECK: memref.subview
// CHECK-SAME: to memref<8x64xf16, strided<[128, 1], offset: ?>
// CHECK: memref.alloc(){{.*}} : memref<8x64xf16
// CHECK: memref.alloc(){{.*}} : memref<64x64xf16
// CHECK: memref.alloc(){{.*}} : memref<64x64xf16
// CHECK: ktdf.buffer_phase
// CHECK: ktdf.data_transfer
// CHECK: dataflow_scheduler.mpe_chunk_profile = "dd2_dl16_chunk"
// CHECK-NOT: ktdf.data_transfer

module {
  ktdf_arch.device @chunk_test attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("Inputs/device.mlir")
  module @resident_bmm {
    func.func @resident_bmm(%b_addr: index) attributes {grid = [1]} {
      %a_addr = arith.constant 1024 : index
      %a = ktdp.construct_memory_view %a_addr, sizes: [2,16,192], strides: [3072,192,1] {memory_space = #ktdp.memory_space<ct_local>, coordinate_set = affine_set<(b,m,k) : (b >= 0, 1-b >= 0, m >= 0, 15-m >= 0, k >= 0, 191-k >= 0)>} : memref<2x16x192xf16, #ktdp.memory_space<ct_local>>
      %b = ktdp.construct_memory_view %b_addr, sizes: [2,192,128], strides: [24576,128,1] {memory_space = #ktdp.memory_space<global>, coordinate_set = affine_set<(b,k,n) : (b >= 0, 1-b >= 0, k >= 0, 191-k >= 0, n >= 0, 127-n >= 0)>} : memref<2x192x128xf16, #ktdp.memory_space<global>>
      %c_addr = arith.constant 16384 : index
      %c = ktdp.construct_memory_view %c_addr, sizes: [2,16,128], strides: [2048,128,1] {memory_space = #ktdp.memory_space<ct_local>, coordinate_set = affine_set<(b,m,n) : (b >= 0, 1-b >= 0, m >= 0, 15-m >= 0, n >= 0, 127-n >= 0)>} : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>
      %z = arith.constant 0 : index
      ktdf.pipeline {
        %p:4 = ktdf.private -> (memref<2x192x128xf16, #ktdp.memory_space<ct_local>>, memref<2x16x128xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
          %v1 = memref.alloc() : memref<2x192x128xf16, #ktdp.memory_space<ct_local>>
          %v2 = memref.alloc() : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>
          %ready = ktdf.create_token : !ktdf.token
          %done = ktdf.create_token : !ktdf.token
          ktdf.private_yield %v1, %v2, %ready, %done : memref<2x192x128xf16, #ktdp.memory_space<ct_local>>, memref<2x16x128xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#2) {
          ktdf.data_transfer from %b[%z, %z, %z] size [2, 192, 128] to %p#0[%z, %z, %z] size [2, 192, 128] : memref<2x192x128xf16, #ktdp.memory_space<global>>, memref<2x192x128xf16, #ktdp.memory_space<ct_local>>
        } {applicable_units = ["LOAD"]}
        ktdf.stage depends_in(%p#2) depends_out(%p#3) {
          %zero = arith.constant 0.0 : f16
          linalg.fill {dataflow_scheduler.mpe_init} ins(%zero : f16) outs(%c : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
            ins(%a, %p#0 : memref<2x16x192xf16, #ktdp.memory_space<ct_local>>, memref<2x192x128xf16, #ktdp.memory_space<ct_local>>) outs(%c : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>)
            attrs = {dataflow_scheduler.mpe_matmul, dataflow_scheduler.requires_target_lowering} {
          ^bb0(%lhs: f16, %rhs: f16, %old: f16):
            %mul = arith.mulf %lhs, %rhs : f16
            %add = arith.addf %mul, %old : f16
            linalg.yield %add : f16
          }
        } {applicable_units = ["MATRIX"]}
      }
      return
    }
  }
}
