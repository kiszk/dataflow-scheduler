// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --mpe-matmul-chunking="chunk-size=64 buffers=1" --canonicalize | FileCheck %s
// CHECK-LABEL: func.func @resident_bmm
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.subview
// CHECK-SAME: to memref<8x64xf16, strided<[192, 1], offset: ?>
// CHECK: ktdf.data_transfer from %arg1
// CHECK: dataflow_scheduler.mpe_chunk_profile = "dd2_dl16_chunk"
// CHECK: ktdf.data_transfer from {{.*}} to %arg2

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module @resident_bmm {
    func.func @resident_bmm(%a: memref<2x16x192xf16, #ktdp.memory_space<ct_local>>, %b: memref<2x192x128xf16, #ktdp.memory_space<global>>, %c: memref<2x16x128xf16, #ktdp.memory_space<global>>) attributes {grid = [1]} {
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
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%p#2) depends_out(%p#3) {
          %zero = arith.constant 0.0 : f16
          linalg.fill {dataflow_scheduler.mpe_init} ins(%zero : f16) outs(%p#1 : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
            ins(%a, %p#0 : memref<2x16x192xf16, #ktdp.memory_space<ct_local>>, memref<2x192x128xf16, #ktdp.memory_space<ct_local>>) outs(%p#1 : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>)
            attrs = {dataflow_scheduler.mpe_matmul, dataflow_scheduler.requires_target_lowering} {
          ^bb0(%lhs: f16, %rhs: f16, %old: f16):
            %mul = arith.mulf %lhs, %rhs : f16
            %add = arith.addf %mul, %old : f16
            linalg.yield %add : f16
          }
        } {applicable_units = ["MPE_ARRAY"]}
        ktdf.stage depends_in(%p#3) depends_out(none) {
          ktdf.data_transfer from %p#1[%z, %z, %z] size [2, 16, 128] to %c[%z, %z, %z] size [2, 16, 128] : memref<2x16x128xf16, #ktdp.memory_space<ct_local>>, memref<2x16x128xf16, #ktdp.memory_space<global>>
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
