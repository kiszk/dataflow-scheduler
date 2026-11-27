// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --mpe-matmul-chunking="chunk-size=64 buffers=1" --canonicalize | FileCheck %s --check-prefix=SINGLE
// RUN: dataflow-scheduler-opt %s --mpe-matmul-chunking="chunk-size=64 buffers=2" --double-buffering --canonicalize | FileCheck %s --check-prefix=DOUBLE
// RUN: not dataflow-scheduler-opt %s --mpe-matmul-chunking="chunk-size=128 buffers=1" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not dataflow-scheduler-opt %s --mpe-matmul-chunking="chunk-size=65 buffers=1" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not dataflow-scheduler-opt %s --mpe-matmul-chunking="chunk-size=64 buffers=3" 2>&1 | FileCheck %s --check-prefix=INVALID
// INVALID: matmul chunk plan requires static, nonaliasing Standard f16 operands, legal K slices


// SINGLE-LABEL: func.func @two_inputs
// SINGLE-NOT: memref.alloc(){{.*}} : memref<192x64
// SINGLE: memref.alloc(){{.*}} : memref<8x64xf16
// SINGLE: memref.alloc(){{.*}} : memref<8x64xf16
// SINGLE: scf.for
// SINGLE: ktdf.data_transfer from %arg0
// SINGLE: ktdf.data_transfer from %arg1
// SINGLE: linalg.fill
// SINGLE: dataflow_scheduler.mpe_chunk_profile = "dd2_dl16_chunk"
// SINGLE: scf.if
// SINGLE: spyreop.identity
// SINGLE: } else {
// SINGLE: arith.addf
// SINGLE: dataflow_scheduler.mpe_chunk_loop
// DOUBLE-LABEL: func.func @two_inputs
// DOUBLE: memref.alloc(){{.*}} : memref<8x64xf16
// DOUBLE: memref.alloc(){{.*}} : memref<8x64xf16
// DOUBLE: memref.alloc(){{.*}} : memref<64x64xf16
// DOUBLE: memref.alloc(){{.*}} : memref<64x64xf16
// DOUBLE: %[[PHASE:.*]] = ktdf.buffer_phase
// DOUBLE: ktdf.select_memref %[[PHASE]]
// DOUBLE: ktdf.select_memref %[[PHASE]]
// DOUBLE: ktdf.pipeline modulo(size : 2)

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module @two_inputs {
    func.func @two_inputs(%a: memref<8x192xf16, #ktdp.memory_space<global>>, %b: memref<192x64xf16, #ktdp.memory_space<global>>, %c: memref<8x64xf16, #ktdp.memory_space<global>>) attributes {grid = [1]} {
      %z = arith.constant 0 : index
      ktdf.pipeline {
        %p:5 = ktdf.private -> (memref<8x192xf16, #ktdp.memory_space<ct_local>>, memref<192x64xf16, #ktdp.memory_space<ct_local>>, memref<8x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
          %v0 = memref.alloc() : memref<8x192xf16, #ktdp.memory_space<ct_local>>
          %v1 = memref.alloc() : memref<192x64xf16, #ktdp.memory_space<ct_local>>
          %v2 = memref.alloc() : memref<8x64xf16, #ktdp.memory_space<ct_local>>
          %ready = ktdf.create_token : !ktdf.token
          %done = ktdf.create_token : !ktdf.token
          ktdf.private_yield %v0, %v1, %v2, %ready, %done : memref<8x192xf16, #ktdp.memory_space<ct_local>>, memref<192x64xf16, #ktdp.memory_space<ct_local>>, memref<8x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#3) {
          ktdf.data_transfer from %a[%z, %z] size [8, 192] to %p#0[%z, %z] size [8, 192] : memref<8x192xf16, #ktdp.memory_space<global>>, memref<8x192xf16, #ktdp.memory_space<ct_local>>
          ktdf.data_transfer from %b[%z, %z] size [192, 64] to %p#1[%z, %z] size [192, 64] : memref<192x64xf16, #ktdp.memory_space<global>>, memref<192x64xf16, #ktdp.memory_space<ct_local>>
        } {applicable_units = ["MNILU"]}
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
        } {applicable_units = ["MPE_ARRAY"]}
        ktdf.stage depends_in(%p#4) depends_out(none) {
          ktdf.data_transfer from %p#2[%z, %z] size [8, 64] to %c[%z, %z] size [8, 64] : memref<8x64xf16, #ktdp.memory_space<ct_local>>, memref<8x64xf16, #ktdp.memory_space<global>>
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
