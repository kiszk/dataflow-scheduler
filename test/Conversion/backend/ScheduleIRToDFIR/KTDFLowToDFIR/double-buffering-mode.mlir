// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -allow-unregistered-dialect %s | FileCheck %s --implicit-check-not=iter_args --implicit-check-not=arith.subi

// Both BufferFirst (L1LU) and BufferLast (MNILU) consume buffer 0 first.
// Expand a complete two-slot cycle before DFIR leaves the scheduler, so the
// address generator does not treat the alternating offset as a linear step.
// The two buffers and the nested loops remain; their address recurrence does not.

// CHECK-LABEL: func.func private @mode_a
// CHECK-DAG: %[[C128_A:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C98432_A:.*]] = arith.constant 98432 : index
// CHECK-DAG: %[[C32_A:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C1_A:.*]] = arith.constant 1 : index
// CHECK: scf.for
// CHECK: scf.for {{.*}} to %[[C32_A]] step %[[C1_A]]
// CHECK: %[[A0:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C128_A]]
// CHECK: "test.use"(%[[A0]])
// CHECK: %[[A1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C98432_A]]
// CHECK: "test.use"(%[[A1]])

// CHECK-LABEL: func.func private @mode_b
// CHECK-DAG: %[[C128_B:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C98432_B:.*]] = arith.constant 98432 : index
// CHECK-DAG: %[[C32_B:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C1_B:.*]] = arith.constant 1 : index
// CHECK: scf.for
// CHECK: scf.for {{.*}} to %[[C32_B]] step %[[C1_B]]
// CHECK: %[[B0:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C128_B]]
// CHECK: "test.use"(%[[B0]])
// CHECK: %[[B1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C98432_B]]
// CHECK: "test.use"(%[[B1]])

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  // Mode A: L1LU uses BufferFirst — subi at yield (end of loop body),
  // iter_arg initialized to offset_0 (the smaller offset, 128).
  func.func private @mode_a() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %c128 = arith.constant 128 : index
    %c98432 = arith.constant 98432 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %c64 = arith.constant 64 : index
    dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
      %view0 = dataflow.get_logical_memory_view %0, %c128 {layout_map = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2 * 64 + d3)>} : index, index, memref<12x1x64x64xf16>
      %view1 = dataflow.get_logical_memory_view %0, %c98432 {layout_map = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2 * 64 + d3)>} : index, index, memref<12x1x64x64xf16>
      scf.for %i = %c0 to %c12 step %c1 {
        scf.for %j = %c0 to %c64 step %c1 {
          %phase = ktdf.buffer_phase(%i, %j) {num_phases = 2 : i64} : index
          %sel = ktdf.select_memref %phase[%view0, %view1] : memref<12x1x64x64xf16>
          "test.use"(%sel) : (memref<12x1x64x64xf16>) -> ()
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    }
    return
  }

  // Mode B: MNILU uses BufferLast — subi at loop-body top (first op),
  // iter_arg initialized to offset_1 (the larger offset, 98432).
  func.func private @mode_b() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %c128 = arith.constant 128 : index
    %c98432 = arith.constant 98432 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %c64 = arith.constant 64 : index
    dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
      %view0 = dataflow.get_logical_memory_view %0, %c128 {layout_map = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2 * 64 + d3)>} : index, index, memref<12x1x64x64xf16>
      %view1 = dataflow.get_logical_memory_view %0, %c98432 {layout_map = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2 * 64 + d3)>} : index, index, memref<12x1x64x64xf16>
      scf.for %i = %c0 to %c12 step %c1 {
        scf.for %j = %c0 to %c64 step %c1 {
          %phase = ktdf.buffer_phase(%i, %j) {num_phases = 2 : i64} : index
          %sel = ktdf.select_memref %phase[%view0, %view1] : memref<12x1x64x64xf16>
          "test.use"(%sel) : (memref<12x1x64x64xf16>) -> ()
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    }
    return
  }
}
