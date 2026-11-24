// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -allow-unregistered-dialect %s | FileCheck %s --implicit-check-not=iter_args --implicit-check-not=arith.subi --implicit-check-not=ktdf.select_memref

// Nonzero lower bound and nonunit step: five iterations alternate A,B,A,B,A.
// The main loop executes two complete cycles; the last iteration uses A.
// Its unit-step IV is denormalized to 6*i + 3 for the original access indices.
// CHECK: #[[INDEX:.*]] = affine_map<(d0) -> (d0 * 6 + 3)>
// CHECK-LABEL: func.func @odd_nonunit
// CHECK-DAG: %[[A:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[B:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C15:.*]] = arith.constant 15 : index
// CHECK: scf.for %[[I:.*]] = %[[ZERO]] to %[[TWO]] step %[[ONE]] {
// CHECK-NEXT: %[[VA:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[A]]
// CHECK-NEXT: %[[IA:.*]] = affine.apply #[[INDEX]](%[[I]])
// CHECK-NEXT: "test.use"(%[[VA]], %[[IA]])
// CHECK-NEXT: %[[BASE_B:.*]] = affine.apply #[[INDEX]](%[[I]])
// CHECK-NEXT: %[[NEXT_I:.*]] = arith.addi %[[BASE_B]], %[[C3]]
// CHECK-NEXT: %[[VB:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[B]]
// CHECK-NEXT: "test.use"(%[[VB]], %[[NEXT_I]])
// CHECK-NEXT: }
// CHECK-NEXT: %[[TAIL:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[A]]
// CHECK-NEXT: "test.use"(%[[TAIL]], %[[C15]])
// CHECK-NOT: "test.use"
// CHECK: return

// An odd inner trip count flips the phase for the next outer iteration.
// Across three outer iterations of three inner iterations, the order must be
// A,B,A / B,A,B / A,B,A. Expanding only the inner loop would leave a nonlinear
// recurrence in the outer loop.
// CHECK-LABEL: func.func @nested_odd
// CHECK-DAG: %[[NA:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[NB:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK: dataflow.program_unit
// CHECK-NEXT: dataflow.program_unit
// CHECK-NEXT: %[[V00:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[NA]]
// CHECK-NEXT: "test.use"(%[[V00]], %[[C0]], %[[C0]])
// CHECK-NEXT: %[[V01:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[NB]]
// CHECK-NEXT: "test.use"(%[[V01]], %[[C0]], %[[C1]])
// CHECK-NEXT: %[[V02:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[NA]]
// CHECK-NEXT: "test.use"(%[[V02]], %[[C0]], %[[C2]])
// CHECK-NEXT: %[[V10:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[NB]]
// CHECK-NEXT: "test.use"(%[[V10]], %[[C1]], %[[C0]])
// CHECK-NEXT: %[[V11:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[NA]]
// CHECK-NEXT: "test.use"(%[[V11]], %[[C1]], %[[C1]])
// CHECK-NEXT: %[[V12:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[NB]]
// CHECK-NEXT: "test.use"(%[[V12]], %[[C1]], %[[C2]])
// CHECK-NEXT: %[[V20:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[NA]]
// CHECK-NEXT: "test.use"(%[[V20]], %[[C2]], %[[C0]])
// CHECK-NEXT: %[[V21:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[NB]]
// CHECK-NEXT: "test.use"(%[[V21]], %[[C2]], %[[C1]])
// CHECK-NEXT: %[[V22:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[NA]]
// CHECK-NEXT: "test.use"(%[[V22]], %[[C2]], %[[C2]])
// CHECK-NEXT: }
// CHECK-NOT: "test.use"
// CHECK: return

module {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @odd_nonunit() attributes {grid = [1]} {
    %unit = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %c3 = arith.constant 3 : index
    %c18 = arith.constant 18 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    dataflow.program_unit iter_arg : %active -> (%unit) : {
      %a = dataflow.get_logical_memory_view %unit, %c128 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      %b = dataflow.get_logical_memory_view %unit, %c256 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      scf.for %i = %c3 to %c18 step %c3 {
        %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
        %selected = ktdf.select_memref %phase[%a, %b] : memref<64xf16>
        "test.use"(%selected, %i) : (memref<64xf16>, index) -> ()
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    }
    return
  }

  func.func @nested_odd() attributes {grid = [1]} {
    %unit = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    dataflow.program_unit iter_arg : %active -> (%unit) : {
      %a = dataflow.get_logical_memory_view %unit, %c128 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      %b = dataflow.get_logical_memory_view %unit, %c256 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      scf.for %i = %c0 to %c3 step %c1 {
        scf.for %j = %c0 to %c3 step %c1 {
          %phase = ktdf.buffer_phase(%i, %j) {num_phases = 2 : i64} : index
          %selected = ktdf.select_memref %phase[%a, %b] : memref<64xf16>
          "test.use"(%selected, %i, %j) : (memref<64xf16>, index, index) -> ()
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    }
    return
  }
}
