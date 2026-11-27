// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktdflowering-to-dfir --allow-unregistered-dialect | FileCheck %s --implicit-check-not=ktdf.buffer_phase --implicit-check-not=ktdf.select_memref --implicit-check-not=iter_args --implicit-check-not=affine.apply --implicit-check-not=arith.addi

// One phase drives two different buffer pairs. This exercises the complete
// loop rewrite, including odd tails and phase reset at every output group.
// CHECK-LABEL: func.func @q1
// CHECK-DAG: %[[C1_128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C1_512:.*]] = arith.constant 512 : index
// CHECK: %[[A1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C1_128]]
// CHECK: %[[B1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C1_512]]
// CHECK: "test.use"(%[[A1]], %[[B1]])
// CHECK-LABEL: func.func @q2
// CHECK-DAG: %[[C2_128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C2_256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C2_512:.*]] = arith.constant 512 : index
// CHECK-DAG: %[[C2_1024:.*]] = arith.constant 1024 : index
// CHECK: %[[A2:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C2_128]]
// CHECK: %[[B2:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C2_512]]
// CHECK: "test.use"(%[[A2]], %[[B2]])
// CHECK: %[[A2_1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C2_256]]
// CHECK: %[[B2_1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C2_1024]]
// CHECK: "test.use"(%[[A2_1]], %[[B2_1]])
// CHECK-LABEL: func.func @q3
// CHECK-DAG: %[[C3_128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C3_256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C3_512:.*]] = arith.constant 512 : index
// CHECK-DAG: %[[C3_1024:.*]] = arith.constant 1024 : index
// CHECK: %[[A3:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C3_128]]
// CHECK: %[[B3:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C3_512]]
// CHECK: "test.use"(%[[A3]], %[[B3]])
// CHECK: %[[A3_1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C3_256]]
// CHECK: %[[B3_1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C3_1024]]
// CHECK: "test.use"(%[[A3_1]], %[[B3_1]])
// CHECK-LABEL: func.func @q100
// CHECK-DAG: %[[C100_128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C100_256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C100_512:.*]] = arith.constant 512 : index
// CHECK-DAG: %[[C100_1024:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[LAST_PAIR:.*]] = arith.constant 49 : index
// CHECK-DAG: %[[PAIRS:.*]] = arith.constant 50 : index
// CHECK: scf.for %[[PAIR:.*]] = %[[ZERO]] to %[[PAIRS]]
// CHECK: %[[A100:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C100_128]]
// CHECK: %[[B100:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C100_512]]
// CHECK: "test.use"(%[[A100]], %[[B100]])
// CHECK: %[[NOT_FIRST:.*]] = arith.cmpi ne, %[[PAIR]], %[[ZERO]] : index
// CHECK: scf.if %[[NOT_FIRST]] {
// CHECK-NEXT: "test.read_partial"()
// CHECK-NEXT: }
// CHECK-NEXT: "test.write_partial"()
// CHECK: %[[A100_1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C100_256]]
// CHECK: %[[B100_1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[C100_1024]]
// CHECK: "test.use"(%[[A100_1]], %[[B100_1]])
// CHECK-NEXT: "test.read_partial"()
// CHECK-NEXT: %[[NOT_LAST:.*]] = arith.cmpi ne, %[[PAIR]], %[[LAST_PAIR]] : index
// CHECK-NEXT: scf.if %[[NOT_LAST]] {
// CHECK-NEXT: "test.write_partial"()

module {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @q1() attributes {grid = [1]} {
    %unit = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %z = arith.constant 0 : index
    %one = arith.constant 1 : index
    %limit = arith.constant 1 : index
    %a0 = arith.constant 128 : index
    %a1 = arith.constant 256 : index
    %b0 = arith.constant 512 : index
    %b1 = arith.constant 1024 : index
    dataflow.program_unit iter_arg : %active -> (%unit) : {
      %av0 = dataflow.get_logical_memory_view %unit, %a0 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      %av1 = dataflow.get_logical_memory_view %unit, %a1 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      %bv0 = dataflow.get_logical_memory_view %unit, %b0 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<32xf16>
      %bv1 = dataflow.get_logical_memory_view %unit, %b1 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<32xf16>
      scf.for %q = %z to %limit step %one {
        %phase = ktdf.buffer_phase(%q) {num_phases = 2 : i64} : index
        %a = ktdf.select_memref %phase[%av0, %av1] : memref<64xf16>
        %b = ktdf.select_memref %phase[%bv0, %bv1] : memref<32xf16>
        "test.use"(%a, %b) : (memref<64xf16>, memref<32xf16>) -> ()
      }
    }
    return
  }
  func.func @q2() attributes {grid = [1]} {
    %unit = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %z = arith.constant 0 : index
    %one = arith.constant 1 : index
    %limit = arith.constant 2 : index
    %a0 = arith.constant 128 : index
    %a1 = arith.constant 256 : index
    %b0 = arith.constant 512 : index
    %b1 = arith.constant 1024 : index
    dataflow.program_unit iter_arg : %active -> (%unit) : {
      %av0 = dataflow.get_logical_memory_view %unit, %a0 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      %av1 = dataflow.get_logical_memory_view %unit, %a1 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      %bv0 = dataflow.get_logical_memory_view %unit, %b0 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<32xf16>
      %bv1 = dataflow.get_logical_memory_view %unit, %b1 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<32xf16>
      scf.for %q = %z to %limit step %one {
        %phase = ktdf.buffer_phase(%q) {num_phases = 2 : i64} : index
        %a = ktdf.select_memref %phase[%av0, %av1] : memref<64xf16>
        %b = ktdf.select_memref %phase[%bv0, %bv1] : memref<32xf16>
        "test.use"(%a, %b) : (memref<64xf16>, memref<32xf16>) -> ()
      }
    }
    return
  }
  func.func @q3() attributes {grid = [1]} {
    %unit = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %z = arith.constant 0 : index
    %one = arith.constant 1 : index
    %limit = arith.constant 3 : index
    %a0 = arith.constant 128 : index
    %a1 = arith.constant 256 : index
    %b0 = arith.constant 512 : index
    %b1 = arith.constant 1024 : index
    dataflow.program_unit iter_arg : %active -> (%unit) : {
      %av0 = dataflow.get_logical_memory_view %unit, %a0 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      %av1 = dataflow.get_logical_memory_view %unit, %a1 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      %bv0 = dataflow.get_logical_memory_view %unit, %b0 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<32xf16>
      %bv1 = dataflow.get_logical_memory_view %unit, %b1 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<32xf16>
      scf.for %q = %z to %limit step %one {
        %phase = ktdf.buffer_phase(%q) {num_phases = 2 : i64} : index
        %a = ktdf.select_memref %phase[%av0, %av1] : memref<64xf16>
        %b = ktdf.select_memref %phase[%bv0, %bv1] : memref<32xf16>
        "test.use"(%a, %b) : (memref<64xf16>, memref<32xf16>) -> ()
      }
    }
    return
  }
  func.func @q100() attributes {grid = [1]} {
    %unit = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %z = arith.constant 0 : index
    %one = arith.constant 1 : index
    %limit = arith.constant 100 : index
    %last = arith.constant 99 : index
    %a0 = arith.constant 128 : index
    %a1 = arith.constant 256 : index
    %b0 = arith.constant 512 : index
    %b1 = arith.constant 1024 : index
    dataflow.program_unit iter_arg : %active -> (%unit) : {
      %av0 = dataflow.get_logical_memory_view %unit, %a0 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      %av1 = dataflow.get_logical_memory_view %unit, %a1 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      %bv0 = dataflow.get_logical_memory_view %unit, %b0 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<32xf16>
      %bv1 = dataflow.get_logical_memory_view %unit, %b1 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<32xf16>
      scf.for %q = %z to %limit step %one {
        %phase = ktdf.buffer_phase(%q) {num_phases = 2 : i64} : index
        %a = ktdf.select_memref %phase[%av0, %av1] : memref<64xf16>
        %b = ktdf.select_memref %phase[%bv0, %bv1] : memref<32xf16>
        "test.use"(%a, %b) : (memref<64xf16>, memref<32xf16>) -> ()
        // Native partial-sum synchronization uses q as a scalar too. After
        // expanding the two phases, compare the normalized counter directly
        // instead of requiring scalar multiplication or register addition.
        %not_first = arith.cmpi ne, %q, %z : index
        scf.if %not_first {
          "test.read_partial"() : () -> ()
        }
        %not_last = arith.cmpi ne, %q, %last : index
        scf.if %not_last {
          "test.write_partial"() : () -> ()
        }
      }
    }
    return
  }
}
