// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s --implicit-check-not=iter_args --implicit-check-not=arith.subi --implicit-check-not=ktdf.select_memref --implicit-check-not='"L1"'

// Source B casts feeding select_memref must become plain logical memory views.
// Both double-buffer phases must retain the complete rank-4 load set and its
// element count after replacing the alternating address with a two-slot cycle.

// CHECK: #[[LAYOUT:.*]] = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2 * 64 + d3)>
// CHECK: #[[ORDER:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[SET:.*]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK-LABEL: func.func private @"local-schedule-0"() attributes {grid = [2]}
// CHECK-DAG: %[[OFFSET0:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[OFFSET1:.*]] = arith.constant 98432 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C12:.*]] = arith.constant 12 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: dataflow.program_unit
// CHECK: scf.for {{.*}} = %[[C0]] to %[[C12]] step %[[C1]]
// CHECK: scf.for {{.*}} = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK: %[[VIEW0:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[OFFSET0]] {layout_map = #[[LAYOUT]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT: %[[LOAD0:.*]] = agen.vector_load %[[VIEW0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {load_order = #[[ORDER]], load_set = #[[SET]]} : memref<12x1x64x64xf16>, vector<49152xf16>
// CHECK: dataflow.send {{.*}}, %[[LOAD0]] : vector<49152xf16>
// CHECK: %[[VIEW1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[OFFSET1]] {layout_map = #[[LAYOUT]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT: %[[LOAD1:.*]] = agen.vector_load %[[VIEW1]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {load_order = #[[ORDER]], load_set = #[[SET]]} : memref<12x1x64x64xf16>, vector<49152xf16>
// CHECK: dataflow.send {{.*}}, %[[LOAD1]] : vector<49152xf16>
// CHECK: return

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func private @"local-schedule-0"() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %2 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %3 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
    %4 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
    %5 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c98432 = arith.constant 98432 : index
    dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
      // Two Source B casts feeding select_memref -> data_transfer
      %l10 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
      %l11 = builtin.unrealized_conversion_cast %c98432 : index to memref<12x1x64x64xf16, "L1">
      %fifo:1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      scf.for %arg1 = %c0 to %c12 step %c1 {
        scf.for %arg2 = %c0 to %c64 step %c1 {
          %phase = ktdf.buffer_phase(%arg1, %arg2) {num_phases = 2 : i64} : index
          %sel = ktdf.select_memref %phase[%l10, %l11] : memref<12x1x64x64xf16, "L1">
          ktdf.data_transfer from %sel[%c0, %c0, %c0, %c0] size [12, 1, 64, 64] to %fifo#0 size [12, 1, 64, 64] : memref<12x1x64x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
      memref.dealloc %l10 : memref<12x1x64x64xf16, "L1">
      memref.dealloc %l11 : memref<12x1x64x64xf16, "L1">
    }
    dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
    }
    dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
    }
    return
  }
}
