// RUN: dataflow-scheduler-opt --path-expansion --split-input-file %s | FileCheck %s

// The two phases use the same SFU and communicate through shared L1 storage.
// Each inner pipeline still gets ordinary route expansion. The outer stages
// preserve their dependency and carry the union of their inner units.
// CHECK-LABEL: func.func @serial_phases(
// CHECK: ktdf.pipeline {
// CHECK: %[[SHARED:.*]]:2 = ktdf.private
// CHECK: ktdf.stage depends_in(none) depends_out(%[[SHARED]]#1) {
// CHECK-NEXT: ktdf.pipeline {
// CHECK: } {applicable_units = ["MNILU"]}
// CHECK: } {applicable_units = ["L1LU"]}
// CHECK: } {applicable_units = ["SFU"]}
// CHECK: } {applicable_units = ["L1SU"]}
// CHECK-NEXT: }
// CHECK-NEXT: } {applicable_units = ["MNILU", "L1LU", "SFU", "L1SU"]}
// CHECK-NEXT: ktdf.stage depends_in(%[[SHARED]]#1) depends_out(none) {
// CHECK-NEXT: %{{.*}} = memref.cast
// CHECK-NEXT: ktdf.pipeline {
// CHECK: } {applicable_units = ["L1LU"]}
// CHECK: } {applicable_units = ["SFU"]}
// CHECK: } {applicable_units = ["L1SU"]}
// CHECK: } {applicable_units = ["MNISU"]}
// CHECK-NEXT: }
// CHECK-NEXT: } {applicable_units = ["L1LU", "SFU", "L1SU", "MNISU"]}
// CHECK-NEXT: }
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @serial_phases(%input: memref<64xf16, "DDR">, %output: memref<64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      %shared:2 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
        %scratch = memref.alloc() : memref<64xf16, "L1">
        %token = ktdf.create_token : !ktdf.token
        ktdf.private_yield %scratch, %token : memref<64xf16, "L1">, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%shared#1) {
        ktdf.pipeline {
          %p:4 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, !ktdf.token, !ktdf.token) {
            %in = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
            %out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>
            %t0 = ktdf.create_token : !ktdf.token
            %t1 = ktdf.create_token : !ktdf.token
            ktdf.private_yield %in, %out, %t0, %t1 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, !ktdf.token, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%p#2) {
            ktdf.data_transfer from %input[%c0] size [64] to %p#0 size [64] : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
          }
          ktdf.stage depends_in(%p#2) depends_out(%p#3) {
            %x = ktdf.read_from_fifo %p#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
            ktdf.write_to_fifo %x, %p#1 : tensor<64xf16>, <"SFU" -> "L1", 64xf16>
          } {applicable_units = ["SFU"]}
          ktdf.stage depends_in(%p#3) depends_out(none) {
            ktdf.data_transfer from %p#1 size [64] to %shared#0[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1", 64xf16>, memref<64xf16, "L1">
          }
        }
      }
      ktdf.stage depends_in(%shared#1) depends_out(none) {
        %view = memref.cast %shared#0 : memref<64xf16, "L1"> to memref<?xf16, "L1">
        ktdf.pipeline {
          %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token) {
            %in = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
            %out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
            %t0 = ktdf.create_token : !ktdf.token
            %t1 = ktdf.create_token : !ktdf.token
            ktdf.private_yield %in, %out, %t0, %t1 : !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%p#2) {
            ktdf.data_transfer from %view[%c0] size [64] to %p#0 size [64] : memref<?xf16, "L1">, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
          }
          ktdf.stage depends_in(%p#2) depends_out(%p#3) {
            %x = ktdf.read_from_fifo %p#0 : <"L1" -> "SFU", 64xf16> -> tensor<64xf16>
            ktdf.write_to_fifo %x, %p#1 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
          } {applicable_units = ["SFU"]}
          ktdf.stage depends_in(%p#3) depends_out(none) {
            ktdf.data_transfer from %p#1 size [64] to %output[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
          }
        }
      }
    }
    return
  }
}

// -----

// A transitive order is sufficient; there need not be an edge between every
// pair. This also covers nested loop wrappers and scalar setup.
// CHECK-LABEL: func.func @transitive_phases()
// CHECK-COUNT-6: } {applicable_units = ["SFU"]}
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @transitive_phases() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %t0 = ktdf.create_token : !ktdf.token
    %t1 = ktdf.create_token : !ktdf.token
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(%t0) {
        scf.for %i = %c0 to %c1 step %c1 {
          %offset = arith.addi %i, %c1 : index
          ktdf.pipeline {
            ktdf.stage depends_in(none) depends_out(none) {
              %value = arith.constant 1.0 : f16
            } {applicable_units = ["SFU"]}
          }
        }
      }
      ktdf.stage depends_in(%t0) depends_out(%t1) {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            %value = arith.constant 2.0 : f16
          } {applicable_units = ["SFU"]}
        }
      }
      ktdf.stage depends_in(%t1) depends_out(none) {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            %value = arith.constant 3.0 : f16
          } {applicable_units = ["SFU"]}
        }
      }
    }
    return
  }
}
