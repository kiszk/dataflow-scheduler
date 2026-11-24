// RUN: dataflow-scheduler-opt --broadcast-promotion --canonicalize --ktdf-to-ktdflowering -allow-unregistered-dialect %s | FileCheck %s

// Hoisting a broadcast load creates a sibling pipeline outside the reader
// loop. Publish the scratchpad data before that loop and wait for all reads
// before the next outer iteration overwrites the same buffer.
// CHECK-LABEL: func.func @promoted_load(
// CHECK: %[[WRITER:.*]] = uniform.query_map
// CHECK: %[[READER:.*]] = uniform.query_map
// CHECK: scf.for
// CHECK: ktdf.data_transfer
// CHECK: ktdf_lowering.signal %[[WRITER]], %[[READER]]
// CHECK-NEXT: scf.for
// CHECK-NOT: ktdf_lowering.signal
// CHECK: ktdf.data_transfer
// CHECK: ktdf_lowering.signal %[[WRITER]], %[[READER]]
// CHECK-NEXT: }
// CHECK-NEXT: return

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @promoted_load(%input: memref<3x64xf16, "DDR">) attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    scf.for %stick = %c0 to %c3 step %c1 {
      scf.for %row = %c0 to %c4 step %c1 {
        ktdf.pipeline {
          %scratch, %ready = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
            %buffer = memref.alloc() : memref<64xf16, "L1">
            %token = ktdf.create_token : !ktdf.token
            ktdf.private_yield %buffer, %token : memref<64xf16, "L1">, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%ready) {
            ktdf.data_transfer from %input[%stick, 0] size [1, 64] to %scratch[0] size [64] : memref<3x64xf16, "DDR">, memref<64xf16, "L1">
          } {applicable_units = ["MNILU"]}
          ktdf.stage depends_in(%ready) depends_out(none) {
            %alias = memref.cast %scratch : memref<64xf16, "L1"> to memref<?xf16, "L1">
            %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
            ktdf.data_transfer from %alias[0] size [64] to %fifo size [64] : memref<?xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
          } {applicable_units = ["L1LU"]}
        }
      }
    }
    return
  }

  // Sharing a memory kind does not require synchronization when the buffers
  // are distinct. These sibling pipelines must remain independent.
  // CHECK-LABEL: func.func @independent_buffers(
  // CHECK-NOT: ktdf_lowering.signal
  // CHECK: return
  func.func @independent_buffers(%input: memref<64xf16, "DDR">,
                                 %unrelated: memref<64xf16, "L1">) attributes {grid = [2]} {
    %scratch = memref.alloc() : memref<64xf16, "L1">
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        ktdf.data_transfer from %input[0] size [64] to %scratch[0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
      } {applicable_units = ["MNILU"]}
    }
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        ktdf.data_transfer from %unrelated[0] size [64] to %fifo size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      } {applicable_units = ["L1LU"]}
    }
    return
  }
}
