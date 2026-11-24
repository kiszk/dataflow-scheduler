// RUN: dataflow-scheduler-opt --ktdf-to-ktdflowering %s | FileCheck %s

// The first signal publishes scratch to its reader. The second signal waits
// for that reader before the next group can overwrite the same allocation.
// Both occur inside the group loop, with completion after the final phase.
// Two aliases of a double-buffer selection still name the same storage pair;
// emit one completion signal, without the unrelated compute unit.
// CHECK-LABEL: func.func @phase_completion(
// CHECK: %[[WRITER:.*]] = uniform.query_map
// CHECK: %[[READER:.*]] = uniform.query_map
// CHECK: %[[COMPUTE:.*]] = uniform.query_map
// CHECK: scf.for
// CHECK: ktdf.data_transfer from %{{.*}} to %{{.*}}
// CHECK: ktdf_lowering.signal %[[WRITER]], %[[READER]]{{$}}
// CHECK: ktdf.data_transfer from %{{.*}} to %{{.*}}
// CHECK: ktdf_lowering.signal %[[WRITER]], %[[READER]]{{$}}
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: return

// A leaf pipeline continues to use only its ordinary RAW synchronization.
// CHECK-LABEL: func.func @leaf_pipeline(
// CHECK: ktdf_lowering.signal
// CHECK-NOT: ktdf_lowering.signal
// CHECK: return
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @phase_completion(%input: memref<4x64xf16, "DDR">, %output: memref<4x64xf16, "DDR">) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %group = %c0 to %c4 step %c1 {
      %phase = ktdf.buffer_phase(%group) {num_phases = 2 : i64} : index
      ktdf.pipeline modulo(size: 2) {
        %shared:3 = ktdf.private -> (memref<64xf16, "L1">, memref<64xf16, "L1">, !ktdf.token) {
          %scratch0 = memref.alloc() : memref<64xf16, "L1">
          %scratch1 = memref.alloc() : memref<64xf16, "L1">
          %done = ktdf.create_token : !ktdf.token
          ktdf.private_yield %scratch0, %scratch1, %done : memref<64xf16, "L1">, memref<64xf16, "L1">, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%shared#2) {
          ktdf.pipeline {
            ktdf.stage depends_in(none) depends_out(none) {
              %selected = ktdf.select_memref %phase[%shared#0, %shared#1] : memref<64xf16, "L1">
              ktdf.data_transfer from %input[%group, %c0] size [1, 64] to %selected[%c0] size [64] : memref<4x64xf16, "DDR">, memref<64xf16, "L1">
            } {applicable_units = ["MNILU"]}
          }
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%shared#2) depends_out(none) {
          ktdf.pipeline {
            ktdf.stage depends_in(none) depends_out(none) {
              %selected = ktdf.select_memref %phase[%shared#0, %shared#1] : memref<64xf16, "L1">
              %alias = memref.cast %selected : memref<64xf16, "L1"> to memref<?xf16, "L1">
              ktdf.data_transfer from %alias[%c0] size [64] to %output[%group, %c0] size [1, 64] : memref<?xf16, "L1">, memref<4x64xf16, "DDR">
            } {applicable_units = ["MNISU"]}
            ktdf.stage depends_in(none) depends_out(none) {
              %value = arith.constant 1.0 : f16
            } {applicable_units = ["SFU"]}
          }
        } {applicable_units = ["MNISU", "SFU"]}
      }
    }
    return
  }

  func.func @leaf_pipeline(%input: memref<64xf16, "DDR">, %output: memref<64xf16, "DDR">) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      %shared:2 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
        %scratch = memref.alloc() : memref<64xf16, "L1">
        %done = ktdf.create_token : !ktdf.token
        ktdf.private_yield %scratch, %done : memref<64xf16, "L1">, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%shared#1) {
        ktdf.data_transfer from %input[%c0] size [64] to %shared#0[%c0] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
      } {applicable_units = ["MNILU"]}
      ktdf.stage depends_in(%shared#1) depends_out(none) {
        ktdf.data_transfer from %shared#0[%c0] size [64] to %output[%c0] size [64] : memref<64xf16, "L1">, memref<64xf16, "DDR">
      } {applicable_units = ["MNISU"]}
    }
    return
  }
}
