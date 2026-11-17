// RUN: dataflow-scheduler-opt --stage-coarsening %s | FileCheck %s

// Each group must finish both phases before its scratch is reused. Coarsening
// the outer pipeline would distribute the group loop across those phases.
// CHECK-LABEL: func.func @phase_boundary(
// CHECK: scf.for %[[GROUP:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT: ktdf.pipeline {
// CHECK: %[[SHARED:.*]]:2 = ktdf.private
// CHECK: memref.alloc() : memref<64xf16, "L1">
// CHECK: ktdf.stage depends_in(none) depends_out(%[[SHARED]]#1) {
// CHECK-NEXT: ktdf.pipeline {
// CHECK-NEXT: ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT: ktdf.data_transfer
// CHECK: ktdf.stage depends_in(%[[SHARED]]#1) depends_out(none) {
// CHECK-NEXT: ktdf.pipeline {
// CHECK-NEXT: ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT: ktdf.data_transfer
// CHECK-NOT: scf.for
// CHECK: return

// An unrelated leaf pipeline still coarsens during the same pass invocation.
// CHECK-LABEL: func.func @leaf_still_coarsens(
// CHECK: %[[VALUE:.*]] = arith.constant {{.*}} : f16
// CHECK-NEXT: ktdf.pipeline {
// CHECK-NEXT: ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT: memref.store %[[VALUE]], %{{.*}}{{\[}}%[[I]]]
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @phase_boundary(%input: memref<4x64xf16, "DDR">, %output: memref<4x64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %group = %c0 to %c4 step %c1 {
      ktdf.pipeline {
        %shared:2 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
          %scratch = memref.alloc() : memref<64xf16, "L1">
          %done = ktdf.create_token : !ktdf.token
          ktdf.private_yield %scratch, %done : memref<64xf16, "L1">, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%shared#1) {
          ktdf.pipeline {
            ktdf.stage depends_in(none) depends_out(none) {
              ktdf.data_transfer from %input[%group, %c0] size [1, 64] to %shared#0[%c0] size [64] : memref<4x64xf16, "DDR">, memref<64xf16, "L1">
            } {applicable_units = ["MNILU"]}
          }
        }
        ktdf.stage depends_in(%shared#1) depends_out(none) {
          ktdf.pipeline {
            ktdf.stage depends_in(none) depends_out(none) {
              ktdf.data_transfer from %shared#0[%c0] size [64] to %output[%group, %c0] size [1, 64] : memref<64xf16, "L1">, memref<4x64xf16, "DDR">
            } {applicable_units = ["MNISU"]}
          }
        }
      }
    }
    return
  }

  func.func @leaf_still_coarsens(%output: memref<4xf16, "SFU_REG">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %value = arith.constant 1.0 : f16
    scf.for %i = %c0 to %c4 step %c1 {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          memref.store %value, %output[%i] : memref<4xf16, "SFU_REG">
        } {applicable_units = ["SFU"]}
      }
    }
    return
  }
}
