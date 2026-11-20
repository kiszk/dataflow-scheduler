// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktdflowering-to-dfir --split-input-file | FileCheck %s --implicit-check-not=ktdf.data_transfer --implicit-check-not=ktdf.read_from_fifo --implicit-check-not=vectorchain.cast
// Bit-preserving four-element copies for both directions of a packet pair.
// Scratch allocation/lifetime is assumed fixed by the preceding KTDF passes.

// CHECK-LABEL: func.func @staggered_to_standard
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[TWO:.*]] = arith.constant 2 : index
// CHECK: %[[FOUR:.*]] = arith.constant 4 : index
// CHECK: scf.for %[[LH:.*]] = %[[ZERO]] to %[[TWO]] step %[[ONE]]
// CHECK: scf.for %[[LG:.*]] = %[[ZERO]] to %[[FOUR]] step %[[ONE]]
// CHECK: scf.for %[[LP:.*]] = %[[ZERO]] to %[[TWO]] step %[[ONE]]
// CHECK: %[[LOAD:.*]] = agen.vector_load %{{.*}}[%[[LP]], %[[LH]] * 16 + %[[LG]] * 4]
// CHECK-SAME: vector<4xf32>
// CHECK: dataflow.send %{{.*}}, %[[LOAD]] : vector<4xf32>
// CHECK: %[[COPY:.*]] = dataflow.receive %{{.*}} : vector<4xf32>
// CHECK: dataflow.send %{{.*}}, %[[COPY]] : vector<4xf32>
// CHECK: scf.for %[[SH:.*]] = %[[ZERO]] to %[[TWO]] step %[[ONE]]
// CHECK: scf.for %[[SG:.*]] = %[[ZERO]] to %[[FOUR]] step %[[ONE]]
// CHECK: scf.for %[[SP:.*]] = %[[ZERO]] to %[[TWO]] step %[[ONE]]
// CHECK: %[[STORE:.*]] = dataflow.receive %{{.*}} : vector<4xf32>
// CHECK: agen.vector_store %[[STORE]], %{{.*}}[%[[SH]], %[[SG]] * 8 + %[[SP]] * 4]
// CHECK-SAME: vector<4xf32>

!in = !ktdf.fifo.slot<"L1LU" -> "SFU", 4xf32>
!out = !ktdf.fifo.slot<"SFU" -> "L1SU", 4xf32>
module {
  ktdf_arch.device @sample_device import("Inputs/device.mlir")
  func.func @staggered_to_standard(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %four = arith.constant 4 : index
    %src = builtin.unrealized_conversion_cast %src_addr : index to memref<2x32xf32, "L1">
    %dst = builtin.unrealized_conversion_cast %dst_addr : index to memref<2x32xf32, "L1">
    %load = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %copy = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %store = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
    %input = ktdf.fifo.allocate() -> !in
    %output = ktdf.fifo.allocate() -> !out
    ktdf_lowering.execute_on %load {
      scf.for %h = %zero to %two step %one {
        scf.for %g = %zero to %four step %one {
          scf.for %p = %zero to %two step %one {
            ktdf.data_transfer from %src[%p, %h * 16 + %g * 4] size [1, 4] to %input size [4]
              : memref<2x32xf32, "L1">, !in
          }
        }
      }
    }
    ktdf_lowering.execute_on %copy {
      scf.for %h = %zero to %two step %one {
        scf.for %g = %zero to %four step %one {
          scf.for %p = %zero to %two step %one {
            %value = ktdf.read_from_fifo %input : !in -> tensor<4xf32>
            ktdf.write_to_fifo %value, %output : tensor<4xf32>, !out
          }
        }
      }
    }
    ktdf_lowering.execute_on %store {
      scf.for %h = %zero to %two step %one {
        scf.for %g = %zero to %four step %one {
          scf.for %p = %zero to %two step %one {
            ktdf.data_transfer from %output size [4] to %dst[%h, %g * 8 + %p * 4] size [1, 4]
              : !out, memref<2x32xf32, "L1">
          }
        }
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @standard_to_staggered
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[TWO:.*]] = arith.constant 2 : index
// CHECK: %[[FOUR:.*]] = arith.constant 4 : index
// CHECK: scf.for %[[LH:.*]] = %[[ZERO]] to %[[TWO]] step %[[ONE]]
// CHECK: scf.for %[[LG:.*]] = %[[ZERO]] to %[[FOUR]] step %[[ONE]]
// CHECK: scf.for %[[LP:.*]] = %[[ZERO]] to %[[TWO]] step %[[ONE]]
// CHECK: %[[LOAD:.*]] = agen.vector_load %{{.*}}[%[[LH]], %[[LG]] * 8 + %[[LP]] * 4]
// CHECK-SAME: vector<4xf32>
// CHECK: dataflow.send %{{.*}}, %[[LOAD]] : vector<4xf32>
// CHECK: %[[COPY:.*]] = dataflow.receive %{{.*}} : vector<4xf32>
// CHECK: dataflow.send %{{.*}}, %[[COPY]] : vector<4xf32>
// CHECK: scf.for %[[SH:.*]] = %[[ZERO]] to %[[TWO]] step %[[ONE]]
// CHECK: scf.for %[[SG:.*]] = %[[ZERO]] to %[[FOUR]] step %[[ONE]]
// CHECK: scf.for %[[SP:.*]] = %[[ZERO]] to %[[TWO]] step %[[ONE]]
// CHECK: %[[STORE:.*]] = dataflow.receive %{{.*}} : vector<4xf32>
// CHECK: agen.vector_store %[[STORE]], %{{.*}}[%[[SP]], %[[SH]] * 16 + %[[SG]] * 4]
// CHECK-SAME: vector<4xf32>

!in = !ktdf.fifo.slot<"L1LU" -> "SFU", 4xf32>
!out = !ktdf.fifo.slot<"SFU" -> "L1SU", 4xf32>
module {
  ktdf_arch.device @sample_device import("Inputs/device.mlir")
  func.func @standard_to_staggered(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %four = arith.constant 4 : index
    %src = builtin.unrealized_conversion_cast %src_addr : index to memref<2x32xf32, "L1">
    %dst = builtin.unrealized_conversion_cast %dst_addr : index to memref<2x32xf32, "L1">
    %load = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %copy = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %store = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
    %input = ktdf.fifo.allocate() -> !in
    %output = ktdf.fifo.allocate() -> !out
    ktdf_lowering.execute_on %load {
      scf.for %h = %zero to %two step %one {
        scf.for %g = %zero to %four step %one {
          scf.for %p = %zero to %two step %one {
            ktdf.data_transfer from %src[%h, %g * 8 + %p * 4] size [1, 4] to %input size [4]
              : memref<2x32xf32, "L1">, !in
          }
        }
      }
    }
    ktdf_lowering.execute_on %copy {
      scf.for %h = %zero to %two step %one {
        scf.for %g = %zero to %four step %one {
          scf.for %p = %zero to %two step %one {
            %value = ktdf.read_from_fifo %input : !in -> tensor<4xf32>
            ktdf.write_to_fifo %value, %output : tensor<4xf32>, !out
          }
        }
      }
    }
    ktdf_lowering.execute_on %store {
      scf.for %h = %zero to %two step %one {
        scf.for %g = %zero to %four step %one {
          scf.for %p = %zero to %two step %one {
            ktdf.data_transfer from %output size [4] to %dst[%p, %h * 16 + %g * 4] size [1, 4]
              : !out, memref<2x32xf32, "L1">
          }
        }
      }
    }
    return
  }
}
