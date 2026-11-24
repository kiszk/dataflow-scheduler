// SPDX-License-Identifier: Apache-2.0
// RUN: not dataflow-scheduler-opt %s --ktdflowering-to-dfir 2>&1 | FileCheck %s

// An arbitrary shuffle is not a hardware MERGE. Reject it rather than leaking
// an unsupported vector operation or inventing a different pack permutation.
// CHECK: failed to legalize operation 'vector.shuffle'
module {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @unsupported_shuffle() attributes {grid = [1]} {
    %sfu = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %data = memref.alloc() : memref<2x32xf32, "SFU_REG">
    ktdf_lowering.execute_on %sfu {
      %a = vector.load %data[%zero, %zero] : memref<2x32xf32, "SFU_REG">, vector<32xf32>
      %b = vector.load %data[%one, %zero] : memref<2x32xf32, "SFU_REG">, vector<32xf32>
      %shuffle = vector.shuffle %a, %b [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] : vector<32xf32>, vector<32xf32>
      vector.store %shuffle, %data[%zero, %zero] : memref<2x32xf32, "SFU_REG">, vector<32xf32>
    }
    return
  }
}
