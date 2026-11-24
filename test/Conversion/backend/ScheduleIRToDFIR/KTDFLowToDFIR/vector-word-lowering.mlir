// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktdflowering-to-dfir | FileCheck %s --implicit-check-not=vector.transfer --implicit-check-not=" vector.load " --implicit-check-not=" vector.store " --implicit-check-not=" vector.shuffle "

// FIFO tensor/vector boundaries must preserve the full word. Scratch accesses
// keep their indices, and a repeated MERGE32 mask becomes a bitwise pack.
// CHECK-LABEL: func.func @vector_words
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK: memref.alloc() : memref<8x32xi32, "SFU_REG">
// CHECK: vectorchain.constant_bitstream {value = [0x0]} : vector<1xf32>
// CHECK: vectorchain.shuffle {{.*}}repetition = 32
// CHECK: agen.vector_store {{.*}} : memref<8x32xi32, "SFU_REG">, vector<32xf32>
// CHECK: %[[IN:.*]] = dataflow.receive {{.*}} : vector<4xi32>
// CHECK: agen.vector_store %[[IN]], %{{.*}}[%[[C2]], %[[C4]]]
// CHECK: vectorchain.pack {{.*}}indices = [0 : i32, 4 : i32, 2 : i32, 6 : i32], repetition = 8 : index
// CHECK: %[[OUT:.*]] = agen.vector_load {{.*}}vector<4xi32>
// CHECK: dataflow.send %{{.*}}, %[[OUT]] : vector<4xi32>
module {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @vector_words() attributes {grid = [1]} {
    %sfu = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %lu = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %three = arith.constant 3 : index
    %four = arith.constant 4 : index
    %five = arith.constant 5 : index
    %pad = arith.constant 0 : i32
    %scratch = memref.alloc() : memref<8x32xi32, "SFU_REG">
    %data = memref.alloc() : memref<32xi32, "L1">
    %input = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 4xi32>
    %output = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 4xi32>
    ktdf_lowering.execute_on %lu {
      ktdf.data_transfer from %data[%zero] size [4] to %input size [4] : memref<32xi32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 4xi32>
    }
    ktdf_lowering.execute_on %sfu {
      %zeros = arith.constant dense<0> : vector<32xi32>
      vector.store %zeros, %scratch[%two, %zero] : memref<8x32xi32, "SFU_REG">, vector<32xi32>
      vector.store %zeros, %scratch[%three, %zero] : memref<8x32xi32, "SFU_REG">, vector<32xi32>
      %word = ktdf.read_from_fifo %input : !ktdf.fifo.slot<"L1LU" -> "SFU", 4xi32> -> tensor<4xi32>
      %value = vector.transfer_read %word[%zero], %pad {in_bounds = [true]} : tensor<4xi32>, vector<4xi32>
      vector.store %value, %scratch[%two, %four] : memref<8x32xi32, "SFU_REG">, vector<4xi32>
      %a = vector.load %scratch[%two, %zero] : memref<8x32xi32, "SFU_REG">, vector<32xi32>
      %b = vector.load %scratch[%three, %zero] : memref<8x32xi32, "SFU_REG">, vector<32xi32>
      %merge = vector.shuffle %a, %b [0, 32, 2, 34, 4, 36, 6, 38, 8, 40, 10, 42, 12, 44, 14, 46, 16, 48, 18, 50, 20, 52, 22, 54, 24, 56, 26, 58, 28, 60, 30, 62] : vector<32xi32>, vector<32xi32>
      vector.store %merge, %scratch[%five, %zero] : memref<8x32xi32, "SFU_REG">, vector<32xi32>
      %result = vector.load %scratch[%five, %zero] : memref<8x32xi32, "SFU_REG">, vector<4xi32>
      %empty = tensor.empty() : tensor<4xi32>
      %tensor = vector.transfer_write %result, %empty[%zero] {in_bounds = [true]} : vector<4xi32>, tensor<4xi32>
      ktdf.write_to_fifo %tensor, %output : tensor<4xi32>, <"SFU" -> "L1LU", 4xi32>
    }
    ktdf_lowering.execute_on %lu {
      ktdf.data_transfer from %output size [4] to %data[%zero] size [4] : !ktdf.fifo.slot<"SFU" -> "L1LU", 4xi32>, memref<32xi32, "L1">
    }
    return
  }
}
