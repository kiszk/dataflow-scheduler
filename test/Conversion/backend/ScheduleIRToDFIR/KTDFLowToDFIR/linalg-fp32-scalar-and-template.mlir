// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -ktdflowering-to-dfir | FileCheck %s --implicit-check-not=linalg.generic

// Scalar immediates must become vectors on either side of a binary operation.
// A register-based template in a buffer generic must stay between its input
// store and output load, just as it does in the tensor-semantics lowering.
ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL: func.func @constant_left
// CHECK: %[[C:.*]] = arith.constant dense<2.000000e+00> : vector<32xf32>
// CHECK: dataflow.program_unit
// CHECK: dataflow.program_unit
// CHECK: %[[IN:.*]] = dataflow.receive
// CHECK: vectorchain.binary %[[C]], %[[IN]]
// CHECK-SAME: binary_operator sub
func.func @constant_left() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  %load = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
  %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 32xf32>
  ktdf_lowering.execute_on %load {
    %zero = arith.constant 0 : index
    %storage = memref.alloc() : memref<32xf32, "L1">
    ktdf.data_transfer from %storage[%zero] size [32] to %fifo size [32]
      : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    ktdf.data_transfer from %out_fifo size [32] to %storage[%zero] size [32]
      : !ktdf.fifo.slot<"SFU" -> "L1LU", 32xf32>, memref<32xf32, "L1">
  }
  ktdf_lowering.execute_on %unit {
    %x = ktdf.read_from_fifo %fifo : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
    %empty = tensor.empty() : tensor<32xf32>
    %c = arith.constant 2.0 : f32
    %result = linalg.generic {
      indexing_maps = [affine_map<(d) -> (d)>, affine_map<(d) -> (d)>],
      iterator_types = ["parallel"]
    } ins(%x : tensor<32xf32>) outs(%empty : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %product = arith.subf %c, %in : f32
      linalg.yield %product : f32
    } -> tensor<32xf32>
    ktdf.write_to_fifo %result, %out_fifo : tensor<32xf32>, <"SFU" -> "L1LU", 32xf32>
  }
  return
}

// CHECK-LABEL: func.func @constant_right
// CHECK: %[[C:.*]] = arith.constant dense<2.000000e+00> : vector<32xf32>
// CHECK: dataflow.program_unit
// CHECK: dataflow.program_unit
// CHECK: %[[IN:.*]] = dataflow.receive
// CHECK: vectorchain.binary %[[IN]], %[[C]]
// CHECK-SAME: binary_operator sub
func.func @constant_right() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  %load = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
  %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 32xf32>
  ktdf_lowering.execute_on %load {
    %zero = arith.constant 0 : index
    %storage = memref.alloc() : memref<32xf32, "L1">
    ktdf.data_transfer from %storage[%zero] size [32] to %fifo size [32]
      : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    ktdf.data_transfer from %out_fifo size [32] to %storage[%zero] size [32]
      : !ktdf.fifo.slot<"SFU" -> "L1LU", 32xf32>, memref<32xf32, "L1">
  }
  ktdf_lowering.execute_on %unit {
    %x = ktdf.read_from_fifo %fifo : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
    %empty = tensor.empty() : tensor<32xf32>
    %c = arith.constant 2.0 : f32
    %result = linalg.generic {
      indexing_maps = [affine_map<(d) -> (d)>, affine_map<(d) -> (d)>],
      iterator_types = ["parallel"]
    } ins(%x : tensor<32xf32>) outs(%empty : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %product = arith.subf %in, %c : f32
      linalg.yield %product : f32
    } -> tensor<32xf32>
    ktdf.write_to_fifo %result, %out_fifo : tensor<32xf32>, <"SFU" -> "L1LU", 32xf32>
  }
  return
}

// CHECK-LABEL: func.func @buffer_template
// CHECK: agen.vector_store
// CHECK: dataflow.opaque
// CHECK-SAME: func_name = "register_template"
// CHECK: agen.vector_load
// CHECK: agen.vector_store
func.func @buffer_template() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  ktdf_lowering.execute_on %unit {
    %x = memref.alloc() : memref<32xf32, "SFU_REG">
    %y = memref.alloc() : memref<32xf32, "SFU_REG">
    %tmp = memref.alloc() : memref<32xf32, "SFU_REG">
    %zero = arith.constant 0 : index
    linalg.generic {
      indexing_maps = [affine_map<(d) -> (d)>, affine_map<(d) -> (d)>],
      iterator_types = ["parallel"]
    } ins(%x : memref<32xf32, "SFU_REG">) outs(%y : memref<32xf32, "SFU_REG">) {
    ^bb0(%in: f32, %out: f32):
      memref.store %in, %tmp[%zero] : memref<32xf32, "SFU_REG">
      dataflow.opaque {func_name = "register_template", parameter_dictionary = {}, read_only_register_dictionary = {}, read_write_register_dictionary = {}}
      %result = memref.load %tmp[%zero] : memref<32xf32, "SFU_REG">
      linalg.yield %result : f32
    }
  }
  return
}
