// SPDX-License-Identifier: Apache-2.0
// RUN: not dataflow-scheduler-opt %s --split-input-file -ktdflowering-to-dfir 2>&1 | FileCheck %s

// Without a matching device pattern, no inner reduction may fall through to
// elementwise lowering, even if the scalar body is otherwise supported.
// A partial register cannot use the full-register SUM template.
// CHECK: error: inner reduction must be lowered by device patterns before DataflowIR conversion
ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")

func.func @inner_sum() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  ktdf_lowering.execute_on %unit {
    %input = memref.alloc() : memref<1x16xf32, "SFU_REG">
    %head = memref.subview %input[0, 0] [1, 1] [1, 1]
        : memref<1x16xf32, "SFU_REG"> to memref<1x1xf32, strided<[16, 1]>, "SFU_REG">
    linalg.generic {
      indexing_maps = [affine_map<(r, l, o) -> (r, l)>,
                       affine_map<(r, l, o) -> (r, o)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%input : memref<1x16xf32, "SFU_REG">)
      outs(%head : memref<1x1xf32, strided<[16, 1]>, "SFU_REG">) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    }
  }
  return
}

// -----

// A subview at a nonzero lane is not the reduction destination.
// CHECK: error: inner reduction must be lowered by device patterns before DataflowIR conversion
ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")

func.func @inner_sum() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  ktdf_lowering.execute_on %unit {
    %input = memref.alloc() : memref<1x32xf32, "SFU_REG">
    %head = memref.subview %input[0, 1] [1, 1] [1, 1]
        : memref<1x32xf32, "SFU_REG"> to memref<1x1xf32, strided<[32, 1], offset: 1>, "SFU_REG">
    linalg.generic {
      indexing_maps = [affine_map<(r, l, o) -> (r, l)>,
                       affine_map<(r, l, o) -> (r, o)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%input : memref<1x32xf32, "SFU_REG">)
      outs(%head : memref<1x1xf32, strided<[32, 1], offset: 1>, "SFU_REG">) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    }
  }
  return
}

// -----

// An unsupported reduction must not become an elementwise multiply.
// CHECK: error: inner reduction must be lowered by device patterns before DataflowIR conversion
ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")

func.func @inner_sum() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  ktdf_lowering.execute_on %unit {
    %input = memref.alloc() : memref<1x32xf32, "SFU_REG">
    %head = memref.subview %input[0, 0] [1, 1] [1, 1]
        : memref<1x32xf32, "SFU_REG"> to memref<1x1xf32, strided<[32, 1]>, "SFU_REG">
    linalg.generic {
      indexing_maps = [affine_map<(r, l, o) -> (r, l)>,
                       affine_map<(r, l, o) -> (r, o)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%input : memref<1x32xf32, "SFU_REG">)
      outs(%head : memref<1x1xf32, strided<[32, 1]>, "SFU_REG">) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.mulf %value, %acc : f32
      linalg.yield %sum : f32
    }
  }
  return
}

// -----

// A reduction left on a load-only unit must also be rejected.
// CHECK: error: inner reduction must be lowered by device patterns before DataflowIR conversion
ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")

func.func @inner_sum() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
  ktdf_lowering.execute_on %unit {
    %input = memref.alloc() : memref<1x32xf32, "SFU_REG">
    %head = memref.subview %input[0, 0] [1, 1] [1, 1]
        : memref<1x32xf32, "SFU_REG"> to memref<1x1xf32, strided<[32, 1]>, "SFU_REG">
    linalg.generic {
      indexing_maps = [affine_map<(r, l, o) -> (r, l)>,
                       affine_map<(r, l, o) -> (r, o)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%input : memref<1x32xf32, "SFU_REG">)
      outs(%head : memref<1x1xf32, strided<[32, 1]>, "SFU_REG">) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    }
  }
  return
}
