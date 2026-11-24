// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --split-input-file -ktdflowering-to-dfir -verify-diagnostics

// As with SUM, an inner MAX without a matching device pattern must be rejected.

ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// expected-error @below {{failed to run operation lowerings}}
func.func @two_packets() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  ktdf_lowering.execute_on %unit {
    %input = memref.alloc() : memref<2x32xf32, "SFU_REG">
    %head = memref.subview %input[0, 0] [2, 1] [1, 1]
        : memref<2x32xf32, "SFU_REG"> to memref<2x1xf32, strided<[32, 1]>, "SFU_REG">
    // expected-error @+2 {{inner reduction must be lowered by device patterns before DataflowIR conversion}}
    // expected-error @below {{failed to legalize operation}}
    linalg.generic {
      indexing_maps = [affine_map<(r, l, o) -> (r, l)>,
                       affine_map<(r, l, o) -> (r, o)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%input : memref<2x32xf32, "SFU_REG">)
      outs(%head : memref<2x1xf32, strided<[32, 1]>, "SFU_REG">) {
    ^bb0(%value: f32, %acc: f32):
      %max = arith.maximumf %value, %acc : f32
      linalg.yield %max : f32
    }
  }
  return
}

// -----

ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// expected-error @below {{failed to run operation lowerings}}
func.func @wrong_head() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  ktdf_lowering.execute_on %unit {
    %input = memref.alloc() : memref<1x32xf32, "SFU_REG">
    %head = memref.subview %input[0, 1] [1, 1] [1, 1]
        : memref<1x32xf32, "SFU_REG"> to memref<1x1xf32, strided<[32, 1], offset: 1>, "SFU_REG">
    // expected-error @+2 {{inner reduction must be lowered by device patterns before DataflowIR conversion}}
    // expected-error @below {{failed to legalize operation}}
    linalg.generic {
      indexing_maps = [affine_map<(r, l, o) -> (r, l)>,
                       affine_map<(r, l, o) -> (r, o)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%input : memref<1x32xf32, "SFU_REG">)
      outs(%head : memref<1x1xf32, strided<[32, 1], offset: 1>, "SFU_REG">) {
    ^bb0(%value: f32, %acc: f32):
      %max = arith.maximumf %value, %acc : f32
      linalg.yield %max : f32
    }
  }
  return
}
