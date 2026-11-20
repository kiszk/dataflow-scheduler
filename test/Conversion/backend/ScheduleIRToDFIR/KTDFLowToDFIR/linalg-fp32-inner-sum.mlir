// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(apply-device-patterns{groups=post_scheduling}),materialize-registers,address-assignment,ktdflowering-to-dfir)' | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=ktdf.opaque --implicit-check-not=vectorchain.scan_with_gap

// Both precisions use the device pattern and common opaque lowering, including
// allocation of the in-place accumulator and its physical register dictionaries.

// CHECK-LABEL: func.func @inner_sum_f16()
// CHECK: dataflow.opaque {dataflow_scheduler.register_names = ["t0", "t0"], func_name = "simdreduction_sum", parameter_dictionary = {unroll = "1"}, read_only_register_dictionary = {}, read_write_register_dictionary = {t0 = "R0"}}
ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")

func.func @inner_sum_f16() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  ktdf_lowering.execute_on %unit {
    %input = memref.alloc() : memref<1x64xf16, "SFU_REG">
    %head = memref.subview %input[0, 0] [1, 1] [1, 1]
        : memref<1x64xf16, "SFU_REG"> to memref<1x1xf16, strided<[64, 1]>, "SFU_REG">
    linalg.generic {
      indexing_maps = [affine_map<(r, l, o) -> (r, l)>,
                       affine_map<(r, l, o) -> (r, o)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%input : memref<1x64xf16, "SFU_REG">)
      outs(%head : memref<1x1xf16, strided<[64, 1]>, "SFU_REG">) {
    ^bb0(%value: f16, %acc: f16):
      %sum = arith.addf %value, %acc : f16
      linalg.yield %sum : f16
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @inner_sum_f32()
// CHECK: dataflow.opaque {dataflow_scheduler.register_names = ["t0", "t0"], func_name = "simdreduction_sum_fp32", parameter_dictionary = {unroll = "1"}, read_only_register_dictionary = {}, read_write_register_dictionary = {t0 = "R0"}}
ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")

func.func @inner_sum_f32() attributes {grid = [1]} {
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
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    }
  }
  return
}
