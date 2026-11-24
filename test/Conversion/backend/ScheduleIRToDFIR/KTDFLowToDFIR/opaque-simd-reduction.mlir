// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --split-input-file -pass-pipeline='builtin.module(materialize-registers,address-assignment,ktdflowering-to-dfir)' | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=ktdf.opaque --implicit-check-not=vectorchain.scan_with_gap

// SUM and MAX in both precisions use common opaque lowering, including
// allocation of the in-place accumulator and its physical register dictionaries.

// CHECK-LABEL: func.func @inner_sum_f16()
// CHECK: dataflow.opaque {dataflow_scheduler.register_names = ["t0", "t0"], func_name = "simdreduction_sum", parameter_dictionary = {unroll = "1"}, read_only_register_dictionary = {}, read_write_register_dictionary = {t0 = "R0"}}
ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

func.func @inner_sum_f16() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  ktdf_lowering.execute_on %unit {
    %input = memref.alloc() : memref<1x64xf16, "SFU_REG">
    ktdf.opaque "simdreduction_sum" {
      dataflow_scheduler.register_names = ["t0", "t0"],
      func_name = "simdreduction_sum", parameter_dictionary = {unroll = "1"}
    } ins(%input : memref<1x64xf16, "SFU_REG">)
      outs(%input : memref<1x64xf16, "SFU_REG">)
  }
  return
}

// -----

// CHECK-LABEL: func.func @inner_sum_f32()
// CHECK: dataflow.opaque {dataflow_scheduler.register_names = ["t0", "t0"], func_name = "simdreduction_sum_fp32", parameter_dictionary = {unroll = "1"}, read_only_register_dictionary = {}, read_write_register_dictionary = {t0 = "R0"}}
ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

func.func @inner_sum_f32() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  ktdf_lowering.execute_on %unit {
    %input = memref.alloc() : memref<1x32xf32, "SFU_REG">
    ktdf.opaque "simdreduction_sum_fp32" {
      dataflow_scheduler.register_names = ["t0", "t0"],
      func_name = "simdreduction_sum_fp32", parameter_dictionary = {unroll = "1"}
    } ins(%input : memref<1x32xf32, "SFU_REG">)
      outs(%input : memref<1x32xf32, "SFU_REG">)
  }
  return
}

// -----

// CHECK-LABEL: func.func @inner_max_f16()
// CHECK: dataflow.opaque {dataflow_scheduler.register_names = ["t0", "t0"], func_name = "simdreduction_minmax", parameter_dictionary = {unroll = "1", x1 = "0", x2 = "8"}, read_only_register_dictionary = {}, read_write_register_dictionary = {t0 = "R0"}}
ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

func.func @inner_max_f16() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  ktdf_lowering.execute_on %unit {
    %input = memref.alloc() : memref<1x64xf16, "SFU_REG">
    ktdf.opaque "simdreduction_minmax" {
      dataflow_scheduler.register_names = ["t0", "t0"],
      func_name = "simdreduction_minmax", parameter_dictionary = {unroll = "1", x1 = "0", x2 = "8"}
    } ins(%input : memref<1x64xf16, "SFU_REG">)
      outs(%input : memref<1x64xf16, "SFU_REG">)
  }
  return
}

// -----

// CHECK-LABEL: func.func @inner_max_f32()
// CHECK: dataflow.opaque {dataflow_scheduler.register_names = ["t0", "t0"], func_name = "simdreduction_minmax_fp32", parameter_dictionary = {unroll = "1", x1 = "0", x2 = "8"}, read_only_register_dictionary = {}, read_write_register_dictionary = {t0 = "R0"}}
ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

func.func @inner_max_f32() attributes {grid = [1]} {
  %unit = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  ktdf_lowering.execute_on %unit {
    %input = memref.alloc() : memref<1x32xf32, "SFU_REG">
    ktdf.opaque "simdreduction_minmax_fp32" {
      dataflow_scheduler.register_names = ["t0", "t0"],
      func_name = "simdreduction_minmax_fp32", parameter_dictionary = {unroll = "1", x1 = "0", x2 = "8"}
    } ins(%input : memref<1x32xf32, "SFU_REG">)
      outs(%input : memref<1x32xf32, "SFU_REG">)
  }
  return
}
