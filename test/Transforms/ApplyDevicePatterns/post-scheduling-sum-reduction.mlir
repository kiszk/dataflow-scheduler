// RUN: dataflow-scheduler-opt %s --split-input-file -allow-unregistered-dialect -apply-device-patterns='groups=post_scheduling' | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=memref.subview --implicit-check-not=ktdf.opaque

// The sample device selects a precision-specific test op and passes the whole
// accumulator register rather than the one-element output subview.

// CHECK-LABEL: func.func @sum_reduction_f16()
// CHECK: %[[REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT: "test.simd_reduction_sum"(%[[REG]], %[[REG]]) : (memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">) -> ()
// CHECK-NEXT: return
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @sum_reduction_f16() {
    %input = memref.alloc() : memref<64xf16, "SFU_REG">
    %head = memref.subview %input[0] [1] [1]
        : memref<64xf16, "SFU_REG"> to memref<1xf16, "SFU_REG">
    linalg.generic {
      indexing_maps = [affine_map<(l, o) -> (l)>, affine_map<(l, o) -> (o)>],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : memref<64xf16, "SFU_REG">)
      outs(%head : memref<1xf16, "SFU_REG">) {
    ^bb0(%value: f16, %acc: f16):
      %sum = arith.addf %value, %acc : f16
      linalg.yield %sum : f16
    }
    return
  }
}

// -----

// CHECK-LABEL: func.func @sum_reduction_f32()
// CHECK: %[[REG:.*]] = memref.alloc() : memref<32xf32, "SFU_REG">
// CHECK-NEXT: "test.simd_reduction_sum_fp32"(%[[REG]], %[[REG]]) : (memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">) -> ()
// CHECK-NEXT: return
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @sum_reduction_f32() {
    %input = memref.alloc() : memref<32xf32, "SFU_REG">
    %head = memref.subview %input[0] [1] [1]
        : memref<32xf32, "SFU_REG"> to memref<1xf32, "SFU_REG">
    linalg.generic {
      indexing_maps = [affine_map<(l, o) -> (l)>, affine_map<(l, o) -> (o)>],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : memref<32xf32, "SFU_REG">)
      outs(%head : memref<1xf32, "SFU_REG">) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    }
    return
  }
}
