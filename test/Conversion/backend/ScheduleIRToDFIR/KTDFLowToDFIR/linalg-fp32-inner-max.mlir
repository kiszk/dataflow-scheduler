// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -ktdflowering-to-dfir | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=dataflow.opaque

// FP32 MAX must fold four lanes per sub-SIMD group, then use FP32 REDUCE.
// CHECK-LABEL: func.func @inner_max_f32()
// CHECK: vectorchain.shuffle {{.*}}indices = [1 : i32, 0 : i32, 3 : i32, 3 : i32]{{.*}}repetition = 8
// CHECK: vectorchain.binary {{.*}}binary_op = #vectorchain<binary_operator max>
// CHECK: vectorchain.shuffle {{.*}}indices = [2 : i32, 0 : i32, 1 : i32, 3 : i32]{{.*}}repetition = 8
// CHECK: vectorchain.binary {{.*}}binary_op = #vectorchain<binary_operator max>
// CHECK: vectorchain.scan_with_gap {{.*}}gap = 8 : index{{.*}}reduction_op = #vectorchain<binary_operator max>
// CHECK: agen.vector_store {{.*}}vector<32xf32>
ktdf_arch.device @sample_device import("../../../../dataflow-scheduler-opt/Inputs/access-order-device.mlir")

func.func @inner_max_f32() attributes {grid = [1]} {
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
      %max = arith.maximumf %value, %acc : f32
      linalg.yield %max : f32
    }
  }
  return
}
