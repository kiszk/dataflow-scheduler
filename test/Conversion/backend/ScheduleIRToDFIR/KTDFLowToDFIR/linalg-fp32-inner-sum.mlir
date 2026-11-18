// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -ktdflowering-to-dfir | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=memref.subview --implicit-check-not=vectorchain.cast

// CHECK-LABEL: func.func @inner_sum()
// CHECK: %[[ONES:.*]] = arith.constant dense<1.000000e+00> : vector<32xf32>
// CHECK: %[[REG:.*]] = memref.alloc() : memref<1x32xf32, "SFU_REG">
// CHECK: %[[INPUT:.*]] = agen.vector_load %[[REG]]{{.*}}vector<32xf32>
// CHECK: %[[NEIGHBOR:.*]] = vectorchain.shuffle {{.*}}indices = [1 : i32, 0 : i32, 3 : i32, 3 : i32], repetition = 8 : i32
// CHECK: %[[PLAIN:.*]] = agen.vector_load %[[REG]]{{.*}}vector<32xf32>
// CHECK: %[[PAIR:.*]] = vectorchain.multiply_and_accumulate %[[NEIGHBOR]], %[[ONES]], %[[PLAIN]] {{.*}}vector<32xf32>
// CHECK: agen.vector_store %[[PAIR]], %[[REG]]
// CHECK: %[[PAIRS:.*]] = agen.vector_load %[[REG]]{{.*}}vector<32xf32>
// CHECK: %[[NEIGHBOR2:.*]] = vectorchain.shuffle {{.*}}indices = [2 : i32, 0 : i32, 1 : i32, 3 : i32], repetition = 8 : i32
// CHECK: %[[PLAIN2:.*]] = agen.vector_load %[[REG]]{{.*}}vector<32xf32>
// CHECK: %[[GROUP:.*]] = vectorchain.multiply_and_accumulate %[[PLAIN2]], %[[ONES]], %[[NEIGHBOR2]] {{.*}}vector<32xf32>
// CHECK: agen.vector_store %[[GROUP]], %[[REG]]
// CHECK: agen.vector_load %[[REG]]{{.*}}vector<32xf32>
// CHECK: %[[SUM:.*]] = vectorchain.scan_with_gap {{.*}}gap = 8 : index, reduction_op = #vectorchain<binary_operator add>{{.*}}vector<32xf32>, vector<32xf32>
// CHECK: agen.vector_store %[[SUM]], %[[REG]]

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
      %sum = arith.addf %value, %acc : f32
      linalg.yield %sum : f32
    }
  }
  return
}
