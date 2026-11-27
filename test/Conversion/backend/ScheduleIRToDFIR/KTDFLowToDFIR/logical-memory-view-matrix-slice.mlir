// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktdflowering-to-dfir | FileCheck %s

// Slice the second batch and K of a resident [2,8,192] matrix. The slice
// retains a row stride of 192, not its Kc extent of 64.
// CHECK-DAG: #[[LAYOUT:.*]] = affine_map<(d0, d1) -> (d0 * 192 + d1)>
// CHECK-DAG: #[[DISPLACEMENT:.*]] = affine_map<(d0) -> (d0 * 64 + 1536)>
// CHECK-DAG: #[[SHIFT:.*]] = affine_map<(d0) -> (d0 * 64 + 7)>
// CHECK-LABEL: func.func @resident_slice
// CHECK: %[[BASE:.*]] = arith.constant 2048 : index
// CHECK: scf.for %[[Q:.*]] =
// CHECK: %[[OFFSET:.*]] = affine.apply #[[DISPLACEMENT]](%[[Q]])
// CHECK: %[[ADDRESS:.*]] = arith.addi %[[OFFSET]], %[[BASE]]
// CHECK: %[[SLICE:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[ADDRESS]] {layout_map = #[[LAYOUT]]} : index, index, memref<8x64xf16>
// CHECK: agen.composite_load_and_store src:%[[SLICE]]
// CHECK-NOT: memref.subview

// A constant in an existing logical layout belongs to the slice's address;
// it must not also appear in the new relative layout (double-counting it).
// CHECK-LABEL: func.func @offset_layout_slice
// CHECK: %[[BASE2:.*]] = arith.constant 2048 : index
// CHECK: scf.for %[[Q2:.*]] =
// CHECK: %[[OFFSET2:.*]] = affine.apply #[[SHIFT]](%[[Q2]])
// CHECK: %[[ADDRESS2:.*]] = arith.addi %[[OFFSET2]], %[[BASE2]]
// CHECK: %[[SLICE2:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[ADDRESS2]] {layout_map = #[[LAYOUT]]} : index, index, memref<8x64xf16>
// CHECK: agen.composite_load_and_store src:%[[SLICE2]]
// CHECK-NOT: memref.subview

#bounds = affine_set<(b,m,k) : (b >= 0, 1-b >= 0, m >= 0, 7-m >= 0, k >= 0, 191-k >= 0)>
module {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @resident_slice() attributes {grid = [1]} {
    %store = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c2048 = arith.constant 2048 : index
    %c8192 = arith.constant 8192 : index
    dataflow.program_unit iter_arg : %unit -> (%store) : {
      %full = ktdp.construct_memory_view %c2048, sizes: [2,8,192], strides: [1536,192,1] {coordinate_set = #bounds, memory_space = #ktdp.memory_space<ct_local>} : memref<2x8x192xf16, "L1">
      %out = builtin.unrealized_conversion_cast %c8192 : index to memref<8x192xf16, "DDR">
      scf.for %q = %c0 to %c3 step %c1 {
        %k = affine.apply affine_map<(q) -> (q * 64)>(%q)
        %slice = memref.subview %full[1, 0, %k] [1, 8, 64] [1, 1, 1] : memref<2x8x192xf16, "L1"> to memref<8x64xf16, strided<[192, 1], offset: ?>, "L1">
        ktdf.data_transfer from %slice[%c0, %c0] size [8,64] to %out[%c0, %k] size [8,64] {dataflow_scheduler.throttle = 64 : i64} : memref<8x64xf16, strided<[192, 1], offset: ?>, "L1">, memref<8x192xf16, "DDR">
      }
    }
    return
  }
  func.func @offset_layout_slice() attributes {grid = [1]} {
    %store = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
    %memory = dataflow.get_unit {core = 0 : i32, name = "C0-L1", type = "L1"} : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c2048 = arith.constant 2048 : index
    %c8192 = arith.constant 8192 : index
    dataflow.program_unit iter_arg : %unit -> (%store) : {
      %full = dataflow.get_logical_memory_view %memory, %c2048 {layout_map = affine_map<(m,k) -> (m * 192 + k + 7)>} : index, index, memref<8x192xf16>
      %out = builtin.unrealized_conversion_cast %c8192 : index to memref<8x192xf16, "DDR">
      scf.for %q = %c0 to %c3 step %c1 {
        %k = affine.apply affine_map<(q) -> (q * 64)>(%q)
        %slice = memref.subview %full[0, %k] [8, 64] [1, 1] : memref<8x192xf16> to memref<8x64xf16, strided<[192, 1], offset: ?>>
        ktdf.data_transfer from %slice[%c0, %c0] size [8,64] to %out[%c0, %k] size [8,64] {dataflow_scheduler.throttle = 64 : i64} : memref<8x64xf16, strided<[192, 1], offset: ?>>, memref<8x192xf16, "DDR">
      }
    }
    return
  }
}
