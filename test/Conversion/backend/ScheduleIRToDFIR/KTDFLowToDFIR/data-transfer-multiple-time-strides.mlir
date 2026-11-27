// RUN: dataflow-scheduler-opt -ktdflowering-to-dfir %s | FileCheck %s

// Both walked axes have different source and destination strides. Preserve
// their ordered extents as two enclosing loops; each vector stays 64 lanes.
// CHECK-LABEL: func.func @two_divergent_walked_dimensions()
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<4x8x64xf16, "DDR">
// CHECK: %[[DST:.*]] = memref.alloc() : memref<4x8x128xf16, "L1">
// CHECK: scf.for %[[OUTER:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK: scf.for %[[INNER:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK: agen.composite_load_and_store src:%[[SRC]]{{\[}}%[[OUTER]], %[[INNER]], 0] dst:%[[DST]]{{\[}}%[[OUTER]], %[[INNER]], 0]
// CHECK: load_iv(%{{.*}}:vector<64xf16>)

// Copyright 2026 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @two_divergent_walked_dimensions() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    // Per step of the outer walked dim: 8*64 = 512 elements in the source,
    // 8*128 = 1024 in the destination. Per step of the inner one: 64 and 128.
    %ddr_buf = memref.alloc() : memref<4x8x64xf16, "DDR">
    %l1_buf = memref.alloc() : memref<4x8x128xf16, "L1">
    ktdf_lowering.execute_on %unit {
      ktdf.data_transfer from %ddr_buf[0, 0, 0] size [4, 8, 64]
                         to %l1_buf[0, 0, 0] size [4, 8, 64]
        {dataflow_scheduler.throttle = 64 : i64}
        : memref<4x8x64xf16, "DDR">, memref<4x8x128xf16, "L1">
    }
    return
  }
}
