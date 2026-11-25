//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
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
//
//===----------------------------------------------------------------------===//

// RUN: dataflow-scheduler-opt %s --ktdflowering-to-dfir --wrap-program-dfir | FileCheck %s

// Both the HBM base and an enclosing loop's IV are supplied by the caller.
// The core-grid quotient/remainder are constants for each unit: core ids 0
// and 3 select rows row and row+5. Only the wrapper computes runtime symbols;
// no call argument or unsigned division can remain in the compiled body.
// CHECK-LABEL: module @copy
// CHECK: func.func @copy(%[[BASE:.*]]: index, %[[ROW:.*]]: index)
// CHECK-DAG: %[[FIVE:.*]] = arith.constant 5 : index
// CHECK: arith.addi %[[ROW]], %[[FIVE]]
// CHECK: symbol.create_id
// CHECK: arith.divsi
// CHECK: symbol.create_id
// CHECK-LABEL: func.func @copy_body()
// CHECK-NOT: arith.divui
// CHECK-NOT: arith.remui
// CHECK: symbol.create_symbol
// CHECK: symbol.create_symbol
// CHECK: uniform.def_immutable_mapping
// CHECK: uniform.query_map
// CHECK: dataflow.get_logical_memory_view
// CHECK-NOT: memref.reinterpret_cast

#domain = affine_set<(r, l) : (r >= 0, 15 - r >= 0, l >= 0, 63 - l >= 0)>
module {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @run(%base: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %row = %c0 to %c4 step %c1 {
        func.call @copy(%base, %row) : (index, index) -> ()
      }
      return
    }
    func.func private @copy(index, index)
  }
  module @copy {
    func.func @copy(%base: index, %row: index) attributes {grid = [2]} {
      %u0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %u1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      dataflow.program_unit iter_arg : %unit -> (%u0, %u1) : {
        %ids = uniform.def_immutable_mapping([%u0 -> %c0], [%u1 -> %c3]):index
        %id = uniform.query_map(map:%ids, key:%unit) : index
        %q = arith.divui %id, %c2 : index
        %r = arith.remui %id, %c2 : index
        %q4 = arith.muli %q, %c4 : index
        %core_row = arith.addi %q4, %r : index
        %start = arith.addi %row, %core_row : index
        %offset = arith.muli %start, %c64 : index
        %view = ktdp.construct_memory_view %base, sizes: [16, 64], strides: [64, 1] {coordinate_set = #domain, memory_space = #ktdp.memory_space<global>} : memref<16x64xf16>
        %global = memref.memory_space_cast %view : memref<16x64xf16> to memref<16x64xf16, "DDR">
        %tile = memref.reinterpret_cast %global to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<16x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
        %local = builtin.unrealized_conversion_cast %c128 : index to memref<1x64xf16, "L1">
        ktdf.data_transfer from %tile[%c0, %c0] size [1, 64] to %local[%c0, %c0] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
      }
      return
    }
  }
}
