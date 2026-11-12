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

// RUN: dataflow-scheduler-opt -ktdflowering-to-dfir %s > %t
// RUN: FileCheck %s --check-prefix=UNITS < %t
// RUN: FileCheck %s --check-prefix=WORDS < %t
// RUN: sed '/^      %logical =/,+2{H;d;}; /^      %canonical =/G; s@../../../../Dialect/KTDFArch/sample_device.mlir@%S/../../../../Dialect/KTDFArch/sample_device.mlir@' %s | sed '/^      %local0 =/{h;d;}; /^      %local1 =/G' > %t.reverse.mlir
// RUN: dataflow-scheduler-opt -ktdflowering-to-dfir %t.reverse.mlir > %t.reverse.out
// RUN: FileCheck %s --check-prefix=UNITS < %t.reverse.out
// RUN: FileCheck %s --check-prefix=WORDS < %t.reverse.out

// Logical and canonical memory spaces share the same hardware unit. The
// symbolic global address must use the device's 64-byte word size for both.
// Repeat with canonical definitions first to check order-independent unit tags.
// UNITS-LABEL: func.func @aliases(
// UNITS: %[[GLOBAL:.*]] = dataflow.get_unit {name = "hbm", type = "hbm"}
// UNITS-NOT: dataflow.get_unit
// UNITS: dataflow.get_unit {core = 0 : i32, name = "C0-lx", type = "lx"}
// UNITS-NOT: dataflow.get_unit
// UNITS: dataflow.program_unit
// UNITS: dataflow.get_logical_memory_view %[[GLOBAL]],
// UNITS: dataflow.get_logical_memory_view %[[GLOBAL]],
// WORDS-LABEL: func.func @aliases(
// WORDS-SAME: %[[BASE:.*]]: index)
// WORDS-DAG: %[[WORD_SIZE:.*]] = arith.constant 64 : index
// WORDS-DAG: arith.divsi %[[BASE]], %[[WORD_SIZE]] : index

#stick = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {
    mem_space_mapping = #ktdf_arch.map<
      #ktdp.memory_space<global> = "DDR",
      #ktdp.memory_space<ct_local> = "L1"
    >
  } import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @aliases(%base: index) attributes {grid = [1]} {
    %dma = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %zero = arith.constant 0 : index
    %next = arith.constant 128 : index
    dataflow.program_unit iter_arg : %unit -> (%dma) : {
      %logical = ktdp.construct_memory_view %base, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = #stick, memory_space = #ktdp.memory_space<global>
      } : memref<1x64xf16, #ktdp.memory_space<global>>
      %view = ktdp.construct_memory_view %base, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = #stick, memory_space = #ktdp.memory_space<global>
      } : memref<1x64xf16, #ktdp.memory_space<global>>
      %canonical = memref.memory_space_cast %view : memref<1x64xf16, #ktdp.memory_space<global>> to memref<1x64xf16, "DDR">
      %local0 = builtin.unrealized_conversion_cast %zero : index to memref<1x64xf16, #ktdp.memory_space<ct_local>>
      %local1 = builtin.unrealized_conversion_cast %next : index to memref<1x64xf16, "L1">
      ktdf.data_transfer from %logical[0, 0] size [1, 64] to %local0[0, 0] size [1, 64]
        {dataflow_scheduler.throttle = 64 : i64}
        : memref<1x64xf16, #ktdp.memory_space<global>>, memref<1x64xf16, #ktdp.memory_space<ct_local>>
      ktdf.data_transfer from %canonical[0, 0] size [1, 64] to %local1[0, 0] size [1, 64]
        {dataflow_scheduler.throttle = 64 : i64}
        : memref<1x64xf16, "DDR">, memref<1x64xf16, "L1">
    }
    return
  }
}
