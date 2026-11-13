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

// RUN: dataflow-scheduler-opt -mlir-print-local-scope -ktdflowering-to-dfir %s | FileCheck %s --implicit-check-not=unrealized_conversion_cast --implicit-check-not=ktdf.select_memref --implicit-check-not=ktdf.buffer_phase

// Address assignment materializes addresses in elements of each buffer's
// element type. The byte ranges [0,256), [256,512), [512,640), [640,768) arrive
// as FP32 element bases 0/64 and f16 element bases 256/320, and the views must
// keep them unchanged: dividing by the element size again would overlap the
// buffers.
// CHECK-LABEL: func.func private @allocated_pairs()
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[INPUT_SUM:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[OUTPUT_START:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[OUTPUT_SUM:.*]] = arith.constant 576 : index
// CHECK: scf.for {{.*}} iter_args(%[[INPUT_ADDRESS:.*]] = %[[ZERO]]) -> (index) {
// CHECK: %[[INPUT_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[INPUT_ADDRESS]] {{.*}}memref<2x32xf32>
// CHECK: agen.vector_load %[[INPUT_VIEW]][0, 0]
// CHECK: agen.vector_load %[[INPUT_VIEW]][1, 0]
// CHECK: %[[NEXT_INPUT:.*]] = arith.subi %[[INPUT_SUM]], %[[INPUT_ADDRESS]] : index
// CHECK-NEXT: scf.yield %[[NEXT_INPUT]] : index
// CHECK: scf.for {{.*}} iter_args(%[[OUTPUT_ADDRESS:.*]] = %[[OUTPUT_START]]) -> (index) {
// CHECK: %[[OUTPUT_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[OUTPUT_ADDRESS]] {{.*}}memref<1x64xf16>
// CHECK: agen.vector_store {{.*}}, %[[OUTPUT_VIEW]][0, 0]
// CHECK: %[[NEXT_OUTPUT:.*]] = arith.subi %[[OUTPUT_SUM]], %[[OUTPUT_ADDRESS]] : index
// CHECK-NEXT: scf.yield %[[NEXT_OUTPUT]] : index

// Explicit construct_memory_view offsets already count elements, even when
// the target memory is the same scratchpad. Do not divide these a second time.
// CHECK-LABEL: func.func private @explicit_element_offsets()
// CHECK-DAG: %[[F16_BASE:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[F32_BASE:.*]] = arith.constant 256 : index
// CHECK: %[[F16_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[F16_BASE]] {{.*}}memref<1x64xf16>
// CHECK: %[[F32_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[F32_BASE]] {{.*}}memref<1x32xf32>
// CHECK: agen.vector_load %[[F16_VIEW]][0, 0]
// CHECK: agen.vector_load %[[F32_VIEW]][0, 0]

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func private @allocated_pairs() attributes {grid = [1]} {
    %load = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %compute = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %store = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c320 = arith.constant 320 : index
    dataflow.program_unit iter_arg : %unit -> (%load) : {
      %input0 = builtin.unrealized_conversion_cast %c0 : index to memref<2x32xf32, "L1">
      %input1 = builtin.unrealized_conversion_cast %c64 : index to memref<2x32xf32, "L1">
      %input_fifo:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      scf.for %i = %c0 to %c3 step %c1 {
        %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
        %input = ktdf.select_memref %phase[%input0, %input1] : memref<2x32xf32, "L1">
        ktdf.data_transfer from %input[0, 0] size [1, 32] to %input_fifo#0 size [32]
          : memref<2x32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
        ktdf.data_transfer from %input[1, 0] size [1, 32] to %input_fifo#1 size [32]
          : memref<2x32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      }
    }
    dataflow.program_unit iter_arg : %unit -> (%compute) : {
    }
    dataflow.program_unit iter_arg : %unit -> (%store) : {
      %output0 = builtin.unrealized_conversion_cast %c256 : index to memref<1x64xf16, "L1">
      %output1 = builtin.unrealized_conversion_cast %c320 : index to memref<1x64xf16, "L1">
      %output_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      scf.for %i = %c0 to %c3 step %c1 {
        %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
        %output = ktdf.select_memref %phase[%output0, %output1] : memref<1x64xf16, "L1">
        ktdf.data_transfer from %output_fifo size [64] to %output[0, 0] size [1, 64]
          : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
      }
    }
    return
  }

  func.func private @explicit_element_offsets() attributes {grid = [1]} {
    %load = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %compute = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    dataflow.program_unit iter_arg : %unit -> (%load) : {
      %f16_view = ktdp.construct_memory_view %c128, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<ct_local>
      } : memref<1x64xf16>
      %f16_memory = memref.memory_space_cast %f16_view : memref<1x64xf16> to memref<1x64xf16, "L1">
      %f32_view = ktdp.construct_memory_view %c256, sizes: [1, 32], strides: [32, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 31 >= 0)>,
        memory_space = #ktdp.memory_space<ct_local>
      } : memref<1x32xf32>
      %f32_memory = memref.memory_space_cast %f32_view : memref<1x32xf32> to memref<1x32xf32, "L1">
      %f16_fifo, %f32_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      ktdf.data_transfer from %f16_memory[0, 0] size [1, 64] to %f16_fifo size [64]
        : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      ktdf.data_transfer from %f32_memory[0, 0] size [1, 32] to %f32_fifo size [32]
        : memref<1x32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    }
    dataflow.program_unit iter_arg : %unit -> (%compute) : {
    }
    return
  }
}
