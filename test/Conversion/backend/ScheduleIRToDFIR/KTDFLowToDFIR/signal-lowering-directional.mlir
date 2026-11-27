// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktdflowering-to-dfir --cse | FileCheck %s

// A ready/credit pair is two one-way signals, not two full barriers. The
// mappings must keep each of the two cores paired with its own reader.
// CHECK-LABEL: func.func @directed_signals
// CHECK-DAG: %[[D0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"}
// CHECK-DAG: %[[D1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"}
// CHECK-DAG: %[[R0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"}
// CHECK-DAG: %[[R1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"}
// CHECK: dataflow.program_unit
// CHECK: uniform.def_immutable_mapping([%[[D0]] -> %[[R0]]], [%[[D1]] -> %[[R1]]])
// CHECK: %[[READER:.*]] = uniform.query_map
// CHECK-NEXT: dataflow.sync_send %[[READER]] {dataflow_scheduler.mpe_input_ready, wait_immediately_for_async_transfers = true}
// CHECK-NEXT: dataflow.sync_recv %[[READER]] {dataflow_scheduler.mpe_input_credit}
// CHECK-NEXT: }
// CHECK: dataflow.program_unit
// CHECK: uniform.def_immutable_mapping([%[[R0]] -> %[[D0]]], [%[[R1]] -> %[[D1]]])
// CHECK: %[[DMA:.*]] = uniform.query_map
// CHECK-NEXT: dataflow.sync_recv %[[DMA]] {dataflow_scheduler.mpe_input_ready}
// CHECK-NEXT: dataflow.sync_send %[[DMA]] {dataflow_scheduler.mpe_input_credit, wait_immediately_for_async_transfers = true}
// CHECK-NEXT: }

module {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @directed_signals() attributes {grid = [2]} {
    %dma0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %dma1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %reader0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %reader1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %core = ktdp.get_compute_tile_id : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %dma_map = uniform.def_immutable_mapping([%zero -> %dma0], [%one -> %dma1]):index
    %reader_map = uniform.def_immutable_mapping([%zero -> %reader0], [%one -> %reader1]):index
    %dma = uniform.query_map(map:%dma_map, key:%core) : index
    %reader = uniform.query_map(map:%reader_map, key:%core) : index
    ktdf_lowering.execute_on %dma {
      ktdf_lowering.signal %dma, %reader {dataflow_scheduler.signal_sender = 0 : i64, dataflow_scheduler.mpe_input_ready}
      ktdf_lowering.signal %dma, %reader {dataflow_scheduler.signal_sender = 1 : i64, dataflow_scheduler.mpe_input_credit}
    }
    ktdf_lowering.execute_on %reader {
      ktdf_lowering.signal %dma, %reader {dataflow_scheduler.signal_sender = 0 : i64, dataflow_scheduler.mpe_input_ready}
      ktdf_lowering.signal %dma, %reader {dataflow_scheduler.signal_sender = 1 : i64, dataflow_scheduler.mpe_input_credit}
    }
    return
  }
}
