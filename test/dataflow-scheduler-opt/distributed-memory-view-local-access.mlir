// RUN: dataflow-scheduler-opt %s --compute-group-extraction --ktir-distributed-memory | FileCheck %s --check-prefix=EXPAND --implicit-check-not=_gather --implicit-check-not=_scatter --implicit-check-not=requires_program_barrier
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR --implicit-check-not=ktdp.construct_distributed_memory_view --implicit-check-not=_gather --implicit-check-not=_scatter --implicit-check-not=requires_program_barrier

// The compute and owner partitions agree. Use the fixed LX view directly,
// keeping two ordered programs and both cores' global row offsets.
// EXPAND-LABEL: func.func @distributed_local_roundtrip(
// EXPAND: call @local_schedule_0(
// EXPAND-NEXT: call @local_schedule_1(
// EXPAND-NEXT: return
// EXPAND-LABEL: func.func @local_schedule_0(
// EXPAND-SAME: attributes {grid = [2]}
// EXPAND: %[[BASE:.*]] = arith.constant 0 : index
// EXPAND: %[[LX:.*]] = ktdp.construct_memory_view %[[BASE]], sizes: [2, 64], strides: [64, 1]
// EXPAND-SAME: memory_space = #ktdp.memory_space<ct_local>
// EXPAND: %[[TILE:.*]] = ktdp.construct_access_tile %[[LX]]
// EXPAND: ktdp.store {{.*}}, %[[TILE]] : tensor<2x64xf16>
// EXPAND-LABEL: func.func @local_schedule_1(
// EXPAND-SAME: attributes {grid = [2]}
// EXPAND: %[[READ_BASE:.*]] = arith.constant 0 : index
// EXPAND: %[[READ_LX:.*]] = ktdp.construct_memory_view %[[READ_BASE]], sizes: [2, 64], strides: [64, 1]
// EXPAND-SAME: memory_space = #ktdp.memory_space<ct_local>
// EXPAND: %[[READ_TILE:.*]] = ktdp.construct_access_tile %[[READ_LX]]
// EXPAND: ktdp.load %[[READ_TILE]] : {{.*}} -> tensor<2x64xf16>
// DFIR-LABEL: func.func @distributed_local_roundtrip(
// DFIR: call @local_schedule_0(
// DFIR-NEXT: call @local_schedule_1(
// DFIR-NEXT: return
// DFIR-LABEL: func.func @local_schedule_0_body()
// DFIR-DAG: dataflow.get_unit {core = 0 : i32, {{.*}}type = "l1su"}
// DFIR-DAG: dataflow.get_unit {core = 1 : i32, {{.*}}type = "l1su"}
// DFIR: agen.vector_store {{.*}} : memref<2x64xf16>, vector<64xf16>
// DFIR-LABEL: func.func @local_schedule_1_body()
// DFIR-DAG: dataflow.get_unit {core = 0 : i32, {{.*}}type = "l1lu"}
// DFIR-DAG: dataflow.get_unit {core = 1 : i32, {{.*}}type = "l1lu"}
// DFIR: agen.vector_load {{.*}} : memref<2x64xf16>, vector<64xf16>

ktdf_arch.device @sample_device attributes {
  dataflow_scheduler.core_transfer = {
    send_unit = "MNISU", receive_unit = "MNILU", bytes = 128 : i64
  },
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">
} import("../Dialect/KTDFArch/sample_device.mlir")

#all = affine_set<(r, l) : (r >= 0, 3 - r >= 0, l >= 0, 63 - l >= 0)>
#first = affine_set<(r, l) : (r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#second = affine_set<(r, l) : (r - 2 >= 0, 3 - r >= 0, l >= 0, 63 - l >= 0)>
#id = affine_map<(r, l) -> (r, l)>

func.func @distributed_local_roundtrip(%src_addr: index, %dst_addr: index) attributes {grid = [2]} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %core = ktdp.get_compute_tile_id : index
  %row = arith.muli %core, %c2 : index
  %src = ktdp.construct_memory_view %src_addr, sizes: [4, 64], strides: [64, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<global>
  } : memref<4x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr, sizes: [4, 64], strides: [64, 1] {
    coordinate_set = #all, memory_space = #ktdp.memory_space<global>
  } : memref<4x64xf16>
  %first = ktdp.construct_memory_view %c0, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #first, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>
  } : memref<2x64xf16>
  %second = ktdp.construct_memory_view %c0, sizes: [2, 64], strides: [64, 1] {
    coordinate_set = #second, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>
  } : memref<2x64xf16>
  %distributed = ktdp.construct_distributed_memory_view (%first, %second : memref<2x64xf16>, memref<2x64xf16>) : memref<4x64xf16>
  %input_tile = ktdp.construct_access_tile %src[%row, 0] {access_tile_order = #id, access_tile_set = #first} : memref<4x64xf16> -> !ktdp.access_tile<2x64xindex>
  %store_tile = ktdp.construct_access_tile %distributed[%row, 0] {access_tile_order = #id, access_tile_set = #first} : memref<4x64xf16> -> !ktdp.access_tile<2x64xindex>
  %input = ktdp.load %input_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>
  %empty0 = tensor.empty() : tensor<2x64xf16>
  %copy0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<2x64xf16>) outs(%empty0 : tensor<2x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %copy = spyreop.identity %value : f16
      linalg.yield %copy : f16
  } -> tensor<2x64xf16>
  ktdp.store %copy0, %store_tile : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>
  %load_tile = ktdp.construct_access_tile %distributed[%row, 0] {access_tile_order = #id, access_tile_set = #first} : memref<4x64xf16> -> !ktdp.access_tile<2x64xindex>
  %output_tile = ktdp.construct_access_tile %dst[%row, 0] {access_tile_order = #id, access_tile_set = #first} : memref<4x64xf16> -> !ktdp.access_tile<2x64xindex>
  %loaded = ktdp.load %load_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>
  %empty1 = tensor.empty() : tensor<2x64xf16>
  %copy1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%loaded : tensor<2x64xf16>) outs(%empty1 : tensor<2x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %copy = spyreop.identity %value : f16
      linalg.yield %copy : f16
  } -> tensor<2x64xf16>
  ktdp.store %copy1, %output_tile : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>
  return
}
