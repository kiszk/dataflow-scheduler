// RUN: dataflow-scheduler-opt -ktir-to-dfir --split-input-file %s | FileCheck %s

// A conversion between f16 and f32 is lowered to the instructions the code
// generator selects the device's FCVT from. One f16 vector converts to two f32
// vectors, or two back to one, so the f32 side of the conversion is a FIFO slot
// moved as two vectors -- at the compute end and at the memory end alike, where
// the second vector is the next 32 lanes of the buffer.

// CHECK-LABEL: func.func @dl16tofp32
// The load unit sends the f16 vector whole.
// CHECK:         dataflow.program_unit
// CHECK:         dataflow.program_unit
// CHECK:           agen.vector_load %{{.+}}[0, 0] {{.+}} : memref<1x64xf16>, vector<64xf16>
// CHECK:           dataflow.send %{{.+}}, %{{.+}} : vector<64xf16>
// The compute unit widens each half from a cast of its own and sends it.
// CHECK:         dataflow.program_unit
// CHECK:           %[[IN:.+]] = dataflow.receive %{{.+}} : vector<64xf16>
// CHECK:           %[[W0:.+]] = vectorchain.cast %[[IN]] : vector<64xf16>, vector<64xf32>
// CHECK-NEXT:      %[[LO:.+]] = vectorchain.shuffle input(%[[W0]]) {indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32], repetition = 8 : i32} : vector<64xf32>, vector<32xf32>
// CHECK-NEXT:      dataflow.send %{{.+}}, %[[LO]] : vector<32xf32>
// CHECK-NEXT:      %[[W1:.+]] = vectorchain.cast %[[IN]] : vector<64xf16>, vector<64xf32>
// CHECK-NEXT:      %[[HI:.+]] = vectorchain.shuffle input(%[[W1]]) {indices = [4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf32>, vector<32xf32>
// CHECK-NEXT:      dataflow.send %{{.+}}, %[[HI]] : vector<32xf32>
// The store unit receives both halves, one after the other in the buffer.
// CHECK:         dataflow.program_unit
// CHECK:           %[[R0:.+]] = dataflow.receive %{{.+}} : vector<32xf32>
// CHECK-NEXT:      agen.vector_store %[[R0]], %{{.+}}[0, 0] {{.+}} : memref<1x64xf32>, vector<32xf32>
// CHECK-NEXT:      %[[R1:.+]] = dataflow.receive %{{.+}} : vector<32xf32>
// CHECK-NEXT:      agen.vector_store %[[R1]], %{{.+}}[0, 32] {{.+}} : memref<1x64xf32>, vector<32xf32>
// CHECK-NOT:       dataflow.receive

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @dl16tofp32(%i: index) attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 3 : index
    %A_start_address = arith.constant 1024 : index
    %C_start_address = arith.constant 18432 : index
    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index
    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>
    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf32>
    %start_row_i = arith.addi %start_row, %i : index
    %A_access_tile = ktdp.construct_access_tile %A_view[%start_row_i, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
    %result_empty = tensor.empty() : tensor<1x64xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
      ins(%A_data_tile : tensor<1x64xf16>) outs(%result_empty : tensor<1x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %cvt = arith.extf %in : f16 to f32
      linalg.yield %cvt : f32
    } -> tensor<1x64xf32>
    %C_access_tile = ktdp.construct_access_tile %C_view[%start_row_i, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf32> -> !ktdp.access_tile<1x64xindex>
    ktdp.store %result, %C_access_tile : tensor<1x64xf32>, !ktdp.access_tile<1x64xindex>
    return
  }
}

// -----

// CHECK-LABEL: func.func @fp32todl16
// The load unit sends both halves, one after the other in the buffer.
// CHECK:         dataflow.program_unit
// CHECK:         dataflow.program_unit
// CHECK:           %[[L0:.+]] = agen.vector_load %{{.+}}[0, 0] {{.+}} : memref<1x64xf32>, vector<32xf32>
// CHECK:           dataflow.send %{{.+}}, %[[L0]] : vector<32xf32>
// CHECK-NEXT:      %[[L1:.+]] = agen.vector_load %{{.+}}[0, 32] {{.+}} : memref<1x64xf32>, vector<32xf32>
// CHECK-NEXT:      dataflow.send %{{.+}}, %[[L1]] : vector<32xf32>
// The compute unit narrows both halves and packs them into one f16 vector.
// CHECK:         dataflow.program_unit
// CHECK:           %[[S0:.+]] = dataflow.receive %{{.+}} : vector<32xf32>
// CHECK-NEXT:      %[[S1:.+]] = dataflow.receive %{{.+}} : vector<32xf32>
// CHECK-NEXT:      %[[N0:.+]] = vectorchain.cast %[[S0]] : vector<32xf32>, vector<32xf16>
// CHECK-NEXT:      %[[N1:.+]] = vectorchain.cast %[[S1]] : vector<32xf32>, vector<32xf16>
// CHECK-NEXT:      %[[P:.+]] = vectorchain.pack %[[N0]], %[[N1]] {indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : index} : vector<32xf16>, vector<32xf16>, vector<64xf16>
// CHECK:           dataflow.send %{{.+}}, %[[P]] : vector<64xf16>
// The store unit stores the f16 vector whole.
// CHECK:         dataflow.program_unit
// CHECK:           %[[R:.+]] = dataflow.receive %{{.+}} : vector<64xf16>
// CHECK-NEXT:      agen.vector_store %[[R]], %{{.+}}[0, 0] {{.+}} : memref<1x64xf16>, vector<64xf16>
// CHECK-NOT:       dataflow.receive

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @fp32todl16(%i: index) attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 3 : index
    %A_start_address = arith.constant 1024 : index
    %C_start_address = arith.constant 18432 : index
    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index
    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf32>
    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>
    %start_row_i = arith.addi %start_row, %i : index
    %A_access_tile = ktdp.construct_access_tile %A_view[%start_row_i, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf32> -> !ktdp.access_tile<1x64xindex>
    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf32>
    %result_empty = tensor.empty() : tensor<1x64xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
      ins(%A_data_tile : tensor<1x64xf32>) outs(%result_empty : tensor<1x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %cvt = arith.truncf %in : f32 to f16
      linalg.yield %cvt : f16
    } -> tensor<1x64xf16>
    %C_access_tile = ktdp.construct_access_tile %C_view[%start_row_i, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
    ktdp.store %result, %C_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    return
  }
}
