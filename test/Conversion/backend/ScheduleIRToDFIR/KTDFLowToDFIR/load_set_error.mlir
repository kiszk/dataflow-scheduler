// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s --implicit-check-not=iter_args --implicit-check-not=arith.subi --implicit-check-not=affine.apply

// Preserve the load/store set rank when the input, local buffers, and output
// have different ranks. Each transfer still contains exactly one 64-element
// packet. Check both phases after expanding the double-buffer address cycle.

// CHECK-DAG: #[[ORDER2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[ORDER4:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[ORDER6:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[ORDER7:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
// CHECK-DAG: #[[ORDER5:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[SET2:.*]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-DAG: #[[SET4:.*]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK-DAG: #[[SET6:.*]] = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0, d4 == 0, d5 >= 0, -d5 + 63 >= 0)>
// CHECK-DAG: #[[SET7:.*]] = affine_set<(d0, d1, d2, d3, d4, d5, d6) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0, d4 == 0, d5 == 0, d6 >= 0, -d6 + 63 >= 0)>
// CHECK-DAG: #[[SET5:.*]] = affine_set<(d0, d1, d2, d3, d4) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0, d4 >= 0, -d4 + 63 >= 0)>
// CHECK-LABEL: func.func @Add_1414()
// CHECK: call @"local-schedule-0"()
// CHECK: func.func private @"local-schedule-0"() attributes {grid = [2]}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[IN0:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[IN1:.*]] = arith.constant 640 : index
// CHECK-DAG: %[[OUT0:.*]] = arith.constant 1152 : index
// CHECK-DAG: %[[OUT1:.*]] = arith.constant 1664 : index
// CHECK-DAG: %[[MNILU0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"}
// CHECK-DAG: %[[MNILU1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"}
// CHECK-DAG: %[[L1LU0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"}
// CHECK-DAG: %[[L1LU1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"}
// CHECK-DAG: %[[SFU0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"}
// CHECK-DAG: %[[SFU1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"}
// CHECK-DAG: %[[L1SU0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"}
// CHECK-DAG: %[[L1SU1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"}
// CHECK-DAG: %[[MNISU0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"}
// CHECK-DAG: %[[MNISU1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"}

// HBM -> LX: rank 4 input and rank 2 broadcast operand, rank 6 input buffers.
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[MNILU0]], %[[MNILU1]])
// CHECK: %[[HBM:.*]] = dataflow.get_logical_memory_view {{.*}}memref<12x1x64x64xf16>
// CHECK: agen.composite_load_and_store
// CHECK-NEXT: time_symbols(), load_iv({{.*}}:vector<64xf16>)
// CHECK-NEXT: {load_order = #[[ORDER2]], load_set = #[[SET2]], {{.*}}store_order = #[[ORDER2]], store_set = #[[SET2]],
// CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK: scf.for %[[J:.*]] = %[[C0]] to %[[C16]] step %[[C1]]
// CHECK: %[[LOADBUF0:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[IN0]] {{.*}}memref<2x2x1x1x1x64xf16>
// CHECK: agen.composite_load_and_store src:%[[HBM]][%[[I]] * 2 + {{.*}}, 0, %[[J]] * 4 + {{.*}}, 0] dst:%[[LOADBUF0]]
// CHECK-NEXT: time_symbols(), load_iv({{.*}}:vector<64xf16>)
// CHECK-NEXT: {load_order = #[[ORDER4]], load_set = #[[SET4]], {{.*}}store_order = #[[ORDER6]], store_set = #[[SET6]],
// CHECK: %[[LOADBUF1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[IN1]] {{.*}}memref<2x2x1x1x1x64xf16>
// CHECK: agen.composite_load_and_store src:%[[HBM]][%[[I]] * 2 + {{.*}}, 0, %[[J]] * 4 + {{.*}} + 2, 0] dst:%[[LOADBUF1]]
// CHECK-NEXT: time_symbols(), load_iv({{.*}}:vector<64xf16>)
// CHECK-NEXT: {load_order = #[[ORDER4]], load_set = #[[SET4]], {{.*}}store_order = #[[ORDER6]], store_set = #[[SET6]],

// LX -> compute: read and send a packet from each bank and the broadcast view.
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[L1LU0]], %[[L1LU1]])
// CHECK: %[[READBUF0:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[IN0]] {{.*}}memref<2x2x1x1x1x64xf16>
// CHECK: %[[READ0:.*]] = agen.vector_load %[[READBUF0]]{{.*}}{load_order = #[[ORDER6]], load_set = #[[SET6]]} : memref<2x2x1x1x1x64xf16>, vector<64xf16>
// CHECK: dataflow.send {{.*}}, %[[READ0]] : vector<64xf16>
// CHECK: agen.vector_load {{.*}}{load_order = #[[ORDER2]], load_set = #[[SET2]]} : memref<1x64xf16>, vector<64xf16>
// CHECK: %[[READBUF1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[IN1]] {{.*}}memref<2x2x1x1x1x64xf16>
// CHECK: %[[READ1:.*]] = agen.vector_load %[[READBUF1]]{{.*}}{load_order = #[[ORDER6]], load_set = #[[SET6]]} : memref<2x2x1x1x1x64xf16>, vector<64xf16>
// CHECK: dataflow.send {{.*}}, %[[READ1]] : vector<64xf16>
// CHECK: agen.vector_load {{.*}}{load_order = #[[ORDER2]], load_set = #[[SET2]]} : memref<1x64xf16>, vector<64xf16>

// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[SFU0]], %[[SFU1]])
// CHECK: %[[LHS:.*]] = dataflow.receive {{.*}} : vector<64xf16>
// CHECK: %[[RHS:.*]] = dataflow.receive {{.*}} : vector<64xf16>
// CHECK: %[[ADD:.*]] = vectorchain.binary %[[LHS]], %[[RHS]] {binary_op = #vectorchain<binary_operator add>{{.*}} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK: dataflow.send {{.*}}, %[[ADD]] : vector<64xf16>

// Compute -> LX: both output banks use rank 7 store sets.
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[L1SU0]], %[[L1SU1]])
// CHECK: %[[STOREBUF0:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[OUT0]] {{.*}}memref<2x2x1x1x1x1x64xf16>
// CHECK: %[[RECV0:.*]] = dataflow.receive {{.*}} : vector<64xf16>
// CHECK: agen.vector_store %[[RECV0]], %[[STOREBUF0]]{{.*}}{store_order = #[[ORDER7]], store_set = #[[SET7]]} : memref<2x2x1x1x1x1x64xf16>, vector<64xf16>
// CHECK: %[[STOREBUF1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[OUT1]] {{.*}}memref<2x2x1x1x1x1x64xf16>
// CHECK: %[[RECV1:.*]] = dataflow.receive {{.*}} : vector<64xf16>
// CHECK: agen.vector_store %[[RECV1]], %[[STOREBUF1]]{{.*}}{store_order = #[[ORDER7]], store_set = #[[SET7]]} : memref<2x2x1x1x1x1x64xf16>, vector<64xf16>

// LX -> HBM: rank 7 loads and rank 5 stores retain their respective ranks.
// CHECK: dataflow.program_unit iter_arg : {{.*}} -> (%[[MNISU0]], %[[MNISU1]])
// CHECK: %[[RESULT:.*]] = dataflow.get_logical_memory_view {{.*}}memref<12x1x1x64x64xf16>
// CHECK: scf.for %[[OI:.*]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK: scf.for %[[OJ:.*]] = %[[C0]] to %[[C16]] step %[[C1]]
// CHECK: %[[WRITEBUF0:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[OUT0]] {{.*}}memref<2x2x1x1x1x1x64xf16>
// CHECK: agen.composite_load_and_store src:%[[WRITEBUF0]]{{.*}}dst:%[[RESULT]][%[[OI]] * 2 + {{.*}}, 0, 0, %[[OJ]] * 4 + {{.*}}, 0]
// CHECK-NEXT: time_symbols(), load_iv({{.*}}:vector<64xf16>)
// CHECK-NEXT: {load_order = #[[ORDER7]], load_set = #[[SET7]], {{.*}}store_order = #[[ORDER5]], store_set = #[[SET5]],
// CHECK: %[[WRITEBUF1:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[OUT1]] {{.*}}memref<2x2x1x1x1x1x64xf16>
// CHECK: agen.composite_load_and_store src:%[[WRITEBUF1]]{{.*}}dst:%[[RESULT]][%[[OI]] * 2 + {{.*}}, 0, 0, %[[OJ]] * 4 + {{.*}} + 2, 0]
// CHECK-NEXT: time_symbols(), load_iv({{.*}}:vector<64xf16>)
// CHECK-NEXT: {load_order = #[[ORDER7]], load_set = #[[SET7]], {{.*}}store_order = #[[ORDER5]], store_set = #[[SET5]],

#map = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @Add_1414() attributes {grid = [2]} {
      call @"local-schedule-0"() : () -> ()
      return
    }
    func.func private @"local-schedule-0"()
  }
  module {
    func.func private @"local-schedule-0"() attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %6 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %7 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %8 = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
      %9 = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
      %10 = ktdp.get_compute_tile_id : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %11 = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
      %12 = uniform.query_map(map:%11, key:%10) : index
      %c0_0 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      %13 = uniform.def_immutable_mapping([%c0_0 -> %2], [%c1_1 -> %3]):index
      %14 = uniform.query_map(map:%13, key:%10) : index
      %c0_2 = arith.constant 0 : index
      %c1_3 = arith.constant 1 : index
      %15 = uniform.def_immutable_mapping([%c0_2 -> %4], [%c1_3 -> %5]):index
      %16 = uniform.query_map(map:%15, key:%10) : index
      %c0_4 = arith.constant 0 : index
      %c1_5 = arith.constant 1 : index
      %17 = uniform.def_immutable_mapping([%c0_4 -> %6], [%c1_5 -> %7]):index
      %18 = uniform.query_map(map:%17, key:%10) : index
      %c0_6 = arith.constant 0 : index
      %c1_7 = arith.constant 1 : index
      %19 = uniform.def_immutable_mapping([%c0_6 -> %8], [%c1_7 -> %9]):index
      %20 = uniform.query_map(map:%19, key:%10) : index
      %c32 = arith.constant 32 : index
      %c6 = arith.constant 6 : index
      %c2 = arith.constant 2 : index
      %c0_8 = arith.constant 0 : index
      %c1_9 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c113216 = arith.constant 113216 : index
      %c113152 = arith.constant 113152 : index
      %c64000 = arith.constant 64000 : index
      %c12 = arith.constant 12 : index
      %21 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
      %memspacecast = memref.memory_space_cast %21 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
      %22 = ktdp.construct_memory_view %c113152, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
      %memspacecast_10 = memref.memory_space_cast %22 : memref<1x64xf16> to memref<1x64xf16, "DDR">
      %reinterpret_cast_11 = memref.reinterpret_cast %memspacecast_10 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<1x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1]>, "DDR">
      %cast_12 = memref.cast %reinterpret_cast_11 : memref<1x64xf16, strided<[64, 1]>, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
      %23 = ktdp.construct_memory_view %c113216, sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<12x1x1x64x64xf16>
      %memspacecast_13 = memref.memory_space_cast %23 : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
      %reinterpret_cast_14 = memref.reinterpret_cast %memspacecast_13 to offset: [0], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR">
      %cast_15 = memref.cast %reinterpret_cast_14 : memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
      %c0_16 = arith.constant 0 : index
      %24 = builtin.unrealized_conversion_cast %c0_16 : index to memref<1x64xf16, "L1">
      ktdf_lowering.execute_on %12 {
        ktdf_lowering.execute_on %12 {
          ktdf.data_transfer from %cast_12[%c0_8, %c0_8] size [1, 64] to %24[0, 0] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
        }
      }
      %c128 = arith.constant 128 : index
      %25 = builtin.unrealized_conversion_cast %c128 : index to memref<2x2x1x1x1x64xf16, "L1">
      %cast_17 = memref.cast %25 : memref<2x2x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x64xf16, "L1">
      %c640 = arith.constant 640 : index
      %26 = builtin.unrealized_conversion_cast %c640 : index to memref<2x2x1x1x1x64xf16, "L1">
      %cast_18 = memref.cast %26 : memref<2x2x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x64xf16, "L1">
      %c1152 = arith.constant 1152 : index
      %27 = builtin.unrealized_conversion_cast %c1152 : index to memref<2x2x1x1x1x1x64xf16, "L1">
      %cast_19 = memref.cast %27 : memref<2x2x1x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x1x64xf16, "L1">
      %c1664 = arith.constant 1664 : index
      %28 = builtin.unrealized_conversion_cast %c1664 : index to memref<2x2x1x1x1x1x64xf16, "L1">
      %cast_20 = memref.cast %28 : memref<2x2x1x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x1x64xf16, "L1">
      scf.for %arg0 = %c0_8 to %c6 step %c1_9 {
        %29 = ktdf.tiling.derive_size [%arg0 : %c2], total_size = %c12 : index
        scf.for %arg1 = %c0_8 to %c32 step %c1_9 {
          %30 = ktdf.tiling.derive_size [%arg1 : %c2], total_size = %c64 : index
          %31 = ktdf.buffer_phase(%arg0, %arg1) {num_phases = 2 : i64} : index
          %32 = ktdf.select_memref %31[%cast_17, %cast_18] : memref<?x?x1x1x1x64xf16, "L1">
          %33 = ktdf.select_memref %31[%cast_19, %cast_20] : memref<?x?x1x1x1x1x64xf16, "L1">
          ktdf_lowering.execute_on %12, %14, %16, %18, %20 {
            %c2176 = arith.constant 2176 : index
            %34 = builtin.unrealized_conversion_cast %c2176 : index to memref<1x64xf16, "L1">
            %35 = ktdf.create_token : !ktdf.token
            %36 = ktdf.create_token : !ktdf.token
            ktdf_lowering.execute_on %12 {
              scf.for %arg2 = %c0_8 to %29 step %c1_9 {
                scf.for %arg3 = %c0_8 to %30 step %c1_9 {
                  %37 = affine.apply #map(%arg0, %arg2)
                  %38 = affine.apply #map(%arg1, %arg3)
                  ktdf.data_transfer from %cast[%arg0 * 2 + %arg2, 0, %arg1 * 2 + %arg3, 0] size [1, 1, 1, 64] to %32[%arg2, %arg3, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<?x?x1x1x1x64xf16, "L1">
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            }
            ktdf_lowering.signal %12, %14, %16, %18
            ktdf_lowering.execute_on %14, %16, %18 {
              scf.for %arg2 = %c0_8 to %29 step %c1_9 {
                scf.for %arg3 = %c0_8 to %30 step %c1_9 {
                  ktdf_lowering.execute_on %14, %16, %18 {
                    %37:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    %38 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                    %39 = ktdf.create_token : !ktdf.token
                    %40 = ktdf.create_token : !ktdf.token
                    ktdf_lowering.execute_on %14 {
                      ktdf.data_transfer from %32[%arg2, %arg3, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] to %37#0 size [64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x?x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                      ktdf.data_transfer from %24[0, 0] size [1, 64] to %37#1 size [64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    }
                    ktdf_lowering.execute_on %16 {
                      %41 = ktdf.read_from_fifo %37#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
                      %42 = ktdf.read_from_fifo %37#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
                      %43 = tensor.empty() : tensor<1x1x1x1x64xf16>
                      %44 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%41, %42 : tensor<1x1x1x64xf16>, tensor<1x64xf16>) outs(%43 : tensor<1x1x1x1x64xf16>) {
                      ^bb0(%in: f16, %in_21: f16, %out: f16):
                        %45 = arith.addf %in, %in_21 : f16
                        linalg.yield %45 : f16
                      } -> tensor<1x1x1x1x64xf16>
                      ktdf.write_to_fifo %44, %38 : tensor<1x1x1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                    }
                    ktdf_lowering.execute_on %18 {
                      ktdf.data_transfer from %38 size [64] to %33[%arg2, %arg3, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<?x?x1x1x1x1x64xf16, "L1">
                    }
                  }
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            }
            ktdf_lowering.signal %14, %16, %18, %20
            ktdf_lowering.execute_on %20 {
              scf.for %arg2 = %c0_8 to %29 step %c1_9 {
                scf.for %arg3 = %c0_8 to %30 step %c1_9 {
                  %37 = affine.apply #map(%arg0, %arg2)
                  %38 = affine.apply #map(%arg1, %arg3)
                  ktdf.data_transfer from %33[%arg2, %arg3, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] to %cast_15[%arg0 * 2 + %arg2, 0, 0, %arg1 * 2 + %arg3, 0] size [1, 1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<?x?x1x1x1x1x64xf16, "L1">, memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            }
          }
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
      return
    }
  }
}
