// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=spyreop.numeric_cast --implicit-check-not=bufferization.to_tensor

// The variance prefix of GPT-OSS RMSNorm, bf16 [1,2,3072], on one core:
// DL16 -> Staggered FP32, square, mean over 3072 elements, and epsilon add.
// Cast views keep the contiguous row-major layout by adding a singleton row
// axis. The mean crosses a memory boundary before fractional scaling; its head
// and the epsilon scalar both require full-stick FP32 broadcast before add.
// Two rows fit sample_device; 3072 features give its two corelets balanced
// packet loops. Hardware coverage uses the original [1,128,2880] program.
// Intermediates use fixed ct_local element offsets. They must survive
// scheduler scratch allocations and the sum -> scale -> add boundaries.

// CHECK-LABEL: func.func @rmsnorm_variance_ct_local
// CHECK: call @local_schedule_0
// CHECK: call @local_schedule_4
// CHECK-LABEL: func.func @local_schedule_0_body
// CHECK: vectorchain.cast
// CHECK-SAME: vector<64xf16>, vector<64xf32>
// CHECK-LABEL: func.func @local_schedule_1_body
// CHECK: vectorchain.binary
// CHECK-SAME: binary_operator mul
// CHECK-LABEL: func.func @local_schedule_2_body
// CHECK: vectorchain.scan_with_gap
// CHECK-LABEL: func.func @local_schedule_3_body
// CHECK: %[[STATS_BASE:.*]] = arith.constant 12288 : index
// CHECK: vectorchain.constant_bitstream {value = [0x39aaaaab]}
// CHECK: vectorchain.binary
// CHECK-SAME: binary_operator mul
// CHECK: %[[READ_BASE:.*]] = arith.addi {{.*}}, %[[STATS_BASE]]
// CHECK: %[[READ_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[READ_BASE]] {{.*}}memref<2x32xf32>
// CHECK: agen.vector_load %[[READ_VIEW]]
// CHECK: %[[WRITE_BASE:.*]] = arith.addi {{.*}}, %[[STATS_BASE]]
// CHECK: %[[WRITE_VIEW:.*]] = dataflow.get_logical_memory_view {{.*}}, %[[WRITE_BASE]] {{.*}}memref<2x32xf32>
// CHECK: agen.vector_store {{.*}}, %[[WRITE_VIEW]]
// CHECK-LABEL: func.func @local_schedule_4_body
// CHECK: %[[HEAD:.*]] = vectorchain.shuffle {{.*}}indices = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
// CHECK: agen.vector_store %[[HEAD]]
// CHECK: %[[EPS:.*]] = vectorchain.shuffle {{.*}}indices = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
// CHECK: agen.vector_store %[[EPS]]
// CHECK: vectorchain.binary
// CHECK-SAME: binary_operator add

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @rmsnorm_variance_ct_local(%arg1: index, %arg2: index, %arg3: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %0 = arith.constant 0 : index
    %1 = arith.constant 6144 : index
    %2 = arith.constant 12288 : index
    %8 = ktdp.construct_memory_view %arg1, sizes: [2, 48, 1, 64], strides: [3072, 64, 64, 1] {coordinate_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 47 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<2x48x1x64xf16>
    %9 = ktdp.construct_access_tile %8[%c0, %c0, %c0, %c0] {access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, access_tile_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 47 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>} : memref<2x48x1x64xf16> -> !ktdp.access_tile<2x48x1x64xindex>
    %10 = ktdp.load %9 : <2x48x1x64xindex> -> tensor<2x48x1x64xf16>
    %cast_init_1 = tensor.empty() : tensor<2x96x1x32xf32>
    %11 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1 * 2 + (d3 mod 8) floordiv 4, d2, (d3 floordiv 8) * 4 + d3 mod 4)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%10 : tensor<2x48x1x64xf16>) outs(%cast_init_1 : tensor<2x96x1x32xf32>) {
    ^bb0(%value: f16, %unused: f32):
      %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
      linalg.yield %cast : f32
    } -> tensor<2x96x1x32xf32>
    %12 = ktdp.construct_memory_view %0, sizes: [2, 96, 1, 32], strides: [3072, 32, 32, 1] {coordinate_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 95 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 31 >= 0)>, memory_space = #ktdp.memory_space<ct_local>} : memref<2x96x1x32xf32>
    %13 = ktdp.construct_access_tile %12[%c0, %c0, %c0, %c0] {access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, access_tile_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 95 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 31 >= 0)>} : memref<2x96x1x32xf32> -> !ktdp.access_tile<2x96x1x32xindex>
    ktdp.store %11, %13 : tensor<2x96x1x32xf32>, <2x96x1x32xindex>
    %14 = ktdp.construct_memory_view %0, sizes: [2, 96, 32], strides: [3072, 32, 1] {coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 95 >= 0, d2 >= 0, -d2 + 31 >= 0)>, memory_space = #ktdp.memory_space<ct_local>} : memref<2x96x32xf32>
    %15 = ktdp.construct_access_tile %14[%c0, %c0, %c0] {access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, access_tile_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 95 >= 0, d2 >= 0, -d2 + 31 >= 0)>} : memref<2x96x32xf32> -> !ktdp.access_tile<2x96x32xindex>
    %16 = ktdp.load %15 : <2x96x32xindex> -> tensor<2x96x32xf32>
    %17 = ktdp.construct_access_tile %14[%c0, %c0, %c0] {access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, access_tile_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 95 >= 0, d2 >= 0, -d2 + 31 >= 0)>} : memref<2x96x32xf32> -> !ktdp.access_tile<2x96x32xindex>
    %18 = ktdp.load %17 : <2x96x32xindex> -> tensor<2x96x32xf32>
    %19 = tensor.empty() : tensor<2x96x32xf32>
    %20 = linalg.mul ins(%16, %18 : tensor<2x96x32xf32>, tensor<2x96x32xf32>) outs(%19 : tensor<2x96x32xf32>) -> tensor<2x96x32xf32>
    %21 = ktdp.construct_memory_view %1, sizes: [2, 96, 32], strides: [3072, 32, 1] {coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 95 >= 0, d2 >= 0, -d2 + 31 >= 0)>, memory_space = #ktdp.memory_space<ct_local>} : memref<2x96x32xf32>
    %22 = ktdp.construct_access_tile %21[%c0, %c0, %c0] {access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, access_tile_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 95 >= 0, d2 >= 0, -d2 + 31 >= 0)>} : memref<2x96x32xf32> -> !ktdp.access_tile<2x96x32xindex>
    ktdp.store %20, %22 : tensor<2x96x32xf32>, <2x96x32xindex>
    %23 = ktdp.construct_memory_view %1, sizes: [2, 96, 32], strides: [3072, 32, 1] {coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 95 >= 0, d2 >= 0, -d2 + 31 >= 0)>, memory_space = #ktdp.memory_space<ct_local>} : memref<2x96x32xf32>
    %24 = ktdp.construct_access_tile %23[%c0, %c0, %c0] {access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, access_tile_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 95 >= 0, d2 >= 0, -d2 + 31 >= 0)>} : memref<2x96x32xf32> -> !ktdp.access_tile<2x96x32xindex>
    %25 = ktdp.load %24 : <2x96x32xindex> -> tensor<2x96x32xf32>
    %26 = tensor.empty() : tensor<2x32xf32>
    %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d3)>], iterator_types = ["parallel", "reduction", "reduction", "parallel"]} ins(%25 : tensor<2x96x32xf32>) outs(%26 : tensor<2x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %85 = arith.addf %in, %out : f32
      linalg.yield %85 : f32
    } -> tensor<2x32xf32>
    %28 = ktdp.construct_memory_view %2, sizes: [2, 32], strides: [32, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>, memory_space = #ktdp.memory_space<ct_local>} : memref<2x32xf32>
    %29 = ktdp.construct_access_tile %28[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>} : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
    ktdp.store %27, %29 : tensor<2x32xf32>, <2x32xindex>
    %30 = ktdp.construct_memory_view %2, sizes: [2, 32], strides: [32, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>, memory_space = #ktdp.memory_space<ct_local>} : memref<2x32xf32>
    %31 = ktdp.construct_access_tile %30[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>} : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
    %32 = ktdp.load %31 : <2x32xindex> -> tensor<2x32xf32>
    %33 = tensor.empty() : tensor<2x32xf32>
    %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%32 : tensor<2x32xf32>) outs(%33 : tensor<2x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst = arith.constant 3.25520843E-4 : f32
      %85 = arith.mulf %in, %cst : f32
      linalg.yield %85 : f32
    } -> tensor<2x32xf32>
    %35 = ktdp.construct_access_tile %30[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>} : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
    ktdp.store %34, %35 : tensor<2x32xf32>, <2x32xindex>
    %36 = ktdp.construct_memory_view %2, sizes: [2, 32], strides: [32, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>, memory_space = #ktdp.memory_space<ct_local>} : memref<2x32xf32>
    %37 = ktdp.construct_access_tile %36[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0)>} : memref<2x32xf32> -> !ktdp.access_tile<2x1xindex>
    %38 = ktdp.load %37 : <2x1xindex> -> tensor<2x1xf32>
    %39 = ktdp.construct_memory_view %arg2, sizes: [32], strides: [1] {coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<32xf32>
    %40 = ktdp.construct_access_tile %39[%c0] {access_tile_order = affine_map<(d0) -> (d0)>, access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 >= 0)>} : memref<32xf32> -> !ktdp.access_tile<1xindex>
    %41 = ktdp.load %40 : <1xindex> -> tensor<1xf32>
    %42 = tensor.empty() : tensor<1x2x32xf32>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, 0)>, affine_map<(d0, d1, d2) -> (0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %41 : tensor<2x1xf32>, tensor<1xf32>) outs(%42 : tensor<1x2x32xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %85 = arith.addf %in, %in_4 : f32
      linalg.yield %85 : f32
    } -> tensor<1x2x32xf32>
    %44 = ktdp.construct_memory_view %arg3, sizes: [1, 2, 32], strides: [64, 32, 1] {coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 31 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<1x2x32xf32>
    %48 = ktdp.construct_access_tile %44[%c0, %c0, %c0] {access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, access_tile_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 31 >= 0)>} : memref<1x2x32xf32> -> !ktdp.access_tile<1x2x32xindex>
    ktdp.store %43, %48 : tensor<1x2x32xf32>, <1x2x32xindex>
    return
  }
}
