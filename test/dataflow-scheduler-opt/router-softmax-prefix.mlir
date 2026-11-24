// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=spyreop.numeric_cast --implicit-check-not=vectorchain.scan_with_gap

// Gemma4 router softmax prefix, bf16 [128,128], with 32-way row division:
// DL16 -> Staggered FP32 -> row MAX -> subtract -> exp -> row SUM.
// Intermediates use ct_local. The opaque exp, MAX and SUM fixtures exercise
// the shared device-pattern path.
// Hardware coverage also checks division and the final Staggered FP32 D2H.
// CHECK-LABEL: func.func @router_softmax_prefix
// CHECK: call @local_schedule_4
// CHECK-LABEL: func.func @local_schedule_0_body
// CHECK: vectorchain.cast
// CHECK-LABEL: func.func @local_schedule_1_body
// CHECK: vectorchain.binary {{.*}}binary_operator max
// CHECK: dataflow.opaque {{.*}}func_name = "simdreduction_minmax_fp32"{{.*}}parameter_dictionary = {unroll = "1", x1 = "0", x2 = "8"}
// CHECK-LABEL: func.func @local_schedule_2_body
// CHECK: vectorchain.binary {{.*}}binary_operator sub
// CHECK-LABEL: func.func @local_schedule_3_body
// CHECK: dataflow.opaque {{.*}}func_name = "fake_f32_exp"
// CHECK-LABEL: func.func @local_schedule_4_body
// CHECK: dataflow.opaque {{.*}}func_name = "simdreduction_sum_fp32"

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d1, 0)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 31 >= 0)>
#set3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set4 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 >= 0)>
#set5 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 31 >= 0)>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

  func.func @router_softmax_prefix(%arg0: index, %arg1: index) attributes {grid = [32]} {
    %c0 = arith.constant 0 : index
    %0 = ktdp.get_compute_tile_id : index
    %c0_0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c640 = arith.constant 640 : index
    %c0_1 = arith.constant 0 : index
    %c512_2 = arith.constant 512 : index
    %c4 = arith.constant 4 : index
    %1 = arith.muli %0, %c4 : index
    %2 = ktdp.construct_memory_view %arg0, sizes: [2, 128, 64], strides: [8192, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x128x64xf16>
    %3 = ktdp.construct_access_tile %2[%c0, %1, %c0] {access_tile_order = #map, access_tile_set = #set1} : memref<2x128x64xf16> -> !ktdp.access_tile<2x4x64xindex>
    %4 = ktdp.load %3 : <2x4x64xindex> -> tensor<2x4x64xf16>
    %5 = tensor.empty() : tensor<4x4x32xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4 : tensor<2x4x64xf16>) outs(%5 : tensor<4x4x32xf32>) attrs =  {dataflow_scheduler.work_divided} {
    ^bb0(%in: f16, %out: f32):
      %51 = spyreop.numeric_cast %in {kind = "dl16tofp32"} : f16 -> f32
      linalg.yield %51 : f32
    } -> tensor<4x4x32xf32>
    %7 = ktdp.construct_memory_view %c0_0, sizes: [4, 4, 32], strides: [128, 32, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<ct_local>} : memref<4x4x32xf32>
    %8 = ktdp.construct_access_tile %7[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<4x4x32xf32> -> !ktdp.access_tile<4x4x32xindex>
    ktdp.store %6, %8 : tensor<4x4x32xf32>, <4x4x32xindex>
    %9 = ktdp.construct_memory_view %c0_0, sizes: [4, 4, 32], strides: [128, 32, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<ct_local>} : memref<4x4x32xf32>
    %10 = ktdp.construct_access_tile %9[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<4x4x32xf32> -> !ktdp.access_tile<4x4x32xindex>
    %11 = ktdp.load %10 : <4x4x32xindex> -> tensor<4x4x32xf32>
    %12 = tensor.empty() : tensor<4x32xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%11 : tensor<4x4x32xf32>) outs(%12 : tensor<4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.maximumf %out, %in : f32
      linalg.yield %51 : f32
    } -> tensor<4x32xf32>
    %14 = ktdp.construct_memory_view %c512, sizes: [4, 32], strides: [32, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<ct_local>} : memref<4x32xf32>
    %15 = ktdp.construct_access_tile %14[%c0, %c0] {access_tile_order = #map4, access_tile_set = #set3} : memref<4x32xf32> -> !ktdp.access_tile<4x32xindex>
    ktdp.store %13, %15 : tensor<4x32xf32>, <4x32xindex>
    %16 = ktdp.construct_memory_view %c0_0, sizes: [4, 4, 32], strides: [128, 32, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<ct_local>} : memref<4x4x32xf32>
    %17 = ktdp.construct_access_tile %16[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<4x4x32xf32> -> !ktdp.access_tile<4x4x32xindex>
    %18 = ktdp.load %17 : <4x4x32xindex> -> tensor<4x4x32xf32>
    %19 = ktdp.construct_memory_view %c512, sizes: [4, 32], strides: [32, 1] {coordinate_set = #set3, memory_space = #ktdp.memory_space<ct_local>} : memref<4x32xf32>
    %20 = ktdp.construct_access_tile %19[%c0, %c0] {access_tile_order = #map4, access_tile_set = #set4} : memref<4x32xf32> -> !ktdp.access_tile<4x1xindex>
    %21 = ktdp.load %20 : <4x1xindex> -> tensor<4x1xf32>
    %22 = tensor.empty() : tensor<4x4x32xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map5, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %21 : tensor<4x4x32xf32>, tensor<4x1xf32>) outs(%22 : tensor<4x4x32xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %51 = arith.subf %in, %in_4 : f32
      linalg.yield %51 : f32
    } -> tensor<4x4x32xf32>
    %24 = ktdp.construct_memory_view %c640, sizes: [4, 4, 32], strides: [128, 32, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<ct_local>} : memref<4x4x32xf32>
    %25 = ktdp.construct_access_tile %24[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<4x4x32xf32> -> !ktdp.access_tile<4x4x32xindex>
    ktdp.store %23, %25 : tensor<4x4x32xf32>, <4x4x32xindex>
    %26 = ktdp.construct_memory_view %c640, sizes: [4, 4, 32], strides: [128, 32, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<ct_local>} : memref<4x4x32xf32>
    %27 = ktdp.construct_access_tile %26[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<4x4x32xf32> -> !ktdp.access_tile<4x4x32xindex>
    %28 = ktdp.load %27 : <4x4x32xindex> -> tensor<4x4x32xf32>
    %29 = tensor.empty() : tensor<4x4x32xf32>
    %30 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28 : tensor<4x4x32xf32>) outs(%29 : tensor<4x4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = spyreop.exp %in : f32
      linalg.yield %51 : f32
    } -> tensor<4x4x32xf32>
    %31 = ktdp.construct_memory_view %c0_1, sizes: [4, 4, 32], strides: [128, 32, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<ct_local>} : memref<4x4x32xf32>
    %32 = ktdp.construct_access_tile %31[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<4x4x32xf32> -> !ktdp.access_tile<4x4x32xindex>
    ktdp.store %30, %32 : tensor<4x4x32xf32>, <4x4x32xindex>
    %33 = ktdp.construct_memory_view %c0_1, sizes: [4, 4, 32], strides: [128, 32, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<ct_local>} : memref<4x4x32xf32>
    %34 = ktdp.construct_access_tile %33[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set2} : memref<4x4x32xf32> -> !ktdp.access_tile<4x4x32xindex>
    %35 = ktdp.load %34 : <4x4x32xindex> -> tensor<4x4x32xf32>
    %36 = tensor.empty() : tensor<4x32xf32>
    %37 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%35 : tensor<4x4x32xf32>) outs(%36 : tensor<4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %51 = arith.addf %out, %in : f32
      linalg.yield %51 : f32
    } -> tensor<4x32xf32>
    %38 = ktdp.construct_memory_view %arg1, sizes: [128, 32], strides: [32, 1] {coordinate_set = affine_set<(r, l) : (r >= 0, -r + 127 >= 0, l >= 0, -l + 31 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<128x32xf32>
    %39 = ktdp.construct_access_tile %38[%1, %c0] {access_tile_order = #map4, access_tile_set = #set3} : memref<128x32xf32> -> !ktdp.access_tile<4x32xindex>
    ktdp.store %37, %39 : tensor<4x32xf32>, <4x32xindex>
    return
  }
