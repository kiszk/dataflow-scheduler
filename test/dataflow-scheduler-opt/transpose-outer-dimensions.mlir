// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --disable-double-buffering --split-dfir-output-dir=%t | FileCheck %s --implicit-check-not=vectorchain.pack --implicit-check-not=vector.shuffle

// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t.double | FileCheck %s --check-prefix=DOUBLE --implicit-check-not=vectorchain.pack --implicit-check-not=iter_args --implicit-check-not=arith.subi

// With double buffering, the three outer iterations exercise a full cycle
// and its odd tail. Buffer offsets must be static before target codegen.
// DOUBLE-LABEL: func.func @local_schedule_0_body
// DOUBLE: agen.composite_load_and_store
// DOUBLE: load_iv(%{{.*}}:vector<64xf16>)
// DOUBLE-LABEL: func.func @local_schedule_1_body
// DOUBLE: agen.composite_load_and_store
// DOUBLE: load_iv(%{{.*}}:vector<32xf32>)
// DOUBLE-LABEL: func.func @local_schedule_2_body
// DOUBLE: agen.composite_load_and_store
// DOUBLE: load_iv(%{{.*}}:vector<32xi32>)

// Changing only outer axes preserves each complete 128-byte stick. No
// lane shuffle or transpose scratch is needed for f16, f32 or i32.
#id = affine_map<(j, i, l) -> (j, i, l)>
#outer = affine_map<(j, i, l) -> (i, j, l)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")

  // CHECK-LABEL: func.func @local_schedule_0_body
  // CHECK: scf.for %[[J0:.*]] =
  // CHECK: scf.for %[[I0:.*]] =
  // CHECK: agen.composite_load_and_store src:%{{.*}}[%[[I0]], %[[J0]], 0]
  // CHECK: load_iv(%{{.*}}:vector<64xf16>)
  // CHECK: } : memref<2x3x64xf16>,
  // CHECK: agen.composite_load_and_store src:{{.*}} dst:{{.*}}
  // CHECK: load_iv(%{{.*}}:vector<64xf16>)
  // CHECK: } : {{.*}}, memref<3x2x64xf16>
  func.func @outer_f16(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
    %src = ktdp.construct_memory_view %src_addr, sizes: [2, 3, 64], strides: [192, 64, 1] {coordinate_set = affine_set<(i, j, l) : (i >= 0, 1-i >= 0, j >= 0, 2-j >= 0, l >= 0, 63-l >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<2x3x64xf16>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [3, 2, 64], strides: [128, 64, 1] {coordinate_set = affine_set<(j, i, l) : (j >= 0, 2-j >= 0, i >= 0, 1-i >= 0, l >= 0, 63-l >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<3x2x64xf16>
    %a = ktdp.construct_access_tile %src[0, 0, 0] {access_tile_order = #id, access_tile_set = affine_set<(i, j, l) : (i >= 0, 1-i >= 0, j >= 0, 2-j >= 0, l >= 0, 63-l >= 0)>} : memref<2x3x64xf16> -> !ktdp.access_tile<2x3x64xindex>
    %x = ktdp.load %a : !ktdp.access_tile<2x3x64xindex> -> tensor<2x3x64xf16>
    %init = tensor.empty() : tensor<3x2x64xf16>
    %y = linalg.generic {indexing_maps = [#outer, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%x : tensor<2x3x64xf16>) outs(%init : tensor<3x2x64xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %copy = spyreop.identity %value : f16
      linalg.yield %copy : f16
    } -> tensor<3x2x64xf16>
    %b = ktdp.construct_access_tile %dst[0, 0, 0] {access_tile_order = #id, access_tile_set = affine_set<(j, i, l) : (j >= 0, 2-j >= 0, i >= 0, 1-i >= 0, l >= 0, 63-l >= 0)>} : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
    ktdp.store %y, %b : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>
    return
  }

  // CHECK-LABEL: func.func @local_schedule_1_body
  // CHECK: scf.for %[[J1:.*]] =
  // CHECK: scf.for %[[I1:.*]] =
  // CHECK: agen.composite_load_and_store src:%{{.*}}[%[[I1]], %[[J1]], 0]
  // CHECK: load_iv(%{{.*}}:vector<32xf32>)
  // CHECK: } : memref<2x3x32xf32>,
  // CHECK: agen.composite_load_and_store src:{{.*}} dst:{{.*}}
  // CHECK: load_iv(%{{.*}}:vector<32xf32>)
  // CHECK: } : {{.*}}, memref<3x2x32xf32>
  func.func @outer_f32(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
    %src = ktdp.construct_memory_view %src_addr, sizes: [2, 3, 32], strides: [96, 32, 1] {coordinate_set = affine_set<(i, j, l) : (i >= 0, 1-i >= 0, j >= 0, 2-j >= 0, l >= 0, 31-l >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<2x3x32xf32>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [3, 2, 32], strides: [64, 32, 1] {coordinate_set = affine_set<(j, i, l) : (j >= 0, 2-j >= 0, i >= 0, 1-i >= 0, l >= 0, 31-l >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<3x2x32xf32>
    %a = ktdp.construct_access_tile %src[0, 0, 0] {access_tile_order = #id, access_tile_set = affine_set<(i, j, l) : (i >= 0, 1-i >= 0, j >= 0, 2-j >= 0, l >= 0, 31-l >= 0)>} : memref<2x3x32xf32> -> !ktdp.access_tile<2x3x32xindex>
    %x = ktdp.load %a : !ktdp.access_tile<2x3x32xindex> -> tensor<2x3x32xf32>
    %init = tensor.empty() : tensor<3x2x32xf32>
    %y = linalg.generic {indexing_maps = [#outer, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%x : tensor<2x3x32xf32>) outs(%init : tensor<3x2x32xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %copy = spyreop.identity %value : f32
      linalg.yield %copy : f32
    } -> tensor<3x2x32xf32>
    %b = ktdp.construct_access_tile %dst[0, 0, 0] {access_tile_order = #id, access_tile_set = affine_set<(j, i, l) : (j >= 0, 2-j >= 0, i >= 0, 1-i >= 0, l >= 0, 31-l >= 0)>} : memref<3x2x32xf32> -> !ktdp.access_tile<3x2x32xindex>
    ktdp.store %y, %b : tensor<3x2x32xf32>, !ktdp.access_tile<3x2x32xindex>
    return
  }

  // CHECK-LABEL: func.func @local_schedule_2_body
  // CHECK: scf.for %[[J2:.*]] =
  // CHECK: scf.for %[[I2:.*]] =
  // CHECK: agen.composite_load_and_store src:%{{.*}}[%[[I2]], %[[J2]], 0]
  // CHECK: load_iv(%{{.*}}:vector<32xi32>)
  // CHECK: } : memref<2x3x32xi32>,
  // CHECK: agen.composite_load_and_store src:{{.*}} dst:{{.*}}
  // CHECK: load_iv(%{{.*}}:vector<32xi32>)
  // CHECK: } : {{.*}}, memref<3x2x32xi32>
  func.func @outer_i32(%src_addr: index, %dst_addr: index) attributes {grid = [1]} {
    %src = ktdp.construct_memory_view %src_addr, sizes: [2, 3, 32], strides: [96, 32, 1] {coordinate_set = affine_set<(i, j, l) : (i >= 0, 1-i >= 0, j >= 0, 2-j >= 0, l >= 0, 31-l >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<2x3x32xi32>
    %dst = ktdp.construct_memory_view %dst_addr, sizes: [3, 2, 32], strides: [64, 32, 1] {coordinate_set = affine_set<(j, i, l) : (j >= 0, 2-j >= 0, i >= 0, 1-i >= 0, l >= 0, 31-l >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<3x2x32xi32>
    %a = ktdp.construct_access_tile %src[0, 0, 0] {access_tile_order = #id, access_tile_set = affine_set<(i, j, l) : (i >= 0, 1-i >= 0, j >= 0, 2-j >= 0, l >= 0, 31-l >= 0)>} : memref<2x3x32xi32> -> !ktdp.access_tile<2x3x32xindex>
    %x = ktdp.load %a : !ktdp.access_tile<2x3x32xindex> -> tensor<2x3x32xi32>
    %init = tensor.empty() : tensor<3x2x32xi32>
    %y = linalg.generic {indexing_maps = [#outer, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%x : tensor<2x3x32xi32>) outs(%init : tensor<3x2x32xi32>) {
    ^bb0(%value: i32, %unused: i32):
      %copy = spyreop.identity %value : i32
      linalg.yield %copy : i32
    } -> tensor<3x2x32xi32>
    %b = ktdp.construct_access_tile %dst[0, 0, 0] {access_tile_order = #id, access_tile_set = affine_set<(j, i, l) : (j >= 0, 2-j >= 0, i >= 0, 1-i >= 0, l >= 0, 31-l >= 0)>} : memref<3x2x32xi32> -> !ktdp.access_tile<3x2x32xindex>
    ktdp.store %y, %b : tensor<3x2x32xi32>, !ktdp.access_tile<3x2x32xindex>
    return
  }
}
