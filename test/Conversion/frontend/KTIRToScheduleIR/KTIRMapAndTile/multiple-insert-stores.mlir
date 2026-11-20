// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-map-and-tile --split-input-file | FileCheck %s --implicit-check-not=tensor.insert_slice
// Sinking each update must preserve packet origins and program order.

#id1 = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0,d1) -> (d0,d1)>
ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL: func.func @separate_packets
// CHECK: %[[FIRST:.*]] = arith.muli
// CHECK: %[[SECOND:.*]] = arith.addi
// CHECK: %[[A:.*]] = linalg.generic
// CHECK: %[[B:.*]] = linalg.generic
// CHECK: %[[EXPANDED_A:.*]] = tensor.expand_shape %[[A]]
// CHECK: ktdp_lowering.store %[[EXPANDED_A]] into %{{.*}}[%[[FIRST]], 0] [1, 32] [1, 1]
// CHECK: %[[EXPANDED_B:.*]] = tensor.expand_shape %[[B]]
// CHECK: ktdp_lowering.store %[[EXPANDED_B]] into %{{.*}}[%[[SECOND]], 0] [1, 32] [1, 1]
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func @separate_packets(%addr: index, %input: tensor<32xf32>, %other: tensor<32xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %mem = ktdp.construct_memory_view %addr, sizes: [6,32], strides: [32,1] {
    coordinate_set = affine_set<(k,e) : (k >= 0, 5-k >= 0, e >= 0, 31-e >= 0)>,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x32xf32>
  %tile = ktdp.construct_access_tile %mem[0,0] {
    access_tile_set = affine_set<(k,e) : (k >= 0, 5-k >= 0, e >= 0, 31-e >= 0)>,
    access_tile_order = #id2
  } : memref<6x32xf32> -> !ktdp.access_tile<6x32xindex>
  %init = tensor.empty() : tensor<6x32xf32>
  %result = scf.for %i = %zero to %three step %one iter_args(%out = %init) -> tensor<6x32xf32> {
    %first = arith.muli %i, %two : index
    %second = arith.addi %first, %one : index
    %a_init = tensor.empty() : tensor<32xf32>
    %a = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
        ins(%input : tensor<32xf32>) outs(%a_init : tensor<32xf32>) {
    ^bb0(%x: f32, %y: f32):
      %v = arith.negf %x : f32
      linalg.yield %v : f32
    } -> tensor<32xf32>
    %b_init = tensor.empty() : tensor<32xf32>
    %b = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
        ins(%other : tensor<32xf32>) outs(%b_init : tensor<32xf32>) {
    ^bb0(%x: f32, %y: f32):
      %v = arith.negf %x : f32
      linalg.yield %v : f32
    } -> tensor<32xf32>
    %a_out = tensor.insert_slice %a into %out[%first,0] [1, 32] [1,1]
      : tensor<32xf32> into tensor<6x32xf32>
    %b_out = tensor.insert_slice %b into %a_out[%second,0] [1,32] [1,1]
      : tensor<32xf32> into tensor<6x32xf32>
    scf.yield %b_out : tensor<6x32xf32>
  }
  ktdp.store %result, %tile : tensor<6x32xf32>, !ktdp.access_tile<6x32xindex>
  return
}

// -----

#id1 = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0,d1) -> (d0,d1)>
ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL: func.func @overlapping_updates
// CHECK: %[[FIRST:.*]] = arith.muli
// CHECK: %[[A:.*]] = linalg.generic
// CHECK: %[[B:.*]] = linalg.generic
// CHECK: ktdp_lowering.store %[[A]] into %{{.*}}[%[[FIRST]], 0] [2, 32] [1, 1]
// CHECK: %[[EXPANDED_B:.*]] = tensor.expand_shape %[[B]]
// CHECK: ktdp_lowering.store %[[EXPANDED_B]] into %{{.*}}[%[[FIRST]], 0] [1, 32] [1, 1]
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func @overlapping_updates(%addr: index, %input: tensor<2x32xf32>, %other: tensor<32xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %mem = ktdp.construct_memory_view %addr, sizes: [6,32], strides: [32,1] {
    coordinate_set = affine_set<(k,e) : (k >= 0, 5-k >= 0, e >= 0, 31-e >= 0)>,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x32xf32>
  %tile = ktdp.construct_access_tile %mem[0,0] {
    access_tile_set = affine_set<(k,e) : (k >= 0, 5-k >= 0, e >= 0, 31-e >= 0)>,
    access_tile_order = #id2
  } : memref<6x32xf32> -> !ktdp.access_tile<6x32xindex>
  %init = tensor.empty() : tensor<6x32xf32>
  %result = scf.for %i = %zero to %three step %one iter_args(%out = %init) -> tensor<6x32xf32> {
    %first = arith.muli %i, %two : index
    %second = arith.addi %first, %one : index
    %a_init = tensor.empty() : tensor<2x32xf32>
    %a = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%input : tensor<2x32xf32>) outs(%a_init : tensor<2x32xf32>) {
    ^bb0(%x: f32, %y: f32):
      %v = arith.negf %x : f32
      linalg.yield %v : f32
    } -> tensor<2x32xf32>
    %b_init = tensor.empty() : tensor<32xf32>
    %b = linalg.generic {indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
        ins(%other : tensor<32xf32>) outs(%b_init : tensor<32xf32>) {
    ^bb0(%x: f32, %y: f32):
      %v = arith.negf %x : f32
      linalg.yield %v : f32
    } -> tensor<32xf32>
    %a_out = tensor.insert_slice %a into %out[%first,0] [2, 32] [1,1]
      : tensor<2x32xf32> into tensor<6x32xf32>
    %b_out = tensor.insert_slice %b into %a_out[%first,0] [1,32] [1,1]
      : tensor<32xf32> into tensor<6x32xf32>
    scf.yield %b_out : tensor<6x32xf32>
  }
  ktdp.store %result, %tile : tensor<6x32xf32>, !ktdp.access_tile<6x32xindex>
  return
}
