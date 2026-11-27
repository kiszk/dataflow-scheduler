// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --cse --ktir-map-and-tile | FileCheck %s

// A partial store preserves the destination loaded before the update. Equal
// per-core origins must not cause a full self-copy, even when their arithmetic
// is emitted separately. Different origins must still preserve the old data.

#id = affine_map<(r, l) -> (r, l)>
#physical = affine_map<(r, s, l) -> (r, s, l)>
#global = affine_set<(r, s, l) : (r >= 0, 127-r >= 0, s >= 0, 1-s >= 0, l >= 0, 63-l >= 0)>
#local = affine_set<(r, s, l) : (r >= 0, 3-r >= 0, s >= 0, 1-s >= 0, l >= 0, 63-l >= 0)>
ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL: func.func @same_origin(
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[WIDTH:.*]] = arith.constant 64 : index
// CHECK: %[[ORIGIN:.*]] = arith.muli
// CHECK: %[[TILE:.*]] = ktdp.construct_access_tile %{{.*}}[%[[ORIGIN]], 0, 0]
// CHECK-NOT: ktdp.load
// CHECK-NOT: iter_args
// CHECK: scf.for %[[ROW:.*]] = {{.*}} {
// CHECK-NEXT: scf.for %[[LANE:.*]] = %[[ZERO]] to %[[WIDTH]] step %[[WIDTH]] {
// CHECK: linalg.generic
// CHECK: ktdp_lowering.store %{{.*}} into %[[TILE]][%[[ROW]], 0, %[[LANE]]] [1, 1, 64] [1, 1, 1]
// CHECK-NOT: ktdp.store
// CHECK-NOT: tensor.insert_slice
// CHECK: return
func.func @same_origin(%addr: index, %input: tensor<4x64xf16>) attributes {grid = [32]} {
  %four = arith.constant 4 : index
  %core = ktdp.get_compute_tile_id : index
  %view = ktdp.construct_memory_view %addr, sizes: [128, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #global, memory_space = #ktdp.memory_space<global>
  } : memref<128x2x64xf16>
  %load_origin = arith.muli %core, %four : index
  %load_tile = ktdp.construct_access_tile %view[%load_origin, 0, 0] {
    access_tile_set = #local, access_tile_order = #physical
  } : memref<128x2x64xf16> -> !ktdp.access_tile<4x2x64xindex>
  %initial = ktdp.load %load_tile : !ktdp.access_tile<4x2x64xindex> -> tensor<4x2x64xf16>
  %empty = tensor.empty() : tensor<4x64xf16>
  %computed = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<4x64xf16>) outs(%empty : tensor<4x64xf16>) {
  ^bb0(%x: f16, %unused: f16):
    %neg = arith.negf %x : f16
    linalg.yield %neg : f16
  } -> tensor<4x64xf16>
  %updated = tensor.insert_slice %computed into %initial[0, 0, 0] [4, 1, 64] [1, 1, 1]
    : tensor<4x64xf16> into tensor<4x2x64xf16>
  %store_origin = arith.muli %core, %four : index
  %store_tile = ktdp.construct_access_tile %view[%store_origin, 0, 0] {
    access_tile_set = #local, access_tile_order = #physical
  } : memref<128x2x64xf16> -> !ktdp.access_tile<4x2x64xindex>
  ktdp.store %updated, %store_tile : tensor<4x2x64xf16>, !ktdp.access_tile<4x2x64xindex>
  return
}

// CHECK-LABEL: func.func @different_origin(
// CHECK: %[[INITIAL:.*]] = ktdp.load
// CHECK: %[[TILE:.*]] = ktdp.construct_access_tile %{{.*}}[4, 0, 0]
// CHECK: ktdp.store %[[INITIAL]], %[[TILE]]
// CHECK-NEXT: ktdp_lowering.store %{{.*}} into %[[TILE]][0, 0, 0] [4, 1, 64] [1, 1, 1]
// CHECK: return
func.func @different_origin(%addr: index, %input: tensor<4x64xf16>) {
  %view = ktdp.construct_memory_view %addr, sizes: [128, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #global, memory_space = #ktdp.memory_space<global>
  } : memref<128x2x64xf16>
  %load_tile = ktdp.construct_access_tile %view[0, 0, 0] {
    access_tile_set = #local, access_tile_order = #physical
  } : memref<128x2x64xf16> -> !ktdp.access_tile<4x2x64xindex>
  %initial = ktdp.load %load_tile : !ktdp.access_tile<4x2x64xindex> -> tensor<4x2x64xf16>
  %empty = tensor.empty() : tensor<4x64xf16>
  %computed = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<4x64xf16>) outs(%empty : tensor<4x64xf16>) {
  ^bb0(%x: f16, %unused: f16):
    %neg = arith.negf %x : f16
    linalg.yield %neg : f16
  } -> tensor<4x64xf16>
  %updated = tensor.insert_slice %computed into %initial[0, 0, 0] [4, 1, 64] [1, 1, 1]
    : tensor<4x64xf16> into tensor<4x2x64xf16>
  %store_tile = ktdp.construct_access_tile %view[4, 0, 0] {
    access_tile_set = #local, access_tile_order = #physical
  } : memref<128x2x64xf16> -> !ktdp.access_tile<4x2x64xindex>
  ktdp.store %updated, %store_tile : tensor<4x2x64xf16>, !ktdp.access_tile<4x2x64xindex>
  return
}
