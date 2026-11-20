// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-bufferize -verify-diagnostics --split-input-file -o /dev/null
// No load/store path may silently erase an unresolved element permutation.

ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
func.func @load(%mem: memref<32xf32>, %value: tensor<32xf32>) {
  // expected-error @+1 {{access_tile_order must be materialized before bufferization; no target data-movement recipe has been selected}}
  %tile = ktdp.construct_access_tile %mem[0] {
    access_tile_set = affine_set<(p) : (p >= 0, 31-p >= 0)>,
    access_tile_order = affine_map<(p) -> (31-p)>
  } : memref<32xf32> -> !ktdp.access_tile<32xindex>
  %v = ktdp.load %tile : !ktdp.access_tile<32xindex> -> tensor<32xf32>
  return
}

// -----

ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
func.func @store(%mem: memref<32xf32>, %value: tensor<32xf32>) {
  // expected-error @+1 {{access_tile_order must be materialized before bufferization; no target data-movement recipe has been selected}}
  %tile = ktdp.construct_access_tile %mem[0] {
    access_tile_set = affine_set<(p) : (p >= 0, 31-p >= 0)>,
    access_tile_order = affine_map<(p) -> (31-p)>
  } : memref<32xf32> -> !ktdp.access_tile<32xindex>
  ktdp.store %value, %tile : tensor<32xf32>, !ktdp.access_tile<32xindex>
  return
}

// -----

ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
func.func @tiled_load(%mem: memref<32xf32>, %value: tensor<32xf32>) {
  // expected-error @+1 {{access_tile_order must be materialized before bufferization; no target data-movement recipe has been selected}}
  %tile = ktdp.construct_access_tile %mem[0] {
    access_tile_set = affine_set<(p) : (p >= 0, 31-p >= 0)>,
    access_tile_order = affine_map<(p) -> (31-p)>
  } : memref<32xf32> -> !ktdp.access_tile<32xindex>
  %v = ktdp_lowering.load %tile [0] [32] [1] : !ktdp.access_tile<32xindex> -> tensor<32xf32>
  return
}

// -----

ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
func.func @tiled_store(%mem: memref<32xf32>, %value: tensor<32xf32>) {
  // expected-error @+1 {{access_tile_order must be materialized before bufferization; no target data-movement recipe has been selected}}
  %tile = ktdp.construct_access_tile %mem[0] {
    access_tile_set = affine_set<(p) : (p >= 0, 31-p >= 0)>,
    access_tile_order = affine_map<(p) -> (31-p)>
  } : memref<32xf32> -> !ktdp.access_tile<32xindex>
  ktdp_lowering.store %value into %tile [0] [32] [1] : tensor<32xf32> into !ktdp.access_tile<32xindex>
  return
}
