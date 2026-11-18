//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

// RUN: dataflow-scheduler-opt %s --ktir-numeric-cast-tiling --split-input-file --verify-diagnostics -o /dev/null
// Per-use recomputation rematerializes input loads. Aliased or derived address
// arguments and unknown effects require preserving a snapshot before this
// optimization is safe, so reject them before any cast is tiled.

ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

// expected-error @+1 {{branched numeric cast requires distinct, unmodified input/output address arguments}}
func.func @input_output_alias(%src_addr: index, %dst_addr: index, %branch_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %branch = ktdp.construct_memory_view %branch_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %branch_tile : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

// expected-error @+1 {{branched numeric cast requires distinct, unmodified input/output address arguments}}
func.func @shared_output_address(%src_addr: index, %dst_addr: index, %branch_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %branch = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %branch_tile : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

// expected-error @+1 {{branched numeric cast requires distinct, unmodified input/output address arguments}}
func.func @derived_input_address(%src_addr: index, %dst_addr: index, %branch_addr: index) attributes {grid = [1]} {
  %offset = arith.constant 128 : index
  %derived = arith.addi %src_addr, %offset : index
  %src = ktdp.construct_memory_view %derived,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %branch = ktdp.construct_memory_view %branch_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %branch_tile : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#cast_order = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r,
    (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4
)>

// expected-error @+1 {{branched numeric cast requires distinct, unmodified input/output address arguments}}
func.func @intervening_unknown_effect(%src_addr: index, %dst_addr: index, %branch_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64x!spyreop.df16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %branch = ktdp.construct_memory_view %branch_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64x!spyreop.df16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64x!spyreop.df16>

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64x!spyreop.df16> -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #cast_order
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  func.call @unknown_effect() : () -> ()
  %branch_tile = ktdp.construct_access_tile %branch[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %branch_tile : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

func.func private @unknown_effect()
