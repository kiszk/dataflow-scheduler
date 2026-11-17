// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --split-input-file --ktir-numeric-cast-tiling --ktir-map-and-tile | FileCheck %s --check-prefix=MAP --implicit-check-not=tensor.insert_slice --implicit-check-not=iter_args
// RUN: dataflow-scheduler-opt %s --split-input-file --ktir-numeric-cast-tiling --ktir-map-and-tile --ktir-bufferize --ktir-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=tensor.insert_slice --implicit-check-not=iter_args
// Six groups on two cores exercise head, middle and tail loop-result chains.
// Core 0 owns head groups 0/1 and tail group 2; core 1 owns head group 3 and
// middle groups 4/5, with an empty tail. Every segment must retain its load,
// native cast and stores after MapAndTile. FP32 uses identity access order
// here to isolate transfer tiling from Standard-order materialization.

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")

// MAP-LABEL: func.func @widen_physical_multicore_segments(
// MAP: ktdp.get_compute_tile_id
// MAP: scf.for %[[HEAD:.*]] = {{.*}} {
// MAP: ktdp_lowering.load {{.*}}[{{.*}}, %[[HEAD]], 0] [1, 1, 64] [1, 1, 1]
// MAP: ktdf_lowering.numeric_cast_value
// MAP-SAME: kind = "dl16tofp32"
// MAP: ktdp_lowering.store {{.*}}[{{.*}}, %[[HEAD]], 0] [1, 1, 32] [1, 1, 1]
// MAP: ktdp_lowering.store {{.*}}[{{.*}}, %[[HEAD]], 0] [1, 1, 32] [1, 1, 1]
// MAP-NEXT: {{^ *}}}
// MAP: scf.for %[[CHUNK:.*]] = {{.*}} {
// MAP: scf.for %[[MIDDLE:.*]] = {{.*}} {
// MAP: ktdp_lowering.load {{.*}}[{{.*}}, %[[MIDDLE]], 0] [1, 1, 64] [1, 1, 1]
// MAP: ktdf_lowering.numeric_cast_value
// MAP-SAME: kind = "dl16tofp32"
// MAP: ktdp_lowering.store {{.*}}[{{.*}}, %[[MIDDLE]], 0] [1, 1, 32] [1, 1, 1]
// MAP: ktdp_lowering.store {{.*}}[{{.*}}, %[[MIDDLE]], 0] [1, 1, 32] [1, 1, 1]
// MAP-NEXT: {{^ *}}}
// MAP-NEXT: {{^ *}}}
// MAP: scf.for %[[TAIL:.*]] = {{.*}} {
// MAP: ktdp_lowering.load {{.*}}[{{.*}}, %[[TAIL]], 0] [1, 1, 64] [1, 1, 1]
// MAP: ktdf_lowering.numeric_cast_value
// MAP-SAME: kind = "dl16tofp32"
// MAP: ktdp_lowering.store {{.*}}[{{.*}}, %[[TAIL]], 0] [1, 1, 32] [1, 1, 1]
// MAP: ktdp_lowering.store {{.*}}[{{.*}}, %[[TAIL]], 0] [1, 1, 32] [1, 1, 1]
// MAP-NEXT: {{^ *}}}
// MAP-NEXT: return
// PIPE-LABEL: func.func @widen_physical_multicore_segments(
// PIPE: scf.for %[[PHEAD:.*]] =
// PIPE: ktdf.pipeline {
// PIPE: ktdf.data_transfer from %{{.*}}[{{.*}}, %[[PHEAD]], 0] size [1, 1, 64]
// PIPE: ktdf_lowering.numeric_cast_value
// PIPE-SAME: kind = "dl16tofp32"
// PIPE: ktdf.data_transfer from %{{.*}} size [1, 1, 32] to %{{.*}}[{{.*}}, %[[PHEAD]], 0] size [1, 1, 32]
// PIPE: ktdf.data_transfer from %{{.*}} size [1, 1, 32] to %{{.*}}[{{.*}}, %[[PHEAD]], 0] size [1, 1, 32]
// PIPE: scf.for %[[PCHUNK:.*]] =
// PIPE: scf.for %[[PMIDDLE:.*]] =
// PIPE: ktdf.pipeline {
// PIPE: ktdf.data_transfer from %{{.*}}[{{.*}}, %[[PMIDDLE]], 0] size [1, 1, 64]
// PIPE: ktdf_lowering.numeric_cast_value
// PIPE-SAME: kind = "dl16tofp32"
// PIPE: ktdf.data_transfer from %{{.*}} size [1, 1, 32] to %{{.*}}[{{.*}}, %[[PMIDDLE]], 0] size [1, 1, 32]
// PIPE: ktdf.data_transfer from %{{.*}} size [1, 1, 32] to %{{.*}}[{{.*}}, %[[PMIDDLE]], 0] size [1, 1, 32]
// PIPE: scf.for %[[PTAIL:.*]] =
// PIPE: ktdf.pipeline {
// PIPE: ktdf.data_transfer from %{{.*}}[{{.*}}, %[[PTAIL]], 0] size [1, 1, 64]
// PIPE: ktdf_lowering.numeric_cast_value
// PIPE-SAME: kind = "dl16tofp32"
// PIPE: ktdf.data_transfer from %{{.*}} size [1, 1, 32] to %{{.*}}[{{.*}}, %[[PTAIL]], 0] size [1, 1, 32]
// PIPE: ktdf.data_transfer from %{{.*}} size [1, 1, 32] to %{{.*}}[{{.*}}, %[[PTAIL]], 0] size [1, 1, 32]
// PIPE: return
func.func @widen_physical_multicore_segments(%src_addr: index, %dst_addr: index) attributes {grid = [2]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>

  %staggered = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64xf16> -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %staggered, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}

// -----

#dl16_domain = affine_set<(s, r, l) :
    (s >= 0, 2 - s >= 0, r >= 0, 1 - r >= 0,
     l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0,
     e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")

// MAP-LABEL: func.func @narrow_physical_multicore_segments(
// MAP: ktdp.get_compute_tile_id
// MAP: scf.for %[[HEAD:.*]] = {{.*}} {
// MAP: ktdp_lowering.load {{.*}}[{{.*}}, %[[HEAD]], 0] [1, 1, 32] [1, 1, 1]
// MAP: ktdp_lowering.load {{.*}}[{{.*}}, %[[HEAD]], 0] [1, 1, 32] [1, 1, 1]
// MAP: ktdf_lowering.numeric_cast_value
// MAP-SAME: kind = "fp32todl16"
// MAP: ktdp_lowering.store {{.*}}[{{.*}}, %[[HEAD]], 0] [1, 1, 64] [1, 1, 1]
// MAP-NEXT: {{^ *}}}
// MAP: scf.for %[[CHUNK:.*]] = {{.*}} {
// MAP: scf.for %[[MIDDLE:.*]] = {{.*}} {
// MAP: ktdp_lowering.load {{.*}}[{{.*}}, %[[MIDDLE]], 0] [1, 1, 32] [1, 1, 1]
// MAP: ktdp_lowering.load {{.*}}[{{.*}}, %[[MIDDLE]], 0] [1, 1, 32] [1, 1, 1]
// MAP: ktdf_lowering.numeric_cast_value
// MAP-SAME: kind = "fp32todl16"
// MAP: ktdp_lowering.store {{.*}}[{{.*}}, %[[MIDDLE]], 0] [1, 1, 64] [1, 1, 1]
// MAP-NEXT: {{^ *}}}
// MAP-NEXT: {{^ *}}}
// MAP: scf.for %[[TAIL:.*]] = {{.*}} {
// MAP: ktdp_lowering.load {{.*}}[{{.*}}, %[[TAIL]], 0] [1, 1, 32] [1, 1, 1]
// MAP: ktdp_lowering.load {{.*}}[{{.*}}, %[[TAIL]], 0] [1, 1, 32] [1, 1, 1]
// MAP: ktdf_lowering.numeric_cast_value
// MAP-SAME: kind = "fp32todl16"
// MAP: ktdp_lowering.store {{.*}}[{{.*}}, %[[TAIL]], 0] [1, 1, 64] [1, 1, 1]
// MAP-NEXT: {{^ *}}}
// MAP-NEXT: return
// PIPE-LABEL: func.func @narrow_physical_multicore_segments(
// PIPE: scf.for %[[PHEAD:.*]] =
// PIPE: ktdf.pipeline {
// PIPE: ktdf.data_transfer from %{{.*}}[{{.*}}, %[[PHEAD]], 0] size [1, 1, 32]
// PIPE: ktdf.data_transfer from %{{.*}}[{{.*}}, %[[PHEAD]], 0] size [1, 1, 32]
// PIPE: ktdf_lowering.numeric_cast_value
// PIPE-SAME: kind = "fp32todl16"
// PIPE: ktdf.data_transfer from %{{.*}} size [1, 1, 64] to %{{.*}}[{{.*}}, %[[PHEAD]], 0] size [1, 1, 64]
// PIPE: scf.for %[[PCHUNK:.*]] =
// PIPE: scf.for %[[PMIDDLE:.*]] =
// PIPE: ktdf.pipeline {
// PIPE: ktdf.data_transfer from %{{.*}}[{{.*}}, %[[PMIDDLE]], 0] size [1, 1, 32]
// PIPE: ktdf.data_transfer from %{{.*}}[{{.*}}, %[[PMIDDLE]], 0] size [1, 1, 32]
// PIPE: ktdf_lowering.numeric_cast_value
// PIPE-SAME: kind = "fp32todl16"
// PIPE: ktdf.data_transfer from %{{.*}} size [1, 1, 64] to %{{.*}}[{{.*}}, %[[PMIDDLE]], 0] size [1, 1, 64]
// PIPE: scf.for %[[PTAIL:.*]] =
// PIPE: ktdf.pipeline {
// PIPE: ktdf.data_transfer from %{{.*}}[{{.*}}, %[[PTAIL]], 0] size [1, 1, 32]
// PIPE: ktdf.data_transfer from %{{.*}}[{{.*}}, %[[PTAIL]], 0] size [1, 1, 32]
// PIPE: ktdf_lowering.numeric_cast_value
// PIPE-SAME: kind = "fp32todl16"
// PIPE: ktdf.data_transfer from %{{.*}} size [1, 1, 64] to %{{.*}}[{{.*}}, %[[PTAIL]], 0] size [1, 1, 64]
// PIPE: return
func.func @narrow_physical_multicore_segments(%src_addr: index, %dst_addr: index) attributes {grid = [2]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain,
    memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #fp32_domain,
    access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  %staggered = ktdp.load %input_tile
      : !ktdp.access_tile<6x2x32xindex> -> tensor<6x2x32xf32>

  %converted = spyreop.numeric_cast %staggered {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64xf16>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #dl16_domain,
    access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  ktdp.store %converted, %output_tile
      : tensor<3x2x64xf16>, !ktdp.access_tile<3x2x64xindex>
  return
}
