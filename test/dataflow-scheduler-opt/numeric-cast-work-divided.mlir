// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-numeric-cast-tiling | FileCheck %s --check-prefix=TILE
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR --implicit-check-not=spyreop.numeric_cast
// Each core already owns two rows. Numeric cast tiling must process the whole
// local tensor rather than dividing its one native stick across two cores.
// TILE-LABEL: func.func @cast_local_rows
// TILE: arith.constant 1 : index
// TILE: arith.constant 2 : index
// TILE: %[[ONE:.*]] = arith.constant 1 : index
// TILE: %[[TWO:.*]] = arith.constant 2 : index
// TILE-NOT: arith.divui
// TILE: scf.for {{.*}} to %[[ONE]]
// TILE: scf.for {{.*}} to %[[TWO]]
// TILE: ktdf_lowering.numeric_cast_value
// TILE-NOT: arith.divui
// DFIR-LABEL: func.func @local_schedule_0_body() attributes {grid = [2]}
// DFIR: dataflow.get_unit {core = 0
// DFIR: dataflow.get_unit {core = 1
// DFIR: vectorchain.cast

#memory_in = affine_set<(s, r, l) : (s == 0, r >= 0, 3 - r >= 0, l >= 0, 63 - l >= 0)>
#memory_out = affine_set<(p, r, l) : (p >= 0, 1 - p >= 0, r >= 0, 3 - r >= 0, l >= 0, 31 - l >= 0)>
#local_in = affine_set<(s, r, l) : (s == 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#local_out = affine_set<(p, r, l) : (p >= 0, 1 - p >= 0, r >= 0, 1 - r >= 0, l >= 0, 31 - l >= 0)>
#id = affine_map<(s, r, l) -> (s, r, l)>
#staggered = affine_map<(s, r, l) -> (s * 2 + (l mod 8) floordiv 4, r, (l floordiv 8) * 4 + l mod 4)>
ktdf_arch.device @sample_device import("Inputs/access-order-device.mlir")
func.func @cast_local_rows(%src: index, %dst: index) attributes {grid = [2]} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %core = ktdp.get_compute_tile_id : index
  %row = arith.muli %core, %c2 : index
  %input_view = ktdp.construct_memory_view %src, sizes: [1, 4, 64], strides: [256, 64, 1] {
    coordinate_set = #memory_in, memory_space = #ktdp.memory_space<global>
  } : memref<1x4x64xf16>
  %input_tile = ktdp.construct_access_tile %input_view[%c0, %row, %c0] {
    access_tile_set = #local_in, access_tile_order = #id
  } : memref<1x4x64xf16> -> !ktdp.access_tile<1x2x64xindex>
  %input = ktdp.load %input_tile : <1x2x64xindex> -> tensor<1x2x64xf16>
  %init = tensor.empty() : tensor<2x2x32xf32>
  %result = linalg.generic {
    indexing_maps = [#id, #staggered], iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<1x2x64xf16>) outs(%init : tensor<2x2x32xf32>) attrs = {dataflow_scheduler.work_divided} {
  ^bb0(%x: f16, %unused: f32):
    %cast = spyreop.numeric_cast %x {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<2x2x32xf32>
  %output_view = ktdp.construct_memory_view %dst, sizes: [2, 4, 32], strides: [128, 32, 1] {
    coordinate_set = #memory_out, memory_space = #ktdp.memory_space<global>
  } : memref<2x4x32xf32>
  %output_tile = ktdp.construct_access_tile %output_view[%c0, %row, %c0] {
    access_tile_set = #local_out, access_tile_order = #id
  } : memref<2x4x32xf32> -> !ktdp.access_tile<2x2x32xindex>
  ktdp.store %result, %output_tile : tensor<2x2x32xf32>, <2x2x32xindex>
  return
}
