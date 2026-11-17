// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-map-and-tile | FileCheck %s --implicit-check-not=tensor.insert_slice --implicit-check-not=iter_args

#id1 = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#domain = affine_set<(d0, d1) : (d0 >= 0, 5 - d0 >= 0, d1 >= 0, 63 - d1 >= 0)>
ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// Each segment initializes the next. The middle has a nested row loop, and
// any segment (in particular the tail) can have zero iterations. Sinking the
// final store must preserve the updates in all three segments.
// CHECK-LABEL: func.func @three_segments(
// CHECK-SAME: %[[ADDR:.*]]: index, %[[DATA:.*]]: tensor<1x64xf16>
// CHECK: %[[TILE:.*]] = ktdp.construct_access_tile
// CHECK: scf.for %[[HEAD:.*]] = {{.*}} {
// CHECK: %[[HEAD_DATA:.*]] = linalg.generic
// CHECK: ktdp_lowering.store %[[HEAD_DATA]] into %[[TILE]][%[[HEAD]], 0] [1, 64] [1, 1]
// CHECK-NEXT: {{^ *}}}
// CHECK: scf.for %[[CHUNK:.*]] = {{.*}} {
// CHECK: scf.for %[[ROW:.*]] = {{.*}} {
// CHECK: %[[BASE:.*]] = arith.muli %[[CHUNK]],
// CHECK: %[[MIDDLE:.*]] = arith.addi %[[BASE]], %[[ROW]]
// CHECK: %[[MIDDLE_DATA:.*]] = linalg.generic
// CHECK: ktdp_lowering.store %[[MIDDLE_DATA]] into %[[TILE]][%[[MIDDLE]], 0] [1, 64] [1, 1]
// CHECK-NEXT: {{^ *}}}
// CHECK-NEXT: {{^ *}}}
// CHECK: scf.for %[[TAIL:.*]] = {{.*}} {
// CHECK: %[[TAIL_DATA:.*]] = linalg.generic
// CHECK: ktdp_lowering.store %[[TAIL_DATA]] into %[[TILE]][%[[TAIL]], 0] [1, 64] [1, 1]
// CHECK-NEXT: {{^ *}}}
// CHECK-NEXT: return
func.func @three_segments(%addr: index, %data: tensor<1x64xf16>,
    %head_begin: index, %head_end: index, %middle_begin: index,
    %middle_end: index, %tail_begin: index, %tail_end: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %mem = ktdp.construct_memory_view %addr, sizes: [6, 64], strides: [64, 1] {
    coordinate_set = #domain, memory_space = #ktdp.memory_space<global>
  } : memref<6x64xf16>
  %tile = ktdp.construct_access_tile %mem[0, 0] {
    access_tile_set = #domain, access_tile_order = #id2
  } : memref<6x64xf16> -> !ktdp.access_tile<6x64xindex>
  %empty = tensor.empty() : tensor<6x64xf16>
  %head = scf.for %i = %head_begin to %head_end step %one
      iter_args(%out = %empty) -> tensor<6x64xf16> {
    %patch_init = tensor.empty() : tensor<1x64xf16>
    %patch = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%data : tensor<1x64xf16>) outs(%patch_init : tensor<1x64xf16>) {
    ^bb0(%x: f16, %y: f16):
      %neg = arith.negf %x : f16
      linalg.yield %neg : f16
    } -> tensor<1x64xf16>
    %updated = tensor.insert_slice %patch into %out[%i, 0] [1, 64] [1, 1]
        : tensor<1x64xf16> into tensor<6x64xf16>
    scf.yield %updated : tensor<6x64xf16>
  }
  %middle = scf.for %i = %middle_begin to %middle_end step %one
      iter_args(%out = %head) -> tensor<6x64xf16> {
    %rows = scf.for %r = %zero to %two step %one
        iter_args(%row_out = %out) -> tensor<6x64xf16> {
      %base = arith.muli %i, %two : index
      %offset = arith.addi %base, %r : index
      %patch_init = tensor.empty() : tensor<1x64xf16>
      %patch = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
          ins(%data : tensor<1x64xf16>) outs(%patch_init : tensor<1x64xf16>) {
      ^bb0(%x: f16, %y: f16):
        %neg = arith.negf %x : f16
        linalg.yield %neg : f16
      } -> tensor<1x64xf16>
      %updated = tensor.insert_slice %patch into %row_out[%offset, 0] [1, 64] [1, 1]
          : tensor<1x64xf16> into tensor<6x64xf16>
      scf.yield %updated : tensor<6x64xf16>
    }
    scf.yield %rows : tensor<6x64xf16>
  }
  %tail = scf.for %i = %tail_begin to %tail_end step %one
      iter_args(%out = %middle) -> tensor<6x64xf16> {
    %patch_init = tensor.empty() : tensor<1x64xf16>
    %patch = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%data : tensor<1x64xf16>) outs(%patch_init : tensor<1x64xf16>) {
    ^bb0(%x: f16, %y: f16):
      %neg = arith.negf %x : f16
      linalg.yield %neg : f16
    } -> tensor<1x64xf16>
    %updated = tensor.insert_slice %patch into %out[%i, 0] [1, 64] [1, 1]
        : tensor<1x64xf16> into tensor<6x64xf16>
    scf.yield %updated : tensor<6x64xf16>
  }
  ktdp.store %tail, %tile : tensor<6x64xf16>, !ktdp.access_tile<6x64xindex>
  return
}

// A non-empty initializer must also preserve the store's reshape and route.
// CHECK-LABEL: func.func @initialized_reshape_via(
// CHECK-SAME: %{{.*}}: index, %[[INIT:.*]]: tensor<1x128xf16>
// CHECK: %[[TILE:.*]] = ktdp.construct_access_tile
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[INIT]]
// CHECK: %[[VIA:.*]] = ktdf.via["A", "B"] %[[FLAT]]
// CHECK-NEXT: ktdp.store %[[VIA]], %[[TILE]]
// CHECK: scf.for {{.*}} {
// CHECK: ktdf.via["A", "B"]
// CHECK: ktdp_lowering.store {{.*}} into %[[TILE]]
// CHECK-NEXT: {{^ *}}}
// CHECK-NEXT: return
func.func @initialized_reshape_via(%addr: index, %initial: tensor<1x128xf16>,
    %data: tensor<1x64xf16>, %begin: index, %end: index) {
  %one = arith.constant 1 : index
  %mem = ktdp.construct_memory_view %addr, sizes: [128], strides: [1] {
    coordinate_set = affine_set<(d0) : (d0 >= 0, 127 - d0 >= 0)>,
    memory_space = #ktdp.memory_space<global>
  } : memref<128xf16>
  %tile = ktdp.construct_access_tile %mem[0] {
    access_tile_set = affine_set<(d0) : (d0 >= 0, 127 - d0 >= 0)>,
    access_tile_order = #id1
  } : memref<128xf16> -> !ktdp.access_tile<128xindex>
  %result = scf.for %i = %begin to %end step %one
      iter_args(%out = %initial) -> tensor<1x128xf16> {
    %patch_init = tensor.empty() : tensor<1x64xf16>
    %patch = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%data : tensor<1x64xf16>) outs(%patch_init : tensor<1x64xf16>) {
    ^bb0(%x: f16, %y: f16):
      %neg = arith.negf %x : f16
      linalg.yield %neg : f16
    } -> tensor<1x64xf16>
    %updated = tensor.insert_slice %patch into %out[0, %i] [1, 64] [1, 1]
        : tensor<1x64xf16> into tensor<1x128xf16>
    scf.yield %updated : tensor<1x128xf16>
  }
  %flat = tensor.collapse_shape %result [[0, 1]]
      : tensor<1x128xf16> into tensor<128xf16>
  %via = ktdf.via["A", "B"] %flat : tensor<128xf16>
  ktdp.store %via, %tile : tensor<128xf16>, !ktdp.access_tile<128xindex>
  return
}
