// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --fuse-linalg | FileCheck %s

#id = affine_map<(r, l) -> (r, l)>
#second = affine_map<(r, l) -> (r + 1, l)>

// The unused scalar init does not make the whole tensor dead: row zero is
// outside the write map and must retain its initial value.
// CHECK-LABEL: func.func @partial_output(
// CHECK-SAME: %[[INIT:.*]]: tensor<2x64xf16>, %[[PATCH:.*]]: tensor<1x64xf16>
// CHECK-NOT: tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[INIT]] : tensor<2x64xf16>)
func.func @partial_output(%initial: tensor<2x64xf16>, %patch: tensor<1x64xf16>) -> tensor<2x64xf16> {
  %result = linalg.generic {indexing_maps = [#id, #second], iterator_types = ["parallel", "parallel"]} ins(%patch : tensor<1x64xf16>) outs(%initial : tensor<2x64xf16>) {
  ^bb0(%x: f16, %unused: f16):
    %copy = spyreop.identity %x : f16
    linalg.yield %copy : f16
  } -> tensor<2x64xf16>
  return %result : tensor<2x64xf16>
}

// An actual full overwrite can still discard the unused initializer.
// CHECK-LABEL: func.func @full_output(
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[EMPTY]] : tensor<2x64xf16>)
func.func @full_output(%initial: tensor<2x64xf16>, %patch: tensor<2x64xf16>) -> tensor<2x64xf16> {
  %result = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%patch : tensor<2x64xf16>) outs(%initial : tensor<2x64xf16>) {
  ^bb0(%x: f16, %unused: f16):
    %copy = spyreop.identity %x : f16
    linalg.yield %copy : f16
  } -> tensor<2x64xf16>
  return %result : tensor<2x64xf16>
}
