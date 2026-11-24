// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -reuse-registers | FileCheck %s
// The pass must not alias overlapping, escaped or concurrently used registers.
#id = affine_map<(i) -> (i)>

// CHECK-LABEL: func.func @live_result(
// CHECK-COUNT-2: memref.alloca
// CHECK-NOT: memref.alloca
// CHECK: return
func.func @live_result(%x0: tensor<32xf32>, %x1: tensor<32xf32>, %out0: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, %out1: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %init = tensor.empty() : tensor<32xf32>
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      %r0 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x0 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r0: memref<32xf32, "SFU_REG">)
        %result = memref.load %r0[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      %r1 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%y0 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r1: memref<32xf32, "SFU_REG">)
        %result = memref.load %r1[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y1, %out1 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// CHECK-LABEL: func.func @alias(
// CHECK-COUNT-2: memref.alloca
// CHECK-NOT: memref.alloca
// CHECK: return
func.func @alias(%x0: tensor<32xf32>, %x1: tensor<32xf32>, %out0: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, %out1: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %init = tensor.empty() : tensor<32xf32>
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      %r0 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x0 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r0: memref<32xf32, "SFU_REG">)
        %result = memref.load %r0[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      %alias = memref.cast %r0 : memref<32xf32, "SFU_REG"> to memref<?xf32, "SFU_REG">
      ktdf.write_to_fifo %y0, %out0 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      %r1 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x1 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r1: memref<32xf32, "SFU_REG">)
        %result = memref.load %r1[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y1, %out1 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// CHECK-LABEL: func.func @parallel_stages(
// CHECK-COUNT-2: memref.alloca
// CHECK-NOT: memref.alloca
// CHECK: return
func.func @parallel_stages(%x0: tensor<32xf32>, %x1: tensor<32xf32>, %out0: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, %out1: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %init = tensor.empty() : tensor<32xf32>
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      %r0 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x0 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r0: memref<32xf32, "SFU_REG">)
        %result = memref.load %r0[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y0, %out0 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      %r1 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x1 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r1: memref<32xf32, "SFU_REG">)
        %result = memref.load %r1[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y1, %out1 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// CHECK-LABEL: func.func @multiple_sends(
// CHECK-COUNT-2: memref.alloca
// CHECK-NOT: memref.alloca
// CHECK: return
func.func @multiple_sends(%x0: tensor<32xf32>, %x1: tensor<32xf32>, %out0: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, %out1: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %init = tensor.empty() : tensor<32xf32>
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      %r0 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x0 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r0: memref<32xf32, "SFU_REG">)
        %result = memref.load %r0[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      %r1 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x1 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r1: memref<32xf32, "SFU_REG">)
        %result = memref.load %r1[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y0, %out0 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      ktdf.write_to_fifo %y0, %out1 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      ktdf.write_to_fifo %y1, %out1 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// CHECK-LABEL: func.func @different_unit(
// CHECK-COUNT-2: memref.alloca
// CHECK-NOT: memref.alloca
// CHECK: return
func.func @different_unit(%x0: tensor<32xf32>, %x1: tensor<32xf32>, %out0: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, %out1: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %init = tensor.empty() : tensor<32xf32>
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      %r0 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x0 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r0: memref<32xf32, "SFU_REG">)
        %result = memref.load %r0[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y0, %out0 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      %r1 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x1 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "OTHER"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r1: memref<32xf32, "SFU_REG">)
        %result = memref.load %r1[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y1, %out1 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// CHECK-LABEL: func.func @sequential_loops(
// CHECK-COUNT-1: memref.alloca
// CHECK-NOT: memref.alloca
// CHECK: return
func.func @sequential_loops(%x0: tensor<32xf32>, %x1: tensor<32xf32>, %out0: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, %out1: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %init = tensor.empty() : tensor<32xf32>
  scf.for %i = %zero to %two step %one {
    ktdf.pipeline modulo(size: 2) {
      ktdf.stage depends_in(none) depends_out(none) {
      %r0 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x0 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r0: memref<32xf32, "SFU_REG">)
        %result = memref.load %r0[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y0, %out0 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      } {applicable_units = ["SFU"]}
    }
  }
  scf.for %i = %zero to %two step %one {
    ktdf.pipeline modulo(size: 2) {
      ktdf.stage depends_in(none) depends_out(none) {
      %r1 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x1 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r1: memref<32xf32, "SFU_REG">)
        %result = memref.load %r1[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y1, %out1 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      } {applicable_units = ["SFU"]}
    }
  }
  return
}

// CHECK-LABEL: func.func @different_constants(
// CHECK-COUNT-2: memref.alloc()
// CHECK-NOT: memref.alloc
// CHECK: return
func.func @different_constants(%x: tensor<32xf32>, %out0: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, %out1: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  %init = tensor.empty() : tensor<32xf32>
  %r0 = memref.alloc() : memref<32xf32, "SFU_REG">
  %r1 = memref.alloc() : memref<32xf32, "SFU_REG">
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      linalg.fill ins(%one : f32) outs(%r0 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%two : f32) outs(%r1 : memref<32xf32, "SFU_REG">)
    } {applicable_units = ["SFU"]}
  }
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      %y0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%in: f32, %unused: f32):
        ktdf.opaque "read_constant" {dataflow_scheduler.register_names = ["constant"]}
          ins(%r0: memref<32xf32, "SFU_REG">) outs()
        linalg.yield %in : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y0, %out0 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      %y1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%in: f32, %unused: f32):
        ktdf.opaque "read_constant" {dataflow_scheduler.register_names = ["constant"]}
          ins(%r1: memref<32xf32, "SFU_REG">) outs()
        linalg.yield %in : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y1, %out1 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// CHECK-LABEL: func.func @written_constant(
// CHECK-COUNT-2: memref.alloc()
// CHECK-NOT: memref.alloc
// CHECK: return
func.func @written_constant(%x: tensor<32xf32>, %out0: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, %out1: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  %init = tensor.empty() : tensor<32xf32>
  %r0 = memref.alloc() : memref<32xf32, "SFU_REG">
  %r1 = memref.alloc() : memref<32xf32, "SFU_REG">
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      linalg.fill ins(%one : f32) outs(%r0 : memref<32xf32, "SFU_REG">)
      linalg.fill ins(%one : f32) outs(%r1 : memref<32xf32, "SFU_REG">)
    } {applicable_units = ["SFU"]}
  }
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      %y0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%in: f32, %unused: f32):
        memref.store %in, %r0[%zero] : memref<32xf32, "SFU_REG">
        ktdf.opaque "read_constant" {dataflow_scheduler.register_names = ["constant"]}
          ins(%r0: memref<32xf32, "SFU_REG">) outs()
        linalg.yield %in : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y0, %out0 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      %y1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%in: f32, %unused: f32):
        ktdf.opaque "read_constant" {dataflow_scheduler.register_names = ["constant"]}
          ins(%r1: memref<32xf32, "SFU_REG">) outs()
        linalg.yield %in : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y1, %out1 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// A loop-invariant allocation may retain its value between iterations.
// CHECK-LABEL: func.func @loop_carried_storage(
// CHECK-COUNT-2: memref.alloca
// CHECK-NOT: memref.alloca
// CHECK: return
func.func @loop_carried_storage(%x0: tensor<32xf32>, %x1: tensor<32xf32>, %out0: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, %out1: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %init = tensor.empty() : tensor<32xf32>
  %r0 = memref.alloca() : memref<32xf32, "SFU_REG">
  scf.for %i = %zero to %two step %one {
    ktdf.pipeline modulo(size: 2) {
      ktdf.stage depends_in(none) depends_out(none) {
      %y0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x0 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r0: memref<32xf32, "SFU_REG">)
        %result = memref.load %r0[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y0, %out0 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      } {applicable_units = ["SFU"]}
    }
  }
  scf.for %i = %zero to %two step %one {
    ktdf.pipeline modulo(size: 2) {
      ktdf.stage depends_in(none) depends_out(none) {
      %r1 = memref.alloca() : memref<32xf32, "SFU_REG">
      %y1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
          ins(%x1 : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
      ^bb0(%x: f32, %unused: f32):
        ktdf.opaque "work" {dataflow_scheduler.register_names = ["out"]}
          ins() outs(%r1: memref<32xf32, "SFU_REG">)
        %result = memref.load %r1[%zero] : memref<32xf32, "SFU_REG">
        linalg.yield %result : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %y1, %out1 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      } {applicable_units = ["SFU"]}
    }
  }
  return
}

// CHECK-LABEL: func.func @conditional_initialization(
// CHECK-COUNT-2: memref.alloc()
// CHECK-NOT: memref.alloc
// CHECK: return
func.func @conditional_initialization(%n: index, %x: tensor<32xf32>, %out0: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, %out1: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %constant = arith.constant 1.0 : f32
  %init = tensor.empty() : tensor<32xf32>
  %r0 = memref.alloc() : memref<32xf32, "SFU_REG">
  %r1 = memref.alloc() : memref<32xf32, "SFU_REG">
  scf.for %iv = %zero to %n step %one {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        linalg.fill ins(%constant : f32) outs(%r0 : memref<32xf32, "SFU_REG">)
        %y = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
            ins(%x : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
        ^bb0(%in: f32, %unused: f32):
          ktdf.opaque "read_constant" {dataflow_scheduler.register_names = ["constant"]}
            ins(%r0: memref<32xf32, "SFU_REG">) outs()
          linalg.yield %in : f32
        } -> tensor<32xf32>
        ktdf.write_to_fifo %y, %out0 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      } {applicable_units = ["SFU"]}
    }
  }
  scf.for %iv = %zero to %one step %one {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        linalg.fill ins(%constant : f32) outs(%r1 : memref<32xf32, "SFU_REG">)
        %y = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
            ins(%x : tensor<32xf32>) outs(%init : tensor<32xf32>) attrs = {ktdf_arch.maps_to = "SFU"} {
        ^bb0(%in: f32, %unused: f32):
          ktdf.opaque "read_constant" {dataflow_scheduler.register_names = ["constant"]}
            ins(%r1: memref<32xf32, "SFU_REG">) outs()
          linalg.yield %in : f32
        } -> tensor<32xf32>
        ktdf.write_to_fifo %y, %out1 : tensor<32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
      } {applicable_units = ["SFU"]}
    }
  }
  return
}
