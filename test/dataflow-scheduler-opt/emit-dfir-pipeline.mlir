// Checks that the `kEmitDFIR` pipeline registered by dataflow-scheduler-opt
// schedules every pass from `scheduler::buildKTDPToDFIRPipeline` (plus the
// leading `ensure-device-declaration` pass and the trailing `emit-split-dfir`
// pass, both added by the pipeline registration itself), in order.
//
// `--dump-pass-pipeline` prints the fully constructed pass manager, which lets
// us assert the passes are invoked without running the full pipeline on real
// KTIR input. The pipeline still starts to run afterwards and fails on the
// empty module (no device), so `-verify-diagnostics` consumes that error.
//
// RUN: dataflow-scheduler-opt -kEmitDFIR --dump-pass-pipeline %s \
// RUN:   -verify-diagnostics 2>&1 | FileCheck %s

// The passes must appear in the exact order established by
// buildKTDPToDFIRPipeline. CHECK directives match in order, and CHECK-NOT
// between the first and last entry would catch reordering.

// CHECK:        builtin.module(
// CHECK-NEXT:   ensure-device-declaration
// CHECK-NEXT:   func.func(
// CHECK-NEXT:   apply-device-patterns{groups={pre_mapping}}
// CHECK-NEXT:   )
// CHECK-NEXT:   ktir-legality-check
// CHECK-NEXT:   compute-group-extraction
// CHECK-NEXT:   ktir-distributed-memory
// CHECK-NEXT:   indirect-compute-group-split
// CHECK-NEXT:   indirect-addr-buf-legalization
// CHECK-NEXT:   indirect-access-loop-materialization
// CHECK-NEXT:   builtin.module(
// CHECK-NEXT:   func.func(
// CHECK-NEXT:   convert-elementwise-to-linalg,
// CHECK-NEXT:   linalg-morph-ops{category-to-generic=true generic-to-named=false named-to-category=false named-to-generic=true},
// CHECK-NEXT:   cse,
// CHECK-NEXT:   ktir-numeric-cast-tiling,
// CHECK-NEXT:   simplify-linalg,
// CHECK-NEXT:   fuse-linalg,
// CHECK-NEXT:   ktir-map-and-tile,
// CHECK-NEXT:   ktir-bufferize,
// CHECK-NEXT:   ktir-pipeline
// CHECK-NEXT:   )
// CHECK-NEXT:   )
// CHECK-NEXT:   prepare-target-operations
// The common optimization passes run inside this coordinator so capacity
// retries restart from the validated prepared IR.
// CHECK-NEXT:   schedule-with-chunk-plan
// CHECK-NEXT:   address-assignment
// CHECK-NEXT:   normalize-grid-to-1d
// CHECK-NEXT:   loop-iv-affine-mul-strength-reduction
// CHECK-NEXT:   ktdf-to-ktdflowering
// CHECK-NEXT:   builtin.module(
// CHECK-NEXT:   func.func(
// CHECK-NEXT:   apply-device-patterns{groups={post_lowering}}
// CHECK-NEXT:   )
// CHECK-NEXT:   )
// CHECK-NEXT:   ktdflowering-to-dfir
// CHECK-NEXT:   wrap-program-dfir
// CHECK-NEXT:   emit-split-dfir

// expected-error @below {{no 'ktdf_arch.device' is present in the module and no device file name is provided to ensure-device-declaration pass}}
module {
}
