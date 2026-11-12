// The three stage dispatchers route numeric casts to their own
// pipelines, and run the standard stage pipelines for other KTIR. Clients may
// invoke these stages separately, as dbo-opt does.
//
// RUN: dataflow-scheduler-opt -kEmitDFIR --dump-pass-pipeline %s \
// RUN:   -verify-diagnostics 2>&1 | FileCheck %s

// CHECK:        builtin.module(
// CHECK-NEXT:   ensure-device-declaration
// CHECK-NEXT:   dispatch-ktir-frontend
// CHECK-NEXT:   dispatch-scheduler-optimization
// CHECK-NEXT:   dispatch-dfir-backend
// CHECK-NEXT:   emit-split-dfir

// expected-error @below {{no 'ktdf_arch.device' is present in the module and no device file name is provided to ensure-device-declaration pass}}
module {
}
