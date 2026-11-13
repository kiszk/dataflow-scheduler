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

// Reject invalid op contracts and other computation beside a standalone cast.
// RUN: sed 's@Inputs/device.mlir@%S/Inputs/device.mlir@' %S/widen.mlir > %t.base.mlir
// RUN: sed 's/kind = "dl16tofp32"/kind = "unknown"/' %t.base.mlir > %t.kind.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.kind.mlir 2>&1 | FileCheck %s --check-prefix=KIND
// RUN: sed '/%result = spyreop.numeric_cast/i\    %extra = arith.addf %value, %value : tensor<1x64xf16>' %t.base.mlir > %t.extra.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.extra.mlir 2>&1 | FileCheck %s --check-prefix=EXTRA
// RUN: sed '/%result = spyreop.numeric_cast/a\    %other = spyreop.numeric_cast %value {kind = "dl16tofp32"} : tensor<1x64xf16> -> tensor<2x32xf32>' %t.base.mlir > %t.duplicate.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.duplicate.mlir 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: sed '/%result = spyreop.numeric_cast/a\    %unused = arith.negf %result : tensor<2x32xf32>' %t.base.mlir > %t.unused.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.unused.mlir 2>&1 | FileCheck %s --check-prefix=UNUSED

// KIND: kind must be dl16tofp32 or fp32todl16
// EXTRA: unsupported operation beside numeric cast
// DUPLICATE: only one numeric cast is supported
// UNUSED: numeric cast requires a single load, unary chain, cast, unary chain and store
