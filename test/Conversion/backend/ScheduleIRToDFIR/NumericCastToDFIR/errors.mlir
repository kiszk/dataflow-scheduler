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

// RUN: sed -n '/^#input =/,$p' %S/widen.mlir | sed 's@Inputs/device.mlir@%S/Inputs/device.mlir@' > %t.widen.mlir

// RUN: sed -e 's/@widen(/@multiple_cores(/' -e 's/grid = \[1\]/grid = [2]/' \
// RUN:   -e '/^  func.func/i\  // expected-error @below {{numeric cast requires whole sticks evenly divided among cores}}' %t.widen.mlir > %t.multiple_cores.mlir
// RUN: grep -q 'expected-error @below' %t.multiple_cores.mlir && dataflow-scheduler-opt -ktir-to-dfir -verify-diagnostics %t.multiple_cores.mlir

// RUN: sed -e 's/@widen(/@strided(/' -e 's/strides: \[64, 1\]/strides: [128, 1]/' \
// RUN:   -e '/^    %input_view =/i\    // expected-error @below {{numeric cast requires a static contiguous global view}}' %t.widen.mlir > %t.strided.mlir
// RUN: grep -q 'expected-error @below' %t.strided.mlir && dataflow-scheduler-opt -ktir-to-dfir -verify-diagnostics %t.strided.mlir

// RUN: sed -e 's/@widen(/@offset(/' -e 's/%input_view\[%zero, %zero\]/%input_view[1, %zero]/' \
// RUN:   -e '/^    %input_tile =/i\    // expected-error @below {{numeric cast requires zero tile offsets}}' %t.widen.mlir > %t.offset.mlir
// RUN: grep -q 'expected-error @below' %t.offset.mlir && dataflow-scheduler-opt -ktir-to-dfir -verify-diagnostics %t.offset.mlir

// RUN: sed -e 's/@widen(/@alias(/' -e 's/construct_memory_view %output,/construct_memory_view %input,/' \
// RUN:   -e '/^  func.func/i\  // expected-error @below {{numeric cast requires distinct base arguments or stick-aligned constant addresses}}' %t.widen.mlir > %t.alias.mlir
// RUN: grep -q 'expected-error @below' %t.alias.mlir && dataflow-scheduler-opt -ktir-to-dfir -verify-diagnostics %t.alias.mlir

// RUN: sed -e 's/@widen(/@extra_op(/' \
// RUN:   -e '/%zero = arith.constant/a\    // expected-error @below {{unsupported operation beside numeric cast}}\n    %%sum = arith.addi %input, %output : index' %t.widen.mlir > %t.extra_op.mlir
// RUN: grep -q 'expected-error @below' %t.extra_op.mlir && dataflow-scheduler-opt -ktir-to-dfir -verify-diagnostics %t.extra_op.mlir
