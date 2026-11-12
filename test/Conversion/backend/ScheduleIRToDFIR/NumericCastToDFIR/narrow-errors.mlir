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

// RUN: sed -n '/^#input =/,$p' %S/narrow.mlir | sed 's@Inputs/device.mlir@%S/Inputs/device.mlir@' > %t.narrow.mlir

// Reject a third allocated packet when the access still covers two packets.
// RUN: sed -e 's/@narrow(/@extra_input_packet(/' \
// RUN:   -e '/^#input =/a\#allocation = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 31 >= 0)>' \
// RUN:   -e 's/coordinate_set = #input,/coordinate_set = #allocation,/' \
// RUN:   -e 's/sizes: \[2, 32\]/sizes: [3, 32]/' -e 's/memref<2x32xf32>/memref<3x32xf32>/g' \
// RUN:   -e '/^    %input_view =/i\    // expected-error @below {{numeric cast requires a static contiguous global view}}' %t.narrow.mlir > %t.extra_input_packet.mlir
// RUN: grep -q 'expected-error @below' %t.extra_input_packet.mlir && dataflow-scheduler-opt -ktir-to-dfir -verify-diagnostics %t.extra_input_packet.mlir

// bf16 is not the f16 storage carrier used by this conversion recipe.
// RUN: sed -e 's/f16/bf16/g' -e 's/@narrow(/@bf16_output(/' \
// RUN:   -e '/^    %result =/i\    // expected-error @below {{requires tensor<Nx64xf16> and tensor<(2*N)x32xf32> without encodings, N > 0}}' %t.narrow.mlir > %t.bf16_output.mlir
// RUN: grep -q 'expected-error @below' %t.bf16_output.mlir && dataflow-scheduler-opt -ktir-to-dfir -verify-diagnostics %t.bf16_output.mlir

// One packet cannot supply the complete 64-value conversion unit.
// RUN: sed -e 's/@narrow(/@one_input_packet(/' -e 's/2x32/1x32/g' -e 's/1x64/1x32/g' \
// RUN:   -e 's/\[2, 32\]/[1, 32]/g' -e 's/\[1, 64\]/[1, 32]/g' -e 's/\[64, 1\]/[32, 1]/g' \
// RUN:   -e 's/-d0 + 1/-d0/' -e 's/-d1 + 63/-d1 + 31/' -e 's/\[#identity, #native\]/[#identity, #identity]/' \
// RUN:   -e '/^    %result =/i\    // expected-error @below {{requires tensor<Nx64xf16> and tensor<(2*N)x32xf32> without encodings, N > 0}}' %t.narrow.mlir > %t.one_input_packet.mlir
// RUN: grep -q 'expected-error @below' %t.one_input_packet.mlir && dataflow-scheduler-opt -ktir-to-dfir -verify-diagnostics %t.one_input_packet.mlir
