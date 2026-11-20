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

// RUN: dataflow-scheduler-opt %s --split-input-file --verify-diagnostics
// Only scalar f16/f32 carriers are accepted.

func.func @df16_input(%input: !spyreop.df16) -> f32 {
  // expected-error @below {{operand #0 must be 16-bit float or 32-bit float, but got '!spyreop.df16'}}
  %result = spyreop.numeric_cast %input {kind = "dl16tofp32"} : !spyreop.df16 -> f32
  return %result : f32
}

// -----

func.func @bf16_input(%input: bf16) -> f32 {
  // expected-error @below {{operand #0 must be 16-bit float or 32-bit float, but got 'bf16'}}
  %result = spyreop.numeric_cast %input {kind = "dl16tofp32"} : bf16 -> f32
  return %result : f32
}

// -----

func.func @df16_output(%input: f32) -> !spyreop.df16 {
  // expected-error @below {{result #0 must be 16-bit float or 32-bit float, but got '!spyreop.df16'}}
  %result = spyreop.numeric_cast %input {kind = "fp32todl16"} : f32 -> !spyreop.df16
  return %result : !spyreop.df16
}

// -----

func.func @tensor_input(%input: tensor<3x2x64xf16>) -> tensor<6x2x32xf32> {
  // expected-error @below {{operand #0 must be 16-bit float or 32-bit float, but got 'tensor<3x2x64xf16>'}}
  %result = spyreop.numeric_cast %input {kind = "dl16tofp32"} : tensor<3x2x64xf16> -> tensor<6x2x32xf32>
  return %result : tensor<6x2x32xf32>
}
