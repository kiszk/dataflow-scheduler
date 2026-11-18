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
// Correct physical shapes must not make IEEE f16 or bf16 valid DL16 types.

func.func @ieee_f16_input(%input: tensor<3x2x64xf16>) -> tensor<6x2x32xf32> {
  // expected-error @below {{requires tensor<Nx64x!spyreop.df16>}}
  %result = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64xf16> -> tensor<6x2x32xf32>
  return %result : tensor<6x2x32xf32>
}

// -----

func.func @bf16_input(%input: tensor<3x2x64xbf16>) -> tensor<6x2x32xf32> {
  // expected-error @below {{requires tensor<Nx64x!spyreop.df16>}}
  %result = spyreop.numeric_cast %input {kind = "dl16tofp32"}
      : tensor<3x2x64xbf16> -> tensor<6x2x32xf32>
  return %result : tensor<6x2x32xf32>
}

// -----

func.func @ieee_f16_output(%input: tensor<6x2x32xf32>) -> tensor<3x2x64xf16> {
  // expected-error @below {{requires tensor<Nx64x!spyreop.df16>}}
  %result = spyreop.numeric_cast %input {kind = "fp32todl16"}
      : tensor<6x2x32xf32> -> tensor<3x2x64xf16>
  return %result : tensor<3x2x64xf16>
}
