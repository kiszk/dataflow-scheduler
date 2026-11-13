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

// RUN: dataflow-scheduler-opt -split-input-file -verify-diagnostics %s

func.func @widen(%input: tensor<64xf16>) {
  %out:2 = ktdf_lowering.native_cast_value ins(%input : tensor<64xf16>)
    outs(tensor<32xf32>, tensor<32xf32>) {kind = "dl16tofp32"}
  return
}

func.func @narrow(%a: tensor<32xf32>, %b: tensor<32xf32>) {
  %out = ktdf_lowering.native_cast_value ins(%a, %b : tensor<32xf32>, tensor<32xf32>)
    outs(tensor<64xf16>) {kind = "fp32todl16"}
  return
}

// -----

func.func @wrong_kind(%input: tensor<64xf16>) {
  // expected-error @below {{requires kind 'dl16tofp32' or 'fp32todl16'}}
  %out:2 = ktdf_lowering.native_cast_value ins(%input : tensor<64xf16>)
    outs(tensor<32xf32>, tensor<32xf32>) {kind = "other"}
  return
}

// -----

func.func @missing_packet(%input: tensor<64xf16>) {
  // expected-error @below {{requires one native and two FP32 packet values}}
  %out = ktdf_lowering.native_cast_value ins(%input : tensor<64xf16>)
    outs(tensor<32xf32>) {kind = "dl16tofp32"}
  return
}

// -----

func.func @short_native(%input: tensor<32xf16>) {
  // expected-error @below {{has an invalid input packet type}}
  %out:2 = ktdf_lowering.native_cast_value ins(%input : tensor<32xf16>)
    outs(tensor<32xf32>, tensor<32xf32>) {kind = "dl16tofp32"}
  return
}

// -----

func.func @bf16_output(%a: tensor<32xf32>, %b: tensor<32xf32>) {
  // expected-error @below {{has an invalid output packet type}}
  %out = ktdf_lowering.native_cast_value ins(%a, %b : tensor<32xf32>, tensor<32xf32>)
    outs(tensor<64xbf16>) {kind = "fp32todl16"}
  return
}
