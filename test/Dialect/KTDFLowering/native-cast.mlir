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

// f16 is the native format's storage carrier. Each FP32 packet holds 32 lanes.
!input16 = !ktdf.fifo.slot<"producer" -> "compute", 64xf16>
!input32 = !ktdf.fifo.slot<"producer" -> "compute", 32xf32>
!output16 = !ktdf.fifo.slot<"compute" -> "consumer", 64xf16>
!output32 = !ktdf.fifo.slot<"compute" -> "consumer", 32xf32>

func.func @widen(%input: !input16, %out0: !output32, %out1: !output32) {
  ktdf_lowering.native_cast ins(%input : !input16)
    outs(%out0, %out1 : !output32, !output32) {kind = "dl16tofp32"}
  return
}

func.func @narrow(%in0: !input32, %in1: !input32, %output: !output16) {
  ktdf_lowering.native_cast ins(%in0, %in1 : !input32, !input32)
    outs(%output : !output16) {kind = "fp32todl16"}
  return
}

// -----

!input = !ktdf.fifo.slot<"producer" -> "compute", 64xf16>
!output = !ktdf.fifo.slot<"compute" -> "consumer", 32xf32>

func.func @unknown_kind(%input: !input, %out0: !output, %out1: !output) {
  // expected-error @below {{requires kind 'dl16tofp32' or 'fp32todl16'}}
  ktdf_lowering.native_cast ins(%input : !input)
    outs(%out0, %out1 : !output, !output) {kind = "f16tofp32"}
  return
}

// -----

!input = !ktdf.fifo.slot<"producer" -> "compute", 64xf16>
!output = !ktdf.fifo.slot<"compute" -> "consumer", 32xf32>

func.func @missing_output_packet(%input: !input, %output: !output) {
  // expected-error @below {{dl16tofp32 requires one input slot and two output slots}}
  ktdf_lowering.native_cast ins(%input : !input)
    outs(%output : !output) {kind = "dl16tofp32"}
  return
}

// -----

!input = !ktdf.fifo.slot<"producer" -> "compute", 32xf32>
!output = !ktdf.fifo.slot<"compute" -> "consumer", 64xf16>

func.func @missing_input_packet(%input: !input, %output: !output) {
  // expected-error @below {{fp32todl16 requires two input slots and one output slot}}
  ktdf_lowering.native_cast ins(%input : !input)
    outs(%output : !output) {kind = "fp32todl16"}
  return
}

// -----

!input = !ktdf.fifo.slot<"producer" -> "compute", 32xf16>
!output = !ktdf.fifo.slot<"compute" -> "consumer", 32xf32>

func.func @short_native_input(%input: !input, %out0: !output, %out1: !output) {
  // expected-error @below {{requires 64xf16 FIFO slots on the DL16 side}}
  ktdf_lowering.native_cast ins(%input : !input)
    outs(%out0, %out1 : !output, !output) {kind = "dl16tofp32"}
  return
}

// -----

!input = !ktdf.fifo.slot<"producer" -> "compute", 32xf32>
!output = !ktdf.fifo.slot<"compute" -> "consumer", 64xbf16>

func.func @wrong_native_carrier(%in0: !input, %in1: !input, %output: !output) {
  // expected-error @below {{requires 64xf16 FIFO slots on the DL16 side}}
  ktdf_lowering.native_cast ins(%in0, %in1 : !input, !input)
    outs(%output : !output) {kind = "fp32todl16"}
  return
}

// -----

!input = !ktdf.fifo.slot<"producer" -> "compute", 64xf16>
!output = !ktdf.fifo.slot<"compute" -> "consumer", 64xf32>

func.func @oversized_fp32_packet(%input: !input, %out0: !output, %out1: !output) {
  // expected-error @below {{requires 32xf32 FIFO slots on the FP32 side}}
  ktdf_lowering.native_cast ins(%input : !input)
    outs(%out0, %out1 : !output, !output) {kind = "dl16tofp32"}
  return
}

// -----

!input = !ktdf.fifo.slot<"producer" -> "compute", 32xf16>
!output = !ktdf.fifo.slot<"compute" -> "consumer", 64xf16>

func.func @wrong_fp32_carrier(%in0: !input, %in1: !input, %output: !output) {
  // expected-error @below {{requires 32xf32 FIFO slots on the FP32 side}}
  ktdf_lowering.native_cast ins(%in0, %in1 : !input, !input)
    outs(%output : !output) {kind = "fp32todl16"}
  return
}

// -----

!input0 = !ktdf.fifo.slot<"producer0" -> "compute", 32xf32>
!input1 = !ktdf.fifo.slot<"producer1" -> "compute", 32xf32>
!output = !ktdf.fifo.slot<"compute" -> "consumer", 64xf16>

func.func @different_input_peers(%in0: !input0, %in1: !input1, %output: !output) {
  // expected-error @below {{requires matching FIFO endpoints within each side}}
  ktdf_lowering.native_cast ins(%in0, %in1 : !input0, !input1)
    outs(%output : !output) {kind = "fp32todl16"}
  return
}

// -----

!input = !ktdf.fifo.slot<"producer" -> "compute", 64xf16>
!output0 = !ktdf.fifo.slot<"compute" -> "consumer0", 32xf32>
!output1 = !ktdf.fifo.slot<"compute" -> "consumer1", 32xf32>

func.func @different_output_peers(%input: !input, %out0: !output0, %out1: !output1) {
  // expected-error @below {{requires matching FIFO endpoints within each side}}
  ktdf_lowering.native_cast ins(%input : !input)
    outs(%out0, %out1 : !output0, !output1) {kind = "dl16tofp32"}
  return
}

// -----

!input = !ktdf.fifo.slot<"producer" -> "compute0", 64xf16>
!output = !ktdf.fifo.slot<"compute1" -> "consumer", 32xf32>

func.func @different_compute_units(%input: !input, %out0: !output, %out1: !output) {
  // expected-error @below {{requires the input destination to equal the output source}}
  ktdf_lowering.native_cast ins(%input : !input)
    outs(%out0, %out1 : !output, !output) {kind = "dl16tofp32"}
  return
}

// -----

!input = !ktdf.fifo.slot<"producer" -> "compute", 32xf32>
!output = !ktdf.fifo.slot<"compute" -> "consumer", 64xf16>

func.func @reused_input_packet(%input: !input, %output: !output) {
  // expected-error @below {{requires distinct input and output FIFO slots}}
  ktdf_lowering.native_cast ins(%input, %input : !input, !input)
    outs(%output : !output) {kind = "fp32todl16"}
  return
}

// -----

!input = !ktdf.fifo.slot<"producer" -> "compute", 64xf16>
!output = !ktdf.fifo.slot<"compute" -> "consumer", 32xf32>

func.func @reused_output_packet(%input: !input, %output: !output) {
  // expected-error @below {{requires distinct input and output FIFO slots}}
  ktdf_lowering.native_cast ins(%input : !input)
    outs(%output, %output : !output, !output) {kind = "dl16tofp32"}
  return
}
