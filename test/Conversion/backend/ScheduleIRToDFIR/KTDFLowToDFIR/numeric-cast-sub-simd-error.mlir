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

// RUN: not dataflow-scheduler-opt -ktdflowering-to-dfir %s 2>&1 | FileCheck %s
// CHECK: numeric_cast requires 8 f16 and 4 f32 sub-SIMD lanes

// incorrect f32 sub-SIMD width
!input = !ktdf.fifo.slot<"MPE_ARRAY" -> "SFU", 64xf16>
!output = !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>

module {
  ktdf_arch.device @sample_device {
    %relay = exec_unit { kind = "MPE_ARRAY" }
    %compute = exec_unit {
      kind = "SFU",
      ktdf_arch.features = {
        ktdf_arch.feature.compute,
        ktdf_arch.feature.simd = {
          lanes = #ktdf_arch.map<f16 = 64, f32 = 32>,
          sub_simd_lanes = #ktdf_arch.map<f16 = 8, f32 = 8>
        }
      }
    }
    %store = exec_unit { kind = "L1SU" }
    datapath %relay to %compute : exec_unit, exec_unit
    datapath %compute to %store : exec_unit, exec_unit
  }
  func.func @numeric_cast_capability() attributes {grid = [1]} {
    %relay = dataflow.get_unit {core = 0 : i32, name = "C0-MPE_ARRAY", type = "MPE_ARRAY"} : index
    %compute = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %store = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
    dataflow.program_unit iter_arg : %unit -> (%compute) : {
      %input = ktdf.fifo.allocate() -> !input
      %output:2 = ktdf.fifo.allocate() -> !output, !output
      ktdf_lowering.numeric_cast ins(%input : !input)
        outs(%output#0, %output#1 : !output, !output) {kind = "dl16tofp32"}
    }
    dataflow.program_unit iter_arg : %unit -> (%relay) : {}
    dataflow.program_unit iter_arg : %unit -> (%store) : {}
    return
  }
}
