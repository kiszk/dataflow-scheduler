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

// RUN: dataflow-scheduler-opt -ktdflowering-to-dfir %s | FileCheck %s --check-prefix=VALID
// RUN: sed 's@../NumericCastToDFIR/Inputs/device.mlir@%t.device.mlir@' %s > %t.input.mlir
// RUN: sed '/ktdf_arch.feature.compute,/d' %S/../NumericCastToDFIR/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktdflowering-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=COMPUTE
// RUN: sed 's/f16 = 64/f16 = 32/g' %S/../NumericCastToDFIR/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktdflowering-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=LANES
// RUN: sed 's/f32 = 4/f32 = 8/g' %S/../NumericCastToDFIR/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktdflowering-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=SUBLANES
// The backend validates unit capabilities even when the frontend is not run.
// VALID: vectorchain.cast
// VALID: indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32], repetition = 8 : i32
// VALID: vectorchain.cast
// VALID: indices = [4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : i32
// VALID-NOT: ktdf_lowering.native_cast
// COMPUTE: no (unambiguous) default compute resource
// LANES: native_cast requires 64 f16 lanes and 32 f32 lanes
// SUBLANES: native_cast requires 8 f16 and 4 f32 sub-SIMD lanes

!input = !ktdf.fifo.slot<"MPE_ARRAY" -> "SFU", 64xf16>
!output = !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>

module {
  ktdf_arch.device @sample_device attributes {} import("../NumericCastToDFIR/Inputs/device.mlir")
  func.func @native_cast_capability() attributes {grid = [1]} {
    %relay = dataflow.get_unit {core = 0 : i32, name = "C0-MPE_ARRAY", type = "MPE_ARRAY"} : index
    %compute = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %store = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
    dataflow.program_unit iter_arg : %unit -> (%compute) : {
      %input = ktdf.fifo.allocate() -> !input
      %output:2 = ktdf.fifo.allocate() -> !output, !output
      ktdf_lowering.native_cast ins(%input : !input)
        outs(%output#0, %output#1 : !output, !output) {kind = "dl16tofp32"}
    }
    dataflow.program_unit iter_arg : %unit -> (%relay) : {}
    dataflow.program_unit iter_arg : %unit -> (%store) : {}
    return
  }
}
