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

// Architecture selection rejects ambiguous compute units, incompatible
// capabilities, and missing routes before emitting a numeric cast pipeline.
// RUN: sed 's@Inputs/device.mlir@%t.device.mlir@' %S/widen.mlir > %t.input.mlir
// RUN: sed '/ktdf_arch.feature.compute/d' %S/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=COMPUTE
// RUN: sed '/^#MPE_ARRAY = {/,/^}/s/ktdf_arch.feature.simd/ktdf_arch.feature.compute, ktdf_arch.feature.simd/' %S/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=COMPUTE
// RUN: sed '/[%]sfu = exec_unit #SFU/a\      %other_sfu = exec_unit #SFU' %S/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=NODES
// RUN: sed '/^#SFU = {/,/^}/s/f16 = 64/f16 = 32/' %S/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=SIMD
// RUN: sed '/^#MPE_ARRAY = {/,/^}/s/f16 = 64/f16 = 32/' %S/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=RELAY
// RUN: sed 's/f16 = 8, f32 = 4/f16 = 4, f32 = 4/' %S/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=SUBSIMD
// RUN: sed '/^#L1LU = {/,/^}/c\#L1LU = { kind = "L1LU" }' %S/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=LOAD
// RUN: sed '/^#L1SU = {/,/^}/c\#L1SU = { kind = "L1SU" }' %S/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=STORE
// RUN: sed '/datapath #L1LU_SUBCORE_FIFO [%]mpe_array to [%]sfu/d' %S/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=ROUTE
// RUN: sed 's/"DDR" = 128/"DDR" = 64/g' %S/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=WORD
// RUN: sed 's/"L1" = 128/"L1" = 64/g' %S/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=WORD
// RUN: sed 's/mem_space_mapping/unused_space_mapping/' %S/Inputs/device.mlir > %t.device.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.input.mlir 2>&1 | FileCheck %s --check-prefix=MEMORY
// RUN: sed -e 's@Inputs/device.mlir@%S/Inputs/device.mlir@' -e 's/kind = "dl16tofp32"/kind = "dl16tofp32", ktdf_arch.maps_to = "MPE_ARRAY"/' %S/widen.mlir > %t.mapped.mlir
// RUN: not dataflow-scheduler-opt -ktir-to-dfir %t.mapped.mlir 2>&1 | FileCheck %s --check-prefix=MAPPED

// COMPUTE: no (unambiguous) default compute resource
// NODES: numeric cast requires unambiguous routing resources
// SIMD: native_cast requires 64 f16 lanes and 32 f32 lanes
// RELAY: numeric cast requires 64 f16 and 32 f32 relay lanes
// SUBSIMD: native_cast requires 8 f16 and 4 f32 sub-SIMD lanes
// LOAD: numeric cast requires a load-capable local load unit
// STORE: numeric cast requires a store-capable local store unit
// ROUTE: numeric cast has no route from "L1" to "SFU"
// WORD: numeric cast requires 128-byte global and scratchpad words
// MEMORY: numeric cast requires global and sized per-core scratchpad memory
// MAPPED: native_cast requires a programmable conversion unit
