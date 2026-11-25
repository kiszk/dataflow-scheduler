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

// RUN: dataflow-scheduler-opt --reduction-dim-chunking="num-chunks=2" %s | FileCheck %s

// A contraction can reuse local operands without adjacent transfer stages.
// It must be excluded from reduction chunking before looking for those stages.
// CHECK-LABEL: func.func @local_contraction
// CHECK: ktdf.pipeline
// CHECK-NEXT: ktdf.stage
// CHECK-NOT: scf.for
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: scf.for
// CHECK: return

#a = affine_map<(m, n, k) -> (m, k)>
#b = affine_map<(m, n, k) -> (k, n)>
#c = affine_map<(m, n, k) -> (m, n)>
func.func @local_contraction(%a: memref<8x64xf16, #ktdp.memory_space<ct_local>>, %b: memref<64x64xf16, #ktdp.memory_space<ct_local>>, %c: memref<8x64xf16, #ktdp.memory_space<ct_local>>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      linalg.generic {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%a, %b : memref<8x64xf16, #ktdp.memory_space<ct_local>>, memref<64x64xf16, #ktdp.memory_space<ct_local>>)
          outs(%c : memref<8x64xf16, #ktdp.memory_space<ct_local>>) {
      ^bb0(%x: f16, %y: f16, %acc: f16):
        %product = arith.mulf %x, %y : f16
        %sum = arith.addf %product, %acc : f16
        linalg.yield %sum : f16
      }
    }
  }
  return
}
