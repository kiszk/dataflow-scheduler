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

// RUN: dataflow-scheduler-opt -ktdflowering-to-dfir -split-input-file -verify-diagnostics %s

module {
  // expected-error @below {{no (unambiguous) default compute resource}}
  ktdf_arch.device @no_compute {
    %unit = exec_unit {kind = "UNIT"}
  }
}

// -----

module {
  // expected-error @below {{no (unambiguous) default compute resource}}
  ktdf_arch.device @ambiguous_compute {
    %first = exec_unit {
      kind = "FIRST",
      ktdf_arch.features = {ktdf_arch.feature.compute}
    }
    %second = exec_unit {
      kind = "SECOND",
      ktdf_arch.features = {ktdf_arch.feature.compute}
    }
  }
}
