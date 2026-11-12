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

// The core-0/corelet-0 path used by numeric casts, extracted from
// spyre_dd2_basic.mlir, using the names and kinds from sample_device.mlir.
// Memory capacities, transfer sizes, and queue features retain the native
// profile's contracts.

#DDR = { kind = "DDR", size = 17179869184 }
#L1 = { kind = "L1", size = 2097152 }

#MNILU = {
  kind = "MNILU",
  ktdf_arch.features = {
    ktdf_arch.feature.load = {
      word_size = #ktdf_arch.map<
        "DDR" = 128,
        "L1" = 128
      >,
      access_granularity = #ktdf_arch.map<
        "DDR" = [{size_in_words = 1, align_in_words = 1}],
        "L1" = [{size_in_words = 1, align_in_words = 1}]
      >
    }
  }
}
#MNISU = {
  kind = "MNISU",
  ktdf_arch.features = {
    ktdf_arch.feature.store = {
      word_size = #ktdf_arch.map<
        "DDR" = 128,
        "L1" = 128
      >,
      access_granularity = #ktdf_arch.map<
        "DDR" = [{size_in_words = 1, align_in_words = 1}],
        "L1" = [{size_in_words = 1, align_in_words = 1}]
      >
    }
  }
}
#L1LU = {
  kind = "L1LU",
  ktdf_arch.features = {
    ktdf_arch.feature.load = {
      word_size = #ktdf_arch.map<"L1" = 1>,
      access_granularity = #ktdf_arch.map<
        "L1" = [
          {size_in_words = 128, align_in_words = 2},
          {size_in_words = 16, align_in_words = 2},
          {size_in_words = 2, align_in_words = 2}
        ]
      >
    },
    ktdf_arch.feature.simd = { splat, zero_pad }
  }
}
#L1SU = {
  kind = "L1SU",
  ktdf_arch.features = {
    ktdf_arch.feature.store = {
      word_size = #ktdf_arch.map<"L1" = 1>,
      access_granularity = #ktdf_arch.map<
        "L1" = [
          {size_in_words = 128, align_in_words = 128},
          {size_in_words = 16, align_in_words = 16},
          {size_in_words = 2, align_in_words = 2}
        ]
      >
    }
  }
}
#MPE_ARRAY = {
  kind = "MPE_ARRAY",
  ktdf_arch.features = {
    ktdf_arch.feature.simd = { lanes = #ktdf_arch.map<f16 = 64, f32 = 32, i32 = 32> }
  }
}
#SFU = {
  kind = "SFU",
  ktdf_arch.features = {
    ktdf_arch.feature.compute,
    ktdf_arch.feature.simd = {
      lanes = #ktdf_arch.map<f16 = 64, f32 = 32, i32 = 32>,
      sub_simd_lanes = #ktdf_arch.map<f16 = 8, f32 = 4, i32 = 4>
    }
  }
}

#MNILU_L1 = { ktdf_arch.transfer_granularity = array<i64: 128> }
#L1_MNISU = #MNILU_L1
#L1LU_CORE_FIFO = {
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 1024, ordered } }
}
#L1LU_SUBCORE_FIFO = {
  ktdf_arch.transfer_granularity = array<i64: 64, 2>,
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 1024, ordered } }
}
#L1SU_CORE_FIFO = #L1LU_CORE_FIFO
#L1SU_SUBCORE_FIFO = #L1LU_SUBCORE_FIFO

ktdf_arch.device @sample_device attributes {
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.memory_space<global> = "DDR",
    #ktdp.memory_space<ct_local> = "L1"
  >
} {
  %ddr = memory #DDR
  group { kind = "core" } share(%ddr) {
    %l1 = memory #L1
    %mnilu = exec_unit #MNILU
    %mnisu = exec_unit #MNISU

    datapath %ddr to %mnilu : memory, exec_unit
    datapath #MNILU_L1 %mnilu to %l1 : exec_unit, memory
    datapath #L1_MNISU %l1 to %mnisu : memory, exec_unit
    datapath %mnisu to %ddr : exec_unit, memory

    group share(%l1) {
      %l1lu = exec_unit #L1LU
      %mpe_array = exec_unit #MPE_ARRAY
      %sfu = exec_unit #SFU
      %l1su = exec_unit #L1SU

      datapath #L1LU_CORE_FIFO %l1 to %l1lu : memory, exec_unit
      datapath #L1LU_SUBCORE_FIFO %l1lu to %mpe_array : exec_unit, exec_unit
      datapath #L1LU_SUBCORE_FIFO %mpe_array to %sfu : exec_unit, exec_unit
      datapath #L1SU_SUBCORE_FIFO %sfu to %l1su : exec_unit, exec_unit
      datapath #L1SU_CORE_FIFO %l1su to %l1 : exec_unit, memory
    }
  }
}
