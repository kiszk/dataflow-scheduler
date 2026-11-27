// SPDX-License-Identifier: Apache-2.0
// Abstract test resources. The Matrix unit preserves target operations; this
// fixture does not implement or claim to emulate native MPE computation.
#load = {kind = "LOAD", ktdf_arch.features = {ktdf_arch.feature.load = {
  word_size = #ktdf_arch.map<"DDR" = 64, "L1" = 64>,
  access_granularity = #ktdf_arch.map<"DDR" = [{size_in_words = 1, align_in_words = 1}], "L1" = [{size_in_words = 1, align_in_words = 1}]>
}}, dataflow_scheduler.double_buffer_last}
#store = {kind = "STORE", ktdf_arch.features = {ktdf_arch.feature.store = {
  word_size = #ktdf_arch.map<"DDR" = 64, "L1" = 64>,
  access_granularity = #ktdf_arch.map<"DDR" = [{size_in_words = 1, align_in_words = 1}], "L1" = [{size_in_words = 1, align_in_words = 1}]>
}}}
#matrix = {kind = "MATRIX", dataflow_scheduler.mpe_matmul, ktdf_arch.features = {ktdf_arch.feature.compute}}
#bus = {ktdf_arch.transfer_granularity = array<i64: 64>}
ktdf_arch.device @chunk_test {
  %hbm = memory {kind = "DDR", size = 1073741824}
  group {kind = "core"} share(%hbm) {
    %lx = memory {kind = "L1", size = 1048576}
    %load = exec_unit #load
    %store = exec_unit #store
    %matrix = exec_unit #matrix
    datapath #bus %hbm to %load : memory, exec_unit
    datapath #bus %load to %lx : exec_unit, memory
    datapath #bus %lx to %store : memory, exec_unit
    datapath #bus %store to %hbm : exec_unit, memory
    datapath %lx to %matrix : memory, exec_unit
    datapath %matrix to %lx : exec_unit, memory
  }
}
