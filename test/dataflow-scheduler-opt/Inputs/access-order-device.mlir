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

// Direct scratch-to-compute route for the access-order copy recipe. Unlike
// sample_device, it supports 128B packets and 16B copies on both memory units.

#DDR = { kind = "DDR", size = 17179869184 }
#L1 = { kind = "L1", size = 2097152 }
#SFU_REG = { kind = "SFU_REG", size = 2048 }

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
      %sfu_reg = memory #SFU_REG
      %l1lu = exec_unit #L1LU
      %sfu = exec_unit #SFU
      %l1su = exec_unit #L1SU

      datapath #L1LU_CORE_FIFO %l1 to %l1lu : memory, exec_unit
      datapath #L1LU_SUBCORE_FIFO %l1lu to %sfu : exec_unit, exec_unit
      datapath #L1SU_SUBCORE_FIFO %sfu to %l1su : exec_unit, exec_unit
      datapath #L1SU_CORE_FIFO %l1su to %l1 : exec_unit, memory
    }
  }

  // Test-only template: exercise an f16 intrinsic and its register transfers.
  // The opaque is checked structurally; it is not a hardware exp recipe.
  patterns ["pre_scheduling"] {
    pdl.pattern : benefit(1) {
      %f16 = type : f16
      %in = operand : %f16
      %exp = operation "spyreop.exp" (%in : !pdl.value) -> (%f16 : !pdl.type)
      rewrite {
        %reg = type : memref<f16, "SFU_REG">
        %no_dyn_operands = attribute = array<i32: 0, 0>
        %in_op = operation "memref.alloca"
          {"operandSegmentSizes" = %no_dyn_operands} -> (%reg : !pdl.type)
        %in_reg = result 0 of %in_op
        %in_store = operation "memref.store"
          (%in, %in_reg : !pdl.value, !pdl.value)
        %out_op = operation "memref.alloca"
          {"operandSegmentSizes" = %no_dyn_operands} -> (%reg : !pdl.type)
        %out_reg = result 0 of %out_op
        %template = attribute = "fake_f16_exp"
        %register_names = attribute = ["in0", "out0"]
        %segments = attribute = array<i32: 1, 1>
        %opaque_op = operation "ktdf.opaque"
          (%in_reg, %out_reg : !pdl.value, !pdl.value)
          {"template_name" = %template, "func_name" = %template,
           "dataflow_scheduler.register_names" = %register_names,
           "operandSegmentSizes" = %segments}
        %load = operation "memref.load" (%out_reg : !pdl.value)
          -> (%f16 : !pdl.type)
        %result = result 0 of %load
        replace %exp with (%result : !pdl.value)
      }
    }
    pdl.pattern : benefit(1) {
      %f32 = type : f32
      %in = operand : %f32
      %exp = operation "spyreop.exp" (%in : !pdl.value) -> (%f32 : !pdl.type)
      rewrite {
        // Match the real FP32 exp register pressure: six constants, input,
        // output and four scratch registers. No hardware template is invoked.
        %reg = type : memref<f32, "SFU_REG">
        %register_mapping = attribute = "SFU_REG"
        %c0_value = attribute = 1.0 : f32
        %c0_op = operation "arith.constant"
          {"value" = %c0_value, "ktdf_arch.maps_to" = %register_mapping}
          -> (%f32 : !pdl.type)
        %c0 = result 0 of %c0_op
        %c1_value = attribute = 2.0 : f32
        %c1_op = operation "arith.constant"
          {"value" = %c1_value, "ktdf_arch.maps_to" = %register_mapping}
          -> (%f32 : !pdl.type)
        %c1 = result 0 of %c1_op
        %c2_value = attribute = 3.0 : f32
        %c2_op = operation "arith.constant"
          {"value" = %c2_value, "ktdf_arch.maps_to" = %register_mapping}
          -> (%f32 : !pdl.type)
        %c2 = result 0 of %c2_op
        %c3_value = attribute = 4.0 : f32
        %c3_op = operation "arith.constant"
          {"value" = %c3_value, "ktdf_arch.maps_to" = %register_mapping}
          -> (%f32 : !pdl.type)
        %c3 = result 0 of %c3_op
        %c4_value = attribute = 5.0 : f32
        %c4_op = operation "arith.constant"
          {"value" = %c4_value, "ktdf_arch.maps_to" = %register_mapping}
          -> (%f32 : !pdl.type)
        %c4 = result 0 of %c4_op
        %c5_value = attribute = 6.0 : f32
        %c5_op = operation "arith.constant"
          {"value" = %c5_value, "ktdf_arch.maps_to" = %register_mapping}
          -> (%f32 : !pdl.type)
        %c5 = result 0 of %c5_op
        %no_dyn_operands = attribute = array<i32: 0, 0>
        %in_op = operation "memref.alloca"
          {"operandSegmentSizes" = %no_dyn_operands} -> (%reg : !pdl.type)
        %in_reg = result 0 of %in_op
        %in_store = operation "memref.store"
          (%in, %in_reg : !pdl.value, !pdl.value)
        %out_op = operation "memref.alloca"
          {"operandSegmentSizes" = %no_dyn_operands} -> (%reg : !pdl.type)
        %out_reg = result 0 of %out_op
        %t0_op = operation "memref.alloca"
          {"operandSegmentSizes" = %no_dyn_operands} -> (%reg : !pdl.type)
        %t0 = result 0 of %t0_op
        %t1_op = operation "memref.alloca"
          {"operandSegmentSizes" = %no_dyn_operands} -> (%reg : !pdl.type)
        %t1 = result 0 of %t1_op
        %t2_op = operation "memref.alloca"
          {"operandSegmentSizes" = %no_dyn_operands} -> (%reg : !pdl.type)
        %t2 = result 0 of %t2_op
        %t3_op = operation "memref.alloca"
          {"operandSegmentSizes" = %no_dyn_operands} -> (%reg : !pdl.type)
        %t3 = result 0 of %t3_op
        %template = attribute = "fake_f32_exp"
        %register_names = attribute = ["c0", "c1", "c2", "c3", "c4", "c5", "in0", "out0", "t0", "t1", "t2", "t3"]
        %segments = attribute = array<i32: 7, 5>
        %opaque_op = operation "ktdf.opaque"
          (%c0, %c1, %c2, %c3, %c4, %c5, %in_reg, %out_reg, %t0, %t1, %t2, %t3 :
           !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value,
           !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value)
          {"template_name" = %template, "func_name" = %template,
           "dataflow_scheduler.register_names" = %register_names,
           "operandSegmentSizes" = %segments}
        %load = operation "memref.load" (%out_reg : !pdl.value)
          -> (%f32 : !pdl.type)
        %result = result 0 of %load
        replace %exp with (%result : !pdl.value)
      }
    }
  }

  patterns ["post_scheduling"] {
    // MAX: select the opaque template by accumulator precision.
    pdl.pattern : benefit(1) {
      %in  = operand
      %acc = operand
      %generic = operation "linalg.generic" (%in, %acc : !pdl.value, !pdl.value)
      apply_native_constraint "ktdf.is_inner_dim_reduction"(%generic : !pdl.operation)
      %expected_kind = attribute = "max"
      apply_native_constraint "ktdf.is_reduction_kind"(%generic, %expected_kind : !pdl.operation, !pdl.attribute)

      rewrite {
        %acc_src = apply_native_rewrite "ktdf.subview_source"(%acc : !pdl.value) : !pdl.value
        %template = attribute = "simdreduction_minmax"
        %func_name = apply_native_rewrite "ktdf.template_for_precision"(%generic, %template : !pdl.operation, !pdl.attribute) : !pdl.attribute
        %reg_names = attribute = ["t0", "t0"]
        %params    = attribute = {x1 = "0", x2 = "8", "unroll" = "1"}
        %segments  = attribute = array<i32: 1, 1>
        %opaque_op = operation "ktdf.opaque"
          (%in, %acc_src : !pdl.value, !pdl.value)
          {"template_name"                     = %func_name,
           "func_name"                         = %func_name,
           "dataflow_scheduler.register_names" = %reg_names,
           "parameter_dictionary"              = %params,
           "operandSegmentSizes"               = %segments}
        replace %generic with %opaque_op
      }
    }

    // SUM: select the opaque template by accumulator precision.
    pdl.pattern : benefit(1) {
      %in  = operand
      %acc = operand
      %generic = operation "linalg.generic" (%in, %acc : !pdl.value, !pdl.value)
      apply_native_constraint "ktdf.is_inner_dim_reduction"(%generic : !pdl.operation)
      %expected_kind = attribute = "sum"
      apply_native_constraint "ktdf.is_reduction_kind"(%generic, %expected_kind : !pdl.operation, !pdl.attribute)

      rewrite {
        %acc_src = apply_native_rewrite "ktdf.subview_source"(%acc : !pdl.value) : !pdl.value
        %template = attribute = "simdreduction_sum"
        %func_name = apply_native_rewrite "ktdf.template_for_precision"(%generic, %template : !pdl.operation, !pdl.attribute) : !pdl.attribute
        %reg_names = attribute = ["t0", "t0"]
        %params    = attribute = {"unroll" = "1"}
        %segments  = attribute = array<i32: 1, 1>
        %opaque_op = operation "ktdf.opaque"
          (%in, %acc_src : !pdl.value, !pdl.value)
          {"template_name"                     = %func_name,
           "func_name"                         = %func_name,
           "dataflow_scheduler.register_names" = %reg_names,
           "parameter_dictionary"              = %params,
           "operandSegmentSizes"               = %segments}
        replace %generic with %opaque_op
      }
    }

  }
}
