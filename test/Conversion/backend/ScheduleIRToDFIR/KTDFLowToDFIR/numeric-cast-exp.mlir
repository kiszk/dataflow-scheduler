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

// KTDFLow numeric cast lowering snapshots.
// RUN: dataflow-scheduler-opt --split-input-file -ktdflowering-to-dfir %s | FileCheck %s --implicit-check-not=ktdf_lowering.numeric_cast_value

// CHECK-LABEL: func.func @widen_exp({{.*}}attributes {grid = [1]}
// CHECK-LABEL: func.func @local_schedule_0(
// CHECK-DAG: vectorchain.cast {{.*}} : vector<64xf16>, vector<64xf32>
// CHECK-DAG: vectorchain.shuffle {{.*}}indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
// CHECK-DAG: vectorchain.shuffle {{.*}}indices = [4 : i32, 5 : i32, 6 : i32, 7 : i32]
// CHECK-DAG: vectorchain.floor
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_exp"

#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
module {
  module {
    func.func @widen_exp(%arg0: index, %arg1: index) attributes {grid = [1]} {
      call @local_schedule_0(%arg0, %arg1) : (index, index) -> ()
      return
    }
    func.func private @local_schedule_0(index, index)
  }
  ktdf_arch.device @sample_device import("Inputs/device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0(%arg0: index, %arg1: index) attributes {grid = [1]} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 1.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
      %1 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnilu", type = "mnilu"} : index
      %2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
      %3 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mpe_array", type = "mpe_array"} : index
      %4 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
      %5 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
      %6 = ktdp.get_compute_tile_id : index
      %7 = uniform.def_immutable_mapping([%c0 -> %0]):index
      %8 = uniform.query_map(map:%7, key:%6) : index
      %9 = uniform.def_immutable_mapping([%c0 -> %1]):index
      %10 = uniform.query_map(map:%9, key:%6) : index
      %11 = uniform.def_immutable_mapping([%c0 -> %2]):index
      %12 = uniform.query_map(map:%11, key:%6) : index
      %13 = uniform.def_immutable_mapping([%c0 -> %3]):index
      %14 = uniform.query_map(map:%13, key:%6) : index
      %15 = uniform.def_immutable_mapping([%c0 -> %4]):index
      %16 = uniform.query_map(map:%15, key:%6) : index
      %17 = uniform.def_immutable_mapping([%c0 -> %5]):index
      %18 = uniform.query_map(map:%17, key:%6) : index
      %19 = ktdp.construct_memory_view %arg0, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
      %20 = ktdp.construct_memory_view %arg1, sizes: [2, 32], strides: [32, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2x32xf32>
      %memspacecast = memref.memory_space_cast %19 : memref<1x64xf16> to memref<1x64xf16, "DDR">
      %21 = tensor.empty() : tensor<64xf16>
      %22 = tensor.empty() : tensor<32xf32>
      %23 = tensor.empty() : tensor<32xf32>
      %memspacecast_0 = memref.memory_space_cast %20 : memref<2x32xf32> to memref<2x32xf32, "DDR">
      %memspacecast_1 = memref.memory_space_cast %20 : memref<2x32xf32> to memref<2x32xf32, "DDR">
      %24 = builtin.unrealized_conversion_cast %c0 : index to memref<64xf16, "SFU_REG">
      ktdf_lowering.execute_on %8 {
        ktdf_lowering.execute_on %8 {
          linalg.fill ins(%cst : f16) outs(%24 : memref<64xf16, "SFU_REG">)
        }
      }
      ktdf_lowering.execute_on %10, %12, %14, %8, %16, %18 {
        %25 = builtin.unrealized_conversion_cast %c0 : index to memref<1x64xf16, "L1">
        %26 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MPE_ARRAY" -> "SFU", 64xf16>
        %27:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
        %28 = builtin.unrealized_conversion_cast %c32 : index to memref<1x32xf32, "L1">
        %29 = builtin.unrealized_conversion_cast %c64 : index to memref<1x32xf32, "L1">
        %30 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "MPE_ARRAY", 64xf16>
        ktdf_lowering.execute_on %10 {
          ktdf.data_transfer from %memspacecast[%c0, 0] size [1, 64] to %25[0, 0] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, "DDR">, memref<1x64xf16, "L1">
        }
        ktdf_lowering.signal %10, %12
        ktdf_lowering.execute_on %12 {
          ktdf.data_transfer from %25[0, 0] size [1, 64] to %30 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "MPE_ARRAY", 64xf16>
        }
        ktdf_lowering.execute_on %14 {
          %31 = ktdf.read_from_fifo %30 : <"L1LU" -> "MPE_ARRAY", 64xf16> -> tensor<64xf16>
          ktdf.write_to_fifo %31, %26 : tensor<64xf16>, <"MPE_ARRAY" -> "SFU", 64xf16>
        }
        ktdf_lowering.execute_on %8 {
          %31 = ktdf.read_from_fifo %26 : <"MPE_ARRAY" -> "SFU", 64xf16> -> tensor<64xf16>
          %32 = builtin.unrealized_conversion_cast %c64 : index to memref<64xf16, "SFU_REG">
          %33 = builtin.unrealized_conversion_cast %c128 : index to memref<64xf16, "SFU_REG">
          %34 = builtin.unrealized_conversion_cast %c192 : index to memref<64xf16, "SFU_REG">
          %35 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%31 : tensor<64xf16>) outs(%21 : tensor<64xf16>) attrs =  {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
          ^bb0(%in: f16, %out: f16):
            memref.store %in, %32[%c0] : memref<64xf16, "SFU_REG">
            ktdf.opaque "fake_exp" {dataflow_scheduler.register_names = ["c0", "in0", "out0", "t0_0"], func_name = "fake_exp"}
              ins(%24, %32: memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">)
              outs(%33, %34: memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">)
            %39 = memref.load %33[%c0] : memref<64xf16, "SFU_REG">
            linalg.yield %39 : f16
          } -> tensor<64xf16>
          %36:2 = ktdf_lowering.numeric_cast_value ins(%35 : tensor<64xf16>) outs(tensor<32xf32>, tensor<32xf32>) {kind = "dl16tofp32", ktdf_arch.maps_to = "SFU"}
          %37 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%36#0 : tensor<32xf32>) outs(%22 : tensor<32xf32>) attrs =  {dataflow_scheduler.throttle = 32 : i64, ktdf_arch.maps_to = "SFU"} {
          ^bb0(%in: f32, %out: f32):
            %39 = math.floor %in : f32
            linalg.yield %39 : f32
          } -> tensor<32xf32>
          %38 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%36#1 : tensor<32xf32>) outs(%23 : tensor<32xf32>) attrs =  {dataflow_scheduler.throttle = 32 : i64, ktdf_arch.maps_to = "SFU"} {
          ^bb0(%in: f32, %out: f32):
            %39 = math.floor %in : f32
            linalg.yield %39 : f32
          } -> tensor<32xf32>
          ktdf.write_to_fifo %37, %27#0 : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
          ktdf.write_to_fifo %38, %27#1 : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
        }
        ktdf_lowering.execute_on %16 {
          ktdf.data_transfer from %27#0 size [32] to %28[0, 0] size [1, 32] : !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, memref<1x32xf32, "L1">
          ktdf.data_transfer from %27#1 size [32] to %29[0, 0] size [1, 32] : !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, memref<1x32xf32, "L1">
        }
        ktdf_lowering.signal %16, %18
        ktdf_lowering.execute_on %18 {
          ktdf.data_transfer from %28[0, 0] size [1, 32] to %memspacecast_0[%c0, 0] size [1, 32] {dataflow_scheduler.throttle = 32 : i64} : memref<1x32xf32, "L1">, memref<2x32xf32, "DDR">
          ktdf.data_transfer from %29[0, 0] size [1, 32] to %memspacecast_1[%c1, 0] size [1, 32] {dataflow_scheduler.throttle = 32 : i64} : memref<1x32xf32, "L1">, memref<2x32xf32, "DDR">
        }
      }
      return
    }
  }
}

// -----

// CHECK-LABEL: func.func @narrow_exp({{.*}}attributes {grid = [1]}
// CHECK-LABEL: func.func @local_schedule_0(
// CHECK-DAG: vectorchain.cast {{.*}} : vector<32xf32>, vector<32xf16>
// CHECK-DAG: vectorchain.pack
// CHECK-DAG: vectorchain.floor
// CHECK-DAG: dataflow.opaque {{.*}}func_name = "fake_exp"

#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  module {
    func.func @narrow_exp(%arg0: index, %arg1: index) attributes {grid = [1]} {
      call @local_schedule_0(%arg0, %arg1) : (index, index) -> ()
      return
    }
    func.func private @local_schedule_0(index, index)
  }
  ktdf_arch.device @sample_device import("Inputs/device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0(%arg0: index, %arg1: index) attributes {grid = [1]} {
      %c192 = arith.constant 192 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 1.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
      %1 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnilu", type = "mnilu"} : index
      %2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
      %3 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mpe_array", type = "mpe_array"} : index
      %4 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
      %5 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
      %6 = ktdp.get_compute_tile_id : index
      %7 = uniform.def_immutable_mapping([%c0 -> %0]):index
      %8 = uniform.query_map(map:%7, key:%6) : index
      %9 = uniform.def_immutable_mapping([%c0 -> %1]):index
      %10 = uniform.query_map(map:%9, key:%6) : index
      %11 = uniform.def_immutable_mapping([%c0 -> %2]):index
      %12 = uniform.query_map(map:%11, key:%6) : index
      %13 = uniform.def_immutable_mapping([%c0 -> %3]):index
      %14 = uniform.query_map(map:%13, key:%6) : index
      %15 = uniform.def_immutable_mapping([%c0 -> %4]):index
      %16 = uniform.query_map(map:%15, key:%6) : index
      %17 = uniform.def_immutable_mapping([%c0 -> %5]):index
      %18 = uniform.query_map(map:%17, key:%6) : index
      %19 = ktdp.construct_memory_view %arg0, sizes: [2, 32], strides: [32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x32xf32>
      %20 = ktdp.construct_memory_view %arg1, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
      %memspacecast = memref.memory_space_cast %19 : memref<2x32xf32> to memref<2x32xf32, "DDR">
      %memspacecast_0 = memref.memory_space_cast %19 : memref<2x32xf32> to memref<2x32xf32, "DDR">
      %21 = tensor.empty() : tensor<32xf32>
      %22 = tensor.empty() : tensor<32xf32>
      %23 = tensor.empty() : tensor<64xf16>
      %memspacecast_1 = memref.memory_space_cast %20 : memref<1x64xf16> to memref<1x64xf16, "DDR">
      %24 = builtin.unrealized_conversion_cast %c0 : index to memref<64xf16, "SFU_REG">
      ktdf_lowering.execute_on %8 {
        ktdf_lowering.execute_on %8 {
          linalg.fill ins(%cst : f16) outs(%24 : memref<64xf16, "SFU_REG">)
        }
      }
      ktdf_lowering.execute_on %10, %12, %14, %8, %16, %18 {
        %25 = builtin.unrealized_conversion_cast %c0 : index to memref<1x32xf32, "L1">
        %26 = builtin.unrealized_conversion_cast %c32 : index to memref<1x32xf32, "L1">
        %27:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MPE_ARRAY" -> "SFU", 32xf32>, !ktdf.fifo.slot<"MPE_ARRAY" -> "SFU", 32xf32>
        %28 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
        %29 = builtin.unrealized_conversion_cast %c128 : index to memref<1x64xf16, "L1">
        %30:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "MPE_ARRAY", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "MPE_ARRAY", 32xf32>
        ktdf_lowering.execute_on %10 {
          ktdf.data_transfer from %memspacecast[%c0, 0] size [1, 32] to %25[0, 0] size [1, 32] {dataflow_scheduler.throttle = 32 : i64} : memref<2x32xf32, "DDR">, memref<1x32xf32, "L1">
          ktdf.data_transfer from %memspacecast_0[%c1, 0] size [1, 32] to %26[0, 0] size [1, 32] {dataflow_scheduler.throttle = 32 : i64} : memref<2x32xf32, "DDR">, memref<1x32xf32, "L1">
        }
        ktdf_lowering.signal %10, %12
        ktdf_lowering.execute_on %12 {
          ktdf.data_transfer from %25[0, 0] size [1, 32] to %30#0 size [32] : memref<1x32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "MPE_ARRAY", 32xf32>
          ktdf.data_transfer from %26[0, 0] size [1, 32] to %30#1 size [32] : memref<1x32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "MPE_ARRAY", 32xf32>
        }
        ktdf_lowering.execute_on %14 {
          %31 = ktdf.read_from_fifo %30#0 : <"L1LU" -> "MPE_ARRAY", 32xf32> -> tensor<32xf32>
          ktdf.write_to_fifo %31, %27#0 : tensor<32xf32>, <"MPE_ARRAY" -> "SFU", 32xf32>
          %32 = ktdf.read_from_fifo %30#1 : <"L1LU" -> "MPE_ARRAY", 32xf32> -> tensor<32xf32>
          ktdf.write_to_fifo %32, %27#1 : tensor<32xf32>, <"MPE_ARRAY" -> "SFU", 32xf32>
        }
        ktdf_lowering.execute_on %8 {
          %31 = ktdf.read_from_fifo %27#0 : <"MPE_ARRAY" -> "SFU", 32xf32> -> tensor<32xf32>
          %32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%31 : tensor<32xf32>) outs(%21 : tensor<32xf32>) attrs =  {dataflow_scheduler.throttle = 32 : i64, ktdf_arch.maps_to = "SFU"} {
          ^bb0(%in: f32, %out: f32):
            %40 = math.floor %in : f32
            linalg.yield %40 : f32
          } -> tensor<32xf32>
          %33 = ktdf.read_from_fifo %27#1 : <"MPE_ARRAY" -> "SFU", 32xf32> -> tensor<32xf32>
          %34 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%33 : tensor<32xf32>) outs(%22 : tensor<32xf32>) attrs =  {dataflow_scheduler.throttle = 32 : i64, ktdf_arch.maps_to = "SFU"} {
          ^bb0(%in: f32, %out: f32):
            %40 = math.floor %in : f32
            linalg.yield %40 : f32
          } -> tensor<32xf32>
          %35 = ktdf_lowering.numeric_cast_value ins(%32, %34 : tensor<32xf32>, tensor<32xf32>) outs(tensor<64xf16>) {kind = "fp32todl16", ktdf_arch.maps_to = "SFU"}
          %36 = builtin.unrealized_conversion_cast %c64 : index to memref<64xf16, "SFU_REG">
          %37 = builtin.unrealized_conversion_cast %c128 : index to memref<64xf16, "SFU_REG">
          %38 = builtin.unrealized_conversion_cast %c192 : index to memref<64xf16, "SFU_REG">
          %39 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%35 : tensor<64xf16>) outs(%23 : tensor<64xf16>) attrs =  {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
          ^bb0(%in: f16, %out: f16):
            memref.store %in, %36[%c0] : memref<64xf16, "SFU_REG">
            ktdf.opaque "fake_exp" {dataflow_scheduler.register_names = ["c0", "in0", "out0", "t0_0"], func_name = "fake_exp"}
              ins(%24, %36: memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">)
              outs(%37, %38: memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">)
            %40 = memref.load %37[%c0] : memref<64xf16, "SFU_REG">
            linalg.yield %40 : f16
          } -> tensor<64xf16>
          ktdf.write_to_fifo %39, %28 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
        }
        ktdf_lowering.execute_on %16 {
          ktdf.data_transfer from %28 size [64] to %29[0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
        }
        ktdf_lowering.signal %16, %18
        ktdf_lowering.execute_on %18 {
          ktdf.data_transfer from %29[0, 0] size [1, 64] to %memspacecast_1[%c0, 0] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, "L1">, memref<1x64xf16, "DDR">
        }
      }
      return
    }
  }
}
