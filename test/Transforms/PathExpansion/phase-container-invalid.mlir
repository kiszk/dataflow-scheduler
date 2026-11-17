// RUN: dataflow-scheduler-opt --path-expansion --split-input-file \
// RUN:   --verify-diagnostics %s

// Reusing a unit in different phases requires a completion dependency.
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @unordered_units() {
    ktdf.pipeline {
      // expected-note @below {{unordered phase is here}}
      ktdf.stage depends_in(none) depends_out(none) {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            %value = arith.constant 1.0 : f16
          } {applicable_units = ["SFU"]}
        }
      }
      // expected-error @below {{path-expansion: phase stages require a transitive token order before reusing an execution unit}}
      ktdf.stage depends_in(none) depends_out(none) {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            %value = arith.constant 2.0 : f16
          } {applicable_units = ["SFU"]}
        }
      }
    }
    return
  }
}

// -----

// Casts and private results do not hide shared scratch memory.
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @unordered_scratch() {
    %c0 = arith.constant 0 : index
    ktdf.pipeline {
      %scratch = ktdf.private -> (memref<64xf16, "L1">) {
        %alloc = memref.alloc() : memref<64xf16, "L1">
        ktdf.private_yield %alloc : memref<64xf16, "L1">
      }
      // expected-note @below {{unordered phase is here}}
      ktdf.stage depends_in(none) depends_out(none) {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            %size = memref.dim %scratch, %c0 : memref<64xf16, "L1">
          } {applicable_units = ["SFU"]}
        }
      }
      // expected-error @below {{path-expansion: phase stages require a transitive token order before accessing shared memory}}
      ktdf.stage depends_in(none) depends_out(none) {
        %view = memref.cast %scratch : memref<64xf16, "L1"> to memref<?xf16, "L1">
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            %size = memref.dim %view, %c0 : memref<?xf16, "L1">
          } {applicable_units = ["SFU"]}
        }
      }
    }
    return
  }
}

// -----

// A token cannot make a cross-phase streaming FIFO safe: a phase must finish
// before its successor starts. Use shared memory for a phase boundary.
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @cross_phase_fifo() {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>, !ktdf.token) {
        %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
        %token = ktdf.create_token : !ktdf.token
        ktdf.private_yield %fifo, %token : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#1) {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            %value = arith.constant dense<1.0> : tensor<64xf16>
            // expected-error @below {{path-expansion: FIFO slots cannot cross phase boundaries}}
            ktdf.write_to_fifo %value, %p#0 : tensor<64xf16>, <"SFU" -> "SFU", 64xf16>
          } {applicable_units = ["SFU"]}
        }
      }
      ktdf.stage depends_in(%p#1) depends_out(none) {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            %value = ktdf.read_from_fifo %p#0 : <"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
          } {applicable_units = ["SFU"]}
        }
      }
    }
    return
  }
}

// -----

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @mixed_stages() {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            %value = arith.constant 1.0 : f16
          } {applicable_units = ["SFU"]}
        }
      }
      // expected-error @below {{path-expansion: cannot mix leaf stages and phase-container stages}}
      ktdf.stage depends_in(none) depends_out(none) {
        %value = arith.constant 2.0 : f16
      } {applicable_units = ["SFU"]}
    }
    return
  }
}

// -----

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @work_outside_pipeline(%output: memref<64xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %value = arith.constant 1.0 : f16
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        // expected-error @below {{path-expansion: phase wrapper contains work outside its nested pipeline}}
        memref.store %value, %output[%c0] : memref<64xf16, "L1">
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            %other = arith.constant 2.0 : f16
          } {applicable_units = ["SFU"]}
        }
      }
    }
    return
  }
}

// -----

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @cyclic_phases() {
    %t0 = ktdf.create_token : !ktdf.token
    %t1 = ktdf.create_token : !ktdf.token
    // expected-error @below {{path-expansion: phase token dependencies contain a cycle}}
    ktdf.pipeline {
      ktdf.stage depends_in(%t1) depends_out(%t0) {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            %value = arith.constant 1.0 : f16
          } {applicable_units = ["SFU"]}
        }
      }
      ktdf.stage depends_in(%t0) depends_out(%t1) {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            %value = arith.constant 2.0 : f16
          } {applicable_units = ["SFU"]}
        }
      }
    }
    return
  }
}
