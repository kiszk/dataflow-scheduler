// RUN: dataflow-scheduler-opt --subsume-linearize-index --allow-unregistered-dialect %s | FileCheck %s

// -----------------------------------------------------------------------------
// Test 1: 1D subscript with strides [12, 1] folds into inline affine expression
//
// tiling.linearize_index [%arg0 : %c12], [%arg1 : %c1] computes 12*arg0 + arg1.
// After folding, the data_transfer source index becomes the inline expression
// "12 * %arg0 + %arg1" via printAffineMapOfSSAIds.
// The tiling.linearize_index op must be gone.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_1d_subscript
// CHECK-NOT:   ktdf.tiling.linearize_index
// CHECK:       ktdf.data_transfer
// CHECK-SAME:  from %{{.*}}[%arg0 * 12 + %arg1]
// CHECK-NOT:   ktdf.tiling.linearize_index

func.func @test_1d_subscript() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %c64 = arith.constant 64 : index
  %src = "test.source"() : () -> memref<768xf16>
  %dst = "test.dest"() : () -> memref<64xf16>
  scf.for %arg0 = %c0 to %c12 step %c1 {
    scf.for %arg1 = %c0 to %c12 step %c1 {
      %idx = ktdf.tiling.linearize_index [%arg0 : %c12], [%arg1 : %c1] : index
      ktdf.data_transfer from %src[%idx] size [64]
                         to %dst[%c0] size [64]
        : memref<768xf16>, memref<64xf16>
    }
  }
  return
}

// -----------------------------------------------------------------------------
// Test 2: two subscripts fold both source dims
//
// tiling.linearize_index for dim0: [%arg0 : %c12], [%arg2 : %c1] -> 12*arg0 + arg2
// tiling.linearize_index for dim1: [%arg1 : %c4],  [%arg3 : %c1] ->  4*arg1 + arg3
// After folding both, the source index list becomes
//   [12 * %arg0 + %arg2, 4 * %arg1 + %arg3]
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_two_subscripts
// CHECK-NOT:   ktdf.tiling.linearize_index
// CHECK:       ktdf.data_transfer
// CHECK-SAME:  from %{{.*}}[%arg0 * 12 + %arg2, %arg1 * 4 + %arg3]
// CHECK-NOT:   ktdf.tiling.linearize_index

func.func @test_two_subscripts() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %src = "test.source"() : () -> memref<12x64xf16>
  %dst = "test.dest"() : () -> memref<64xf16>
  scf.for %arg0 = %c0 to %c12 step %c1 {
    scf.for %arg1 = %c0 to %c12 step %c1 {
      scf.for %arg2 = %c0 to %c12 step %c1 {
        scf.for %arg3 = %c0 to %c12 step %c1 {
          %idx0 = ktdf.tiling.linearize_index [%arg0 : %c12], [%arg2 : %c1] : index
          %idx1 = ktdf.tiling.linearize_index [%arg1 : %c4], [%arg3 : %c1] : index
          ktdf.data_transfer from %src[%idx0, %idx1] size [1, 64]
                             to %dst[%c0] size [64]
            : memref<12x64xf16>, memref<64xf16>
        }
      }
    }
  }
  return
}

// -----------------------------------------------------------------------------
// Test 3: subscript mixed with constant %c0 — the constant folds to 0 in map
//
// source indices: [%idx, %c0] where %idx = tiling.linearize_index [%arg0 : %c12], [%arg1 : %c1] : index
// fullyComposeAffineMapAndOperands folds the arith.constant 0 into the map,
// so the second result dimension becomes the literal integer 0.
// After folding, the source index becomes [12 * %arg0 + %arg1, 0].
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_mixed_with_constant
// CHECK-NOT:   ktdf.tiling.linearize_index
// CHECK:       ktdf.data_transfer
// CHECK-SAME:  from %{{.*}}[%arg0 * 12 + %arg1, 0]
// CHECK-NOT:   ktdf.tiling.linearize_index

func.func @test_mixed_with_constant() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %src = "test.source"() : () -> memref<12x64xf16>
  %dst = "test.dest"() : () -> memref<64xf16>
  scf.for %arg0 = %c0 to %c12 step %c1 {
    scf.for %arg1 = %c0 to %c12 step %c1 {
      %idx = ktdf.tiling.linearize_index [%arg0 : %c12], [%arg1 : %c1] : index
      ktdf.data_transfer from %src[%idx, %c0] size [1, 64]
                         to %dst[%c0] size [64]
        : memref<12x64xf16>, memref<64xf16>
    }
  }
  return
}

// -----------------------------------------------------------------------------
// Test 4: subscript on the dest side (not just source)
//
// The pass must fold tiling.linearize_index used as a dest index too.
// tiling.linearize_index [%arg0 : %c12], [%arg1 : %c1] -> 12*arg0 + arg1
// After folding, the dest index becomes 12 * %arg0 + %arg1.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_dest_subscript
// CHECK-NOT:   ktdf.tiling.linearize_index
// CHECK:       ktdf.data_transfer
// CHECK-SAME:  to %{{.*}}[%arg0 * 12 + %arg1]
// CHECK-NOT:   ktdf.tiling.linearize_index

func.func @test_dest_subscript() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %src = "test.source"() : () -> memref<64xf16>
  %dst = "test.dest"() : () -> memref<768xf16>
  scf.for %arg0 = %c0 to %c12 step %c1 {
    scf.for %arg1 = %c0 to %c12 step %c1 {
      %idx = ktdf.tiling.linearize_index [%arg0 : %c12], [%arg1 : %c1] : index
      ktdf.data_transfer from %src[%c0] size [64]
                         to %dst[%idx] size [64]
        : memref<64xf16>, memref<768xf16>
    }
  }
  return
}

// CHECK-LABEL: func.func @test_two_packet_destination
// CHECK:       ktdf.data_transfer
// CHECK-SAME:  to %{{.*}}[%arg0 * 2, %arg1, 0]
// CHECK:       ktdf.data_transfer
// CHECK-SAME:  to %{{.*}}[%arg0 * 2 + 1, %arg1, 0]
func.func @test_two_packet_destination() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %src = "test.source"() : () -> memref<2x32xf32>
  %dst = "test.dest"() : () -> memref<6x2x32xf32>
  scf.for %arg0 = %c0 to %c3 step %c1 {
    %first = arith.muli %arg0, %c2 : index
    %second = arith.addi %first, %c1 : index
    scf.for %arg1 = %c0 to %c2 step %c1 {
      ktdf.data_transfer from %src[%arg1, %c0] size [1, 32]
                         to %dst[%first, %arg1, %c0] size [1, 1, 32]
        : memref<2x32xf32>, memref<6x2x32xf32>
      ktdf.data_transfer from %src[%arg1, %c0] size [1, 32]
                         to %dst[%second, %arg1, %c0] size [1, 1, 32]
        : memref<2x32xf32>, memref<6x2x32xf32>
    }
  }
  return
}

// -----------------------------------------------------------------------------
// Test 5: single-IV stride-1 subscript (degenerate case)
//
// tiling.linearize_index [%arg : %c1] is simply %arg itself (1 * arg + 0 = arg).
// After folding, the source index should be just %arg.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_stride1_single_iv
// CHECK-NOT:   ktdf.tiling.linearize_index
// CHECK:       ktdf.data_transfer
// CHECK-SAME:  from %{{.*}}[%arg0]
// CHECK-NOT:   ktdf.tiling.linearize_index

func.func @test_stride1_single_iv() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %src = "test.source"() : () -> memref<64xf16>
  %dst = "test.dest"() : () -> memref<64xf16>
  scf.for %arg0 = %c0 to %c64 step %c1 {
    %idx = ktdf.tiling.linearize_index [%arg0 : %c1] : index
    ktdf.data_transfer from %src[%idx] size [1]
                       to %dst[%c0] size [1]
      : memref<64xf16>, memref<64xf16>
  }
  return
}

// -----------------------------------------------------------------------------
// Test 6: NEGATIVE — dynamic stride is NOT folded
//
// tiling.linearize_index [%arg1 : %arg0], [%arg2 : %c1] — %arg0 is a runtime value,
// not a compile-time constant. The pass must leave this subscript intact.
// The data_transfer must still reference the tiling.linearize_index result directly.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_nonfold_dynamic_stride
// CHECK:       %[[IDX:.*]] = ktdf.tiling.linearize_index
// CHECK:       ktdf.data_transfer
// CHECK-SAME:  from %{{.*}}[%[[IDX]]]

func.func @test_nonfold_dynamic_stride(%arg0 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %src = "test.source"() : () -> memref<768xf16>
  %dst = "test.dest"() : () -> memref<64xf16>
  scf.for %arg1 = %c0 to %c12 step %c1 {
    scf.for %arg2 = %c0 to %c12 step %c1 {
      %idx = ktdf.tiling.linearize_index [%arg1 : %arg0], [%arg2 : %c1] : index
      ktdf.data_transfer from %src[%idx] size [64]
                         to %dst[%c0] size [64]
        : memref<768xf16>, memref<64xf16>
    }
  }
  return
}

// -----------------------------------------------------------------------------
// Test 7: memref→FIFO transfer — subscript folds on the source (memref) side
//
// The dest is a FIFO slot; it has no indices and no dest_map attribute.
// The pass must fold the source subscript even though dest_map is absent.
// tiling.linearize_index [%arg0 : %c12], [%arg1 : %c1] -> 12*arg0 + arg1
// After folding, the source index becomes 12 * %arg0 + %arg1.
// The dest FIFO slot is untouched.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_memref_to_fifo
// CHECK-NOT:   ktdf.tiling.linearize_index
// CHECK:       ktdf.data_transfer
// CHECK-SAME:  from %{{.*}}[%arg0 * 12 + %arg1]
// CHECK-SAME:  to %{{.*}} size [64]
// CHECK-NOT:   ktdf.tiling.linearize_index

func.func @test_memref_to_fifo() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %src = "test.source"() : () -> memref<768xf16>
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
  scf.for %arg0 = %c0 to %c12 step %c1 {
    scf.for %arg1 = %c0 to %c12 step %c1 {
      %idx = ktdf.tiling.linearize_index [%arg0 : %c12], [%arg1 : %c1] : index
      ktdf.data_transfer from %src[%idx] size [64]
                         to %fifo size [64]
        : memref<768xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
  }
  return
}

// -----------------------------------------------------------------------------
// Test 8: FIFO→memref transfer — subscript folds on the dest (memref) side
//
// The source is a FIFO slot; it has no indices and no source_map attribute.
// The pass must fold the dest subscript even though source_map is absent.
// tiling.linearize_index [%arg0 : %c12], [%arg1 : %c1] -> 12*arg0 + arg1
// After folding, the dest index becomes 12 * %arg0 + %arg1.
// The source FIFO slot is untouched.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @test_fifo_to_memref
// CHECK-NOT:   ktdf.tiling.linearize_index
// CHECK:       ktdf.data_transfer
// CHECK-SAME:  to %{{.*}}[%arg0 * 12 + %arg1]
// CHECK-NOT:   ktdf.tiling.linearize_index

func.func @test_fifo_to_memref() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
  %dst = "test.dest"() : () -> memref<768xf16>
  scf.for %arg0 = %c0 to %c12 step %c1 {
    scf.for %arg1 = %c0 to %c12 step %c1 {
      %idx = ktdf.tiling.linearize_index [%arg0 : %c12], [%arg1 : %c1] : index
      ktdf.data_transfer from %fifo size [64]
                         to %dst[%idx] size [64]
        : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, memref<768xf16>
    }
  }
  return
}
