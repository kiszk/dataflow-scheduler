//===-- KTDFLowering.td ------------------------------------*- tablegen -*-===//
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
//
// This file defines KTDFLowering operations in the MLIR operation set.
// These are intermediate operations used only during the lowering from
// KTIR into Dataflow IR.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDFLOWERING_KTDFLOWERING_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDFLOWERING_KTDFLOWERING_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLoweringDialect.td"
include "dataflow-scheduler/Dialect/KTDF/KTDFEnums.td"
include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.td"
include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.td"

class KTDFLowering_Op<string mnemonic, list<Trait> traits = []> :
    Op<KTDFLowering_Dialect, mnemonic, traits> {
}

def KTDFLowering_ExecuteOnOp : KTDFLowering_Op<"execute_on", [
    NoTerminator,
    SingleBlock
  ]> {
  let summary = "Represents execution on one or more hardware units with a body region";
  let description = [{
    The `ktdf_lowering.execute_on` operation represents execution on one or more
    hardware units. It wraps a body region containing operations that should be
    executed on the specified units.

    This operation takes index-typed SSA values as operands representing the units
    to execute on. These are typically results from dataflow.get_unit operations
    and uniform.query_map results in Phase 1.5.

    This is a temporary operation used during the lowering process from KTIR to
    Dataflow IR. In Phase 2, pipelines and stages are replaced with execute_on
    operations. In Phase 3, these are used to create separate program_unit
    operations for each unit type.

    Example:
    ```mlir
    // Single unit operand
    ktdf_lowering.execute_on(%sfu_unit) {
      %result = arith.addf %a, %b : f32
    }

    // Multiple unit operands
    ktdf_lowering.execute_on(%mnilu_unit, %mnisu_unit) {
      // Operations for multiple units
    }
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$units
  );

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the block containing the body.
    Block *getBodyBlock() { return &getBody().front(); }

    /// Get the number of unit operands
    size_t getNumUnits() { return getUnits().size(); }
  }];
}

def KTDFLowering_SignalOp : KTDFLowering_Op<"signal", []> {
  let summary = "Synchronization signal between units";
  let description = [{
    The `ktdf_lowering.signal` operation represents a synchronization point
    between hardware units. It takes index-typed SSA values as operands representing
    the units that need to synchronize.

    This is a temporary operation used during the lowering process. In Phase 2,
    signals are inserted between stages based on their dependencies and scratchpad
    conflicts. In Phase 3, they are lowered to dataflow.sync_send/recv operations.

    This operation has side effects to prevent DCE (Dead Code Elimination) from
    removing it before it can be lowered to sync operations.

    Example:
    ```mlir
    // Signal between MNILU and L1LU
    ktdf_lowering.signal(%mnilu_unit, %l1lu_unit)

    // Signal between L1LU and SFU
    ktdf_lowering.signal(%l1lu_unit, %sfu_unit)
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$units
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the number of unit operands
    size_t getNumUnits() { return getUnits().size(); }
  }];
}

def KTDFLowering_NativeCastOp : KTDFLowering_Op<"native_cast", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Convert one native stick through ordered FIFO packets";
  let description = [{
    The `ktdf_lowering.native_cast` operation consumes input FIFO slots and
    writes output FIFO slots in operand order. It converts one 64-element
    native stick, using `f16` as the storage carrier for SEN169/DL16. The
    `kind` selects native numeric conversion, not an IEEE f16 conversion.

    `dl16tofp32` consumes one `64xf16` slot and writes two `32xf32` slots.
    `fp32todl16` consumes two `32xf32` slots and writes one `64xf16` slot.
    For packet p, group q and lane r, the staggered FP32 arrangement is:

    ```text
    fp32[p][4*q+r] = native_cast(dl16[8*q+4*p+r])
    p = 0..1, q = 0..7, r = 0..3
    ```

    Narrowing applies the reverse map after numeric conversion. Every lane
    is valid; neither direction inserts padding. Input slots share their
    source and destination, output slots share theirs, and the input
    destination equals the output source. The operation must be assigned to
    that execution unit before lowering to DataflowIR.
  }];

  let arguments = (ins
    Variadic<KTDF_FifoSlotType>:$inputs,
    Variadic<KTDF_FifoSlotType>:$outputs,
    StrAttr:$kind
  );

  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outputs `:` type($outputs) `)` attr-dict
  }];

  let hasVerifier = 1;
}

def KTDFLowering_NativeCastValueOp : KTDFLowering_Op<"native_cast_value", [Pure]> {
  let summary = "Convert native DL16 and staggered FP32 tensor packets";
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs, StrAttr:$kind);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = [{
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` type($outputs) `)` attr-dict
  }];
  let hasVerifier = 1;
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDFLOWERING_KTDFLOWERING_TD
