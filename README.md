# `DataflowScheduler`

## Table of Contents

- [Description](#description)
  - [Design principles](#design-principles)
  - [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Build Instructions](#build-instructions)
  - [Linking against shared LLVM/MLIR](#linking-against-shared-llvmmlir)
- [Usage](#usage)
  - [Options](#options)
  - [Debugging](#debugging)
- [Documentation](#documentation)
- [Running Tests](#running-tests)
- [Installing](#installing)

## Description

`DataflowScheduler` is an MLIR-based compiler infrastructure that transforms a Kernel Tile IR (KTIR) program into a Dataflow IR (DFIR) execution schedule for dataflow hardware accelerators. It is the scheduling stage of a dataflow compiler backend, sitting ahead of code generation and binary generation, both of which are kept agnostic of the scheduler and its intermediate schedule representation.

Dataflow accelerators are built from a distributed set of compute engines and scratchpad memories connected by wide data paths with FIFOs, and follow a decoupled access-execute paradigm in which compute is separated from load/store (DMA) movement across distinct engines. In this setting, *scheduling* means constructing and optimizing a **dataflow pipeline**—operating across space and time—that wires together these decoupled compute and memory-movement engines to realize the original computation.

### Design principles

- **Architecture-driven, not hardware-hardcoded.** The scheduler takes an *architecture description* (an MLIR file) as input alongside the KTIR program, so it is not tied to any single accelerator and can target other or future architectures.
- **Pluggable backends.** The flow is organized so that different code-generation backends can be plugged in below the schedule representation.
- **Built on open-source LLVM/MLIR.** Analyses and transformations reuse upstream MLIR dialects and utilities (e.g. `linalg`, `scf`) wherever possible.
- **A single execution model.** All analyses and optimizations are expressed in terms of a *pipeline + parallel* execution model.

### Overview

The scheduler takes a KTIR program—a tile-based, block-structured IR expressing kernels over a data-parallel hardware abstraction—together with an architecture file, lowers it through an internal Schedule IR (KTDF) and a KTDFLow representation, and emits Dataflow IR (DFIR) for the downstream code generator. The guiding strategy is to *start with a simple pipeline, legalize it, and incrementally enlarge and refine it*: the program is partitioned into compute groups at memory boundaries, lowered to `linalg.generic` and fused, and materialized into an initial load → compute → store pipeline over FIFO slots. Successive transformations then make data movement hardware-legal by routing it through the required memory hierarchy, tile the loops for multi-granularity scheduling, build hierarchical and sibling pipelines, hoist loop-invariant transfers, apply double buffering, parallelize work across compute instances, and select concrete tile sizes. Finally, the logical multi-dimensional launch grid is normalized to the physical 1-D grid, hardware units are made explicit, and the program is split into one independent dataflow unit per engine before being emitted as Dataflow IR.

> **Status.** This is version 1 of the scheduler. It currently supports element-wise operations (e.g. add/mul/sub) with broadcast and reduction dimensions, handles multi-core KTIR programs, and performs the optimizations above (multiple corelets, double buffering, hierarchical and sibling pipelines, tiling, transfer/reuse promotion, and more). The V1 flow has been functionally validated end-to-end—from KTIR translation through scheduling to code generation—on representative transformer operations.

## Prerequisites

Before building `DataflowScheduler`, you need the following dependencies:

1. **Build Tools**
   - Git
   - CMake >= 3.21.0
   - C++17 compiler (clang >= 21.0.0 recommended)
   - Ninja build (optional, recommended)
   - Python >= 3.12 (optional)

2. **LLVM Project** - The LLVM/MLIR infrastructure
   
   You can build this project using a local installation of LLVM & MLIR. However, this project requires the specific revision `llvmorg-22.1.3` (SHA: `e9846648`).

   If you do not have such an installation, you can build it using:
  
   ```sh
   # Clone the repository as lean as possible.
   git clone https://github.com/llvm/llvm-project.git \
     --branch llvmorg-22.1.3 --sparse --depth 1
   cd llvm-project
   git sparse-checkout add cmake libc llvm mlir runtimes third-party

   # Configure LLVM & MLIR.
   cmake -S ./llvm -B build -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_PROJECTS=mlir \
     -DLLVM_TARGETS_TO_BUILD="host"
   # If you want to build the Python bindings, add:
     # -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
     # -DMLIR_PYTHON_STUBGEN_ENABLED=ON
   # If you are a developer, add:
     # -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     # -DBUILD_SHARED_LIBS=ON \
     # -DLLVM_ENABLE_ASSERTIONS=ON

   # Build LLVM. This may take a while, and consume a lot of space.
   cmake --build build
   ```

## Build Instructions

Building `DataflowScheduler` automatically fetches and builds the required KTIR submodules, but requires a working build or installation of LLVM and MLIR to be present on your system.

```sh
# Clone `dataflow-scheduler` and all its submodules.
git clone --recursive https://github.com/torch-spyre/dataflow-scheduler.git
cd dataflow-scheduler

# Configure the project.
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=$MLIR_INSTALL_PREFIX/lib/cmake/mlir

# Build everything.
cmake --build build --target all
```

The following CMake variables can be configured:

|              Name | Type      | Description |
| ----------------: | :-------- | --- |
| `MLIR_DIR` <br/>*(required)* | `STRING` | Path to the CMake directory of an **MLIR** installation. <br/> *e.g. `~/tools/llvm-12.0.1/lib/cmake/mlir`* |
| `LLVM_EXTERNAL_LIT` <br/>*(optional)* | `STRING` | Path to a `lit` executable, required for testing. |
| `DataflowScheduler_ENABLE_PYTHON_BINDINGS` <br/>*(default: `OFF`)* | `BOOL` | Whether to build the Python bindings. |
| `DataflowScheduler_BUILD_TOOLS` <br/>*(default: `ON`)* | `BOOL` | Whether to build the tool executables. |
| `DataflowScheduler_KTIR_SOURCE_DIR` <br/>*(default: `external/ktir-mlir-frontend`)* | `PATH` | KTIR frontend source checkout to build; use an absolute path to integrate local frontend changes without modifying the submodule. |
| `MLIR_LINK_MLIR_DYLIB` <br/>*(optional)* | `BOOL` | Whether to link against a shared MLIR library. |

### **Linking against shared LLVM/MLIR**

This project supports linking against a version of LLVM/MLIR that was build using the `LLVM_BUILD_LLVM_DYLIB` etc. flags. Since MLIR does not export the `MLIR_LINK_MLIR_DYLIB` flag, it is inferred from the presence of the `MLIR` library target when it is not specified at configure time. When linking against shared MLIR, CMake will set up an RPATH to your MLIR install destination.

### DL16 numeric casts

`spyreop.numeric_cast` uses `!spyreop.df16` for SEN169/DL16 and `f32` for IEEE FP32. The scheduler preserves `!spyreop.df16` through KTDF and KTDFLow, including memory views, FIFO slots, and register storage. KTDFLow-to-DFIR lowering maps it to DFIR's existing `f16` storage carrier and emits the native conversion recipe. This representation change does not convert SEN169 data to IEEE f16. Existing device descriptions can continue to name the native 16-bit SIMD width with `f16`; an explicit `!spyreop.df16` lane entry takes precedence when sizing scheduler tiles. IEEE-only `arith` and `math` operations cannot take `!spyreop.df16`; use supported SpyreOp intrinsics for DL16 arithmetic.

## Usage

Once built, the `dataflow-scheduler` driver lowers a single KTIR program to Dataflow IR (DFIR), given an architecture (device) file:

```sh
dataflow-scheduler -kEmitDFIR --device=<path-to-device.mlir> <input-ktir.mlir>
```

### Options

- `-kEmitDFIR` — the pass pipeline that lowers KTIR → DFIR (the full `buildKTDPToDFIRPipeline`).
- `--device=<file>` — the architecture/device description file. Use an **absolute path**; the driver resolves and stores it as absolute.
- `--split-dfir-output-dir=<dir>` — where the split DFIR output files are written (default: the input file's parent directory).

**Output.** The tool writes a `global.mlir` plus one `.mlir` file per function (e.g. `local-schedule-0.mlir`) into the output directory.

### Debugging

These standard MLIR flags are useful when a pipeline fails:

```sh
# Print the IR after each pass (find which pass fails — look for "... Failed (pass-name)")
dataflow-scheduler -kEmitDFIR --device=<dev> in.mlir --mlir-print-ir-after-all

# Print the IR before each pass
dataflow-scheduler -kEmitDFIR --device=<dev> in.mlir --mlir-print-ir-before-all

# Trace a specific pass (e.g. why address assignment fails)
dataflow-scheduler -kEmitDFIR --device=<dev> in.mlir --debug-only=address-assignment
```

Run `dataflow-scheduler --help` to list every available flag.

> **Note:** `dataflow-scheduler-opt` is a separate tool used by the `lit` regression tests. Unlike the `dataflow-scheduler` driver, it takes an explicit `-pass-pipeline="builtin.module(...)"` and obtains the device from a `ktdf_arch.device` operation embedded in the input, rather than via `--device`.

## Documentation

In addition to the documentation found in the project sources, there are auto-generated documentations for the MLIR dialects, interfaces and passes. These can be generated to the build directory using:

```sh
cmake --build build --target dataflow-scheduler-doc
```

## Running Tests

The regression tests can be run with the following command:

```sh
cmake --build build --target check-dataflow-scheduler
```

## Installing

You can install the generated package to `$INSTALL_PREFIX` using the comamnd:

```sh
cmake --install build --prefix $INSTALL_PREFIX
```

Note that the contents of your install will depend on the targets you have built so far. In particular, building the documentation target before installing will cause the docs to be installed. You can also select specific components for installation.
