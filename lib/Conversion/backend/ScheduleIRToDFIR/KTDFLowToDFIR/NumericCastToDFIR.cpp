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

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"
#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/NativeCastLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/PipelineBuilder.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace scheduler {
#define GEN_PASS_DEF_NUMERICCASTPIPELINEPASS
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h.inc"
}  // namespace scheduler

namespace {

using namespace mlir;

struct Buffer {
  RankedTensorType type;
  int argument = -1;
  int64_t address = 0;
};

struct CastPlan {
  func::FuncOp function;
  spyreop::NumericCast conversion;
  Buffer input;
  Buffer output;
  int64_t cores;
};

// Accept the two equivalent spellings of a unit extent: d == 0, or
// 0 <= d <= 0. Other sets need an access lowering, not this fixed recipe.
bool isBox(IntegerSet set, ArrayRef<int64_t> shape) {
  if (set.getNumDims() != shape.size() || set.getNumSymbols()) return false;
  SmallVector<bool> lower(shape.size()), upper(shape.size());
  for (auto [expr, equality] :
       llvm::zip(set.getConstraints(), set.getEqFlags())) {
    bool matched = false;
    for (auto [axis, size] : llvm::enumerate(shape)) {
      auto dim = getAffineDimExpr(axis, set.getContext());
      if (expr == dim && (!equality || size == 1)) {
        lower[axis] = true;
        if (equality) upper[axis] = true;
        matched = true;
        break;
      }
      if (!equality && expr == size - 1 - dim) {
        upper[axis] = true;
        matched = true;
        break;
      }
    }
    if (!matched) return false;
  }
  return llvm::all_of(lower, [](bool value) { return value; }) &&
         llvm::all_of(upper, [](bool value) { return value; });
}

LogicalResult readBuffer(Value access, RankedTensorType type,
                         func::FuncOp function, Buffer& buffer) {
  auto tile = access.getDefiningOp<ktdp::ConstructAccessTilesOp>();
  auto view = tile ? tile.getBase().getDefiningOp<ktdp::ConstructMemoryViewOp>()
                   : ktdp::ConstructMemoryViewOp();
  if (!view || !tile.getAccessTileOrder().isIdentity() ||
      !tile.getSymbolOperands().empty() ||
      !isBox(tile.getAccessTileSet().getValue(), type.getShape())) {
    return function.emitError(
        "numeric cast requires a complete, identity access tile");
  }
  SmallVector<Attribute> indices;
  for (Value index : tile.getIndices()) {
    auto value = getConstantIntValue(index);
    if (!value) {
      return tile.emitError("numeric cast requires constant indices");
    }
    indices.push_back(
        IntegerAttr::get(IndexType::get(function.getContext()), *value));
  }
  SmallVector<Attribute> offsets;
  if (failed(tile.getBaseMap().constantFold(indices, offsets)) ||
      offsets.size() != 2 || !llvm::all_of(offsets, [](Attribute attr) {
        return cast<IntegerAttr>(attr).getInt() == 0;
      })) {
    return tile.emitError("numeric cast requires zero tile offsets");
  }
  const auto shape = type.getShape();
  const SmallVector<int64_t> strides{shape.back(), 1};
  auto memory_type = cast<MemRefType>(view.getResult().getType());
  if (view.getMemorySpace().getKind() != ktdp::MemorySpaceKind::global ||
      view.getStaticSizes() != shape ||
      view.getStaticStrides() != ArrayRef<int64_t>(strides) ||
      memory_type.getShape() != shape ||
      memory_type.getElementType() != type.getElementType() ||
      !memory_type.getLayout().isIdentity() ||
      !isBox(view.getCoordinateSet().getValue(), shape)) {
    return view.emitError(
        "numeric cast requires a static contiguous global view");
  }
  buffer.type = type;
  if (auto argument = dyn_cast<BlockArgument>(view.getOffset());
      argument && argument.getOwner() == &function.getBody().front()) {
    buffer.argument = argument.getArgNumber();
  } else if (auto address = getConstantIntValue(view.getOffset());
             address && *address >= 0) {
    buffer.address = *address;
  } else {
    return view.emitError(
        "numeric cast requires a base argument or nonnegative "
        "constant element address");
  }
  return success();
}

FailureOr<CastPlan> matchCast(ModuleOp module) {
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (functions.size() != 1) {
    return module.emitError("numeric cast requires one function");
  }
  auto function = functions.front();
  auto grid = function->getAttrOfType<ArrayAttr>("grid");
  if (function.empty() || !llvm::hasSingleElement(function.getBody()) ||
      function.getNumResults() || !grid || grid.size() != 1 ||
      !isa<IntegerAttr>(grid[0]) || cast<IntegerAttr>(grid[0]).getInt() < 1 ||
      (function.getNumArguments() != 0 && function.getNumArguments() != 2) ||
      !llvm::all_of(function.getArgumentTypes(),
                    [](Type type) { return type.isIndex(); })) {
    return function.emitError(
        "numeric cast requires a one-dimensional grid and zero or two index "
        "arguments");
  }
  spyreop::NumericCast conversion;
  unsigned loads = 0, stores = 0, views = 0, tiles = 0;
  for (Operation& op : function.getBody().front()) {
    if (auto cast_op = dyn_cast<spyreop::NumericCast>(op)) {
      if (conversion) {
        return cast_op.emitError("only one numeric cast is supported");
      }
      conversion = cast_op;
    } else if (isa<ktdp::LoadOp>(op)) {
      ++loads;
    } else if (isa<ktdp::StoreOp>(op)) {
      ++stores;
    } else if (isa<ktdp::ConstructMemoryViewOp>(op)) {
      ++views;
    } else if (isa<ktdp::ConstructAccessTilesOp>(op)) {
      ++tiles;
    } else if (!isa<func::ReturnOp, arith::ConstantIndexOp>(op)) {
      return op.emitError("unsupported operation beside numeric cast");
    }
  }
  if (!conversion || loads != 1 || stores != 1 || views != 2 || tiles != 2) {
    return function.emitError(
        "numeric cast requires one global load, cast and global "
        "store");
  }
  bool widening = conversion.getKind() == "dl16tofp32";
  auto input = cast<RankedTensorType>(conversion.getInput().getType());
  auto output = cast<RankedTensorType>(conversion.getResult().getType());
  auto native = widening ? input : output;
  auto wide = widening ? output : input;
  if (!native.hasStaticShape() || !wide.hasStaticShape() ||
      native.getRank() != 2 || wide.getRank() != 2 ||
      native.getDimSize(0) <= 0 || wide.getDimSize(0) % 2 ||
      native.getDimSize(0) != wide.getDimSize(0) / 2 ||
      native.getDimSize(1) != 64 || wide.getDimSize(1) != 32 ||
      !native.getElementType().isF16() || native.getEncoding() ||
      wide.getEncoding() || !wide.getElementType().isF32()) {
    return conversion.emitError(
        "numeric cast requires tensor<Nx64xf16> and "
        "tensor<(2*N)x32xf32>, N > 0");
  }
  auto load = conversion.getInput().getDefiningOp<ktdp::LoadOp>();
  auto store =
      conversion.getResult().hasOneUse()
          ? dyn_cast<ktdp::StoreOp>(*conversion.getResult().user_begin())
          : ktdp::StoreOp();
  if (!load || !load->hasOneUse() || !store ||
      store.getDataTile() != conversion.getResult()) {
    return conversion.emitError(
        "numeric cast must connect one load directly to one store");
  }
  CastPlan plan{
      function, conversion, {}, {}, cast<IntegerAttr>(grid[0]).getInt()};
  if (native.getDimSize(0) % plan.cores) {
    return function.emitError(
        "numeric cast requires whole sticks evenly divided among cores");
  }
  if (failed(readBuffer(load.getAccessTile(), input, function, plan.input)) ||
      failed(
          readBuffer(store.getAccessTile(), output, function, plan.output))) {
    return failure();
  }
  if ((function.getNumArguments() == 2 &&
       (plan.input.argument < 0 || plan.output.argument < 0 ||
        plan.input.argument == plan.output.argument)) ||
      (function.getNumArguments() == 0 &&
       (plan.input.address % input.getShape().back() ||
        plan.output.address % output.getShape().back()))) {
    return function.emitError(
        "numeric cast requires distinct base arguments or "
        "stick-aligned constant addresses");
  }
  return plan;
}

struct CastRoute {
  StringAttr fetch, drain;
  SmallVector<StringAttr> input, output;
};

FailureOr<CastRoute> findCastRoute(const CastPlan& plan,
                                   const ktdf_arch::DeviceRef& device) {
  auto module = plan.function->getParentOfType<ModuleOp>();
  ktdf_arch::Mapping mapping(device);
  auto compute_count = llvm::count_if(mapping.byKind(), [](const auto& kind) {
    ktdf_arch::Resource resource = kind.getExemplar();
    return isa<ktdf_arch::ExecutionUnitOp>(resource) &&
           resource.getFeature<ktdf_arch::feature::Compute>();
  });
  if (compute_count != 1) {
    return module.emitError("no (unambiguous) default compute resource");
  }
  auto compute = mapping.getOrMap<ktdf_arch::ExecutionUnitOp>(
      plan.conversion, mapping.byKind().getDefaultCompute().getKind());
  if (failed(scheduler::verifyNativeCastUnit(compute, plan.conversion)))
    return failure();
  if (compute != mapping.byKind().getInstance<ktdf_arch::ExecutionUnitOp>(
                     cast<ktdf_arch::KindAttr>(compute.getKind()))) {
    return plan.conversion->emitError(
        "numeric cast requires the representative compute "
        "instance");
  }

  auto& memories = device.getOrCreateView<scheduler::arch_view::MemoryTree>();
  auto memory = [&](ktdp::MemorySpaceKind kind) {
    return memories.getCanonicalResource(cast<ktdf_arch::KindAttr>(
        ktdp::MemorySpaceAttr::get(module.getContext(), kind, -1)));
  };
  auto global = memory(ktdp::MemorySpaceKind::global);
  auto scratch = memory(ktdp::MemorySpaceKind::ct_local);
  if (!memories.getCapacity(scratch) || !memories.isGlobalMemory(global) ||
      !memories.isPerCoreScratchPadMemory(scratch)) {
    return module.emitError(
        "numeric cast requires global and sized "
        "per-core scratchpad memory");
  }
  using Graph = scheduler::arch_view::RoutingGraph;
  auto& graph = device.getOrCreateView<Graph>();
  auto path = [&](scheduler::ResourceType source, scheduler::ResourceType dest)
      -> FailureOr<SmallVector<Graph::ResourceNode>> {
    auto sources = graph.getNodeIdsForResource(source);
    auto destinations = graph.getNodeIdsForResource(dest);
    if (sources.size() != 1 || destinations.size() != 1) {
      return module.emitError(
          "numeric cast requires unambiguous routing resources");
    }
    auto route = graph.findShortestPath(sources.front(), destinations.front());
    if (!route) {
      return module.emitError("numeric cast has no route from ")
             << source << " to " << dest;
    }
    SmallVector<Graph::ResourceNode> result;
    for (auto id : *route) {
      auto node = *graph.getNode(id);
      if (graph.getNodeIdsForResource(node.resource).size() != 1) {
        return module.emitError(
            "numeric cast requires unambiguous routing resources");
      }
      result.push_back(node);
    }
    return result;
  };
  auto fetch = path(global, scratch);
  auto input = path(scratch, compute.getKind());
  auto output = path(compute.getKind(), scratch);
  auto drain = path(scratch, global);
  if (failed(fetch) || failed(input) || failed(output) || failed(drain))
    return failure();
  if (fetch->size() != 3 || drain->size() != 3) {
    return module.emitError(
        "numeric cast requires a single global/scratchpad "
        "transfer unit");
  }
  auto unit = [&](const Graph::ResourceNode& node) {
    return mapping.byKind().getInstance<ktdf_arch::ExecutionUnitOp>(
        cast<ktdf_arch::KindAttr>(node.resource));
  };
  auto transfer = [&](const Graph::ResourceNode& node, bool load,
                      bool global_transfer) -> FailureOr<StringAttr> {
    auto op = unit(node);
    auto kind = op ? dyn_cast<StringAttr>(op.getKind()) : StringAttr();
    if (!kind) {
      return module.emitError("numeric cast requires a named transfer unit");
    }
    auto load_feature = op.getFeature<ktdf_arch::feature::Load>();
    auto store_feature = op.getFeature<ktdf_arch::feature::Store>();
    if (load ? !load_feature : !store_feature) {
      return op.emitError(load ? "numeric cast requires a "
                                 "load-capable local load unit"
                               : "numeric cast requires a "
                                 "store-capable local store unit");
    }
    auto word_size = [&](scheduler::ResourceType space) {
      return load ? load_feature.getWordSize(space)
                  : store_feature.getWordSize(space);
    };
    if (global_transfer &&
        (word_size(scratch) != 128 || word_size(global) != 128)) {
      return op.emitError(
          "numeric cast requires 128-byte global "
          "and scratchpad words");
    }
    return kind;
  };
  auto fetch_unit = transfer((*fetch)[1], true, true);
  auto drain_unit = transfer((*drain)[1], false, true);
  if (failed(fetch_unit) || failed(drain_unit)) return failure();
  CastRoute result{*fetch_unit, *drain_unit, {}, {}};
  auto local = [&](ArrayRef<Graph::ResourceNode> nodes, bool load,
                   SmallVectorImpl<StringAttr>& units) -> LogicalResult {
    if (nodes.size() < 3) {
      return module.emitError("numeric cast requires a local load/store unit");
    }
    nodes = load ? nodes.drop_front() : nodes.drop_back();
    if (failed(transfer(load ? nodes.front() : nodes.back(), load, false)))
      return failure();
    for (auto [index, node] : llvm::enumerate(nodes)) {
      auto op = unit(node);
      auto kind = op ? dyn_cast<StringAttr>(op.getKind()) : StringAttr();
      if (!kind) {
        return module.emitError(
            "numeric cast requires execution-unit-only local "
            "transfer paths");
      }
      if (index != 0 && index + 1 != nodes.size()) {
        auto simd = op.getFeature<ktdf_arch::feature::SIMD>();
        if (!simd ||
            simd.getLanes(Float16Type::get(module.getContext())) != 64 ||
            simd.getLanes(Float32Type::get(module.getContext())) != 32) {
          return op.emitError(
              "numeric cast requires 64 f16 and 32 f32 relay "
              "lanes");
        }
      }
      units.push_back(kind);
    }
    return success();
  };
  if (failed(local(*input, true, result.input)) ||
      failed(local(*output, false, result.output)))
    return failure();
  return result;
}

// This is the bufferization boundary: f16 carries SEN169 bits in memory and
// FIFOs; native_cast retains the numerical conversion and staggered packet
// order.
void emitPipeline(func::FuncOp program, const CastPlan& plan,
                  const CastRoute& route) {
  OpBuilder builder(program.getContext());
  auto loc = program.getLoc();
  auto tile_type = [&](bool wide) {
    auto memory = ktdp::MemorySpaceAttr::get(
        builder.getContext(), ktdp::MemorySpaceKind::ct_local, -1);
    return MemRefType::get({wide ? 2 : 1, wide ? 32 : 64},
                           wide ? builder.getF32Type() : builder.getF16Type(),
                           nullptr, memory);
  };

  auto external = [&](OpBuilder& b, const Buffer& buffer) -> Value {
    Value address;
    if (buffer.argument < 0)
      address = arith::ConstantIndexOp::create(b, loc, buffer.address);
    else
      address = program.getArgument(buffer.argument);
    if (plan.cores > 1) {
      auto tile_id = ktdp::GetComputeTileIdOp::create(b, loc, b.getIndexType())
                         .getResult()[0];
      int64_t stride = buffer.type.getNumElements() / plan.cores;
      if (buffer.argument >= 0)
        stride *= buffer.type.getElementType().getIntOrFloatBitWidth() / 8;
      auto offset = arith::MulIOp::create(
          b, loc, tile_id, arith::ConstantIndexOp::create(b, loc, stride));
      address = arith::AddIOp::create(b, loc, address, offset);
    }
    auto memory = ktdp::MemorySpaceAttr::get(b.getContext(),
                                             ktdp::MemorySpaceKind::global, -1);
    SmallVector<int64_t> shape(buffer.type.getShape());
    shape[0] /= plan.cores;
    auto element = buffer.type.getElementType();
    return ktdp::ConstructMemoryViewOp::create(
        b, loc, MemRefType::get(shape, element, nullptr, memory), address,
        ValueRange{}, ValueRange{}, shape, ArrayRef<int64_t>{shape.back(), 1},
        memory,
        IntegerSetAttr::get(
            scheduler::buildIntegerSetFromSizes(b.getContext(), shape)));
  };

  builder.setInsertionPointToStart(&program.getBody().front());
  Value input, output;
  if (plan.cores == 1) {
    input = external(builder, plan.input);
    output = external(builder, plan.output);
  }
  auto zero = arith::ConstantIndexOp::create(builder, loc, 0);
  auto one = arith::ConstantIndexOp::create(builder, loc, 1);
  auto count = arith::ConstantIndexOp::create(
      builder, loc, plan.input.type.getNumElements() / (64 * plan.cores));
  auto loop = scf::ForOp::create(builder, loc, zero, count, one);
  builder.setInsertionPointToStart(loop.getBody());
  ktdf::PipelineBuilder pipeline(builder, loc);
  auto& private_builder = pipeline.getPrivateBuilder();
  bool widening = !plan.input.type.getElementType().isF32();
  auto input_type = tile_type(!widening);
  auto output_type = tile_type(widening);
  auto input_buffer = memref::AllocOp::create(private_builder, input_type);
  auto output_buffer = memref::AllocOp::create(private_builder, output_type);
  input_buffer.setAlignment(128);
  output_buffer.setAlignment(128);
  auto fifo = [&](StringAttr source, StringAttr dest, bool wide) {
    SmallVector<ktdf::FifoSlotType> slots(
        wide ? 2 : 1, ktdf::FifoSlotType::get(
                          builder.getContext(), source, dest, wide ? 32 : 64,
                          wide ? builder.getF32Type() : builder.getF16Type()));
    return SmallVector<Value>(private_builder.createFifo(slots));
  };
  auto edges = [&](ArrayRef<StringAttr> units, bool wide) {
    SmallVector<SmallVector<Value>> result;
    for (auto [source, dest] : llvm::zip(units.drop_back(), units.drop_front()))
      result.push_back(fifo(source, dest, wide));
    return result;
  };
  auto inputs = edges(route.input, !widening);
  auto outputs = edges(route.output, widening);
  SmallVector<ktdf::StageOp> stages;
  auto stage = [&](StringAttr unit, auto body) {
    stages.push_back(
        pipeline.createStage(builder.getArrayAttr({unit}), loc, body));
  };
  auto zero_expr = builder.getAffineConstantExpr(0);
  auto dma = [&](OpBuilder& b, Value global, Value scratch, bool load) {
    auto type = cast<MemRefType>(scratch.getType());
    auto packets = type.getDimSize(0);
    auto map = AffineMap::get(
        1, 0, {b.getAffineDimExpr(0) * packets, zero_expr}, b.getContext());
    auto scratch_map =
        AffineMap::get(0, 0, {zero_expr, zero_expr}, b.getContext());
    SmallVector<Value> global_indices{loop.getInductionVar()};
    auto transfer = ktdf::DataTransferOp::create(
        b, loc, load ? global : scratch, load ? map : scratch_map,
        load ? ValueRange(global_indices) : ValueRange{}, type.getShape(),
        load ? scratch : global, load ? scratch_map : map,
        load ? ValueRange{} : ValueRange(global_indices), type.getShape());
    transfer->setAttr("dataflow_scheduler.throttle",
                      b.getI64IntegerAttr(type.getDimSize(1)));
  };
  auto packets = [&](OpBuilder& b, Value scratch, ArrayRef<Value> slots,
                     bool load) {
    int64_t lanes = cast<MemRefType>(scratch.getType()).getDimSize(1);
    for (auto [part, slot] : llvm::enumerate(slots)) {
      auto map = AffineMap::get(
          0, 0, {b.getAffineConstantExpr(part), zero_expr}, b.getContext());
      SmallVector<int64_t> memory_size{1, lanes}, fifo_size{lanes};
      ktdf::DataTransferOp::create(
          b, loc, load ? scratch : slot, load ? map : AffineMap{}, ValueRange{},
          load ? memory_size : fifo_size, load ? slot : scratch,
          load ? AffineMap{} : map, ValueRange{},
          load ? fifo_size : memory_size);
    }
  };
  stage(route.fetch, [&](OpBuilder& b, Location) {
    dma(b, plan.cores == 1 ? input : external(b, plan.input), input_buffer,
        true);
  });
  stage(route.input.front(), [&](OpBuilder& b, Location) {
    packets(b, input_buffer, inputs.front(), true);
  });
  auto relays = [&](ArrayRef<StringAttr> units,
                    ArrayRef<SmallVector<Value>> slots) {
    for (size_t i = 1; i + 1 < units.size(); ++i) {
      stage(units[i], [&](OpBuilder& b, Location) {
        for (auto [source, dest] : llvm::zip(slots[i - 1], slots[i])) {
          auto type = cast<ktdf::FifoSlotType>(source.getType());
          auto value = ktdf::ReadFromFifoOp::create(
              b, loc,
              RankedTensorType::get({type.getNumElements()},
                                    type.getElementType()),
              source);
          ktdf::WriteToFifoOp::create(b, loc, value, dest);
        }
      });
    }
  };
  relays(route.input, inputs);
  stage(route.input.back(), [&](OpBuilder& b, Location) {
    ktdf_lowering::NativeCastOp::create(
        b, loc, inputs.back(), outputs.front(),
        b.getStringAttr(widening ? "dl16tofp32" : "fp32todl16"));
  });
  relays(route.output, outputs);
  stage(route.output.back(), [&](OpBuilder& b, Location) {
    packets(b, output_buffer, outputs.back(), false);
  });
  stage(route.drain, [&](OpBuilder& b, Location) {
    dma(b, plan.cores == 1 ? output : external(b, plan.output), output_buffer,
        false);
  });
  for (auto [producer, consumer] :
       llvm::zip(ArrayRef(stages).drop_back(), ArrayRef(stages).drop_front()))
    pipeline.addDependency(producer, consumer);
  auto built = pipeline.build();
  for (auto ordered : stages) {
    ordered->moveBefore(built.getBody(), built.getBody()->end());
  }
  builder.setInsertionPointAfter(loop);
  func::ReturnOp::create(builder, loc);
}

struct NumericCastPipelinePass
    : scheduler::impl::NumericCastPipelinePassBase<NumericCastPipelinePass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto plan = matchCast(module);
    if (failed(plan)) return signalPassFailure();
    const auto& default_device = getAnalysis<ktdf_arch::DefaultDevice>();
    if (!default_device) return signalPassFailure();
    auto& device = default_device.getRef();
    auto route = findCastRoute(*plan, device);
    if (failed(route)) return signalPassFailure();
    OpBuilder builder(module.getContext());
    builder.setInsertionPoint(plan->function);
    auto declarations = ModuleOp::create(builder, module.getLoc());
    auto implementation = ModuleOp::create(builder, module.getLoc());
    implementation.setName("local_schedule_0");
    auto function = plan->function;
    function->moveBefore(declarations.getBody(), declarations.getBody()->end());
    function.getBody().front().clear();
    builder.setInsertionPointToEnd(&function.getBody().front());
    func::CallOp::create(builder, function.getLoc(), "local_schedule_0",
                         TypeRange{}, function.getArguments());
    func::ReturnOp::create(builder, function.getLoc());
    builder.setInsertionPointToEnd(declarations.getBody());
    auto forward =
        func::FuncOp::create(builder, function.getLoc(), "local_schedule_0",
                             function.getFunctionType());
    forward.setPrivate();
    builder.setInsertionPointToEnd(implementation.getBody());
    auto program =
        func::FuncOp::create(builder, function.getLoc(), "local_schedule_0",
                             function.getFunctionType());
    program->setAttr("grid", function->getAttr("grid"));
    program.addEntryBlock();
    emitPipeline(program, *plan, *route);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createNumericCastPipelinePass() {
  return std::make_unique<NumericCastPipelinePass>();
}
