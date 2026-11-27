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

#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/DistributedMemory.h"

#include <limits>

#include "dataflow-scheduler/Analysis/MemoryTrackerAnalysis.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

namespace scheduler {
#define GEN_PASS_DEF_KTIRDISTRIBUTEDMEMORYPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

struct Partition {
  ktdp::ConstructMemoryViewOp view;
  SmallVector<int64_t> origin;
  SmallVector<int64_t> shape;
  int64_t owner;
};

struct DistributedView {
  ktdp::ConstructDistributedMemoryViewOp op;
  SmallVector<Partition> partitions;
};

presburger::IntegerPolyhedron makeBox(ArrayRef<int64_t> origin,
                                      ArrayRef<int64_t> shape) {
  presburger::IntegerPolyhedron box(
      presburger::PresburgerSpace::getSetSpace(shape.size()));
  for (auto [dim, size] : llvm::enumerate(shape)) {
    box.addBound(presburger::BoundType::LB, dim, origin[dim]);
    box.addBound(presburger::BoundType::UB, dim, origin[dim] + size - 1);
  }
  return box;
}

/// A bounding box alone would silently accept triangles and strided sets.
/// Compare the integer sets, including local variables introduced by mod/div.
FailureOr<Partition> analyzePartition(ktdp::ConstructMemoryViewOp view,
                                      MemRefType global_type) {
  auto fail = [&](StringRef message) -> FailureOr<Partition> {
    view.emitError(message);
    return failure();
  };
  auto type = view.getType();
  const unsigned rank = global_type.getRank();
  if (!type.hasStaticShape() ||
      llvm::any_of(type.getShape(), [](int64_t size) { return size <= 0; }) ||
      type.getShape() != view.getStaticSizes())
    return fail(
        "distributed partition requires positive static sizes matching "
        "its memref shape");
  if (llvm::any_of(view.getStaticStrides(),
                   [](int64_t stride) { return stride <= 0; }))
    return fail("distributed partition requires positive static strides");

  auto space = view.getMemorySpace();
  int64_t owner = space.getCtId();
  if (space.getKind() == ktdp::MemorySpaceKind::ct_local && owner < 0)
    return fail("ct_local distributed partition requires an explicit ct_id");

  IntegerSet set = view.getCoordinateSet().getValue();
  if (set.getNumDims() != rank || set.getNumSymbols())
    return fail(
        "distributed partition requires a static coordinate_set with "
        "one dimension per memref dimension and no symbols");
  affine::FlatAffineValueConstraints constraints(set);
  Partition partition{view, {}, {}, owner};
  for (unsigned dim = 0; dim < rank; ++dim) {
    auto lower = constraints.getConstantBound64(presburger::BoundType::LB, dim);
    auto upper = constraints.getConstantBound64(presburger::BoundType::UB, dim);
    if (!lower || !upper || *lower > *upper)
      return fail(
          "distributed partition requires a nonempty bounded "
          "rectangular coordinate_set");
    if (*lower < 0 || *upper >= global_type.getDimSize(dim))
      return fail(
          "distributed partition coordinate_set is outside the "
          "distributed view shape");
    partition.origin.push_back(*lower);
    partition.shape.push_back(*upper - *lower + 1);
  }
  if (!presburger::PresburgerSet(constraints)
           .isEqual(presburger::PresburgerSet(
               makeBox(partition.origin, partition.shape))))
    return fail("distributed partition coordinate_set must be rectangular");
  if (ArrayRef<int64_t>(partition.shape) != type.getShape())
    return fail(
        "distributed partition extents must match its local memref "
        "shape; coordinate_set lower bounds are the local origin");
  return partition;
}

FailureOr<DistributedView> analyzeView(
    ktdp::ConstructDistributedMemoryViewOp view) {
  auto fail = [&](StringRef message) -> FailureOr<DistributedView> {
    view.emitError(message);
    return failure();
  };
  auto type = cast<MemRefType>(view.getResult().getType());
  if (type.getRank() == 0 || !type.hasStaticShape() ||
      llvm::any_of(type.getShape(), [](int64_t size) { return size <= 0; }))
    return fail("distributed view requires a positive static shape");
  if (view.getMemrefs().empty())
    return fail("distributed view requires at least one partition");

  DistributedView result{view, {}};
  auto covered = presburger::PresburgerSet::getEmpty(
      presburger::PresburgerSpace::getSetSpace(type.getRank()));
  for (Value value : view.getMemrefs()) {
    auto source = value.getDefiningOp<ktdp::ConstructMemoryViewOp>();
    if (!source)
      return fail(
          "distributed partitions must be defined by "
          "ktdp.construct_memory_view");
    auto partition = analyzePartition(source, type);
    if (failed(partition)) return failure();
    auto box =
        presburger::PresburgerSet(makeBox(partition->origin, partition->shape));
    if (!covered.intersect(box).isIntegerEmpty())
      return fail("distributed partition coordinate_sets must not overlap");
    covered.unionInPlace(box);
    result.partitions.push_back(std::move(*partition));
  }
  // Dense staging represents every coordinate in the declared memref. Reject
  // holes rather than introducing reads of unspecified storage there.
  if (!covered.isEqual(presburger::PresburgerSet(
          makeBox(SmallVector<int64_t>(type.getRank(), 0), type.getShape()))))
    return fail(
        "distributed lowering currently requires partitions to cover "
        "the entire distributed view shape");

  for (Operation* user : view.getResult().getUsers()) {
    auto tile = dyn_cast<ktdp::ConstructAccessTilesOp>(user);
    if (!tile)
      return fail(
          "distributed view currently supports only direct access "
          "tiles consumed by ktdp.load or ktdp.store");
    for (Operation* access : tile.getResult().getUsers()) {
      if (!isa<ktdp::LoadOp, ktdp::StoreOp>(access))
        return fail(
            "distributed access tile must be consumed by ktdp.load "
            "or ktdp.store");
      // Compute-group extraction groups uses of the same tile and hoists its
      // loads. Diagnose a read-after-write before that information is lost.
      if (isa<ktdp::LoadOp>(access))
        for (Operation* other : tile.getResult().getUsers())
          if (isa<ktdp::StoreOp>(other) &&
              other->getBlock() == access->getBlock() &&
              other->isBeforeInBlock(access)) {
            access->emitError(
                "distributed loads must precede stores within a compute "
                "group; use separate access tiles for a later load");
            return failure();
          }
    }
  }
  return result;
}

IntegerSet localDomain(ArrayRef<int64_t> shape, MLIRContext* context) {
  SmallVector<AffineExpr> constraints;
  for (auto [dim, size] : llvm::enumerate(shape)) {
    auto index = getAffineDimExpr(dim, context);
    constraints.push_back(index);
    constraints.push_back(size - 1 - index);
  }
  return IntegerSet::get(shape.size(), 0, constraints,
                         SmallVector<bool>(constraints.size(), false));
}

/// Addresses may be runtime function arguments, but must denote the same
/// owner-local allocation regardless of which core accesses the logical view.
bool hasStableAddress(Value value, func::FuncOp func,
                      DenseSet<Value>& visited) {
  if (!visited.insert(value).second) return true;
  if (auto arg = dyn_cast<BlockArgument>(value))
    return arg.getOwner() == &func.getBody().front();
  auto* op = value.getDefiningOp();
  return op && !isa<ktdp::GetComputeTileIdOp>(op) && op->getNumRegions() == 0 &&
         isMemoryEffectFree(op) &&
         llvm::all_of(op->getOperands(), [&](Value operand) {
           return hasStableAddress(operand, func, visited);
         });
}

Value cloneDependency(Value value, IRMapping& mapping, OpBuilder& builder) {
  if (Value mapped = mapping.lookupOrNull(value)) return mapped;
  auto* op = value.getDefiningOp();
  assert(op && "function arguments must already be mapped");
  for (Value operand : op->getOperands())
    cloneDependency(operand, mapping, builder);
  builder.clone(*op, mapping);
  return mapping.lookup(value);
}

Value createView(OpBuilder& builder, Location loc, Value address,
                 ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
                 Type element, ktdp::MemorySpaceKind kind) {
  auto space = ktdp::MemorySpaceAttr::get(builder.getContext(), kind, -1);
  return ktdp::ConstructMemoryViewOp::create(
      builder, loc, MemRefType::get(shape, element), address, ValueRange{},
      ValueRange{}, builder.getDenseI64ArrayAttr(shape),
      builder.getDenseI64ArrayAttr(strides), space,
      IntegerSetAttr::get(localDomain(shape, builder.getContext())));
}

Value createTile(OpBuilder& builder, Location loc, Value memory,
                 ArrayRef<int64_t> origin, ArrayRef<int64_t> shape) {
  SmallVector<Value> offsets;
  for (int64_t offset : origin)
    offsets.push_back(arith::ConstantIndexOp::create(builder, loc, offset));
  auto identity = builder.getMultiDimIdentityMap(shape.size());
  return ktdp::ConstructAccessTilesOp::create(
      builder, loc, ktdp::AccessTileType::get(shape, builder.getIndexType()),
      memory, identity, offsets, localDomain(shape, builder.getContext()),
      identity);
}

SmallVector<int64_t> denseStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int dim = static_cast<int>(shape.size()) - 2; dim >= 0; --dim)
    strides[dim] = strides[dim + 1] * shape[dim + 1];
  return strides;
}

/// Fold an index expression for one logical grid coordinate without changing
/// the compute function. Runtime arguments deliberately remain unknown.
Attribute evaluateIndex(Value value, ArrayRef<int64_t> coordinate,
                        DenseMap<Value, Attribute>& constants) {
  if (auto found = constants.find(value); found != constants.end())
    return found->second;
  auto* op = value.getDefiningOp();
  if (!op) return {};
  if (auto id = dyn_cast<ktdp::GetComputeTileIdOp>(op)) {
    if (id->getNumResults() != coordinate.size()) return {};
    for (auto [result, index] : llvm::zip(id->getResults(), coordinate))
      constants[result] = IntegerAttr::get(result.getType(), index);
    return constants.lookup(value);
  }
  if (op->getNumRegions() || !isMemoryEffectFree(op)) return {};
  SmallVector<Attribute> operands;
  for (Value operand : op->getOperands()) {
    auto constant = evaluateIndex(operand, coordinate, constants);
    if (!constant) return {};
    operands.push_back(constant);
  }
  // A fold hook may modify its operation in place.
  OwningOpRef<Operation*> copy(op->clone());
  SmallVector<OpFoldResult> folded;
  if (failed(copy->fold(operands, folded)) ||
      folded.size() != op->getNumResults())
    return {};
  for (auto [result, constant] : llvm::zip(op->getResults(), folded)) {
    if (auto attr = dyn_cast<Attribute>(constant))
      constants[result] = attr;
    else if (auto operand = dyn_cast<Value>(constant);
             operand && operand.getDefiningOp() != copy.get())
      constants[result] = evaluateIndex(operand, coordinate, constants);
  }
  return constants.lookup(value);
}

FailureOr<SmallVector<int64_t>> getAccessOrigin(
    ktdp::ConstructAccessTilesOp tile, ArrayRef<int64_t> coordinate,
    DenseMap<Value, Attribute>& constants) {
  auto shape = tile.getType().getShape();
  auto set = tile.getAccessTileSet().getValue();
  if (!tile.getAccessTileOrder().isIdentity() || set.getNumSymbols() ||
      shape.size() != set.getNumDims() ||
      llvm::any_of(shape, [](int64_t size) { return size <= 0; }) ||
      !presburger::PresburgerSet(affine::FlatAffineValueConstraints(set))
           .isEqual(presburger::PresburgerSet(
               makeBox(SmallVector<int64_t>(shape.size(), 0), shape))))
    return failure();
  SmallVector<Attribute> operands, offsets;
  for (Value index : tile.getIndices()) {
    auto constant = evaluateIndex(index, coordinate, constants);
    if (!constant) return failure();
    operands.push_back(constant);
  }
  if (failed(tile.getBaseMap().constantFold(operands, offsets)))
    return failure();
  SmallVector<int64_t> origin;
  for (Attribute offset : offsets) {
    auto integer = dyn_cast<IntegerAttr>(offset);
    if (!integer) return failure();
    origin.push_back(integer.getInt());
  }
  return origin;
}

struct CoreTransfer {
  StringAttr sender;
  StringAttr receiver;
  scheduler::ResourceType memory;
  int64_t bytes = 128;
};

Value createUnit(OpBuilder& builder, Location loc, StringAttr kind,
                 int64_t core) {
  auto name = kind.getValue().lower();
  auto unit = dataflow::GetUnitOp::create(
      builder, loc, TypeRange{builder.getIndexType()},
      "C" + std::to_string(core) + "-" + name, name);
  unit->setAttr("core", builder.getI32IntegerAttr(core));
  unit->setAttr("corelet", builder.getI32IntegerAttr(0));
  return unit.getUnit();
}

/// The sender reads its own memory; the receiver stores into its own memory.
/// Peers are selected on send/receive, never on a remote memory view. The
/// architecture explicitly opts in to this transport and sets its granularity.
void createExchange(OpBuilder& builder, Location loc, Value source,
                    Value destination, ArrayRef<int64_t> shape,
                    ArrayRef<int64_t> source_origin,
                    ArrayRef<int64_t> destination_origin, bool gather,
                    int64_t owner, int64_t consumer,
                    const CoreTransfer& transfer) {
  auto type = cast<MemRefType>(source.getType());
  int64_t lanes =
      transfer.bytes / *scheduler::tryGetSizeInBytes(type.getElementType());
  auto vector_type = VectorType::get({lanes}, type.getElementType());
  auto cast_space = [&](Value value) -> Value {
    auto original = cast<MemRefType>(value.getType());
    auto mapped =
        MemRefType::get(original.getShape(), original.getElementType(),
                        original.getLayout(), transfer.memory);
    return memref::MemorySpaceCastOp::create(builder, loc, mapped, value);
  };
  source = cast_space(source);
  destination = cast_space(destination);
  Value sender =
      createUnit(builder, loc, transfer.sender, gather ? owner : consumer);
  Value receiver =
      createUnit(builder, loc, transfer.receiver, gather ? consumer : owner);
  SmallVector<int64_t> packet_shape(shape.size(), 1);
  packet_shape.back() = lanes;
  auto set = localDomain(packet_shape, builder.getContext());
  auto identity = builder.getMultiDimIdentityMap(shape.size());
  auto address_map = [&](ArrayRef<int64_t> origin) {
    SmallVector<AffineExpr> indices;
    for (unsigned dim = 0; dim < shape.size(); ++dim) {
      auto index = builder.getAffineDimExpr(dim);
      if (dim + 1 == shape.size()) index = index * lanes;
      index = index + origin[dim];
      indices.push_back(index);
    }
    return AffineMap::get(shape.size(), 0, indices, builder.getContext());
  };
  for (bool send : {true, false}) {
    auto execute = ktdf_lowering::ExecuteOnOp::create(
        builder, loc, ValueRange{send ? sender : receiver});
    auto* block = new Block;
    execute.getBody().push_back(block);
    OpBuilder body = OpBuilder::atBlockBegin(block);
    Value zero = arith::ConstantIndexOp::create(body, loc, 0);
    Value one = arith::ConstantIndexOp::create(body, loc, 1);
    SmallVector<Value> upper;
    for (auto [dim, extent] : llvm::enumerate(shape))
      upper.push_back(arith::ConstantIndexOp::create(
          body, loc, dim + 1 == shape.size() ? extent / lanes : extent));
    scf::buildLoopNest(
        body, loc, SmallVector<Value>(shape.size(), zero), upper,
        SmallVector<Value>(shape.size(), one),
        [&](OpBuilder& inner, Location loc, ValueRange indices) {
          if (send) {
            Value data = agen::VectorLoadOp::create(
                inner, loc, vector_type, source, nullptr,
                address_map(source_origin), indices, set, identity);
            dataflow::SendOp::create(inner, loc, receiver, data, nullptr,
                                     nullptr);
          } else {
            Value data = dataflow::ReceiveOp::create(inner, loc, vector_type,
                                                     sender, nullptr);
            agen::VectorStoreOp::create(inner, loc, data, destination, nullptr,
                                        address_map(destination_origin),
                                        indices, set, identity);
          }
        });
  }
}

void createLocalCopy(OpBuilder& builder, Location loc, Value source,
                     Value destination, ArrayRef<int64_t> source_origin,
                     ArrayRef<int64_t> destination_origin,
                     ArrayRef<int64_t> shape) {
  Value input_tile = createTile(builder, loc, source, source_origin, shape);
  Value output_tile =
      createTile(builder, loc, destination, destination_origin, shape);
  auto element_type = cast<MemRefType>(source.getType()).getElementType();
  auto tensor_type = RankedTensorType::get(shape, element_type);
  Value input = ktdp::LoadOp::create(builder, loc, tensor_type, input_tile);
  Value init = tensor::EmptyOp::create(builder, loc, shape, element_type);
  auto identity = builder.getMultiDimIdentityMap(shape.size());
  auto copy = linalg::GenericOp::create(
      builder, loc, TypeRange{tensor_type}, ValueRange{input}, ValueRange{init},
      SmallVector<AffineMap>{identity, identity},
      SmallVector<utils::IteratorType>(shape.size(),
                                       utils::IteratorType::parallel),
      [](OpBuilder& body, Location loc, ValueRange args) {
        Value result = spyreop::Identity::create(body, loc, args[0]);
        linalg::YieldOp::create(body, loc, result);
      });
  ktdp::StoreOp::create(builder, loc, copy.getResult(0), output_tile);
}

/// Build an ordinary KTIR copy so mapping, tiling, bufferization, path
/// expansion, address calculation and DFIR generation are shared with all
/// other transfers. Placement is applied only after the one-core program has
/// finished lowering, keeping its logical grid coordinate zero.
func::FuncOp createCopyProgram(ModuleOp top, SymbolTable& symbols,
                               func::FuncOp compute, const Partition& partition,
                               ArrayRef<int64_t> origin,
                               ArrayRef<int64_t> shape, int64_t address,
                               MemRefType snapshot_type, bool gather,
                               int64_t consumer, const CoreTransfer& transfer,
                               int64_t receive_address = -1) {
  OpBuilder builder(top.getContext());
  auto loc = compute.getLoc();
  builder.setInsertionPointToEnd(top.getBody());
  auto module = ModuleOp::create(
      builder, loc,
      compute.getName().str() + (gather ? "_gather" : "_scatter"));
  symbols.insert(module);
  builder.setInsertionPointToStart(module.getBody());
  auto func = func::FuncOp::create(builder, loc, *module.getSymName(),
                                   compute.getFunctionType());
  auto source = partition.view;
  bool remote =
      source.getMemorySpace().getKind() == ktdp::MemorySpaceKind::ct_local &&
      partition.owner != consumer;
  func->setAttr("grid",
                builder.getI64ArrayAttr(
                    {remote ? std::max(partition.owner, consumer) + 1 : 1}));
  if (!remote)
    func->setAttr("dataflow_scheduler.physical_core",
                  builder.getI32IntegerAttr(consumer));
  builder.setInsertionPointToStart(func.addEntryBlock());
  IRMapping mapping;
  mapping.map(compute.getArguments(), func.getArguments());
  Value local_address = cloneDependency(source.getOffset(), mapping, builder);
  Value local = createView(
      builder, loc, local_address, partition.shape, source.getStaticStrides(),
      snapshot_type.getElementType(), source.getMemorySpace().getKind());
  SmallVector<int64_t> local_origin;
  for (auto [index, lower] : llvm::zip(origin, partition.origin))
    local_origin.push_back(index - lower);
  if (receive_address >= 0) {
    // A short store travels as a whole packet into an owner-local relay.
    // A subsequent ordinary copy publishes only its written elements.
    assert(remote && !gather);
    Value base = arith::ConstantIndexOp::create(builder, loc, receive_address);
    local = createView(builder, loc, base, snapshot_type.getShape(),
                       denseStrides(snapshot_type.getShape()),
                       snapshot_type.getElementType(),
                       ktdp::MemorySpaceKind::ct_local);
    local_origin.assign(origin.begin(), origin.end());
  }
  Value snapshot_address =
      arith::ConstantIndexOp::create(builder, loc, address);
  Value snapshot = createView(
      builder, loc, snapshot_address, snapshot_type.getShape(),
      denseStrides(snapshot_type.getShape()), snapshot_type.getElementType(),
      ktdp::MemorySpaceKind::ct_local);
  if (remote) {
    createExchange(builder, loc, gather ? local : snapshot,
                   gather ? snapshot : local, shape,
                   gather ? ArrayRef<int64_t>(local_origin) : origin,
                   gather ? origin : ArrayRef<int64_t>(local_origin), gather,
                   partition.owner, consumer, transfer);
    func::ReturnOp::create(builder, loc);
    return func;
  }
  createLocalCopy(builder, loc, gather ? local : snapshot,
                  gather ? snapshot : local,
                  gather ? ArrayRef<int64_t>(local_origin) : origin,
                  gather ? origin : ArrayRef<int64_t>(local_origin), shape);
  func::ReturnOp::create(builder, loc);
  return func;
}

void insertCall(OpBuilder& builder, func::FuncOp helper, func::CallOp anchor) {
  auto module = anchor->getParentOfType<ModuleOp>();
  auto declaration = module.lookupSymbol<func::FuncOp>(helper.getName());
  if (!declaration) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    declaration = func::FuncOp::create(
        builder, helper.getLoc(), helper.getName(), helper.getFunctionType());
    declaration.setPrivate();
  }
  func::CallOp::create(builder, anchor.getLoc(), declaration,
                       anchor.getOperands());
}

struct TransferRegion {
  unsigned partition;
  SmallVector<int64_t> origin;
  SmallVector<int64_t> shape;
};

struct ViewPlan {
  struct LocalAccess {
    int64_t address;
    SmallVector<int64_t> shape;
    SmallVector<int64_t> strides;
    DenseMap<Operation*, SmallVector<int64_t>> origins;
  };
  DistributedView view;
  bool read = false;
  bool written = false;
  SmallVector<SmallVector<TransferRegion, 0>, 0> reads;
  SmallVector<SmallVector<TransferRegion, 0>, 0> writes;
  int64_t read_address = 0;
  int64_t write_address = 0;
  int64_t relay_address = -1;
  std::optional<LocalAccess> local_read;
  std::optional<LocalAccess> local_write;
};

struct PartitionCopy {
  Partition source, destination;
  SmallVector<int64_t> source_origin, destination_origin, shape;
};

struct GroupPlan {
  func::FuncOp func;
  func::CallOp call;
  int64_t cores = 1;
  SmallVector<ViewPlan, 0> views;
  ktdp::ConstructAccessTilesOp initialization_source;
  SmallVector<PartitionCopy, 0> direct_copies;
};

SmallVector<int64_t> coreCoordinate(const GroupPlan& group, int64_t core) {
  SmallVector<int64_t> grid{1};
  if (auto attr = group.func->getAttrOfType<ArrayAttr>("grid")) {
    grid.clear();
    for (Attribute size : attr)
      grid.push_back(cast<IntegerAttr>(size).getInt());
  }
  SmallVector<int64_t> coordinate(grid.size());
  for (int dim = static_cast<int>(grid.size()) - 1; dim >= 0; --dim) {
    coordinate[dim] = core % grid[dim];
    core /= grid[dim];
  }
  return coordinate;
}

/// A core can use its owner-local view without a staging copy when every
/// access fits that partition and has the same local coordinates on all
/// cores. Keep the computation's logical grid and global operands unchanged.
std::optional<ViewPlan::LocalAccess> localAccess(const GroupPlan& group,
                                                 const ViewPlan& plan,
                                                 bool read) {
  std::optional<ViewPlan::LocalAccess> result;
  for (int64_t core = 0; core < group.cores; ++core) {
    DenseMap<Value, Attribute> constants;
    auto coordinate = coreCoordinate(group, core);
    for (Operation* user : plan.view.op->getResult(0).getUsers()) {
      auto tile = cast<ktdp::ConstructAccessTilesOp>(user);
      if (llvm::none_of(tile.getResult().getUsers(), [&](Operation* access) {
            return read ? isa<ktdp::LoadOp>(access)
                        : isa<ktdp::StoreOp>(access);
          }))
        continue;
      auto origin = getAccessOrigin(tile, coordinate, constants);
      if (failed(origin)) return std::nullopt;
      const Partition* owner = nullptr;
      for (const auto& partition : plan.view.partitions) {
        auto view = partition.view;
        if (partition.owner != core ||
            view.getMemorySpace().getKind() != ktdp::MemorySpaceKind::ct_local)
          continue;
        bool contains = true;
        for (auto [index, size, lower, extent] :
             llvm::zip(*origin, tile.getType().getShape(), partition.origin,
                       partition.shape))
          contains &= index >= lower && index + size <= lower + extent;
        if (contains) owner = &partition;
      }
      if (!owner) return std::nullopt;
      auto view = owner->view;
      auto base = getConstantIntValue(view.getOffset());
      if (!base) return std::nullopt;
      SmallVector<int64_t> strides(view.getStaticStrides());
      if (!result) {
        result.emplace();
        result->address = *base;
        result->shape = owner->shape;
        result->strides = strides;
      }
      if (result->address != *base || result->shape != owner->shape ||
          result->strides != strides)
        return std::nullopt;
      SmallVector<int64_t> local;
      for (auto [index, lower] : llvm::zip(*origin, owner->origin))
        local.push_back(index - lower);
      auto [it, inserted] = result->origins.try_emplace(tile, local);
      if (!inserted && it->second != local) return std::nullopt;
    }
  }
  return result;
}

void planLocalAccesses(GroupPlan& group) {
  bool global_inputs = true, staged_outputs = true;
  group.func.walk([&](ktdp::LoadOp load) {
    auto tile =
        load.getAccessTile().getDefiningOp<ktdp::ConstructAccessTilesOp>();
    auto view =
        tile ? tile.getBase().getDefiningOp<ktdp::ConstructMemoryViewOp>()
             : nullptr;
    global_inputs &= view && view.getMemorySpace().getKind() ==
                                 ktdp::MemorySpaceKind::global;
  });
  group.func.walk([&](ktdp::StoreOp store) {
    auto tile =
        store.getAccessTile().getDefiningOp<ktdp::ConstructAccessTilesOp>();
    if (!tile) {
      staged_outputs = false;
      return;
    }
    if (tile.getBase().getDefiningOp<ktdp::ConstructDistributedMemoryViewOp>())
      return;
    auto view = tile.getBase().getDefiningOp<ktdp::ConstructMemoryViewOp>();
    staged_outputs &= view && view.getMemorySpace().getKind() ==
                                  ktdp::MemorySpaceKind::global;
  });
  for (auto& plan : group.views) {
    // In-place writes still go to a separate allocation. The program boundary
    // before scatter protects every old-value reader, including other cores.
    if (plan.read && staged_outputs)
      plan.local_read = localAccess(group, plan, true);
    // Ordinary ct_local outputs may alias the owner view. Keep their stores
    // ordered before scatter unless their nonoverlap is established.
    if (plan.written && global_inputs && staged_outputs)
      plan.local_write = localAccess(group, plan, false);
  }
}

bool overlaps(const TransferRegion& a, const TransferRegion& b) {
  for (auto [x, nx, y, ny] : llvm::zip(a.origin, a.shape, b.origin, b.shape))
    if (x + nx <= y || y + ny <= x) return false;
  return true;
}

/// Independent copies can share a program. A short remote store and its relay
/// publication retain their program boundary, as do overlapping writers.
bool canBatchTransfers(const GroupPlan& group, bool gather,
                       const CoreTransfer& transfer) {
  for (const auto& plan : group.views) {
    if (gather ? bool(plan.local_read) : bool(plan.local_write)) continue;
    const auto& accesses = gather ? plan.reads : plan.writes;
    auto type = cast<MemRefType>(plan.view.op->getResult(0).getType());
    int64_t lanes =
        transfer.bytes / *scheduler::tryGetSizeInBytes(type.getElementType());
    for (auto [core, regions] : llvm::enumerate(accesses)) {
      for (auto [i, region] : llvm::enumerate(regions)) {
        if (region.origin.back() % lanes || region.shape.back() % lanes)
          return false;
        for (size_t other = 0; other <= core; ++other) {
          if (gather && other != core) continue;
          for (auto [j, previous] : llvm::enumerate(accesses[other])) {
            if (other == core && j >= i) break;
            if ((gather || region.partition == previous.partition) &&
                overlaps(region, previous))
              return false;
          }
        }
      }
    }
  }
  return true;
}

/// Recognize an unobservable initialization copy, not an arbitrary compute.
/// Requiring a global source also rules out an intervening LX alias reader.
ktdp::ConstructAccessTilesOp globalCopySource(GroupPlan& group) {
  if (group.views.size() != 1 || group.views.front().read) return {};
  SmallVector<ktdp::LoadOp> loads;
  SmallVector<ktdp::StoreOp> stores;
  SmallVector<linalg::GenericOp> generics;
  group.func.walk([&](ktdp::LoadOp op) { loads.push_back(op); });
  group.func.walk([&](ktdp::StoreOp op) { stores.push_back(op); });
  group.func.walk([&](linalg::GenericOp op) { generics.push_back(op); });
  if (loads.size() != 1 || stores.size() != 1 || generics.size() != 1)
    return {};
  auto generic = generics.front();
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
      generic.getDpsInputs().front() != loads.front().getResult() ||
      stores.front().getDataTile() != generic.getResult(0) ||
      !llvm::all_of(generic.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); }) ||
      !llvm::all_of(generic.getIteratorTypesArray(), [](auto kind) {
        return kind == utils::IteratorType::parallel;
      }))
    return {};
  auto* body = generic.getBody();
  if (body->getOperations().size() != 2) return {};
  auto identity = dyn_cast<spyreop::Identity>(body->front());
  if (!identity || identity->getOperand(0) != body->getArgument(0) ||
      body->getTerminator()->getOperand(0) != identity->getResult(0))
    return {};
  for (auto& op : group.func.getBody().front())
    if (!isa<ktdp::LoadOp, ktdp::StoreOp, linalg::GenericOp, func::ReturnOp>(
            op) &&
        !isMemoryEffectFree(&op))
      return {};
  auto tile = loads.front()
                  .getAccessTile()
                  .getDefiningOp<ktdp::ConstructAccessTilesOp>();
  auto destination = stores.front()
                         .getAccessTile()
                         .getDefiningOp<ktdp::ConstructAccessTilesOp>();
  auto view = tile ? tile.getBase().getDefiningOp<ktdp::ConstructMemoryViewOp>()
                   : nullptr;
  DenseSet<Value> visited;
  if (!view ||
      view.getMemorySpace().getKind() != ktdp::MemorySpaceKind::global ||
      !destination ||
      destination.getBase() != group.views.front().view.op.getResult() ||
      !hasStableAddress(view.getOffset(), group.func, visited))
    return {};
  for (int64_t core = 0; core < group.cores; ++core) {
    DenseMap<Value, Attribute> constants;
    auto coordinate = coreCoordinate(group, core);
    if (failed(getAccessOrigin(tile, coordinate, constants)) ||
        failed(getAccessOrigin(destination, coordinate, constants)))
      return {};
  }
  return tile;
}

SmallVector<TransferRegion> subtractRegion(TransferRegion region,
                                           const TransferRegion& removed) {
  if (!overlaps(region, removed)) return {region};
  SmallVector<TransferRegion> remaining;
  for (unsigned dim = 0; dim < region.shape.size(); ++dim) {
    int64_t lower = std::max(region.origin[dim], removed.origin[dim]);
    int64_t upper = std::min(region.origin[dim] + region.shape[dim],
                             removed.origin[dim] + removed.shape[dim]);
    if (region.origin[dim] < lower) {
      auto piece = region;
      piece.shape[dim] = lower - region.origin[dim];
      remaining.push_back(std::move(piece));
      region.shape[dim] -= lower - region.origin[dim];
      region.origin[dim] = lower;
    }
    if (upper < region.origin[dim] + region.shape[dim]) {
      auto piece = region;
      piece.origin[dim] = upper;
      piece.shape[dim] = region.origin[dim] + region.shape[dim] - upper;
      remaining.push_back(std::move(piece));
      region.shape[dim] = upper - region.origin[dim];
    }
  }
  return remaining;
}

void trimOverwrittenInitializations(MutableArrayRef<GroupPlan> groups) {
  for (auto& group : groups) {
    auto source = globalCopySource(group);
    if (!source || !group.views.front().local_write) continue;
    Operation* next = group.call->getNextNode();
    while (next && !isa<func::CallOp>(next) && isMemoryEffectFree(next))
      next = next->getNextNode();
    auto found = llvm::find_if(
        groups, [&](GroupPlan& candidate) { return candidate.call == next; });
    if (found == groups.end() || !globalCopySource(*found)) continue;
    auto& initial = group.views.front();
    auto& update = found->views.front();
    auto trimmed = initial.writes;
    bool changed = false;
    for (auto& regions : trimmed) {
      SmallVector<TransferRegion, 0> live;
      for (auto region : regions) {
        const auto& partition = initial.view.partitions[region.partition];
        auto view = partition.view;
        SmallVector<TransferRegion> pieces{region};
        for (const auto& writes : update.writes) {
          for (const auto& write : writes) {
            const auto& other = update.view.partitions[write.partition];
            auto other_view = other.view;
            if (partition.owner != other.owner ||
                partition.shape != other.shape ||
                view.getMemorySpace() != other_view.getMemorySpace() ||
                view.getType().getElementType() !=
                    other_view.getType().getElementType() ||
                view.getStaticStrides() != other_view.getStaticStrides() ||
                getConstantIntValue(view.getOffset()) !=
                    getConstantIntValue(other_view.getOffset()))
              continue;
            auto removed = write;
            for (unsigned dim = 0; dim < removed.origin.size(); ++dim)
              removed.origin[dim] += partition.origin[dim] - other.origin[dim];
            SmallVector<TransferRegion> rest;
            for (auto piece : pieces) {
              changed |= overlaps(piece, removed);
              llvm::append_range(rest, subtractRegion(piece, removed));
            }
            pieces = std::move(rest);
          }
        }
        llvm::append_range(live, pieces);
      }
      regions = std::move(live);
    }
    // A single local copy per core needs no new synchronization within the
    // batch. More fragmented initialization retains the original safe path.
    if (changed && llvm::all_of(trimmed, [](const auto& regions) {
          return regions.size() <= 1;
        })) {
      initial.writes = std::move(trimmed);
      group.initialization_source = source;
    }
  }
}

/// Distinct logical partitions must have distinct physical storage. Reject
/// unknown aliasing rather than overwrite a different partition's elements.
bool disjointStorage(const Partition& lhs, const Partition& rhs) {
  auto lhs_view = lhs.view, rhs_view = rhs.view;
  auto left = lhs_view.getMemorySpace();
  auto right = rhs_view.getMemorySpace();
  if (left.getKind() != right.getKind() ||
      (left.getKind() == ktdp::MemorySpaceKind::ct_local &&
       lhs.owner != rhs.owner))
    return true;
  auto interval = [](ktdp::ConstructMemoryViewOp view)
      -> std::optional<std::pair<int64_t, int64_t>> {
    auto base = getConstantIntValue(view.getOffset());
    auto bytes = scheduler::tryGetSizeInBytes(view.getType().getElementType());
    if (!base || *base < 0 || !bytes) return std::nullopt;
    int64_t begin, end = *base;
    for (auto [size, stride] :
         llvm::zip(view.getStaticSizes(), view.getStaticStrides())) {
      int64_t span;
      if (__builtin_mul_overflow(size - 1, stride, &span) ||
          __builtin_add_overflow(end, span, &end))
        return std::nullopt;
    }
    if (__builtin_add_overflow(end, int64_t{1}, &end) ||
        __builtin_mul_overflow(*base, *bytes, &begin) ||
        __builtin_mul_overflow(end, *bytes, &end))
      return std::nullopt;
    return std::pair{begin, end};
  };
  auto a = interval(lhs.view), b = interval(rhs.view);
  return a && b && (a->second <= b->first || b->second <= a->first);
}

/// A pure, out-of-place identity needs no per-core global-sized snapshot.
/// Prove the whole group before changing it: multiple identities may copy one
/// input to distinct replicas, but no destination may clobber any input.
void planDirectCopies(GroupPlan& group, const CoreTransfer& transfer) {
  SmallVector<PartitionCopy, 0> copies;
  DenseSet<Operation*> matched;
  auto find_view = [&](Value value) -> const DistributedView* {
    for (const auto& plan : group.views)
      if (plan.view.op->getResult(0) == value) return &plan.view;
    return nullptr;
  };
  for (const auto& output : group.views) {
    if (!output.written) continue;
    for (const auto& input : group.views) {
      if (!input.read) continue;
      for (const auto& to : output.view.partitions)
        for (const auto& from : input.view.partitions)
          if (!disjointStorage(from, to)) return;
    }
  }
  for (auto store : group.func.getOps<ktdp::StoreOp>()) {
    auto generic = store.getDataTile().getDefiningOp<linalg::GenericOp>();
    if (!generic || generic.getNumDpsInputs() != 1 ||
        generic.getNumDpsInits() != 1 || generic->getNumResults() != 1 ||
        !llvm::all_of(generic.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); }) ||
        !llvm::all_of(generic.getIteratorTypesArray(), [](auto kind) {
          return kind == utils::IteratorType::parallel;
        }))
      return;
    auto* body = generic.getBody();
    if (body->getOperations().size() != 2) return;
    auto identity = dyn_cast<spyreop::Identity>(body->front());
    if (!identity || identity->getOperand(0) != body->getArgument(0) ||
        body->getTerminator()->getOperand(0) != identity->getResult(0))
      return;
    auto load = generic.getDpsInputs().front().getDefiningOp<ktdp::LoadOp>();
    if (!load || load.getResult().getType() != generic.getResult(0).getType())
      return;
    auto from_tile =
        load.getAccessTile().getDefiningOp<ktdp::ConstructAccessTilesOp>();
    auto to_tile =
        store.getAccessTile().getDefiningOp<ktdp::ConstructAccessTilesOp>();
    auto* source = from_tile ? find_view(from_tile.getBase()) : nullptr;
    auto* destination = to_tile ? find_view(to_tile.getBase()) : nullptr;
    if (!source || !destination) return;
    auto shape = from_tile.getType().getShape();
    auto element =
        cast<RankedTensorType>(load.getResult().getType()).getElementType();
    auto bytes = scheduler::tryGetSizeInBytes(element);
    if (!bytes || transfer.bytes % *bytes) return;
    int64_t lanes = transfer.bytes / *bytes;
    if (shape != to_tile.getType().getShape() || shape.back() % lanes) return;
    for (int64_t core = 0; core < group.cores; ++core) {
      DenseMap<Value, Attribute> constants;
      auto coordinate = coreCoordinate(group, core);
      auto from = getAccessOrigin(from_tile, coordinate, constants);
      auto to = getAccessOrigin(to_tile, coordinate, constants);
      if (failed(from) || failed(to)) return;
      for (const auto& src : source->partitions) {
        for (const auto& dst : destination->partitions) {
          auto src_view = src.view, dst_view = dst.view;
          if (src_view.getMemorySpace().getKind() !=
                  ktdp::MemorySpaceKind::ct_local ||
              dst_view.getMemorySpace().getKind() !=
                  ktdp::MemorySpaceKind::ct_local)
            return;
          PartitionCopy copy{src, dst, {}, {}, {}};
          for (unsigned dim = 0; dim < shape.size(); ++dim) {
            int64_t lower =
                std::max({int64_t{0}, src.origin[dim] - (*from)[dim],
                          dst.origin[dim] - (*to)[dim]});
            int64_t upper = std::min(
                {shape[dim], src.origin[dim] + src.shape[dim] - (*from)[dim],
                 dst.origin[dim] + dst.shape[dim] - (*to)[dim]});
            copy.source_origin.push_back((*from)[dim] + lower -
                                         src.origin[dim]);
            copy.destination_origin.push_back((*to)[dim] + lower -
                                              dst.origin[dim]);
            copy.shape.push_back(std::max(int64_t{0}, upper - lower));
          }
          if (llvm::is_contained(copy.shape, int64_t{0})) continue;
          auto aligned = [&](const Partition& part, ArrayRef<int64_t> origin) {
            auto view = part.view;
            auto base = getConstantIntValue(view.getOffset());
            if (!base || *base % lanes || view.getStaticStrides().back() != 1 ||
                origin.back() % lanes || copy.shape.back() % lanes)
              return false;
            for (auto [extent, stride] :
                 llvm::zip(ArrayRef<int64_t>(part.shape).drop_back(),
                           view.getStaticStrides().drop_back()))
              if (extent > 1 && stride % lanes) return false;
            return true;
          };
          if (!aligned(src, copy.source_origin) ||
              !aligned(dst, copy.destination_origin))
            return;
          // Repeated whole-tensor stores on different compute cores are not
          // an independent transfer batch, even when they copy equal values.
          for (const auto& previous : copies)
            if (previous.destination.view == dst.view &&
                overlaps({0, previous.destination_origin, previous.shape},
                         {0, copy.destination_origin, copy.shape}))
              return;
          copies.push_back(std::move(copy));
        }
      }
    }
    matched.insert(store);
    matched.insert(generic);
    matched.insert(load);
  }
  if (copies.empty()) return;
  for (auto& op : group.func.getBody().front()) {
    if (matched.contains(&op)) continue;
    if (isa<ktdp::LoadOp, ktdp::StoreOp, linalg::GenericOp>(op) ||
        op.getNumRegions() || !isMemoryEffectFree(&op))
      return;
  }
  group.direct_copies = std::move(copies);
}

LogicalResult verifyGroupAccesses(const GroupPlan& group) {
  SmallVector<const Partition*> written;
  for (const auto& plan : group.views) {
    if (!plan.written) continue;
    for (const auto& partition : plan.view.partitions) {
      auto view = partition.view;
      // Positive strides alone do not imply an injective storage layout.
      SmallVector<std::pair<int64_t, int64_t>> dims;
      for (auto [size, stride] :
           llvm::zip(partition.shape, view.getStaticStrides()))
        if (size > 1) dims.emplace_back(stride, size);
      llvm::sort(dims);
      int64_t span = 1;
      for (auto [stride, size] : dims) {
        int64_t increment;
        if (stride < span ||
            __builtin_mul_overflow(size - 1, stride, &increment) ||
            __builtin_add_overflow(span, increment, &span))
          return view.emitError(
              "distributed stores require nonoverlapping bounded partition "
              "storage strides");
      }
      for (const Partition* previous : written)
        if (!disjointStorage(*previous, partition))
          return view.emitError(
              "distributed stores require provably disjoint physical "
              "partition storage within a compute group");
      written.push_back(&partition);
    }
  }
  bool stored = false;
  auto func = group.func;
  auto result = func.walk([&](Operation* op) {
    Value tile;
    if (auto load = dyn_cast<ktdp::LoadOp>(op)) tile = load.getAccessTile();
    if (auto store = dyn_cast<ktdp::StoreOp>(op)) {
      stored = true;
      tile = store.getAccessTile();
    }
    auto construct =
        tile ? tile.getDefiningOp<ktdp::ConstructAccessTilesOp>() : nullptr;
    if (!construct ||
        !construct.getBase()
             .getDefiningOp<ktdp::ConstructDistributedMemoryViewOp>())
      return WalkResult::advance();
    if (op->getBlock() != &func.getBody().front()) {
      op->emitError(
          "distributed accesses must be in the extracted compute "
          "group's entry block");
      return WalkResult::interrupt();
    }
    if (isa<ktdp::LoadOp>(op) && stored) {
      op->emitError(
          "distributed loads must precede stores within a compute "
          "group; use separate access tiles for a later load");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

/// Intersect each core's actual memory accesses with the fixed partitions.
/// In particular, never enlarge a write footprint to its partition's box.
LogicalResult planAccesses(GroupPlan& group) {
  SmallVector<int64_t> grid{1};
  if (auto attr = group.func->getAttrOfType<ArrayAttr>("grid")) {
    grid.clear();
    for (Attribute size : attr)
      grid.push_back(cast<IntegerAttr>(size).getInt());
  }
  for (auto& plan : group.views) {
    plan.reads.resize(group.cores);
    plan.writes.resize(group.cores);
  }
  for (int64_t core = 0; core < group.cores; ++core) {
    SmallVector<int64_t> coordinate(grid.size());
    int64_t remaining = core;
    for (int dim = static_cast<int>(grid.size()) - 1; dim >= 0; --dim) {
      coordinate[dim] = remaining % grid[dim];
      remaining /= grid[dim];
    }
    DenseMap<Value, Attribute> constants;
    for (auto& plan : group.views) {
      auto type = cast<MemRefType>(plan.view.op.getResult().getType());
      for (Operation* user : plan.view.op.getResult().getUsers()) {
        auto tile = cast<ktdp::ConstructAccessTilesOp>(user);
        bool read = false, write = false;
        for (Operation* access : tile.getResult().getUsers()) {
          read |= isa<ktdp::LoadOp>(access);
          write |= isa<ktdp::StoreOp>(access);
        }
        auto origin = getAccessOrigin(tile, coordinate, constants);
        auto shape = tile.getType().getShape();
        if (failed(origin) && write)
          return tile.emitError(
              "distributed stores require a zero-based rectangular tile, "
              "identity access_tile_order and offsets constant for each core");
        if (succeeded(origin)) {
          if (origin->size() != type.getShape().size() ||
              shape.size() != type.getShape().size())
            return tile.emitError(
                "distributed access rank must match its view");
          for (auto [start, extent, bound] :
               llvm::zip(*origin, shape, type.getShape()))
            if (start < 0 || extent > bound || start > bound - extent)
              return tile.emitError()
                     << "distributed access is outside its view on core "
                     << core;
        }
        auto add = [&](SmallVectorImpl<TransferRegion>& regions,
                       const TransferRegion& region) {
          if (llvm::none_of(regions, [&](const TransferRegion& previous) {
                return previous.partition == region.partition &&
                       previous.origin == region.origin &&
                       previous.shape == region.shape;
              }))
            regions.push_back(region);
        };
        for (auto [index, partition] : llvm::enumerate(plan.view.partitions)) {
          TransferRegion region{static_cast<unsigned>(index), partition.origin,
                                partition.shape};
          if (succeeded(origin)) {
            for (unsigned dim = 0; dim < shape.size(); ++dim) {
              int64_t end =
                  std::min((*origin)[dim] + shape[dim],
                           partition.origin[dim] + partition.shape[dim]);
              region.origin[dim] =
                  std::max((*origin)[dim], partition.origin[dim]);
              region.shape[dim] =
                  std::max(int64_t{0}, end - region.origin[dim]);
            }
          }
          if (llvm::is_contained(region.shape, int64_t{0})) continue;
          // Unknown read indices retain the existing whole-view snapshot.
          if (read) add(plan.reads[core], region);
          if (write) add(plan.writes[core], region);
        }
      }
    }
  }
  return success();
}

FailureOr<SmallVector<GroupPlan>> analyzeGroups(ModuleOp top) {
  SmallVector<GroupPlan> groups;
  DenseMap<Operation*, unsigned> indices;
  auto result = top.walk([&](ktdp::ConstructDistributedMemoryViewOp op) {
    if (op.getResult().use_empty()) return WalkResult::advance();
    auto view = analyzeView(op);
    if (failed(view)) return WalkResult::interrupt();
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func || !llvm::hasSingleElement(func.getBody())) {
      op.emitError(
          "distributed lowering requires a single-block compute "
          "function");
      return WalkResult::interrupt();
    }
    auto [it, inserted] = indices.try_emplace(func, groups.size());
    if (inserted) {
      GroupPlan group{func, {}, 1, {}};
      unsigned calls = 0;
      top.walk([&](func::CallOp call) {
        if (call.getCallee() == func.getName()) {
          group.call = call;
          ++calls;
        }
      });
      if (calls != 1) {
        op.emitError(
            "distributed lowering requires an extracted compute "
            "group with exactly one orchestrator call");
        return WalkResult::interrupt();
      }
      if (auto grid = func->getAttrOfType<ArrayAttr>("grid")) {
        for (Attribute dim : grid) {
          auto size = dyn_cast<IntegerAttr>(dim);
          if (!size || size.getInt() <= 0 ||
              __builtin_mul_overflow(group.cores, size.getInt(),
                                     &group.cores) ||
              group.cores > std::numeric_limits<int32_t>::max()) {
            func.emitError(
                "distributed lowering requires a positive static grid");
            return WalkResult::interrupt();
          }
        }
      }
      groups.push_back(std::move(group));
    }
    ViewPlan plan{std::move(*view)};
    for (auto& partition : plan.view.partitions) {
      DenseSet<Value> visited;
      if (!hasStableAddress(partition.view.getOffset(), func, visited)) {
        partition.view.emitError(
            "distributed partition address must be "
            "independent of the executing core");
        return WalkResult::interrupt();
      }
    }
    for (auto* user : op.getResult().getUsers()) {
      auto tile = cast<ktdp::ConstructAccessTilesOp>(user);
      for (auto* access : tile.getResult().getUsers()) {
        plan.read |= isa<ktdp::LoadOp>(access);
        if (isa<ktdp::StoreOp>(access)) {
          plan.written = true;
        }
      }
    }
    groups[it->second].views.push_back(std::move(plan));
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return failure();
  for (auto& group : groups)
    if (failed(verifyGroupAccesses(group)) || failed(planAccesses(group)))
      return failure();
  return groups;
}

FailureOr<CoreTransfer> resolveTransfer(const ktdf_arch::Device& device,
                                        scheduler::ResourceType memory,
                                        ArrayRef<GroupPlan> groups) {
  CoreTransfer transfer;
  transfer.memory = memory;
  bool remote = false;
  for (const auto& group : groups)
    for (const auto& view : group.views)
      for (const auto& partition : view.view.partitions) {
        auto source = partition.view;
        remote |= source.getMemorySpace().getKind() ==
                      ktdp::MemorySpaceKind::ct_local &&
                  (group.cores > 1 || partition.owner != 0);
      }
  if (!remote) return transfer;
  auto capability = dyn_cast_or_null<DictionaryAttr>(
      device.getAttr("dataflow_scheduler.core_transfer"));
  if (capability) {
    transfer.sender = capability.getAs<StringAttr>("send_unit");
    transfer.receiver = capability.getAs<StringAttr>("receive_unit");
    if (auto bytes = capability.getAs<IntegerAttr>("bytes"))
      transfer.bytes = bytes.getInt();
    else
      transfer.bytes = 0;
  }
  const ktdf_arch::ResourceKinds kinds(device);
  if (!transfer.sender || !transfer.receiver ||
      transfer.sender == transfer.receiver || transfer.bytes <= 0 ||
      !llvm::isPowerOf2_64(transfer.bytes) ||
      !kinds.getInstance<ktdf_arch::ExecutionUnitOp>(
          cast<scheduler::ResourceType>(transfer.sender)) ||
      !kinds.getInstance<ktdf_arch::ExecutionUnitOp>(
          cast<scheduler::ResourceType>(transfer.receiver))) {
    device.getDeclaration().emitError(
        "remote distributed access requires dataflow_scheduler.core_transfer "
        "with send_unit, receive_unit and a positive power-of-two bytes value");
    return failure();
  }
  for (const auto& group : groups) {
    for (const auto& view : group.views) {
      auto op = view.view.op;
      auto type = cast<MemRefType>(op.getResult().getType());
      auto bytes = scheduler::tryGetSizeInBytes(type.getElementType());
      if (!bytes || transfer.bytes % *bytes) {
        op.emitError(
            "core transfer width must be divisible by the element size");
        return failure();
      }
      int64_t lanes = transfer.bytes / *bytes;
      for (const auto& partition : view.view.partitions) {
        auto source = partition.view;
        if (source.getMemorySpace().getKind() !=
                ktdp::MemorySpaceKind::ct_local ||
            (group.cores == 1 && partition.owner == 0))
          continue;
        auto base = getConstantIntValue(source.getOffset());
        bool aligned = base && *base >= 0 && *base % lanes == 0 &&
                       source.getStaticStrides().back() == 1 &&
                       partition.shape.back() % lanes == 0 &&
                       partition.origin.back() % lanes == 0 &&
                       type.getShape().back() % lanes == 0;
        for (auto [extent, stride] :
             llvm::zip(ArrayRef<int64_t>(partition.shape).drop_back(),
                       source.getStaticStrides().drop_back()))
          aligned &= extent == 1 || stride % lanes == 0;
        if (!aligned) {
          source.emitError(
              "remote distributed partition requires aligned whole transfer "
              "packets, unit innermost stride and a constant local address");
          return failure();
        }
      }
    }
  }
  return transfer;
}

/// Protect caller-owned LX across every transfer and compute program. This
/// initial allocator requires bounded local views; a runtime local pointer has
/// no capacity/alias bound from which a safe snapshot address can be selected.
LogicalResult reserveLocalViews(ModuleOp top,
                                scheduler::MemoryTrackerAnalysis& tracker,
                                scheduler::ResourceType memory) {
  size_t end = tracker.getNextAvailableAddress(memory);
  auto result = top.walk([&](ktdp::ConstructMemoryViewOp view) {
    if (view.getMemorySpace().getKind() != ktdp::MemorySpaceKind::ct_local)
      return WalkResult::advance();
    auto base = getConstantIntValue(view.getOffset());
    auto bytes = scheduler::tryGetSizeInBytes(view.getType().getElementType());
    size_t last = base && *base >= 0 ? *base : 0;
    bool invalid = !base || *base < 0 || !bytes;
    for (auto [size, stride] :
         llvm::zip(view.getStaticSizes(), view.getStaticStrides())) {
      size_t span = 0;
      if (size <= 0 || stride <= 0 ||
          __builtin_mul_overflow(size - 1, stride, &span) ||
          __builtin_add_overflow(last, span, &last))
        invalid = true;
    }
    if (invalid || __builtin_add_overflow(last, size_t{1}, &last) ||
        __builtin_mul_overflow(last, *bytes, &last)) {
      view.emitError(
          "distributed snapshots require constant, bounded ct_local "
          "view addresses, sizes and strides");
      return WalkResult::interrupt();
    }
    end = std::max(end, last);
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return failure();
  size_t next = tracker.getNextAvailableAddress(memory);
  if (end > next) {
    auto reserved = tracker.allocate(memory, end - next);
    if (!reserved) {
      top.emitError() << llvm::toString(reserved.takeError());
      return failure();
    }
  }
  return success();
}

LogicalResult allocateSnapshots(GroupPlan& group,
                                scheduler::MemoryTrackerAnalysis& tracker,
                                const CoreTransfer& transfer) {
  for (auto& plan : group.views) {
    auto op = plan.view.op;
    auto type = cast<MemRefType>(op.getResult().getType());
    auto bytes = scheduler::tryGetSizeInBytes(type.getElementType());
    int64_t size = bytes.value_or(0);
    for (int64_t dim : type.getShape())
      if (__builtin_mul_overflow(size, dim, &size)) size = 0;
    if (!size || transfer.bytes % *bytes) {
      return op.emitError(
          "distributed snapshot requires a bounded byte size "
          "and an element size dividing the transfer width");
    }
    auto allocate = [&](int64_t& address) -> LogicalResult {
      auto result = tracker.allocate(transfer.memory, size, transfer.bytes);
      if (!result) {
        op.emitError() << llvm::toString(result.takeError());
        return failure();
      }
      address = *result / *bytes;
      return success();
    };
    if (plan.read && !plan.local_read && failed(allocate(plan.read_address)))
      return failure();
    if (plan.written && !plan.local_write &&
        failed(allocate(plan.write_address)))
      return failure();
    int64_t lanes = transfer.bytes / *bytes;
    bool relay = false;
    for (auto [core, regions] : llvm::enumerate(plan.writes))
      for (const auto& region : regions) {
        const auto& partition = plan.view.partitions[region.partition];
        auto view = partition.view;
        relay |= view.getMemorySpace().getKind() ==
                     ktdp::MemorySpaceKind::ct_local &&
                 partition.owner != static_cast<int64_t>(core) &&
                 (region.origin.back() % lanes || region.shape.back() % lanes);
      }
    if (relay && failed(allocate(plan.relay_address))) return failure();
  }
  return success();
}

void replaceAccesses(func::FuncOp func, ArrayRef<ViewPlan> plans) {
  SmallVector<ktdp::ConstructDistributedMemoryViewOp> views;
  func.walk([&](ktdp::ConstructDistributedMemoryViewOp op) {
    if (!op.getResult().use_empty()) views.push_back(op);
  });
  assert(views.size() == plans.size());
  for (auto [op, plan] : llvm::zip(views, plans)) {
    OpBuilder builder(op);
    auto loc = op.getLoc();
    auto type = cast<MemRefType>(op.getResult().getType());
    auto snapshot = [&](int64_t address) {
      Value base = arith::ConstantIndexOp::create(builder, loc, address);
      return createView(builder, loc, base, type.getShape(),
                        denseStrides(type.getShape()), type.getElementType(),
                        ktdp::MemorySpaceKind::ct_local);
    };
    auto local = [&](const ViewPlan::LocalAccess& access) {
      Value base = arith::ConstantIndexOp::create(builder, loc, access.address);
      return createView(builder, loc, base, access.shape, access.strides,
                        type.getElementType(), ktdp::MemorySpaceKind::ct_local);
    };
    Value read = plan.local_read ? local(*plan.local_read)
                 : plan.read     ? snapshot(plan.read_address)
                                 : Value{};
    Value write = plan.local_write ? local(*plan.local_write)
                  : plan.written   ? snapshot(plan.write_address)
                                   : Value{};
    for (auto* user : llvm::make_early_inc_range(op.getResult().getUsers())) {
      auto tile = cast<ktdp::ConstructAccessTilesOp>(user);
      Value read_tile, write_tile;
      builder.setInsertionPoint(tile);
      for (auto* access :
           llvm::make_early_inc_range(tile.getResult().getUsers())) {
        bool load = isa<ktdp::LoadOp>(access);
        Value& replacement = load ? read_tile : write_tile;
        if (!replacement) {
          const auto& direct = load ? plan.local_read : plan.local_write;
          if (direct) {
            replacement = createTile(builder, loc, load ? read : write,
                                     direct->origins.lookup(tile),
                                     tile.getType().getShape());
          } else {
            IRMapping mapping;
            mapping.map(op.getResult(), load ? read : write);
            replacement = builder.clone(*tile, mapping)->getResult(0);
          }
        }
        access->setOperand(load ? 0 : 1, replacement);
      }
      tile.erase();
    }
    op.erase();
  }
}

void materializeInitialization(ModuleOp top, SymbolTable& symbols,
                               GroupPlan& group) {
  auto& plan = group.views.front();
  auto input = group.initialization_source;
  auto source = input.getBase().getDefiningOp<ktdp::ConstructMemoryViewOp>();
  ktdp::ConstructAccessTilesOp output;
  for (Operation* user : plan.view.op.getResult().getUsers())
    output = cast<ktdp::ConstructAccessTilesOp>(user);
  auto loc = group.func.getLoc();
  for (auto [core, regions] : llvm::enumerate(plan.writes)) {
    DenseMap<Value, Attribute> constants;
    auto coordinate = coreCoordinate(group, core);
    auto input_origin = *getAccessOrigin(input, coordinate, constants);
    auto output_origin = *getAccessOrigin(output, coordinate, constants);
    for (const auto& region : regions) {
      const auto& partition = plan.view.partitions[region.partition];
      auto destination = partition.view;
      OpBuilder builder(top.getContext());
      builder.setInsertionPointToEnd(top.getBody());
      auto module = ModuleOp::create(
          builder, loc, group.func.getName().str() + "_initialize");
      symbols.insert(module);
      builder.setInsertionPointToStart(module.getBody());
      auto func = func::FuncOp::create(builder, loc, *module.getSymName(),
                                       group.func.getFunctionType());
      func->setAttr("grid", builder.getI64ArrayAttr({1}));
      func->setAttr("dataflow_scheduler.physical_core",
                    builder.getI32IntegerAttr(core));
      func->setAttr(
          "dataflow_scheduler.distributed_transfer_batch",
          builder.getStringAttr(group.func.getName().str() + "_initialize"));
      builder.setInsertionPointToStart(func.addEntryBlock());
      IRMapping mapping;
      mapping.map(group.func.getArguments(), func.getArguments());
      auto view = [&](ktdp::ConstructMemoryViewOp original) {
        Value base = cloneDependency(original.getOffset(), mapping, builder);
        return createView(builder, loc, base, original.getStaticSizes(),
                          original.getStaticStrides(),
                          original.getType().getElementType(),
                          original.getMemorySpace().getKind());
      };
      SmallVector<int64_t> from, to;
      for (unsigned dim = 0; dim < region.origin.size(); ++dim) {
        from.push_back(region.origin[dim] - output_origin[dim] +
                       input_origin[dim]);
        to.push_back(region.origin[dim] - partition.origin[dim]);
      }
      Value src = createTile(builder, loc, view(source), from, region.shape);
      Value dst = createTile(builder, loc, view(destination), to, region.shape);
      auto type = RankedTensorType::get(region.shape,
                                        source.getType().getElementType());
      Value loaded = ktdp::LoadOp::create(builder, loc, type, src);
      Value empty = tensor::EmptyOp::create(builder, loc, region.shape,
                                            type.getElementType());
      auto identity = builder.getMultiDimIdentityMap(region.shape.size());
      auto copy = linalg::GenericOp::create(
          builder, loc, TypeRange{type}, ValueRange{loaded}, ValueRange{empty},
          SmallVector<AffineMap>{identity, identity},
          SmallVector<utils::IteratorType>(region.shape.size(),
                                           utils::IteratorType::parallel),
          [](OpBuilder& body, Location loc, ValueRange args) {
            Value value = spyreop::Identity::create(body, loc, args[0]);
            linalg::YieldOp::create(body, loc, value);
          });
      ktdp::StoreOp::create(builder, loc, copy.getResult(0), dst);
      func::ReturnOp::create(builder, loc);
      builder.setInsertionPoint(group.call);
      insertCall(builder, func, group.call);
    }
  }
  auto caller = group.call->getParentOfType<ModuleOp>();
  if (auto declaration =
          caller.lookupSymbol<func::FuncOp>(group.func.getName()))
    declaration.erase();
  group.call.erase();
  symbols.erase(group.func->getParentOfType<ModuleOp>());
}

void materializeDirectCopies(ModuleOp top, SymbolTable& symbols,
                             GroupPlan& group, const CoreTransfer& transfer) {
  OpBuilder builder(top.getContext());
  auto loc = group.func.getLoc();
  std::string batch = group.func.getName().str() + "_relayout";
  for (const auto& copy : group.direct_copies) {
    builder.setInsertionPointToEnd(top.getBody());
    auto module = ModuleOp::create(builder, loc, batch);
    symbols.insert(module);
    builder.setInsertionPointToStart(module.getBody());
    auto func = func::FuncOp::create(builder, loc, *module.getSymName(),
                                     group.func.getFunctionType());
    bool remote = copy.source.owner != copy.destination.owner;
    func->setAttr(
        "grid",
        builder.getI64ArrayAttr(
            {remote ? std::max(copy.source.owner, copy.destination.owner) + 1
                    : 1}));
    if (!remote)
      func->setAttr("dataflow_scheduler.physical_core",
                    builder.getI32IntegerAttr(copy.destination.owner));
    func->setAttr("dataflow_scheduler.distributed_transfer_batch",
                  builder.getStringAttr(batch));
    builder.setInsertionPointToStart(func.addEntryBlock());
    IRMapping mapping;
    mapping.map(group.func.getArguments(), func.getArguments());
    auto view = [&](const Partition& partition) {
      auto original = partition.view;
      Value address = cloneDependency(original.getOffset(), mapping, builder);
      return createView(
          builder, loc, address, partition.shape, original.getStaticStrides(),
          original.getType().getElementType(), ktdp::MemorySpaceKind::ct_local);
    };
    Value source = view(copy.source), destination = view(copy.destination);
    if (remote)
      createExchange(builder, loc, source, destination, copy.shape,
                     copy.source_origin, copy.destination_origin, true,
                     copy.source.owner, copy.destination.owner, transfer);
    else
      createLocalCopy(builder, loc, source, destination, copy.source_origin,
                      copy.destination_origin, copy.shape);
    func::ReturnOp::create(builder, loc);
    builder.setInsertionPoint(group.call);
    insertCall(builder, func, group.call);
  }
  group.call.erase();
  symbols.erase(group.func->getParentOfType<ModuleOp>());
}

void materializeGroup(ModuleOp top, SymbolTable& symbols, GroupPlan& group,
                      const CoreTransfer& transfer) {
  if (!group.direct_copies.empty()) {
    materializeDirectCopies(top, symbols, group, transfer);
    return;
  }
  if (group.initialization_source) {
    materializeInitialization(top, symbols, group);
    return;
  }
  OpBuilder builder(group.call);
  bool batch_gather = canBatchTransfers(group, true, transfer);
  bool batch_scatter = canBatchTransfers(group, false, transfer);
  auto copy = [&](ViewPlan& plan, int64_t core, int64_t address, bool gather) {
    auto type = cast<MemRefType>(plan.view.op.getResult().getType());
    int64_t lanes =
        transfer.bytes / *scheduler::tryGetSizeInBytes(type.getElementType());
    for (const auto& region : gather ? plan.reads[core] : plan.writes[core]) {
      const auto& partition = plan.view.partitions[region.partition];
      auto view = partition.view;
      bool remote =
          view.getMemorySpace().getKind() == ktdp::MemorySpaceKind::ct_local &&
          partition.owner != core;
      auto packet = region;
      bool partial = remote && (region.origin.back() % lanes ||
                                region.shape.back() % lanes);
      if (partial) {
        int64_t end = region.origin.back() + region.shape.back();
        packet.origin.back() = region.origin.back() / lanes * lanes;
        packet.shape.back() = llvm::alignTo(end, lanes) - packet.origin.back();
      }
      auto helper =
          createCopyProgram(top, symbols, group.func, partition, packet.origin,
                            packet.shape, address, type, gather, core, transfer,
                            partial && !gather ? plan.relay_address : -1);
      if (gather ? batch_gather : batch_scatter)
        helper->setAttr(
            "dataflow_scheduler.distributed_transfer_batch",
            builder.getStringAttr(group.func.getName().str() +
                                  (gather ? "_gather" : "_scatter")));
      insertCall(builder, helper, group.call);
      if (partial && !gather) {
        auto merge = createCopyProgram(
            top, symbols, group.func, partition, region.origin, region.shape,
            plan.relay_address, type, false, partition.owner, transfer);
        insertCall(builder, merge, group.call);
      }
    }
  };
  // Preserve the old values needed by in-place operations. Only accessed
  // regions are gathered when their coordinates are known for this core.
  for (int64_t core = 0; core < group.cores; ++core)
    for (auto& plan : group.views)
      if (plan.read && !plan.local_read)
        copy(plan, core, plan.read_address, true);
  // Every core writes its own staging allocation, so the original grid can
  // compute concurrently. No old owner contents are needed for a store.
  insertCall(builder, group.func, group.call);
  for (int64_t core = 0; core < group.cores; ++core)
    for (auto& plan : group.views)
      if (plan.written && !plan.local_write)
        copy(plan, core, plan.write_address, false);
  group.call.erase();
  replaceAccesses(group.func, group.views);
}

struct KTIRDistributedMemoryPass
    : scheduler::impl::KTIRDistributedMemoryPassBase<
          KTIRDistributedMemoryPass> {
  void runOnOperation() final {
    ModuleOp top = getOperation();
    auto groups = analyzeGroups(top);
    if (failed(groups)) return signalPassFailure();
    if (groups->empty()) return;
    const auto& device = getAnalysis<ktdf_arch::DefaultDevice>();
    if (!device) return signalPassFailure();
    auto& tracker = getAnalysis<scheduler::MemoryTrackerAnalysis>();
    const auto& tree = tracker.getMemoryTree();
    auto space = ktdp::MemorySpaceAttr::get(
        &getContext(), ktdp::MemorySpaceKind::ct_local, -1);
    auto node_id =
        tree.getNodeIdForMemory(cast<scheduler::ResourceType>(space));
    auto node = node_id ? tree.getNode(*node_id) : std::nullopt;
    if (!node || !node->capacity_in_bytes) {
      top.emitError(
          "distributed lowering requires ct_local memory with a "
          "declared capacity and memory-space mapping");
      return signalPassFailure();
    }
    auto transfer =
        resolveTransfer(device.getRef(), node->memory_resource, *groups);
    if (failed(transfer) ||
        failed(reserveLocalViews(top, tracker, node->memory_resource)))
      return signalPassFailure();
    for (auto& group : *groups) {
      planDirectCopies(group, *transfer);
      if (group.direct_copies.empty()) planLocalAccesses(group);
    }
    trimOverwrittenInitializations(*groups);
    for (auto& group : *groups)
      if (!group.initialization_source && group.direct_copies.empty() &&
          failed(allocateSnapshots(group, tracker, *transfer)))
        return signalPassFailure();
    SymbolTable symbols(top);
    for (auto& group : *groups)
      materializeGroup(top, symbols, group, *transfer);
    // Sequential program calls retain completion ordering, including when
    // Autopilot places the programs in one launch. Independent transfer
    // batches can be combined after their physical unit mapping is resolved.
    IRRewriter rewriter(&getContext());
    (void)runRegionDCE(rewriter, top->getRegions());
  }
};

}  // namespace

LogicalResult scheduler::verifyDistributedMemoryView(
    ktdp::ConstructDistributedMemoryViewOp view) {
  return success(succeeded(analyzeView(view)));
}

std::unique_ptr<Pass> scheduler::createKTIRDistributedMemoryPass() {
  return std::make_unique<KTIRDistributedMemoryPass>();
}
