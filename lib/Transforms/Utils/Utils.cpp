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
//
// Utility functions for transform passes.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Transforms/Utils/Utils.h"

#include <algorithm>
#include <cassert>

#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace scheduler;

std::optional<int64_t> scheduler::getStaticTripCount(mlir::scf::ForOp loop) {
  if (auto tc = loop.getStaticTripCount()) return tc->getSExtValue();
  return std::nullopt;
}

mlir::LogicalResult scheduler::extractGridSize(mlir::func::FuncOp func,
                                               int& grid_size) {
  auto grid_attr = func->template getAttrOfType<mlir::ArrayAttr>("grid");
  if (!grid_attr) {
    return func.emitError("Function missing 'grid' attribute");
  }

  if (grid_attr.size() != 1) {
    return func.emitError("Only 1D grids supported in Phase 1 (got ")
           << grid_attr.size() << "D grid)";
  }

  auto size_attr = mlir::dyn_cast<mlir::IntegerAttr>(grid_attr[0]);
  if (!size_attr) {
    return func.emitError("Grid size must be integer attribute");
  }

  grid_size = size_attr.getInt();
  if (grid_size <= 0) {
    return func.emitError("Grid size must be positive (got ")
           << grid_size << ")";
  }

  return mlir::success();
}

mlir::scf::ForOp scheduler::createForOpWithAdditionalIterArgs(
    mlir::scf::ForOp loop_op, int n_values, mlir::IRMapping& ir_map,
    bool delete_op) {
  mlir::OpBuilder builder(loop_op);
  llvm::SmallVector<mlir::Value, 8> iter_args;

  // Copy existing iterator arguments
  for (auto operand : loop_op.getInits()) {
    iter_args.push_back(operand);
  }

  // Add new iterator arguments with dummy constant initial values
  for (int i = 0; i < n_values; i++) {
    auto const_op =
        mlir::arith::ConstantIndexOp::create(builder, loop_op.getLoc(), 0);
    iter_args.push_back(const_op);
  }

  // Create new loop with additional iter args
  auto new_loop = mlir::scf::ForOp::create(
      builder, loop_op.getLoc(), loop_op.getLowerBound(),
      loop_op.getUpperBound(), loop_op.getStep(), iter_args);

  // Copy attributes from old loop to new loop
  for (auto attr : loop_op->getAttrs()) {
    // Skip operandSegmentSizes as it will be automatically computed
    if (attr.getName() != "operandSegmentSizes") {
      new_loop->setAttr(attr.getName(), attr.getValue());
    }
  }

  // Clone the loop body
  builder.setInsertionPointToStart(new_loop.getBody());

  // Map the induction variable
  ir_map.map(loop_op.getInductionVar(), new_loop.getInductionVar());

  // Map existing region iter args
  for (auto [old_arg, new_arg] :
       llvm::zip(loop_op.getRegionIterArgs(), new_loop.getRegionIterArgs())) {
    ir_map.map(old_arg, new_arg);
  }

  // Clone operations in the loop body
  for (auto& op : loop_op.getBody()->without_terminator()) {
    builder.clone(op, ir_map);
  }

  // Create new yield operation
  llvm::SmallVector<mlir::Value> yield_operands;
  auto old_yield =
      mlir::cast<mlir::scf::YieldOp>(loop_op.getBody()->getTerminator());

  // Map existing yield operands
  for (auto operand : old_yield.getOperands()) {
    yield_operands.push_back(ir_map.lookup(operand));
  }

  // Add new yield operands (pass through the new iter args)
  for (size_t i = loop_op.getNumRegionIterArgs();
       i < new_loop.getNumRegionIterArgs(); i++) {
    yield_operands.push_back(new_loop.getRegionIterArgs()[i]);
  }

  mlir::scf::YieldOp::create(builder, old_yield.getLoc(), yield_operands);

  // Replace uses of old loop results with new loop results
  for (size_t i = 0; i < loop_op->getNumResults(); i++) {
    loop_op->getResult(i).replaceAllUsesWith(new_loop->getResult(i));
  }

  // Erase the old loop if requested
  if (delete_op) {
    loop_op->erase();
  }

  return new_loop;
}

void scheduler::eraseDeadAncestorOps(llvm::ArrayRef<mlir::Operation*> roots) {
  llvm::SmallVector<mlir::Operation*> worklist(roots);
  llvm::DenseSet<mlir::Operation*> erased;
  while (!worklist.empty()) {
    mlir::Operation* op = worklist.pop_back_val();
    // An op reached twice is on the worklist twice, and the second visit would
    // be to freed memory.
    if (erased.contains(op) || !mlir::isOpTriviallyDead(op)) continue;
    for (mlir::Value operand : op->getOperands())
      if (mlir::Operation* def = operand.getDefiningOp())
        worklist.push_back(def);
    erased.insert(op);
    op->erase();
  }
}

int64_t scheduler::getVectorLanes(mlir::Type elem_type,
                                  mlir::ktdf_arch::ExecutionUnitOp compute) {
  auto simd = compute.getFeature<mlir::ktdf_arch::feature::SIMD>();
  int64_t lanes = simd.getLanes(elem_type);
  // Existing devices describe the native 16-bit datapath under the DFIR f16
  // carrier. This lookup sizes packets; it does not change the numeric type.
  if (!lanes && mlir::isa<mlir::spyreop::DF16Type>(elem_type))
    lanes = simd.getLanes(mlir::Float16Type::get(elem_type.getContext()));
  return std::max(lanes, int64_t{1});
}

void scheduler::composeIndexMap(mlir::AffineMap& map,
                                llvm::SmallVectorImpl<mlir::Value>& operands) {
  auto* ctx = map.getContext();
  llvm::SmallVector<mlir::Value> leaves;
  auto index_expr = [&](auto&& self, mlir::Value value) -> mlir::AffineExpr {
    if (auto constant = value.getDefiningOp<mlir::arith::ConstantIndexOp>())
      return mlir::getAffineConstantExpr(constant.value(), ctx);
    if (auto apply = value.getDefiningOp<mlir::affine::AffineApplyOp>()) {
      auto nested = apply.getAffineMap();
      auto replacements = llvm::map_to_vector(
          apply.getMapOperands(), [&](mlir::Value v) { return self(self, v); });
      auto expressions = llvm::ArrayRef(replacements);
      return nested.getResult(0).replaceDimsAndSymbols(
          expressions.take_front(nested.getNumDims()),
          expressions.drop_front(nested.getNumDims()));
    }
    if (auto add = value.getDefiningOp<mlir::arith::AddIOp>())
      return self(self, add.getLhs()) + self(self, add.getRhs());
    if (auto mul = value.getDefiningOp<mlir::arith::MulIOp>()) {
      if (auto constant =
              mul.getLhs().getDefiningOp<mlir::arith::ConstantIndexOp>())
        return self(self, mul.getRhs()) * constant.value();
      if (auto constant =
              mul.getRhs().getDefiningOp<mlir::arith::ConstantIndexOp>())
        return self(self, mul.getLhs()) * constant.value();
    }
    auto it = llvm::find(leaves, value);
    if (it != leaves.end())
      return mlir::getAffineDimExpr(std::distance(leaves.begin(), it), ctx);
    leaves.push_back(value);
    return mlir::getAffineDimExpr(leaves.size() - 1, ctx);
  };
  auto replacements = llvm::map_to_vector(
      operands, [&](mlir::Value v) { return index_expr(index_expr, v); });
  auto expressions = llvm::ArrayRef(replacements);
  map = map.replaceDimsAndSymbols(expressions.take_front(map.getNumDims()),
                                  expressions.drop_front(map.getNumDims()),
                                  leaves.size(), 0);
  operands.assign(leaves.begin(), leaves.end());
  mlir::affine::canonicalizeMapAndOperands(&map, &operands);
  // Canonicalization may promote loop-invariant indices to symbols. Access
  // maps retain the dimensions-only contract in dims/symbols operand order.
  auto inputs = mlir::AffineMap::getMultiDimIdentityMap(map.getNumInputs(), ctx)
                    .getResults();
  map = mlir::simplifyAffineMap(map.replaceDimsAndSymbols(
      inputs.take_front(map.getNumDims()), inputs.drop_front(map.getNumDims()),
      map.getNumInputs(), 0));
}

bool scheduler::isIdentityGeneric(mlir::linalg::GenericOp generic) {
  if (!generic.hasPureTensorSemantics() || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 || generic.getNumResults() != 1 ||
      !llvm::all_of(generic.getIteratorTypesArray(), [](auto it) {
        return it == mlir::utils::IteratorType::parallel;
      }))
    return false;
  auto* body = generic.getBody();
  auto yielded = body->getTerminator()->getOperand(0);
  if (body->getOperations().size() == 1) return yielded == body->getArgument(0);
  auto identity = mlir::dyn_cast<mlir::spyreop::Identity>(body->front());
  return body->getOperations().size() == 2 && identity &&
         identity.getOperand() == body->getArgument(0) &&
         yielded == identity.getResult();
}

int64_t scheduler::getWordTransposeWidth(mlir::linalg::GenericOp generic) {
  if (!isIdentityGeneric(generic)) return 0;
  auto type =
      mlir::cast<mlir::RankedTensorType>(generic.getResult(0).getType());
  auto element = type.getElementType();
  int64_t width = element.isF16() ? 8 : 4;
  if ((!element.isF16() && !element.isF32() &&
       !element.isSignlessInteger(32)) ||
      type.getEncoding() ||
      type.getShape() != llvm::ArrayRef<int64_t>{width, width} ||
      generic.getDpsInputs()[0].getType() != type)
    return 0;
  auto maps = generic.getIndexingMapsArray();
  auto transpose = mlir::AffineMap::getPermutationMap(
      llvm::ArrayRef<unsigned>{1, 0}, generic.getContext());
  return maps[0] == transpose && maps[1].isIdentity() ? width : 0;
}
