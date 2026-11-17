//===-- SubsumeLinearizeIndex.cpp -------------------------------*- c++ -*-===//
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
// SubsumeLinearizeIndex: fold index calculations into data transfer maps.
//
// Fold constant-stride linearized indices and constant-scale loop-IV arithmetic
// into data_transfer source_map/dest_map. AGEN then receives loop IV operands.
//
//===----------------------------------------------------------------------===//

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "subsume-linearize-index"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Subsume Linearize Index pass"),
    llvm::cl::init(false));

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_SUBSUMELINEARIZEINDEXPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

namespace {

/// Returns true if all stride operands of op are arith.constant ops.
static bool hasAllConstantStrides(ktdf::TilingLinearizeIndexOp op) {
  return llvm::all_of(op.getStrides(), [](Value stride) {
    return stride.getDefiningOp<arith::ConstantOp>() != nullptr;
  });
}

/// Builds an affine.apply encoding s[0]*iv[0] + s[1]*iv[1] + ... for op,
/// inserted immediately before insert_before.
static affine::AffineApplyOp buildSubscriptApply(
    IRRewriter& rewriter, ktdf::TilingLinearizeIndexOp op,
    Operation* insert_before) {
  MLIRContext* ctx = op.getContext();
  ValueRange ivs = op.getIvs();
  ValueRange strides = op.getStrides();
  unsigned n = ivs.size();

  AffineExpr expr = getAffineConstantExpr(0, ctx);
  for (unsigned i = 0; i < n; ++i) {
    int64_t s =
        cast<IntegerAttr>(
            strides[i].getDefiningOp<arith::ConstantOp>()->getAttr("value"))
            .getInt();
    expr = expr + s * getAffineDimExpr(i, ctx);
  }

  AffineMap map = AffineMap::get(n, 0, expr, ctx);
  rewriter.setInsertionPoint(insert_before);
  return affine::AffineApplyOp::create(rewriter, op.getLoc(), map, ivs);
}

/// Compose tiled subscripts, then move linear IV arithmetic into the map.
static std::pair<AffineMap, llvm::SmallVector<Value>> rewriteMapOperands(
    IRRewriter& rewriter, AffineMap map, ValueRange operands,
    ktdf::DataTransferOp transfer) {
  llvm::SmallVector<Value> new_operands(operands.begin(), operands.end());

  for (Value& operand : new_operands) {
    auto subscript = operand.getDefiningOp<ktdf::TilingLinearizeIndexOp>();
    if (subscript && hasAllConstantStrides(subscript))
      operand = buildSubscriptApply(rewriter, subscript, transfer);
  }

  affine::fullyComposeAffineMapAndOperands(&map, &new_operands);
  llvm::SmallVector<Value> leaves;
  auto index_expr = [&](auto&& self, Value value) -> AffineExpr {
    if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
      return getAffineConstantExpr(constant.value(), transfer.getContext());
    if (auto add = value.getDefiningOp<arith::AddIOp>())
      return self(self, add.getLhs()) + self(self, add.getRhs());
    if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
      if (auto constant = mul.getLhs().getDefiningOp<arith::ConstantIndexOp>())
        return self(self, mul.getRhs()) * constant.value();
      if (auto constant = mul.getRhs().getDefiningOp<arith::ConstantIndexOp>())
        return self(self, mul.getLhs()) * constant.value();
    }
    auto it = llvm::find(leaves, value);
    if (it != leaves.end())
      return getAffineDimExpr(std::distance(leaves.begin(), it),
                              transfer.getContext());
    leaves.push_back(value);
    return getAffineDimExpr(leaves.size() - 1, transfer.getContext());
  };
  llvm::SmallVector<AffineExpr> replacements;
  for (auto operand : new_operands)
    replacements.push_back(index_expr(index_expr, operand));
  // Affine composition may promote core-uniform operands to symbols.
  // DataTransfer maps carry all operands as dimensions, in dims/symbols order.
  auto expressions = llvm::ArrayRef(replacements);
  map = map.replaceDimsAndSymbols(expressions.take_front(map.getNumDims()),
                                  expressions.drop_front(map.getNumDims()),
                                  leaves.size(), 0);
  new_operands = std::move(leaves);
  affine::canonicalizeMapAndOperands(&map, &new_operands);
  // Keep the dims-only DataTransfer contract after affine canonicalization.
  auto inputs = mlir::AffineMap::getMultiDimIdentityMap(map.getNumInputs(),
                                                        map.getContext())
                    .getResults();
  map = map.replaceDimsAndSymbols(inputs.take_front(map.getNumDims()),
                                  inputs.drop_front(map.getNumDims()),
                                  map.getNumInputs(), 0);
  return {map, new_operands};
}

struct SubsumeLinearizeIndexPass
    : public ktdf::impl::SubsumeLinearizeIndexPassBase<
          SubsumeLinearizeIndexPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    IRRewriter rewriter(&getContext());
    llvm::SmallVector<ktdf::DataTransferOp> transfers;
    getOperation()->walk(
        [&](ktdf::DataTransferOp op) { transfers.push_back(op); });

    auto is_foldable = [](Value v) {
      auto op = v.getDefiningOp<ktdf::TilingLinearizeIndexOp>();
      if (op && hasAllConstantStrides(op)) return true;
      return llvm::isa_and_nonnull<arith::AddIOp, arith::MulIOp,
                                   affine::AffineApplyOp>(v.getDefiningOp());
    };

    for (ktdf::DataTransferOp transfer : transfers) {
      // Process source and destination independently, including implicit
      // identity maps on memrefs.
      if (isa<MemRefType>(transfer.getSource().getType())) {
        if (llvm::any_of(transfer.getSourceIndices(), is_foldable)) {
          auto src_attr = transfer.getSourceMapAttr();
          auto map = src_attr ? src_attr.getValue()
                              : AffineMap::getMultiDimIdentityMap(
                                    transfer.getSourceIndices().size(),
                                    transfer.getContext());
          auto result_src = rewriteMapOperands(
              rewriter, map, transfer.getSourceIndices(), transfer);
          auto src_map = result_src.first;
          auto src_ops = result_src.second;
          rewriter.modifyOpInPlace(transfer, [&]() {
            transfer.setSourceMapAttr(AffineMapAttr::get(src_map));
            transfer.getSourceIndicesMutable().assign(src_ops);
          });
        }
      }

      if (isa<MemRefType>(transfer.getDestination().getType())) {
        if (llvm::any_of(transfer.getDestIndices(), is_foldable)) {
          auto dst_attr = transfer.getDestMapAttr();
          auto map = dst_attr ? dst_attr.getValue()
                              : AffineMap::getMultiDimIdentityMap(
                                    transfer.getDestIndices().size(),
                                    transfer.getContext());
          auto result_dst = rewriteMapOperands(
              rewriter, map, transfer.getDestIndices(), transfer);
          auto dst_map = result_dst.first;
          auto dst_ops = result_dst.second;
          rewriter.modifyOpInPlace(transfer, [&]() {
            transfer.setDestMapAttr(AffineMapAttr::get(dst_map));
            transfer.getDestIndicesMutable().assign(dst_ops);
          });
        }
      }
    }

    // Replace any remaining constant-stride linearize_index ops with
    // affine.apply.
    llvm::SmallVector<ktdf::TilingLinearizeIndexOp> remaining;
    getOperation()->walk([&](ktdf::TilingLinearizeIndexOp op) {
      if (hasAllConstantStrides(op)) remaining.push_back(op);
    });
    for (ktdf::TilingLinearizeIndexOp op : remaining) {
      auto apply = buildSubscriptApply(rewriter, op, op);
      rewriter.replaceOp(op, apply.getResult());
    }

    // Erase dead linearize ops.
    llvm::SmallVector<ktdf::TilingLinearizeIndexOp> dead;
    getOperation()->walk([&](ktdf::TilingLinearizeIndexOp op) {
      if (op->use_empty()) dead.push_back(op);
    });
    for (ktdf::TilingLinearizeIndexOp op : dead) op->erase();
  }
};

}  // namespace

auto ktdf::createSubsumeLinearizeIndexPass() -> std::unique_ptr<Pass> {
  return std::make_unique<SubsumeLinearizeIndexPass>();
}
