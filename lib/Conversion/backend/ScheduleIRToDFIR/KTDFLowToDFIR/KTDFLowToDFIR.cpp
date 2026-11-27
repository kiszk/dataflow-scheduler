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
/// KTDFLowToDFIR pass: creates one dataflow.program_unit per component type
/// from the KTDFLowering execute_on / signal IR.
///
//
//===----------------------------------------------------------------------===//

#include <mlir/Transforms/RegionUtils.h>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/LogicalMemoryViewBuilder.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/OperationLowerings.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/PreludeWorkPartition.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/ProgramUnitBuilder.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/QueryMapArithCollapse.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/UnitTypeDiscovery.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "dataflow-scheduler/Dialect/Symbol/Symbol.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Transforms/Utils/SCFTilingUtils.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/SpyreOp/SpyreOpTypes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define PASS_NAME "ktdflowering-to-dfir"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTDFLOWTODFIRPASS
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h.inc"
}  // namespace scheduler

namespace {

// Run the full canonicalization pattern set (all loaded dialects + registered
// ops) with region simplification (DCE) on `func`. Mirrors the upstream
// Canonicalizer pass; best-effort (non-convergence is not a failure).
static void canonicalizeFunc(mlir::func::FuncOp func) {
  mlir::MLIRContext* ctx = func.getContext();
  mlir::RewritePatternSet patterns(ctx);
  for (mlir::Dialect* dialect : ctx->getLoadedDialects()) {
    dialect->getCanonicalizationPatterns(patterns);
  }
  for (mlir::RegisteredOperationName op : ctx->getRegisteredOperations()) {
    op.getCanonicalizationPatterns(patterns, ctx);
  }
  mlir::FrozenRewritePatternSet frozen(std::move(patterns));
  // Default GreedyRewriteConfig enables region simplification (DCE).
  (void)mlir::applyPatternsGreedily(func, frozen);
}

// DistributedMemory marks only mutually independent, whole-packet transfers.
// Keep one ordered body per physical unit instead of a program per core pair.
// Do this after one-core helpers have resolved their logical grid and
// placement; widening their grid before mapping would change their tensor
// coordinates.
static mlir::LogicalResult batchDistributedTransfers(mlir::ModuleOp top) {
  constexpr llvm::StringLiteral marker =
      "dataflow_scheduler.distributed_transfer_batch";
  llvm::MapVector<mlir::StringAttr, llvm::SmallVector<mlir::func::FuncOp>>
      batches;
  top.walk([&](mlir::func::FuncOp func) {
    if (auto batch = func->getAttrOfType<mlir::StringAttr>(marker))
      batches[batch].push_back(func);
  });
  for (auto& [name, members] : batches) {
    for (auto func : members) func->removeAttr(marker);
    if (members.size() < 2) continue;
    llvm::SmallVector<mlir::func::CallOp> calls;
    llvm::SmallVector<mlir::func::FuncOp> ordered;
    top.walk([&](mlir::func::CallOp call) {
      for (auto func : members)
        if (call.getCallee() == func.getName()) {
          calls.push_back(call);
          ordered.push_back(func);
        }
    });
    if (calls.size() != members.size())
      return top.emitError(
          "distributed transfer batch requires one call per helper");
    auto first = calls.front();
    auto type = ordered.front().getFunctionType();
    for (auto [call, func] : llvm::zip(calls, ordered)) {
      if (call->getBlock() != first->getBlock() ||
          call.getOperands() != first.getOperands() ||
          func.getFunctionType() != type)
        return call.emitError(
            "distributed batch must have identical call arguments");
    }
    // An intervening effect would invalidate the frontend's independence proof.
    for (mlir::Operation* op = first; op != calls.back();
         op = op->getNextNode()) {
      if (!llvm::is_contained(calls, mlir::dyn_cast<mlir::func::CallOp>(op)) &&
          !mlir::isMemoryEffectFree(op))
        return op->emitError(
            "unexpected effect within distributed transfer batch");
    }

    mlir::OpBuilder builder(top.getContext());
    builder.setInsertionPointToEnd(top.getBody());
    auto module = mlir::ModuleOp::create(builder, first.getLoc(),
                                         name.getValue().str() + "_batch");
    mlir::SymbolTable(top).insert(module);
    builder.setInsertionPointToStart(module.getBody());
    auto target = mlir::func::FuncOp::create(builder, first.getLoc(),
                                             *module.getSymName(), type);
    builder.setInsertionPointToStart(target.addEntryBlock());
    llvm::SmallVector<mlir::IRMapping> mappings(ordered.size());
    llvm::SmallVector<mlir::dataflow::GetUnitOp> units;
    int64_t max_core = 0;
    // All preludes must dominate all unit bodies, including those appended to
    // an already existing body. Intern physical unit identities across helpers.
    for (auto [i, func] : llvm::enumerate(ordered)) {
      auto& mapping = mappings[i];
      mapping.map(func.getArguments(), target.getArguments());
      for (auto& op : func.getBody().front().without_terminator()) {
        if (mlir::isa<mlir::dataflow::ProgramUnitOp>(op)) continue;
        if (auto unit = mlir::dyn_cast<mlir::dataflow::GetUnitOp>(op)) {
          auto found = llvm::find_if(units, [&](auto previous) {
            return previous->getAttr("type") == unit->getAttr("type") &&
                   previous->getAttr("core") == unit->getAttr("core") &&
                   previous->getAttr("corelet") == unit->getAttr("corelet");
          });
          if (found != units.end()) {
            mapping.map(unit.getResults(), found->getResults());
            continue;
          }
          auto cloned =
              mlir::cast<mlir::dataflow::GetUnitOp>(builder.clone(op, mapping));
          units.push_back(cloned);
          if (auto core = cloned->getAttrOfType<mlir::IntegerAttr>("core"))
            max_core = std::max(max_core, core.getInt());
        } else {
          builder.clone(op, mapping);
        }
      }
    }
    target->setAttr("grid", builder.getI64ArrayAttr({max_core + 1}));
    llvm::MapVector<mlir::Value, mlir::dataflow::ProgramUnitOp> programs;
    for (auto [i, func] : llvm::enumerate(ordered)) {
      for (auto source : func.getOps<mlir::dataflow::ProgramUnitOp>()) {
        for (mlir::Value old_unit : source.getUnits()) {
          auto mapping = mappings[i];
          mlir::Value unit = mapping.lookup(old_unit);
          auto& destination = programs[unit];
          if (!destination) {
            destination = mlir::dataflow::ProgramUnitOp::create(
                builder, source.getLoc(), mlir::ValueRange{unit},
                [](mlir::OpBuilder& body, mlir::Location loc,
                   mlir::ValueRange) {
                  mlir::dataflow::ReturnOp::create(body, loc);
                });
          }
          // Specialize the helper's uniform queries for this concrete unit.
          mapping.map(source.getRegion().front().getArgument(0), unit);
          mlir::OpBuilder body(destination.getRegion().front().getTerminator());
          for (auto& op : source.getRegion().front().without_terminator())
            body.clone(op, mapping);
        }
      }
    }
    mlir::func::ReturnOp::create(builder, target.getLoc());
    canonicalizeFunc(target);

    auto caller = first->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(caller.getBody());
    auto declaration = mlir::func::FuncOp::create(builder, target.getLoc(),
                                                  target.getName(), type);
    declaration.setPrivate();
    builder.setInsertionPoint(first);
    mlir::func::CallOp::create(builder, first.getLoc(), declaration,
                               first.getOperands());
    for (auto call : calls) call.erase();
    for (auto func : ordered) {
      if (auto declaration =
              caller.lookupSymbol<mlir::func::FuncOp>(func.getName()))
        declaration.erase();
      func->getParentOfType<mlir::ModuleOp>().erase();
    }
  }
  return mlir::success();
}

// Loop normalization introduces index arithmetic. AGEN operands must be loop
// IVs or constants, so reuse the index composition used by KTDF transfers.
static void composeAgenIndexMaps(mlir::func::FuncOp func) {
  auto compose = [](mlir::Operation* op, mlir::StringAttr name,
                    mlir::AffineMap map, mlir::MutableOperandRange indices) {
    auto values = indices.getAsOperandRange();
    if (llvm::none_of(values, [](mlir::Value value) {
          return mlir::isa_and_nonnull<mlir::arith::AddIOp, mlir::arith::MulIOp,
                                       mlir::affine::AffineApplyOp>(
              value.getDefiningOp());
        }))
      return;
    llvm::SmallVector<mlir::Value> operands(values.begin(), values.end());
    composeIndexMap(map, operands);
    op->setAttr(name, mlir::AffineMapAttr::get(map));
    indices.assign(operands);
  };
  func.walk([&](mlir::agen::VectorLoadOp op) {
    compose(op, op.getAffineMapAttrName(), op.getAffineMap(),
            op.getMapOperandsMutable());
  });
  func.walk([&](mlir::agen::VectorStoreOp op) {
    compose(op, op.getAffineMapAttrName(), op.getAffineMap(),
            op.getMapOperandsMutable());
  });
  func.walk([&](mlir::agen::CompositeLoadAndStoreOp op) {
    compose(op, op.getSrcMapAttrName(), op.getSrcMap(),
            op.getSrcMapOperandsMutable());
    compose(op, op.getDstMapAttrName(), op.getDstMap(),
            op.getDstMapOperandsMutable());
  });
}

// Compare first/last-iteration guards against the normalized counter itself.
// For example, 2*q != 0 becomes q != 0, while 2*q+1 != 0 is always true.
// Keeping the denormalized counter as scalar arithmetic would require target
// instructions that are not available on every unit executing these guards.
static void foldExpandedLoopGuards(llvm::ArrayRef<mlir::scf::ForOp> loops,
                                   mlir::IRRewriter& rewriter) {
  auto d0 = rewriter.getAffineDimExpr(0);
  for (auto loop : loops) {
    mlir::Value iv = loop.getInductionVar();
    loop.walk([&](mlir::arith::CmpIOp compare) {
      auto predicate = compare.getPredicate();
      if (predicate != mlir::arith::CmpIPredicate::eq &&
          predicate != mlir::arith::CmpIPredicate::ne)
        return;
      mlir::Value value = compare.getLhs();
      auto constant =
          compare.getRhs().getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (!constant) {
        constant =
            compare.getLhs().getDefiningOp<mlir::arith::ConstantIndexOp>();
        value = compare.getRhs();
      }
      if (!constant) return;
      auto map = rewriter.getDimIdentityMap();
      llvm::SmallVector<mlir::Value> operands{value};
      composeIndexMap(map, operands);
      if (operands.size() != 1 || operands.front() != iv) return;
      auto expr = mlir::simplifyAffineMap(map).getResult(0);
      auto bias = mlir::dyn_cast<mlir::AffineConstantExpr>(
          expr.replaceDimsAndSymbols({rewriter.getAffineConstantExpr(0)}, {}));
      if (!bias || (expr != d0 + bias && expr != d0 * 2 + bias)) return;
      int64_t scale = expr == d0 * 2 + bias ? 2 : 1;
      int64_t difference;
      if (llvm::SubOverflow(constant.value(), bias.getValue(), difference))
        return;
      auto lower =
          loop.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
      auto upper =
          loop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
      bool impossible = difference % scale != 0 ||
                        (lower && difference / scale < lower.value()) ||
                        (upper && difference / scale >= upper.value());
      rewriter.setInsertionPoint(compare);
      if (impossible) {
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(
            compare, predicate == mlir::arith::CmpIPredicate::ne, 1);
      } else {
        auto bound = mlir::arith::ConstantIndexOp::create(
            rewriter, compare.getLoc(), difference / scale);
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(compare, predicate, iv,
                                                         bound);
      }
    });
  }
}

// Keep both buffers, but expose a complete ping-pong cycle to the address
// generator. A loop-carried `sum - offset` is not an arithmetic progression:
// treating it as a fixed address increment reads outside the buffers after
// the second iteration. Unrolling two iterations cancels the recurrence;
// ordinary SCF canonicalization also handles an odd final iteration. Normalize
// the expanded loops again because the target requires a unit step.
static mlir::LogicalResult expandAlternatingBufferOffsets(
    mlir::func::FuncOp func) {
  mlir::IRRewriter rewriter(func.getContext());
  while (true) {
    mlir::scf::ForOp alternating;
    func.walk<mlir::WalkOrder::PostOrder>([&](mlir::scf::ForOp loop) {
      if (!loop.getStaticTripCount()) return mlir::WalkResult::advance();
      for (auto [arg, yielded] :
           llvm::zip(loop.getRegionIterArgs(),
                     loop.getBody()->getTerminator()->getOperands())) {
        auto sub = yielded.getDefiningOp<mlir::arith::SubIOp>();
        if (sub && sub.getRhs() == arg &&
            mlir::matchPattern(sub.getLhs(), mlir::m_Constant())) {
          alternating = loop;
          return mlir::WalkResult::interrupt();
        }
      }
      return mlir::WalkResult::advance();
    });
    if (!alternating) return mlir::success();
    auto unrolled = mlir::loopUnrollByFactor(alternating, 2);
    if (mlir::failed(unrolled))
      return alternating.emitError(
          "could not expand alternating buffer offsets");
    llvm::SmallVector<mlir::scf::ForOp> loops;
    if (unrolled->mainLoopOp) loops.push_back(*unrolled->mainLoopOp);
    if (unrolled->epilogueLoopOp) loops.push_back(*unrolled->epilogueLoopOp);
    normalizeSCFLoops(loops, rewriter);
    composeAgenIndexMaps(func);
    foldExpandedLoopGuards(loops, rewriter);
    canonicalizeFunc(func);
  }
}

// AGEN subscripts accept induction variables and constants. Resolve per-unit
// coordinates before handing such accesses to the target backend. This also
// lets ordinary canonicalization remove empty core partitions and fold their
// bounds; it does not change the ownership or ordering of any transfer.
static void specializeUniformAccesses(mlir::func::FuncOp func) {
  bool needs_specialization = false;
  func.walk([&](mlir::Operation* op) {
    if (!mlir::isa<mlir::agen::VectorLoadOp, mlir::agen::VectorStoreOp,
                   mlir::agen::CompositeLoadAndStoreOp>(op))
      return;
    for (mlir::Value operand : op->getOperands()) {
      if (operand.getDefiningOp<mlir::uniform::QueryMapOp>())
        needs_specialization = true;
    }
  });
  if (!needs_specialization) return;
  llvm::SmallVector<mlir::dataflow::ProgramUnitOp> programs;
  func.walk([&](mlir::dataflow::ProgramUnitOp op) { programs.push_back(op); });
  mlir::IRRewriter rewriter(func.getContext());
  for (auto program : programs) {
    rewriter.setInsertionPoint(program);
    for (mlir::Value unit : program.getUnits()) {
      auto copy =
          mlir::cast<mlir::dataflow::ProgramUnitOp>(rewriter.clone(*program));
      copy.getUnitsMutable().assign(mlir::ValueRange{unit});
      copy.getRegion().front().getArgument(0).replaceAllUsesWith(unit);
      copy.walk([&](mlir::uniform::QueryMapOp query) {
        auto map = query.getMap()
                       .getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
        if (map)
          if (auto value = map.getValue(query.getKey()))
            rewriter.replaceOp(query, *value);
      });
    }
    rewriter.eraseOp(program);
  }
}

struct KTDFLowToDFIRPass
    : public impl::KTDFLowToDFIRPassBase<KTDFLowToDFIRPass> {
  KTDFLowToDFIRPass(const SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void runOnOperation() override {
    LDBG(1) << "========= " PASS_NAME " =========";
    mlir::ModuleOp module_op = getOperation();

    // Obtain the default device, emitting a diagnostic on failure.
    const auto& default_device = getAnalysis<mlir::ktdf_arch::DefaultDevice>();
    if (!default_device) {
      signalPassFailure();
      return;
    }

    // The lowerings ask the mapping which resource runs an op, so they can size
    // vectors and address memory the way that resource does.
    mlir::ktdf_arch::Mapping mapping(default_device.getRef());
    if (!getUniqueComputeUnit(mapping.byKind())) {
      mapping.getDevice().getDeclaration().emitError(
          "no (unambiguous) default compute resource");
      signalPassFailure();
      return;
    }
    auto& memory_tree =
        default_device.getRef().getOrCreateView<arch_view::MemoryTree>();

    // The run's symbols, numbered once for the whole module: nothing sets a
    // range of ids aside, so two functions cannot each start from the top. This
    // also declares in the IR which symbol each of the run's inputs is, which
    // is what lets a consumer read the numbering rather than work it out again.
    SymbolAllocator symbols(module_op);

    llvm::SmallVector<mlir::func::FuncOp, 4> funcs;
    module_op.walk([&](mlir::func::FuncOp func) { funcs.push_back(func); });
    for (auto func : funcs) {
      LDBG(1) << "Running " << PASS_NAME << " on " << func.getName();

      // DFIR uses f16 as its native SEN169 storage carrier. Keep df16 through
      // scheduling, then lower its representation here. numeric_cast_value
      // still selects native FCVT; this is not an IEEE numerical conversion.
      mlir::AttrTypeReplacer storage_types;
      storage_types.addReplacement(
          [](mlir::spyreop::DF16Type type) -> mlir::Type {
            return mlir::Float16Type::get(type.getContext());
          });
      storage_types.recursivelyReplaceElementsIn(func, /*replaceAttrs=*/true,
                                                 /*replaceLocs=*/false,
                                                 /*replaceTypes=*/true);

      auto split = partitionPreludeAndWork(func);
      if (split.work_ops.empty()) {
        LDBG(1) << "  no work ops; skipping function";
        continue;
      }

      ResourceToUnits components;
      if (mlir::failed(discoverUnitTypes(split.work_ops, components))) {
        return signalPassFailure();
      }
      if (components.empty()) {
        LDBG(1) << "  no ktdf_lowering.execute_on ops; skipping function";
      } else {
        if (mlir::failed(buildProgramUnits(func, split.work_ops, components))) {
          return signalPassFailure();
        }
      }

      // Clean up side-effect-free ops left dead after extraction (e.g. ops that
      // were cloned into child modules but whose originals are now unused in
      // the top-level module). runRegionDCE recurses into nested regions, so
      // passing the module's regions covers child modules created above.
      mlir::IRRewriter rewriter(module_op.getContext());
      (void)mlir::runRegionDCE(rewriter, func.getBody());

      if (mlir::failed(buildLogicalMemoryViews(
              func, memory_tree, mapping.byKind(), scheduler_ctx_, symbols))) {
        return signalPassFailure();
      }
      // run arith folding into query maps after logical mem view is built to
      // cleanup any extraneous ops.
      if (mlir::failed(scheduler::collapseArithOverQueryMaps(func))) {
        func.emitError("failed to collapse arith ops over query maps");
        return signalPassFailure();
      }

      // Run operation lowerings after program units have been created
      if (mlir::failed(runOperationLowerings(func, schedulerExtContext(),
                                             components, mapping, symbols))) {
        func.emitError("failed to run operation lowerings for ")
            << func.getName();
        return signalPassFailure();
      }

      if (mlir::failed(schedulerExtContext().lowerTargetOperations(func)))
        return signalPassFailure();

      // Final cleanup: canonicalize + DCE the fully-lowered function so the
      // emitted DFIR has no dead/duplicate/foldable leftovers (duplicate
      // folded constants from query-map collapse, addi(const,const), dead
      // NoMemoryEffect ops).
      specializeUniformAccesses(func);
      canonicalizeFunc(func);
      if (mlir::failed(expandAlternatingBufferOffsets(func)))
        return signalPassFailure();

      // Owner-local distributed-copy programs have a one-element logical
      // grid. Relocate that whole program, including its local memory units,
      // only after grid queries and addresses have been resolved. Changing a
      // memory view's core alone would still execute the load on the wrong LX.
      if (auto core = func->getAttrOfType<mlir::IntegerAttr>(
              "dataflow_scheduler.physical_core")) {
        auto result = func.walk([&](mlir::dataflow::GetUnitOp unit) {
          auto old = unit->getAttrOfType<mlir::IntegerAttr>("core");
          if (!old) return mlir::WalkResult::advance();
          if (old.getInt() != 0 || core.getInt() < 0) {
            unit.emitError(
                "fixed placement requires a one-core logical program "
                "and a nonnegative physical core");
            return mlir::WalkResult::interrupt();
          }
          unit->setAttr("core", core);
          std::string name = unit.getName().str();
          if (llvm::StringRef(name).starts_with("C0-"))
            name.replace(0, 3, "C" + std::to_string(core.getInt()) + "-");
          unit.setName(name);
          return mlir::WalkResult::advance();
        });
        if (result.wasInterrupted()) return signalPassFailure();
        func->removeAttr("dataflow_scheduler.physical_core");
      }
    }
    if (mlir::failed(batchDistributedTransfers(module_op)))
      return signalPassFailure();
  }

 private:
  const SchedulerExtContext& schedulerExtContext() const {
    return scheduler_ctx_;
  }

  const SchedulerExtContext& scheduler_ctx_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createKTDFLowToDFIRPass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<KTDFLowToDFIRPass>(scheduler_ctx);
}

std::unique_ptr<mlir::Pass> scheduler::createKTDFLowToDFIRPass() {
  return std::make_unique<KTDFLowToDFIRPass>(
      SchedulerExtContext::dummyContext());
}
