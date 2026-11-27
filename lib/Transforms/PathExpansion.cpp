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
// Path Expansion Pass
//
// This pass legalizes data movement in ktdf.pipeline operations using an
// explicit architecture graph. It expands illegal direct transfers into
// legal multi-hop paths by inserting intermediate stages and resources.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"
#include "dataflow-scheduler/Analysis/PipelineTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageGraph.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/PathExpansion/Materializer.h"
#include "dataflow-scheduler/Transforms/PathExpansion/Planner.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "path-expansion"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Path Expansion pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_PATHEXPANSIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

bool containsPipeline(mlir::ktdf::StageOp stage) {
  return stage
      .walk(
          [](mlir::ktdf::PipelineOp) { return mlir::WalkResult::interrupt(); })
      .wasInterrupted();
}

bool isPhaseContainer(mlir::ktdf::PipelineOp pipeline) {
  return llvm::any_of(pipeline.getStages(), containsPipeline);
}

/// A final export can be a single already-routed memory copy, with no compute
/// anchor. Accept it only when every transfer has the named unit's direct
/// source -> unit -> destination datapaths in the architecture graph.
bool isDirectTransferPipeline(mlir::ktdf::PipelineOp pipeline,
                              const arch_view::RoutingGraph& graph,
                              const arch_view::MemoryTree& memories) {
  if (pipeline.getNumStages() != 1) return false;
  auto stage = *pipeline.getStages().begin();
  auto units = stage.getApplicableUnitsAttr();
  if (!units || units.size() != 1) return false;
  auto unit = mlir::dyn_cast<ResourceType>(units[0]);
  auto unit_ids = graph.getNodeIdsForResource(unit);
  if (unit_ids.empty()) return false;
  auto resource = [&](mlir::Value buffer) -> ResourceType {
    auto type = mlir::dyn_cast<mlir::MemRefType>(buffer.getType());
    if (!type) return {};
    auto space = mlir::dyn_cast_or_null<ResourceType>(type.getMemorySpace());
    auto id = space ? memories.getNodeIdForMemory(space) : std::nullopt;
    return id ? memories.getNode(*id)->memory_resource : ResourceType{};
  };
  bool found = false;
  for (auto& op : *stage.getBody()) {
    auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op);
    if (!transfer) {
      if (!mlir::isMemoryEffectFree(&op)) return false;
      continue;
    }
    found = true;
    auto src = resource(transfer.getSource());
    auto dst = resource(transfer.getDestination());
    if (!src || !dst) return false;
    bool direct = llvm::any_of(unit_ids, [&](auto unit_id) {
      if (graph.getNode(unit_id)->kind !=
          arch_view::RoutingGraph::ResourceNode::ResourceKind::LoadStoreUnit)
        return false;
      return llvm::any_of(graph.getNodeIdsForResource(src), [&](auto src_id) {
        return llvm::any_of(graph.getNodeIdsForResource(dst), [&](auto dst_id) {
          return graph.getEdgeInfo(src_id, unit_id).has_value() &&
                 graph.getEdgeInfo(unit_id, dst_id).has_value();
        });
      });
    });
    if (!direct) return false;
  }
  return found;
}

/// A phase wrapper schedules one nested pipeline, possibly in a loop nest.
/// Scalar/index calculations and memory views are setup, but actual work must
/// remain inside the nested pipeline so it receives ordinary route validation.
mlir::LogicalResult validatePhaseBody(mlir::Block& body,
                                      unsigned& pipeline_count) {
  for (mlir::Operation& op : body) {
    if (mlir::isa<mlir::ktdf::PipelineOp>(op)) {
      ++pipeline_count;
      continue;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      if (loop.getNumResults() != 0)
        return op.emitError(
            "path-expansion: phase loops must have no loop-carried values");
      if (mlir::failed(validatePhaseBody(*loop.getBody(), pipeline_count)))
        return mlir::failure();
      continue;
    }
    if (mlir::isa<mlir::scf::YieldOp>(op) && op.getNumOperands() == 0) continue;
    auto setup_type = [](mlir::Type type) {
      return mlir::isa<mlir::IntegerType, mlir::IndexType, mlir::FloatType,
                       mlir::MemRefType>(type);
    };
    if (op.getNumRegions() == 0 && mlir::isMemoryEffectFree(&op) &&
        llvm::all_of(op.getOperandTypes(), setup_type) &&
        llvm::all_of(op.getResultTypes(), setup_type))
      continue;
    return op.emitError(
        "path-expansion: phase wrapper contains work outside its nested "
        "pipeline");
  }
  return mlir::success();
}

/// Follow private results and ordinary views to their storage origin. Unknown
/// aliasing remains safe because phase validation conservatively requires a
/// total token order, even when two phases appear to use distinct resources.
mlir::Value phaseResourceOrigin(mlir::Value value) {
  llvm::DenseSet<mlir::Value> seen;
  while (seen.insert(value).second) {
    auto* op = value.getDefiningOp();
    if (!op) break;
    if (auto priv = mlir::dyn_cast<mlir::ktdf::PrivateOp>(op)) {
      value = priv.getYieldOp().getOperand(
          mlir::cast<mlir::OpResult>(value).getResultNumber());
    } else if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) {
      value = view.getViewSource();
    } else if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(op)) {
      value = cast.getSource();
    } else if (auto cast =
                   mlir::dyn_cast<mlir::memref::MemorySpaceCastOp>(op)) {
      value = cast.getSource();
    } else {
      break;
    }
  }
  return value;
}

mlir::LogicalResult validatePhaseContainer(
    mlir::ktdf::PipelineOp pipeline,
    const scheduler::arch_view::RoutingGraph* routing_graph = nullptr) {
  auto phases = llvm::to_vector(pipeline.getStages());
  llvm::SmallVector<llvm::DenseSet<mlir::Value>> buffers(phases.size());
  llvm::SmallVector<llvm::DenseSet<mlir::Attribute>> units(phases.size());
  llvm::DenseMap<mlir::Operation*, unsigned> indices;
  for (auto [index, phase] : llvm::enumerate(phases)) {
    indices[phase] = index;
    if (!containsPipeline(phase))
      return phase.emitError(
          "path-expansion: cannot mix leaf stages and phase-container stages "
          "in one pipeline");
    unsigned pipeline_count = 0;
    if (mlir::failed(validatePhaseBody(*phase.getBody(), pipeline_count)))
      return mlir::failure();
    if (pipeline_count != 1)
      return phase.emitError(
          "path-expansion: each phase must contain exactly one nested "
          "pipeline");

    auto result = phase.walk([&](mlir::Operation* op) -> mlir::WalkResult {
      for (mlir::Value operand : op->getOperands()) {
        if (mlir::isa<mlir::MemRefType>(operand.getType()))
          buffers[index].insert(phaseResourceOrigin(operand));
        if (!mlir::isa<mlir::ktdf::FifoSlotType>(operand.getType())) continue;
        auto* origin = phaseResourceOrigin(operand).getDefiningOp();
        if (!origin || !phase->isAncestor(origin)) {
          op->emitError(
              "path-expansion: FIFO slots cannot cross phase boundaries; "
              "use a shared memref and a token dependency");
          return mlir::WalkResult::interrupt();
        }
      }
      auto stage = mlir::dyn_cast<mlir::ktdf::StageOp>(op);
      if (!stage || containsPipeline(stage)) return mlir::WalkResult::advance();
      auto applicable = stage.getApplicableUnits();
      if (routing_graph && (!applicable || applicable->empty())) {
        stage.emitError(
            "path-expansion: nested leaf stage has no assigned execution unit");
        return mlir::WalkResult::interrupt();
      }
      if (applicable) {
        for (mlir::Attribute attr : *applicable) {
          auto unit = mlir::dyn_cast<mlir::StringAttr>(attr);
          // A kind may name one execution unit per core or sub-core. Routing
          // assigned the kind, not a unique physical instance.
          if (routing_graph &&
              (!unit || routing_graph->getNodeIdsForResource(unit).empty())) {
            stage.emitError(
                "path-expansion: nested leaf stage has an unknown execution "
                "unit");
            return mlir::WalkResult::interrupt();
          }
          units[index].insert(attr);
        }
      }
      return mlir::WalkResult::advance();
    });
    if (result.wasInterrupted()) return mlir::failure();
  }

  // Keep token edges local to this container. StageGraph supplies the same
  // token dependencies used by the leaf scheduler; closure also detects cycles.
  llvm::SmallVector<llvm::SmallVector<bool>> precedes(
      phases.size(), llvm::SmallVector<bool>(phases.size(), false));
  for (auto [index, phase] : llvm::enumerate(phases)) {
    for (mlir::Value token : phase.getDependsIn()) {
      bool has_producer =
          llvm::any_of(token.getUses(), [&](mlir::OpOperand& use) {
            auto producer = mlir::dyn_cast<mlir::ktdf::StageOp>(use.getOwner());
            return producer && producer.isOutDependency(use);
          });
      if (!has_producer)
        return phase.emitError(
            "path-expansion: phase dependency has no producer");
    }
    mlir::ktdf::StageGraph::Node node(phase);
    for (auto producer : node.getProducers()) {
      auto found = indices.find(producer);
      if (found == indices.end())
        return phase.emitError(
            "path-expansion: phase token dependencies must stay within their "
            "container");
      precedes[found->second][index] = true;
    }
    for (auto consumer : node.getConsumers()) {
      if (!indices.contains(consumer))
        return phase.emitError(
            "path-expansion: phase token dependencies must stay within their "
            "container");
    }
  }
  for (unsigned via = 0; via < phases.size(); ++via)
    for (unsigned from = 0; from < phases.size(); ++from)
      for (unsigned to = 0; to < phases.size(); ++to)
        precedes[from][to] =
            precedes[from][to] || (precedes[from][via] && precedes[via][to]);
  for (unsigned first = 0; first < phases.size(); ++first) {
    if (precedes[first][first])
      return pipeline.emitError(
          "path-expansion: phase token dependencies contain a cycle");
    for (unsigned second = first + 1; second < phases.size(); ++second) {
      if (precedes[first][second] || precedes[second][first]) continue;
      auto error = phases[second].emitError(
          "path-expansion: phase stages require a transitive token order");
      if (llvm::any_of(buffers[first], [&](mlir::Value buffer) {
            return buffers[second].contains(buffer);
          }))
        error << " before accessing shared memory";
      else if (llvm::any_of(units[first], [&](mlir::Attribute unit) {
                 return units[second].contains(unit);
               }))
        error << " before reusing an execution unit";
      error.attachNote(phases[first].getLoc()) << "unordered phase is here";
      return mlir::failure();
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Path Expansion Pass
//===----------------------------------------------------------------------===//

struct PathExpansionPass
    : public impl::PathExpansionPassBase<PathExpansionPass> {
  PathExpansionPass() : scheduler_ctx_(SchedulerExtContext::dummyContext()) {}
  explicit PathExpansionPass(const SchedulerExtContext& ctx)
      : scheduler_ctx_(ctx) {}

  void runOnOperation() override;

 private:
  /// Process a single pipeline operation
  mlir::LogicalResult processPipeline(
      mlir::ktdf::PipelineOp pipeline_op,
      const scheduler::arch_view::RoutingGraph& routing_graph);

  const SchedulerExtContext& scheduler_ctx_;
};

void PathExpansionPass::runOnOperation() {
  if (DisableThisPass) return;
  LDBG(1) << "========= " PASS_NAME " =========";

  mlir::ModuleOp module_op = getOperation();

  auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
  auto* const device = device_manager.getOrImportDevice();
  if (!device) {
    module_op->emitError(
        "Unable to import the device specification for path expansion. This "
        "could happen if the device spec file is empty or contains multiple "
        "devices");
    signalPassFailure();
    return;
  }
  auto& routing_graph =
      device_manager.getOrCreateView<arch_view::RoutingGraph>(*device);

  LLVM_DEBUG({
    llvm::dbgs() << "Routing Graph:\n";
    routing_graph.dump();
  });

  // Walk all pipeline operations in the module
  llvm::SmallVector<mlir::ktdf::PipelineOp> candidates;
  module_op.walk<mlir::WalkOrder::PostOrder>(
      [&](mlir::ktdf::PipelineOp pipeline_op) {
        candidates.push_back(pipeline_op);
      });
  // Validate boundaries before modifying leaf pipelines. In particular, a
  // cross-phase FIFO must never reach the leaf planner's local FIFO logic.
  for (mlir::ktdf::PipelineOp cand : llvm::reverse(candidates)) {
    if (isPhaseContainer(cand) && mlir::failed(validatePhaseContainer(cand))) {
      signalPassFailure();
      return;
    }
  }
  for (mlir::ktdf::PipelineOp cand : candidates) {
    if (mlir::failed(processPipeline(cand, routing_graph))) {
      signalPassFailure();
      return;
    }
  }
}

mlir::LogicalResult PathExpansionPass::processPipeline(
    mlir::ktdf::PipelineOp pipeline_op,
    const scheduler::arch_view::RoutingGraph& routing_graph) {
  LDBG(1) << "Processing pipeline at " << pipeline_op.getLoc() << "";

  // Descendants have already been routed by the postorder driver. A phase
  // wrapper schedules their completion; it is not itself a compute stage.
  if (isPhaseContainer(pipeline_op)) {
    if (mlir::failed(validatePhaseContainer(pipeline_op, &routing_graph)))
      return mlir::failure();
    // Downstream mapping uses a wrapper's unit union to select every unit
    // participating in its body, as for wrappers created by stage coarsening.
    mlir::OpBuilder builder(pipeline_op.getContext());
    for (auto phase : pipeline_op.getStages()) {
      llvm::SmallSetVector<mlir::Attribute, 8> units;
      phase.walk([&](mlir::ktdf::StageOp leaf) {
        if (containsPipeline(leaf)) return;
        for (mlir::Attribute unit : leaf.getApplicableUnitsAttr())
          units.insert(unit);
      });
      phase.setApplicableUnitsAttr(builder.getArrayAttr(units.getArrayRef()));
    }
    return mlir::success();
  }

  auto& manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
  auto& memories = manager.getOrCreateView<arch_view::MemoryTree>(
      *manager.getOrImportDevice());
  if (isDirectTransferPipeline(pipeline_op, routing_graph, memories))
    return mlir::success();

  // Build PipelineTree for this pipeline
  PipelineTree tree;
  mlir::Operation* parent = pipeline_op->getParentOp();
  if (!parent) {
    pipeline_op->emitError("path-expansion: pipeline has no parent operation");
    return mlir::failure();
  }

  tree.compute(*parent);

  // Find the pipeline node in the tree
  PipelineNode* pipeline_node =
      static_cast<PipelineNode*>(tree.getNodeForOp(pipeline_op));

  // Plan path expansion
  auto plan = planPathExpansion(tree, pipeline_node, routing_graph);
  if (!plan) {
    pipeline_op->emitError("path-expansion: failed to plan path expansion");
    return mlir::failure();
  }

  if (!plan->changed) {
    LDBG(1) << "  Pipeline already legal, no changes needed";
    return mlir::success();
  }

  LDBG(1) << "  Pipeline needs expansion";

  // Materialize the modified pipeline tree
  mlir::OpBuilder builder(pipeline_op.getContext());
  mlir::IRMapping value_map;

  PathExpansionMaterializer materializer(builder, value_map, plan->stage_info,
                                         plan->resource_factory);

  // Set insertion point before the original pipeline
  builder.setInsertionPoint(pipeline_op);

  // Materialize the new pipeline from the modified tree
  mlir::Operation* new_pipeline_op = materializer.materialize(*pipeline_node);

  if (!new_pipeline_op) {
    pipeline_op->emitError(
        "path-expansion: failed to materialize expanded pipeline");
    return mlir::failure();
  }

  LDBG(1) << "  Materialization succeeded";

  auto new_pipeline = mlir::dyn_cast<mlir::ktdf::PipelineOp>(new_pipeline_op);
  if (!new_pipeline) {
    new_pipeline_op->emitError(
        "path-expansion: materialized operation is not a pipeline");
    return mlir::failure();
  }

  mlir::IRRewriter rewriter(new_pipeline);
  mlir::ktdf::PrivateBuilder::canonicalize(rewriter, new_pipeline);

  // Erase the original pipeline
  pipeline_op.erase();

  return mlir::success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> scheduler::createPathExpansionPass() {
  return std::make_unique<PathExpansionPass>();
}

std::unique_ptr<mlir::Pass> scheduler::createPathExpansionPass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<PathExpansionPass>(scheduler_ctx);
}

// Made with Bob
