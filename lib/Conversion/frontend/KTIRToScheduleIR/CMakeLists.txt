add_dataflow_scheduler_library(DataflowSchedulerKTIRToScheduleIR
    RelayoutMaterialization.cpp
    ComputeGroupExtraction.cpp
    DistributedMemory.cpp
    FuseLinalg.cpp
    IndirectComputeGroupSplit.cpp
    KTIRBufferize.cpp
    KTIRLegalityCheck.cpp
    KTIRMapAndTile.cpp
    KTIRPipeline.cpp
    MpeMatmul.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR

  GROUP CONVERSION
  EXPORT_NAME KTIRToScheduleIR

  DEPENDS
    DataflowSchedulerKTIRToScheduleIRPassesIncGen

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    KTIR::SpyreOpDialect
    DataflowSchedulerDialects::AgenDialect
    DataflowSchedulerDialects::DataflowDialect
    DataflowScheduler::Analysis
    DataflowScheduler::ArchViews
    DataflowScheduler::KTDFAnalysis
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    DataflowScheduler::KTDFTransforms
    DataflowScheduler::KTDFUtils
    DataflowScheduler::TransformUtils
    DataflowScheduler::Utils
    DataflowScheduler::KTDPLoweringDialect
    DataflowScheduler::KTDFLoweringDialect

  LINK_MLIR_LIBS
    MLIRIR
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRMathDialect
    MLIRPass
    MLIRPresburger
    MLIRSCFDialect
    MLIRSCFUtils
    MLIRTensorDialect
    MLIRTransformUtils
)
