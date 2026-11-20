# Note that these are _not_ exclude from all because we currently intend to
# install them together with the scheduler.
set(DataflowScheduler_DIALECTS_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/dataflow-scheduler-dialects"
  CACHE PATH "Dataflow scheduler dialects source directory")
add_subdirectory("${DataflowScheduler_DIALECTS_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/dataflow-scheduler-dialects")
set(DataflowScheduler_KTIR_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/ktir-mlir-frontend"
  CACHE PATH "KTIR frontend source directory")
add_subdirectory("${DataflowScheduler_KTIR_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/ktir-mlir-frontend")
