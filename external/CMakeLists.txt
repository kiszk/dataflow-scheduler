# Note that these are _not_ exclude from all because we currently intend to
# install them together with the scheduler.
set(DataflowScheduler_DIALECTS_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/dataflow-scheduler-dialects"
  CACHE PATH "Dataflow scheduler dialects source directory")
add_subdirectory("${DataflowScheduler_DIALECTS_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/dataflow-scheduler-dialects")
add_subdirectory(ktir-mlir-frontend)
