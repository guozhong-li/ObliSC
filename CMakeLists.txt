cmake_minimum_required(VERSION 3.16)
project(PPSC VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(PPSC_NATIVE "Optimize for the build machine (do not move its binaries to another CPU)" OFF)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(ppsc_common STATIC
  src/algorithm2_index_construct.cpp
  src/dataset.cpp
  src/fo_primitives.cpp
  src/metrics.cpp)
target_include_directories(ppsc_common PUBLIC include)
target_link_libraries(ppsc_common PUBLIC OpenMP::OpenMP_CXX)

add_library(ppsc_fo_core STATIC
  src/algorithm3_fo_canactive.cpp
  src/algorithm4_fo_topk_refine.cpp)
target_link_libraries(ppsc_fo_core PUBLIC ppsc_common)

add_library(ppsc_fo_par_core STATIC
  src/parallel/algorithm3_optimized.cpp
  src/parallel/algorithm4_fo_topk_refine.cpp)
target_compile_definitions(ppsc_fo_par_core PRIVATE PACKED_HEAP)
target_link_libraries(ppsc_fo_par_core PUBLIC ppsc_common)

add_executable(ppsc_fo src/run_benchmark.cpp src/algorithm1_framework.cpp)
target_link_libraries(ppsc_fo PRIVATE ppsc_fo_core)
add_executable(ppsc_fo_par src/run_benchmark.cpp src/parallel/algorithm1_framework.cpp)
target_compile_definitions(ppsc_fo_par PRIVATE PPSC_PARALLEL_EXECUTABLE)
target_link_libraries(ppsc_fo_par PRIVATE ppsc_fo_par_core)

set(ppsc_targets ppsc_common ppsc_fo_core ppsc_fo_par_core ppsc_fo ppsc_fo_par)
foreach(target IN LISTS ppsc_targets)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra)
    if(PPSC_NATIVE)
      target_compile_options(${target} PRIVATE -march=native)
    endif()
  endif()
endforeach()

include(CTest)
if(BUILD_TESTING)
  add_library(ppsc_reference OBJECT
    src/algorithm3_fo_canactive.cpp
    src/algorithm4_fo_topk_refine.cpp)
  target_link_libraries(ppsc_reference PRIVATE ppsc_common)
  target_compile_definitions(ppsc_reference PRIVATE
    algorithm3_fo_candidate_activation=baseline3
    algorithm4_fo_topk_refinement=baseline4)
  add_executable(ppsc_equivalence_tests tests/test_equivalence.cpp
    $<TARGET_OBJECTS:ppsc_reference>)
  target_link_libraries(ppsc_equivalence_tests PRIVATE ppsc_fo_par_core)
  add_test(NAME fo_parallel_equivalence COMMAND ppsc_equivalence_tests)
  set_tests_properties(fo_parallel_equivalence PROPERTIES
    ENVIRONMENT "OMP_THREAD_LIMIT=100;OMP_DYNAMIC=FALSE;OMP_MAX_ACTIVE_LEVELS=3"
    TIMEOUT 300)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME cli_end_to_end
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cli.py
      $<TARGET_FILE:ppsc_fo> $<TARGET_FILE:ppsc_fo_par>)
    set_tests_properties(cli_end_to_end PROPERTIES TIMEOUT 120)
  endif()
endif()

install(TARGETS ppsc_fo ppsc_fo_par RUNTIME DESTINATION bin)
