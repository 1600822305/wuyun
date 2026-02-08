cmake_minimum_required(VERSION 3.16)
project(wuyun VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Options
option(WUYUN_BUILD_TESTS "Build C++ unit tests" ON)
option(WUYUN_BUILD_PYTHON "Build pybind11 Python bindings" OFF)

# Core C++ library
add_subdirectory(src)

# Evolution runner (Step 16: Genome Layer v1 — 直接编码)
add_executable(run_evolution tools/run_evolution.cpp)
target_link_libraries(run_evolution PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(run_evolution PRIVATE /utf-8)
endif()

# DevEvolution runner (Step 49: 间接编码发育基因组)
add_executable(run_dev_evolution tools/run_dev_evolution.cpp)
target_link_libraries(run_dev_evolution PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(run_dev_evolution PRIVATE /utf-8)
endif()

# Brain topology visualizer (Step 54)
add_executable(visualize_brain tools/visualize_brain.cpp)
target_link_libraries(visualize_brain PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(visualize_brain PRIVATE /utf-8)
endif()

# v55 Continuous movement benchmark (A/B comparison)
add_executable(benchmark_continuous tools/benchmark_continuous.cpp)
target_link_libraries(benchmark_continuous PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(benchmark_continuous PRIVATE /utf-8)
endif()

# Quick scratch pad (类似 python -c 的快速实验)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/scratch.cpp")
    add_executable(scratch tools/scratch.cpp)
    target_link_libraries(scratch PRIVATE wuyun_core)
    if(MSVC)
        target_compile_options(scratch PRIVATE /utf-8)
    endif()
    set_target_properties(scratch PROPERTIES
        EXCLUDE_FROM_ALL TRUE   # 不随 ALL 构建, 需要时才编译
    )
endif()

# Tests
if(WUYUN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
