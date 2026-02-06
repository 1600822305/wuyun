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

# Tests
if(WUYUN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
