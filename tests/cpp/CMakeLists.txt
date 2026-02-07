# WuYun C++ tests

add_executable(test_neuron test_neuron.cpp)
target_link_libraries(test_neuron PRIVATE wuyun_core)

if(MSVC)
    target_compile_options(test_neuron PRIVATE /utf-8)
endif()

# Column tests
add_executable(test_column test_column.cpp)
target_link_libraries(test_column PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_column PRIVATE /utf-8)
endif()
add_test(NAME column_tests COMMAND test_column)

# Benchmark
add_executable(test_benchmark test_benchmark.cpp)
target_link_libraries(test_benchmark PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_benchmark PRIVATE /utf-8)
endif()

# Debug
add_executable(test_debug_column test_debug_column.cpp)
target_link_libraries(test_debug_column PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_debug_column PRIVATE /utf-8)
endif()

# Minimal brain end-to-end tests (Step 3)
add_executable(test_minimal_brain test_minimal_brain.cpp)
target_link_libraries(test_minimal_brain PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_minimal_brain PRIVATE /utf-8)
endif()
add_test(NAME minimal_brain_tests COMMAND test_minimal_brain)

# Foundation tests (NMDA, STP, SpikeBus, DA-STDP, Neuromodulator)
add_executable(test_foundation test_foundation.cpp)
target_link_libraries(test_foundation PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_foundation PRIVATE /utf-8)
endif()
add_test(NAME foundation_tests COMMAND test_foundation)

# Register as CTest
add_test(NAME neuron_tests COMMAND test_neuron)
