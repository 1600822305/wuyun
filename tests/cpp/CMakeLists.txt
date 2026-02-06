# WuYun C++ tests

add_executable(test_neuron test_neuron.cpp)
target_link_libraries(test_neuron PRIVATE wuyun_core)

if(MSVC)
    target_compile_options(test_neuron PRIVATE /utf-8)
endif()

# Benchmark
add_executable(test_benchmark test_benchmark.cpp)
target_link_libraries(test_benchmark PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_benchmark PRIVATE /utf-8)
endif()

# Register as CTest
add_test(NAME neuron_tests COMMAND test_neuron)
