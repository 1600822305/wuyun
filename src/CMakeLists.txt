# WuYun C++ core library

set(WUYUN_CORE_SOURCES
    core/neuron.cpp
    core/population.cpp
    core/synapse_group.cpp
    core/spike_queue.cpp
    plasticity/stdp.cpp
    plasticity/stp.cpp
)

add_library(wuyun_core STATIC ${WUYUN_CORE_SOURCES})

target_include_directories(wuyun_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# MSVC specific
if(MSVC)
    target_compile_options(wuyun_core PRIVATE /W4 /utf-8)
    target_compile_definitions(wuyun_core PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(wuyun_core PRIVATE -Wall -Wextra -O2)
endif()

# pybind11 module (optional)
if(WUYUN_BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(pywuyun bindings/pywuyun.cpp)
    target_link_libraries(pywuyun PRIVATE wuyun_core)
endif()
