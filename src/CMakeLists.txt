# WuYun C++ core library

set(WUYUN_CORE_SOURCES
    core/neuron.cpp
    core/population.cpp
    core/synapse_group.cpp
    core/spike_queue.cpp
    core/neuromodulator.cpp
    core/spike_bus.cpp
    core/oscillation.cpp
    core/gap_junction.cpp
    plasticity/stdp.cpp
    plasticity/stp.cpp
    plasticity/da_stdp.cpp
    plasticity/istdp.cpp
    plasticity/homeostatic.cpp
    circuit/cortical_column.cpp
    region/brain_region.cpp
    region/cortical_region.cpp
    region/subcortical/thalamic_relay.cpp
    region/subcortical/basal_ganglia.cpp
    region/subcortical/nucleus_accumbens.cpp
    region/subcortical/superior_colliculus.cpp
    region/subcortical/periaqueductal_gray.cpp
    region/subcortical/cerebellum.cpp
    region/limbic/hippocampus.cpp
    region/limbic/amygdala.cpp
    region/limbic/septal_nucleus.cpp
    region/limbic/mammillary_body.cpp
    region/limbic/hypothalamus.cpp
    region/limbic/lateral_habenula.cpp
    region/anterior_cingulate.cpp
    region/neuromod/vta_da.cpp
    region/neuromod/snc_da.cpp
    region/neuromod/lc_ne.cpp
    region/neuromod/drn_5ht.cpp
    region/neuromod/nbm_ach.cpp
    engine/simulation_engine.cpp
    engine/global_workspace.cpp
    engine/sensory_input.cpp
    engine/sleep_cycle.cpp
    engine/grid_world.cpp
    engine/closed_loop_agent.cpp
    genome/genome.cpp
    genome/evolution.cpp
)

add_library(wuyun_core STATIC ${WUYUN_CORE_SOURCES})

target_include_directories(wuyun_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# OpenMP for brain-level parallelism (region stepping + neuron updates)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(wuyun_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(wuyun_core PUBLIC WUYUN_OPENMP=1)
    message(STATUS "WuYun: OpenMP enabled (${OpenMP_CXX_FLAGS})")
else()
    message(STATUS "WuYun: OpenMP not found, single-threaded fallback")
endif()

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
    if(MSVC)
        target_compile_options(pywuyun PRIVATE /utf-8)
    endif()
endif()
