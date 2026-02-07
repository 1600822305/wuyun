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

# Memory & Emotion tests (Step 4)
add_executable(test_memory_emotion test_memory_emotion.cpp)
target_link_libraries(test_memory_emotion PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_memory_emotion PRIVATE /utf-8)
endif()
add_test(NAME memory_emotion_tests COMMAND test_memory_emotion)

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

# Integrated brain tests (Step 4.5)
add_executable(test_integrated_brain test_integrated_brain.cpp)
target_link_libraries(test_integrated_brain PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_integrated_brain PRIVATE /utf-8)
endif()
add_test(NAME integrated_brain_tests COMMAND test_integrated_brain)

# Predictive coding tests (Step 6)
add_executable(test_predictive_coding test_predictive_coding.cpp)
target_link_libraries(test_predictive_coding PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_predictive_coding PRIVATE /utf-8)
endif()
add_test(NAME predictive_coding_tests COMMAND test_predictive_coding)

# Decision cortex + Dorsal visual tests (Step 5c+5d)
add_executable(test_decision_dorsal test_decision_dorsal.cpp)
target_link_libraries(test_decision_dorsal PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_decision_dorsal PRIVATE /utf-8)
endif()
add_test(NAME decision_dorsal_tests COMMAND test_decision_dorsal)

# Cerebellum tests (Step 5b)
add_executable(test_cerebellum test_cerebellum.cpp)
target_link_libraries(test_cerebellum PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_cerebellum PRIVATE /utf-8)
endif()
add_test(NAME cerebellum_tests COMMAND test_cerebellum)

# Visual hierarchy tests (Step 5a)
add_executable(test_visual_hierarchy test_visual_hierarchy.cpp)
target_link_libraries(test_visual_hierarchy PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_visual_hierarchy PRIVATE /utf-8)
endif()
add_test(NAME visual_hierarchy_tests COMMAND test_visual_hierarchy)

# Neuromodulator broadcast tests
add_executable(test_neuromod_broadcast test_neuromod_broadcast.cpp)
target_link_libraries(test_neuromod_broadcast PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_neuromod_broadcast PRIVATE /utf-8)
endif()
add_test(NAME neuromod_broadcast_tests COMMAND test_neuromod_broadcast)

# E2E learning demo (Step 4.9)
add_executable(test_e2e_learning test_e2e_learning.cpp)
target_link_libraries(test_e2e_learning PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_e2e_learning PRIVATE /utf-8)
endif()
add_test(NAME e2e_learning_tests COMMAND test_e2e_learning)

# BG DA-STDP learning tests (Step 4.8)
add_executable(test_bg_learning test_bg_learning.cpp)
target_link_libraries(test_bg_learning PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_bg_learning PRIVATE /utf-8)
endif()
add_test(NAME bg_learning_tests COMMAND test_bg_learning)

# Cortical STDP tests (Step 4.7)
add_executable(test_cortical_stdp test_cortical_stdp.cpp)
target_link_libraries(test_cortical_stdp PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_cortical_stdp PRIVATE /utf-8)
endif()
add_test(NAME cortical_stdp_tests COMMAND test_cortical_stdp)

# Learning tests (Step 4.6)
add_executable(test_learning test_learning.cpp)
target_link_libraries(test_learning PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_learning PRIVATE /utf-8)
endif()
add_test(NAME learning_tests COMMAND test_learning)

# GNW global workspace tests
add_executable(test_global_workspace test_global_workspace.cpp)
target_link_libraries(test_global_workspace PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_global_workspace PRIVATE /utf-8)
endif()
add_test(NAME global_workspace_tests COMMAND test_global_workspace)

# Step 6 hypothalamus drive tests
add_executable(test_hypothalamus test_hypothalamus.cpp)
target_link_libraries(test_hypothalamus PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_hypothalamus PRIVATE /utf-8)
endif()
add_test(NAME hypothalamus_tests COMMAND test_hypothalamus)

# Step 5 full brain system tests
add_executable(test_full_brain test_full_brain.cpp)
target_link_libraries(test_full_brain PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_full_brain PRIVATE /utf-8)
endif()
add_test(NAME full_brain_tests COMMAND test_full_brain)

# Step 12 attention mechanism tests
add_executable(test_attention test_attention.cpp)
target_link_libraries(test_attention PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_attention PRIVATE /utf-8)
endif()
add_test(NAME attention_tests COMMAND test_attention)

# Step 11 cognitive tasks (trained Go/NoGo, DMTS, Papez, emotional memory)
add_executable(test_cognitive_tasks test_cognitive_tasks.cpp)
target_link_libraries(test_cognitive_tasks PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_cognitive_tasks PRIVATE /utf-8)
endif()
add_test(NAME cognitive_tasks_tests COMMAND test_cognitive_tasks)

# Step 4 completion tests (Septal/MB/Papez/Presub/HATA/Amyg expansion)
add_executable(test_step4_completion test_step4_completion.cpp)
target_link_libraries(test_step4_completion PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_step4_completion PRIVATE /utf-8)
endif()
add_test(NAME step4_completion_tests COMMAND test_step4_completion)

# Working memory + BG online learning tests (Step 10)
add_executable(test_working_memory test_working_memory.cpp)
target_link_libraries(test_working_memory PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_working_memory PRIVATE /utf-8)
endif()
add_test(NAME working_memory_tests COMMAND test_working_memory)

# Step 8 sleep/hippocampal replay tests
add_executable(test_sleep_replay test_sleep_replay.cpp)
target_link_libraries(test_sleep_replay PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_sleep_replay PRIVATE /utf-8)
endif()
add_test(NAME sleep_replay_tests COMMAND test_sleep_replay)

# Step 9 sensory input interface tests
add_executable(test_sensory_input test_sensory_input.cpp)
target_link_libraries(test_sensory_input PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_sensory_input PRIVATE /utf-8)
endif()
add_test(NAME sensory_input_tests COMMAND test_sensory_input)

# Step 10 scale emergent property tests
add_executable(test_scale_emergent test_scale_emergent.cpp)
target_link_libraries(test_scale_emergent PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_scale_emergent PRIVATE /utf-8)
endif()
add_test(NAME scale_emergent_tests COMMAND test_scale_emergent)

# Step 11 REM sleep + dream tests
add_executable(test_rem_sleep test_rem_sleep.cpp)
target_link_libraries(test_rem_sleep PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_rem_sleep PRIVATE /utf-8)
endif()
add_test(NAME rem_sleep_tests COMMAND test_rem_sleep)

# Step 13-A Homeostatic plasticity integration tests
add_executable(test_homeostatic test_homeostatic.cpp)
target_link_libraries(test_homeostatic PRIVATE wuyun_core)
if(MSVC)
    target_compile_options(test_homeostatic PRIVATE /utf-8)
endif()
add_test(NAME homeostatic_tests COMMAND test_homeostatic)

# Register as CTest
add_test(NAME neuron_tests COMMAND test_neuron)
