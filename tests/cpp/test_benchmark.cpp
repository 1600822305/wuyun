/**
 * WuYun C++ Performance Benchmark
 *
 * Measures: neurons/step time for various population sizes.
 */

#include "core/types.h"
#include "core/population.h"
#include "core/synapse_group.h"
#include "core/spike_queue.h"
#include "plasticity/stdp.h"
#include "plasticity/stp.h"

#include <cstdio>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>

using namespace wuyun;
using Clock = std::chrono::high_resolution_clock;

// =============================================================================
// Benchmark 1: NeuronPopulation step speed
// =============================================================================

void bench_population(size_t n_neurons, int n_steps) {
    auto params = L23_PYRAMIDAL_PARAMS();
    NeuronPopulation pop(n_neurons, params);

    // Inject constant current to ~30% of neurons (realistic sparse activity)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    auto t0 = Clock::now();
    size_t total_spikes = 0;

    for (int t = 0; t < n_steps; ++t) {
        // Inject current to random subset
        for (size_t i = 0; i < n_neurons; ++i) {
            if (dist(rng) < 0.3f) {
                pop.inject_basal(i, 12.0f);
            }
            if (dist(rng) < 0.1f) {
                pop.inject_apical(i, 15.0f);
            }
        }
        total_spikes += pop.step(t);
    }

    auto t1 = Clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_step_ms = elapsed_ms / n_steps;
    double per_step_us = per_step_ms * 1000.0;
    double neurons_per_sec = (double)n_neurons * n_steps / (elapsed_ms / 1000.0);
    double firing_rate = (double)total_spikes / ((double)n_neurons * n_steps) * 1000.0; // Hz approx

    printf("  Population  %7zu neurons x %4d steps | %8.2f ms total | %7.2f us/step | %.1f M neurons/s | ~%.1f Hz\n",
           n_neurons, n_steps, elapsed_ms, per_step_us, neurons_per_sec / 1e6, firing_rate);
}

// =============================================================================
// Benchmark 2: SynapseGroup deliver + compute
// =============================================================================

void bench_synapse(size_t n_pre, size_t n_post, size_t synapses_per_pre, int n_steps) {
    size_t n_syn = n_pre * synapses_per_pre;

    std::mt19937 rng(123);
    std::uniform_int_distribution<int32_t> post_dist(0, static_cast<int32_t>(n_post - 1));

    std::vector<int32_t> pre_ids(n_syn);
    std::vector<int32_t> post_ids(n_syn);
    std::vector<float> weights(n_syn, 0.5f);
    std::vector<int32_t> delays(n_syn, 1);

    for (size_t pre = 0; pre < n_pre; ++pre) {
        for (size_t s = 0; s < synapses_per_pre; ++s) {
            size_t idx = pre * synapses_per_pre + s;
            pre_ids[idx] = static_cast<int32_t>(pre);
            post_ids[idx] = post_dist(rng);
        }
    }

    SynapseGroup sg(n_pre, n_post, pre_ids, post_ids, weights, delays, AMPA_PARAMS);

    // Simulate ~5% pre neurons firing each step
    std::vector<uint8_t> fired(n_pre, 0);
    std::vector<int8_t> spike_type(n_pre, 0);
    std::vector<float> v_post(n_post, -65.0f);
    std::uniform_real_distribution<float> fdist(0.0f, 1.0f);

    auto t0 = Clock::now();

    for (int t = 0; t < n_steps; ++t) {
        // Random firing
        for (size_t i = 0; i < n_pre; ++i) {
            fired[i] = (fdist(rng) < 0.05f) ? 1 : 0;
        }
        sg.deliver_spikes(fired, spike_type);
        sg.step_and_compute(v_post);
    }

    auto t1 = Clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_step_us = elapsed_ms / n_steps * 1000.0;

    printf("  Synapse  %6zu pre x %3zu syn/pre = %7zu synapses x %4d steps | %8.2f ms | %7.2f us/step\n",
           n_pre, synapses_per_pre, n_syn, n_steps, elapsed_ms, per_step_us);
}

// =============================================================================
// Benchmark 3: STDP update
// =============================================================================

void bench_stdp(size_t n_syn, int n_steps) {
    std::mt19937 rng(456);

    std::vector<float> weights(n_syn, 0.5f);
    std::vector<float> pre_times(n_syn / 10, -1.0f);
    std::vector<float> post_times(n_syn / 10, -1.0f);
    std::vector<int> pre_ids(n_syn);
    std::vector<int> post_ids(n_syn);

    size_t n_neurons = n_syn / 10;
    std::uniform_int_distribution<int> ndist(0, static_cast<int>(n_neurons - 1));

    for (size_t s = 0; s < n_syn; ++s) {
        pre_ids[s] = ndist(rng);
        post_ids[s] = ndist(rng);
    }

    STDPParams params;
    std::uniform_real_distribution<float> tdist(0.0f, 100.0f);

    auto t0 = Clock::now();

    for (int t = 0; t < n_steps; ++t) {
        // Simulate some spike times
        for (size_t i = 0; i < n_neurons; ++i) {
            pre_times[i] = tdist(rng);
            post_times[i] = tdist(rng);
        }
        stdp_update_batch(weights.data(), n_syn, pre_times.data(), post_times.data(),
                         pre_ids.data(), post_ids.data(), params);
    }

    auto t1 = Clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_step_us = elapsed_ms / n_steps * 1000.0;

    printf("  STDP  %7zu synapses x %4d steps | %8.2f ms | %7.2f us/step\n",
           n_syn, n_steps, elapsed_ms, per_step_us);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== WuYun C++ Performance Benchmark ===\n");
    printf("(Release build, single thread, CPU only)\n\n");

    printf("[NeuronPopulation step]\n");
    bench_population(100, 1000);
    bench_population(1000, 1000);
    bench_population(10000, 1000);
    bench_population(100000, 100);
    bench_population(1000000, 10);

    printf("\n[SynapseGroup deliver + compute]\n");
    bench_synapse(1000, 1000, 100, 1000);
    bench_synapse(10000, 10000, 100, 100);
    bench_synapse(100000, 100000, 10, 10);

    printf("\n[STDP batch update]\n");
    bench_stdp(100000, 100);
    bench_stdp(1000000, 10);

    printf("\n=== Done ===\n");
    return 0;
}
