/**
 * 单神经元 + NeuronPopulation 单元测试
 *
 * 验证三种发放模式:
 *   1. REGULAR — 只有前馈输入
 *   2. BURST — 前馈 + 反馈同时激活
 *   3. SILENCE — 无输入
 *
 * 不依赖 Google Test，使用简单 assert 宏。
 * 后续可迁移到 GTest。
 */

#include "core/types.h"
#include "core/neuron.h"
#include "core/population.h"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace wuyun;

// =============================================================================
// 辅助
// =============================================================================

#define WTEST(name) static void name()
#define RUN(name) do { printf("  [RUN]  %s ...", #name); name(); printf(" PASS\n"); } while(0)

// =============================================================================
// 测试: 单神经元
// =============================================================================

WTEST(test_silence) {
    auto params = L23_PYRAMIDAL_PARAMS();
    NeuronState state;
    state.v_soma   = params.somatic.v_rest;
    state.v_apical = params.somatic.v_rest;

    // 无输入, 100 步应该一直沉默
    for (int t = 0; t < 100; ++t) {
        SpikeType st = neuron_step(state, params, 0.0f, 0.0f, 0.0f, t);
        assert(st == SpikeType::NONE);
    }
    // 膜电位应该接近静息电位
    assert(std::abs(state.v_soma - params.somatic.v_rest) < 1.0f);
}

WTEST(test_regular_spike) {
    auto params = L23_PYRAMIDAL_PARAMS();
    NeuronState state;
    state.v_soma   = params.somatic.v_rest;
    state.v_apical = params.somatic.v_rest;

    // 只注入基底树突电流 (前馈), 不注入顶端树突 (无反馈)
    // 应该产生 REGULAR 脉冲
    bool found_regular = false;
    for (int t = 0; t < 200; ++t) {
        SpikeType st = neuron_step(state, params, 15.0f, 0.0f, 0.0f, t);
        if (st == SpikeType::REGULAR) {
            found_regular = true;
            // 发放后膜电位应该被重置
            assert(std::abs(state.v_soma - params.somatic.v_reset) < 0.1f);
            break;
        }
    }
    assert(found_regular);
}

WTEST(test_burst_spike) {
    auto params = L23_PYRAMIDAL_PARAMS();
    NeuronState state;
    state.v_soma   = params.somatic.v_rest;
    state.v_apical = params.somatic.v_rest;

    // 同时注入基底树突 + 顶端树突电流 (前馈 + 反馈)
    // 应该产生 BURST_START → BURST_CONTINUE → BURST_END 序列
    bool found_burst_start    = false;
    bool found_burst_continue = false;
    bool found_burst_end      = false;

    for (int t = 0; t < 300; ++t) {
        SpikeType st = neuron_step(state, params, 15.0f, 20.0f, 0.0f, t);
        if (st == SpikeType::BURST_START)    found_burst_start = true;
        if (st == SpikeType::BURST_CONTINUE) found_burst_continue = true;
        if (st == SpikeType::BURST_END)      found_burst_end = true;

        if (found_burst_end) break;
    }
    assert(found_burst_start);
    assert(found_burst_continue);
    assert(found_burst_end);
}

WTEST(test_refractory_period) {
    auto params = L23_PYRAMIDAL_PARAMS();
    NeuronState state;
    state.v_soma   = params.somatic.v_rest;
    state.v_apical = params.somatic.v_rest;

    // 找到第一个 REGULAR 脉冲
    int spike_time = -1;
    for (int t = 0; t < 200; ++t) {
        SpikeType st = neuron_step(state, params, 15.0f, 0.0f, 0.0f, t);
        if (st == SpikeType::REGULAR) {
            spike_time = t;
            break;
        }
    }
    assert(spike_time >= 0);

    // 不应期内不应发放 (refractory_period = 3)
    for (int t = spike_time + 1; t <= spike_time + params.somatic.refractory_period; ++t) {
        SpikeType st = neuron_step(state, params, 15.0f, 0.0f, 0.0f, t);
        assert(st == SpikeType::NONE);
    }
}

WTEST(test_adaptation) {
    auto params = L23_PYRAMIDAL_PARAMS();
    NeuronState state;
    state.v_soma   = params.somatic.v_rest;
    state.v_apical = params.somatic.v_rest;

    // 持续注入恒定电流, 记录发放间隔 (ISI)
    // 适应机制应该使 ISI 逐渐增大
    std::vector<int> spike_times;
    for (int t = 0; t < 500; ++t) {
        SpikeType st = neuron_step(state, params, 12.0f, 0.0f, 0.0f, t);
        if (st == SpikeType::REGULAR) {
            spike_times.push_back(t);
        }
    }

    // 至少应该有 3 个脉冲来比较 ISI
    if (spike_times.size() >= 3) {
        int isi_first = spike_times[1] - spike_times[0];
        int isi_last  = spike_times[spike_times.size()-1] - spike_times[spike_times.size()-2];
        // 后期 ISI >= 前期 ISI (适应导致减慢)
        assert(isi_last >= isi_first);
    }
}

// =============================================================================
// 测试: NeuronPopulation 向量化
// =============================================================================

WTEST(test_population_silence) {
    auto params = L23_PYRAMIDAL_PARAMS();
    NeuronPopulation pop(100, params);

    // 无输入, 不应有任何发放
    for (int t = 0; t < 50; ++t) {
        size_t n_fired = pop.step(t);
        assert(n_fired == 0);
    }
}

WTEST(test_population_regular) {
    auto params = L23_PYRAMIDAL_PARAMS();
    NeuronPopulation pop(10, params);

    // 只注入基底树突电流
    bool found_regular = false;
    for (int t = 0; t < 200; ++t) {
        for (size_t i = 0; i < pop.size(); ++i) {
            pop.inject_basal(i, 15.0f);
        }
        size_t n_fired = pop.step(t);
        if (n_fired > 0) {
            // 检查发放类型是 REGULAR
            for (size_t i = 0; i < pop.size(); ++i) {
                if (pop.fired()[i]) {
                    assert(pop.spike_type()[i] == static_cast<int8_t>(SpikeType::REGULAR));
                    found_regular = true;
                }
            }
            break;
        }
    }
    assert(found_regular);
}

WTEST(test_population_burst) {
    auto params = L23_PYRAMIDAL_PARAMS();
    NeuronPopulation pop(10, params);

    // 同时注入基底 + 顶端电流
    bool found_burst = false;
    for (int t = 0; t < 300; ++t) {
        for (size_t i = 0; i < pop.size(); ++i) {
            pop.inject_basal(i, 15.0f);
            pop.inject_apical(i, 20.0f);
        }
        pop.step(t);
        for (size_t i = 0; i < pop.size(); ++i) {
            if (pop.spike_type()[i] == static_cast<int8_t>(SpikeType::BURST_START)) {
                found_burst = true;
                break;
            }
        }
        if (found_burst) break;
    }
    assert(found_burst);
}

WTEST(test_population_consistency) {
    // 验证 Population 和单神经元 step 结果一致
    auto params = L23_PYRAMIDAL_PARAMS();

    NeuronPopulation pop(1, params);
    NeuronState single;
    single.v_soma   = params.somatic.v_rest;
    single.v_apical = params.somatic.v_rest;

    for (int t = 0; t < 100; ++t) {
        pop.inject_basal(0, 10.0f);
        pop.step(t);

        SpikeType st = neuron_step(single, params, 10.0f, 0.0f, 0.0f, t);

        // 膜电位应该非常接近
        float diff = std::abs(pop.v_soma()[0] - single.v_soma);
        assert(diff < 0.01f);

        // 脉冲类型应该一致
        assert(pop.spike_type()[0] == static_cast<int8_t>(st));
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== WuYun C++ Neuron Tests ===\n");

    printf("\n[Single Neuron]\n");
    RUN(test_silence);
    RUN(test_regular_spike);
    RUN(test_burst_spike);
    RUN(test_refractory_period);
    RUN(test_adaptation);

    printf("\n[NeuronPopulation]\n");
    RUN(test_population_silence);
    RUN(test_population_regular);
    RUN(test_population_burst);
    RUN(test_population_consistency);

    printf("\n=== ALL %d TESTS PASSED ===\n", 9);
    return 0;
}
