/**
 * Step 12: 注意力机制测试
 *
 * 测试:
 *   1. 基础注意力增益: gain>1 → V1响应增强
 *   2. 选择性注意: V1注意+V2忽略 → V1>V2
 *   3. VIP去抑制回路: attention → VIP→SST↓ → L2/3 burst↑
 *   4. 注意力+预测编码交互: 注意力增强误差传播
 *   5. ACh精度调制: 高ACh → prior↓ → 更多感觉驱动
 *   6. NE感觉精度: 高NE → sensory精度↑ → 响应增强
 *   7. 向后兼容: gain=1.0 行为不变
 */

#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#endif
#include <vector>
#include <cmath>
#include <memory>

#include "engine/simulation_engine.h"
#include "core/neuromodulator.h"
#include "region/cortical_region.h"
#include "region/subcortical/thalamic_relay.h"

using namespace wuyun;

static int tests_passed = 0;
static int tests_failed = 0;

static size_t count_fired(const std::vector<uint8_t>& f) {
    size_t n = 0; for (auto x : f) if (x) n++; return n;
}

static ColumnConfig make_v1_cfg() {
    ColumnConfig c;
    c.n_l4_stellate = 50; c.n_l23_pyramidal = 100;
    c.n_l5_pyramidal = 50; c.n_l6_pyramidal = 40;
    c.n_pv_basket = 15; c.n_sst_martinotti = 10; c.n_vip = 5;
    return c;
}

// =============================================================================
// Test 1: Basic attention gain — higher gain → more spikes
// =============================================================================
static void test_basic_gain() {
    printf("\n--- 测试1: 基础注意力增益 ---\n");
    printf("    原理: gain>1 → L4 PSP放大 + VIP去抑制 → 响应增强\n");

    auto run_with_gain = [](float gain) -> size_t {
        SimulationEngine eng(10);
        ThalamicConfig lgn; lgn.name = "LGN"; lgn.n_relay = 50; lgn.n_trn = 15;
        eng.add_region(std::make_unique<ThalamicRelay>(lgn));
        eng.add_region(std::make_unique<CorticalRegion>("V1", make_v1_cfg()));
        eng.add_projection("LGN", "V1", 2);

        auto* v1 = dynamic_cast<CorticalRegion*>(eng.find_region("V1"));
        v1->set_attention_gain(gain);

        size_t total = 0;
        for (int t = 0; t < 100; ++t) {
            eng.find_region("LGN")->inject_external(std::vector<float>(50, 30.0f));
            eng.step();
            total += count_fired(v1->fired());
        }
        return total;
    };

    size_t normal = run_with_gain(1.0f);
    size_t attend = run_with_gain(1.5f);
    size_t ignore = run_with_gain(0.5f);

    printf("    V1(忽略0.5)=%zu  V1(正常1.0)=%zu  V1(注意1.5)=%zu\n",
           ignore, normal, attend);

    bool ok = attend > normal && normal > ignore;
    printf("  [%s] 注意力增益: attend > normal > ignore\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 2: Selective attention — attend V1, ignore V2
// =============================================================================
static void test_selective_attention() {
    printf("\n--- 测试2: 选择性注意 ---\n");
    printf("    原理: V1 gain=1.5 + V2 gain=0.7 → V1放大, V2抑制\n");

    SimulationEngine eng(10);
    ThalamicConfig lgn; lgn.name = "LGN"; lgn.n_relay = 50; lgn.n_trn = 15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn));
    eng.add_region(std::make_unique<CorticalRegion>("V1", make_v1_cfg()));
    eng.add_region(std::make_unique<CorticalRegion>("V2", make_v1_cfg()));
    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("LGN", "V2", 2);

    auto* v1 = dynamic_cast<CorticalRegion*>(eng.find_region("V1"));
    auto* v2 = dynamic_cast<CorticalRegion*>(eng.find_region("V2"));

    // Same input, different attention
    v1->set_attention_gain(1.5f);  // Attend
    v2->set_attention_gain(0.7f);  // Ignore

    size_t v1_spikes = 0, v2_spikes = 0;
    for (int t = 0; t < 100; ++t) {
        eng.find_region("LGN")->inject_external(std::vector<float>(50, 30.0f));
        eng.step();
        v1_spikes += count_fired(v1->fired());
        v2_spikes += count_fired(v2->fired());
    }

    printf("    V1(注意)=%zu  V2(忽略)=%zu  比率=%.2f\n",
           v1_spikes, v2_spikes,
           v2_spikes > 0 ? (float)v1_spikes / v2_spikes : 999.0f);

    bool ok = v1_spikes > v2_spikes;
    printf("  [%s] 选择性注意: V1(注意) > V2(忽略)\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 3: VIP disinhibition circuit (via SpikeBus path)
// =============================================================================
static void test_vip_disinhibition() {
    printf("\n--- 测试3: VIP去抑制回路 ---\n");
    printf("    原理: LGN→V1(SpikeBus) + attention→VIP→SST↓→L2/3增强\n");

    auto run_with_gain = [](float gain) -> size_t {
        SimulationEngine eng(10);
        ThalamicConfig lgn; lgn.name = "LGN"; lgn.n_relay = 50; lgn.n_trn = 15;
        eng.add_region(std::make_unique<ThalamicRelay>(lgn));
        eng.add_region(std::make_unique<CorticalRegion>("V1", make_v1_cfg()));
        eng.add_projection("LGN", "V1", 2);

        auto* v1 = dynamic_cast<CorticalRegion*>(eng.find_region("V1"));
        v1->set_attention_gain(gain);

        size_t total = 0;
        for (int t = 0; t < 100; ++t) {
            eng.find_region("LGN")->inject_external(std::vector<float>(50, 30.0f));
            eng.step();
            total += count_fired(v1->fired());
        }
        return total;
    };

    size_t no_att  = run_with_gain(1.0f);
    size_t med_att = run_with_gain(1.3f);
    size_t hi_att  = run_with_gain(2.0f);

    printf("    V1(无注意)=%zu  V1(中注意1.3)=%zu  V1(高注意2.0)=%zu\n",
           no_att, med_att, hi_att);

    bool ok = hi_att > no_att;
    printf("  [%s] VIP去抑制: 高注意 > 无注意\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 4: Attention + Predictive coding — attend amplifies sensory response
// =============================================================================
static void test_attention_pc() {
    printf("\n--- 测试4: 注意力×预测编码 ---\n");
    printf("    原理: 注意力 + PC启用 → 感觉精度增强 → V1响应增强\n");

    auto run_pc = [](float att_gain) -> size_t {
        SimulationEngine eng(10);
        ThalamicConfig lgn; lgn.name = "LGN"; lgn.n_relay = 50; lgn.n_trn = 15;
        eng.add_region(std::make_unique<ThalamicRelay>(lgn));
        eng.add_region(std::make_unique<CorticalRegion>("V1", make_v1_cfg()));
        eng.add_projection("LGN", "V1", 2);

        auto* v1 = dynamic_cast<CorticalRegion*>(eng.find_region("V1"));
        v1->enable_predictive_coding();
        v1->set_attention_gain(att_gain);

        size_t total = 0;
        for (int t = 0; t < 100; ++t) {
            eng.find_region("LGN")->inject_external(std::vector<float>(50, 30.0f));
            eng.step();
            total += count_fired(v1->fired());
        }
        return total;
    };

    size_t v1_normal = run_pc(1.0f);
    size_t v1_attend = run_pc(1.5f);

    printf("    V1(正常+PC)=%zu  V1(注意+PC)=%zu\n", v1_normal, v1_attend);

    bool ok = v1_attend > v1_normal;
    printf("  [%s] 注意力+PC增强感觉响应\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 5: ACh prior precision — high ACh reduces prediction suppression
// =============================================================================
static void test_ach_precision() {
    printf("\n--- 测试5: ACh先验精度调制 ---\n");
    printf("    原理: 高ACh → prior精度↓ → 预测抑制减弱 → 更多感觉驱动\n");

    auto run_with_ach = [](float ach) -> size_t {
        SimulationEngine eng(10);
        ThalamicConfig lgn; lgn.name = "LGN"; lgn.n_relay = 50; lgn.n_trn = 15;
        eng.add_region(std::make_unique<ThalamicRelay>(lgn));
        ColumnConfig v2_cfg = make_v1_cfg();
        eng.add_region(std::make_unique<CorticalRegion>("V1", make_v1_cfg()));
        eng.add_region(std::make_unique<CorticalRegion>("V2", v2_cfg));
        eng.add_projection("LGN", "V1", 2);
        eng.add_projection("V1", "V2", 2);   // feedforward
        eng.add_projection("V2", "V1", 3);   // feedback (prediction)

        auto* v1 = dynamic_cast<CorticalRegion*>(eng.find_region("V1"));
        v1->enable_predictive_coding();
        v1->add_feedback_source(eng.find_region("V2")->region_id());

        NeuromodulatorLevels levels;
        levels.ach = ach;
        v1->neuromod().set_tonic(levels);

        size_t total = 0;
        for (int t = 0; t < 150; ++t) {
            eng.find_region("LGN")->inject_external(std::vector<float>(50, 30.0f));
            eng.step();
            if (t >= 50) total += count_fired(v1->fired());  // Skip transient
        }
        return total;
    };

    size_t low_ach  = run_with_ach(0.1f);
    size_t high_ach = run_with_ach(0.8f);

    printf("    V1(ACh=0.1)=%zu  V1(ACh=0.8)=%zu\n", low_ach, high_ach);
    printf("    prior精度: ACh=0.1→%.2f  ACh=0.8→%.2f\n",
           1.0f - 0.8f * 0.1f, 1.0f - 0.8f * 0.8f);

    bool ok = high_ach > low_ach;
    printf("  [%s] ACh↑ → prior↓ → 更多感觉驱动\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 6: NE sensory precision — via SpikeBus PSP path
// =============================================================================
static void test_ne_precision() {
    printf("\n--- 测试6: NE感觉精度调制 ---\n");
    printf("    原理: 高NE → sensory精度↑ → PSP放大 → V1响应增强\n");

    auto run_with_ne = [](float ne) -> size_t {
        SimulationEngine eng(10);
        ThalamicConfig lgn; lgn.name = "LGN"; lgn.n_relay = 50; lgn.n_trn = 15;
        eng.add_region(std::make_unique<ThalamicRelay>(lgn));
        eng.add_region(std::make_unique<CorticalRegion>("V1", make_v1_cfg()));
        eng.add_projection("LGN", "V1", 2);

        auto* v1 = dynamic_cast<CorticalRegion*>(eng.find_region("V1"));
        NeuromodulatorLevels levels;
        levels.ne = ne;
        v1->neuromod().set_tonic(levels);

        size_t total = 0;
        for (int t = 0; t < 100; ++t) {
            eng.find_region("LGN")->inject_external(std::vector<float>(50, 30.0f));
            eng.step();
            total += count_fired(v1->fired());
        }
        return total;
    };

    size_t low_ne  = run_with_ne(0.1f);
    size_t mid_ne  = run_with_ne(0.5f);
    size_t high_ne = run_with_ne(0.9f);

    printf("    V1(NE=0.1)=%zu  V1(NE=0.5)=%zu  V1(NE=0.9)=%zu\n",
           low_ne, mid_ne, high_ne);

    bool ok = high_ne > low_ne;
    printf("  [%s] NE↑ → sensory精度↑ → 响应增强\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 7: Backward compatibility — gain=1.0 doesn't change behavior
// =============================================================================
static void test_backward_compat() {
    printf("\n--- 测试7: 向后兼容 ---\n");
    printf("    原理: gain=1.0时, 行为与不设置注意力完全一致\n");

    ColumnConfig cfg = make_v1_cfg();

    // Run A: explicit gain=1.0
    CorticalRegion a("V1_a", cfg);
    a.set_attention_gain(1.0f);
    size_t spikes_a = 0;
    for (int t = 0; t < 80; ++t) {
        a.inject_external(std::vector<float>(cfg.n_l4_stellate, 25.0f));
        a.step(t);
        spikes_a += count_fired(a.fired());
    }

    // Run B: default (no set_attention_gain call)
    CorticalRegion b("V1_b", cfg);
    size_t spikes_b = 0;
    for (int t = 0; t < 80; ++t) {
        b.inject_external(std::vector<float>(cfg.n_l4_stellate, 25.0f));
        b.step(t);
        spikes_b += count_fired(b.fired());
    }

    printf("    V1(gain=1.0)=%zu  V1(默认)=%zu\n", spikes_a, spikes_b);

    bool ok = spikes_a == spikes_b;
    printf("  [%s] 向后兼容: gain=1.0 == 默认\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) Step 12 注意力机制测试\n");
    printf("  Top-down gain + VIP去抑制 + ACh/NE精度\n");
    printf("============================================\n");

    test_basic_gain();
    test_selective_attention();
    test_vip_disinhibition();
    test_attention_pc();
    test_ach_precision();
    test_ne_precision();
    test_backward_compat();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("============================================\n");

    return tests_failed > 0 ? 1 : 0;
}
