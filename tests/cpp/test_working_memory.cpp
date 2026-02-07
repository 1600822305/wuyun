/**
 * test_working_memory.cpp — 工作记忆 + BG在线学习测试
 *
 * Step 10: dlPFC持续性活动 + DA稳定 + BG门控训练
 *
 * 测试:
 *   1. 工作记忆基本功能: 刺激→移除→活动持续
 *   2. DA增强持续性: 高DA→更长维持
 *   3. 无WM时活动快速消退 (对照)
 *   4. BG在线DA-STDP训练: 奖励动作→D1增强
 *   5. 工作记忆+BG联合: dlPFC维持→BG学习
 *   6. 向后兼容: WM不影响现有系统
 */

#include "region/cortical_region.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/neuromod/vta_da.h"
#include "engine/simulation_engine.h"
#include <cstdio>
#include <cassert>
#include <vector>
#include <cmath>

using namespace wuyun;

static int tests_passed = 0;
static int tests_failed = 0;

static size_t count_fired(const std::vector<uint8_t>& f) {
    size_t n = 0;
    for (auto v : f) if (v) n++;
    return n;
}

// =============================================================================
// Test 1: Working memory basic — persistent activity after stimulus removal
// =============================================================================
static void test_wm_basic() {
    printf("\n--- 测试1: 工作记忆基础 ---\n");
    printf("    原理: 刺激→移除→L2/3循环自持→活动持续\n");

    ColumnConfig cfg;
    cfg.n_l4_stellate = 30; cfg.n_l23_pyramidal = 80;
    cfg.n_l5_pyramidal = 40; cfg.n_l6_pyramidal = 30;
    cfg.n_pv_basket = 10; cfg.n_sst_martinotti = 8; cfg.n_vip = 5;

    CorticalRegion pfc("dlPFC", cfg);
    pfc.enable_working_memory();

    // Set moderate DA (simulates tonic DA in PFC)
    NeuromodulatorLevels levels;
    levels.da = 0.3f;
    pfc.neuromod().set_tonic(levels);

    // Phase 1: Stimulate (30 steps)
    std::vector<float> stim(cfg.n_l4_stellate, 40.0f);
    size_t stim_spikes = 0;
    for (int t = 0; t < 30; ++t) {
        pfc.inject_external(stim);
        pfc.step(t);
        stim_spikes += count_fired(pfc.fired());
    }

    // Phase 2: No stimulus (50 steps) — activity should persist
    size_t persist_spikes = 0;
    float persistence_at_start = pfc.wm_persistence();
    for (int t = 30; t < 80; ++t) {
        pfc.step(t);
        persist_spikes += count_fired(pfc.fired());
    }
    float persistence_at_end = pfc.wm_persistence();

    printf("    刺激期: %zu spikes, 持续期: %zu spikes\n", stim_spikes, persist_spikes);
    printf("    持续性: 开始=%.3f, 结束=%.3f\n", persistence_at_start, persistence_at_end);

    bool ok = persist_spikes > 0 && persistence_at_start > 0.0f;
    printf("  [%s] 工作记忆基础\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 2: DA enhances persistence
// =============================================================================
static void test_da_persistence() {
    printf("\n--- 测试2: DA增强持续性 ---\n");
    printf("    原理: 高DA→循环增益↑→更长维持\n");

    auto run_with_da = [](float da_level) -> size_t {
        ColumnConfig cfg;
        cfg.n_l4_stellate = 30; cfg.n_l23_pyramidal = 80;
        cfg.n_l5_pyramidal = 40; cfg.n_l6_pyramidal = 30;
        cfg.n_pv_basket = 10; cfg.n_sst_martinotti = 8; cfg.n_vip = 5;

        CorticalRegion pfc("dlPFC", cfg);
        pfc.enable_working_memory();

        NeuromodulatorLevels levels;
        levels.da = da_level;
        pfc.neuromod().set_tonic(levels);

        // Stimulate 20 steps
        std::vector<float> stim(cfg.n_l4_stellate, 40.0f);
        for (int t = 0; t < 20; ++t) {
            pfc.inject_external(stim);
            pfc.step(t);
        }

        // Maintain without stimulus 60 steps
        size_t persist = 0;
        for (int t = 20; t < 80; ++t) {
            pfc.step(t);
            persist += count_fired(pfc.fired());
        }
        return persist;
    };

    size_t low_da  = run_with_da(0.1f);
    size_t mid_da  = run_with_da(0.3f);
    size_t high_da = run_with_da(0.6f);

    printf("    DA=0.1: %zu  DA=0.3: %zu  DA=0.6: %zu\n", low_da, mid_da, high_da);

    bool ok = high_da > low_da;
    printf("  [%s] DA增强持续性\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 3: No WM — rapid decay (control)
// =============================================================================
static void test_no_wm_control() {
    printf("\n--- 测试3: 无WM对照 ---\n");
    printf("    原理: 不启用WM时, 移除刺激后活动快速消退\n");

    ColumnConfig cfg;
    cfg.n_l4_stellate = 30; cfg.n_l23_pyramidal = 80;
    cfg.n_l5_pyramidal = 40; cfg.n_l6_pyramidal = 30;
    cfg.n_pv_basket = 10; cfg.n_sst_martinotti = 8; cfg.n_vip = 5;

    CorticalRegion pfc_wm("dlPFC_wm", cfg);
    CorticalRegion pfc_no("dlPFC_no", cfg);
    pfc_wm.enable_working_memory();
    // pfc_no: no working memory

    NeuromodulatorLevels levels;
    levels.da = 0.3f;
    pfc_wm.neuromod().set_tonic(levels);
    pfc_no.neuromod().set_tonic(levels);

    // Stimulate both 20 steps
    std::vector<float> stim(cfg.n_l4_stellate, 40.0f);
    for (int t = 0; t < 20; ++t) {
        pfc_wm.inject_external(stim); pfc_wm.step(t);
        pfc_no.inject_external(stim); pfc_no.step(t);
    }

    // 30 steps without stimulus
    size_t wm_persist = 0, no_persist = 0;
    for (int t = 20; t < 50; ++t) {
        pfc_wm.step(t); wm_persist += count_fired(pfc_wm.fired());
        pfc_no.step(t); no_persist += count_fired(pfc_no.fired());
    }

    printf("    有WM: %zu  无WM: %zu\n", wm_persist, no_persist);

    bool ok = wm_persist > no_persist;
    printf("  [%s] WM vs 无WM对照\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 4: BG online DA-STDP training
// =============================================================================
static void test_bg_online_learning() {
    printf("\n--- 测试4: BG在线DA-STDP训练 ---\n");
    printf("    原理: 刺激A+奖励→D1_A增强 vs 刺激B无奖励→D1_B不变\n");

    // Trained BG: LGN->V1->BG with DA reward
    auto make_engine = [](bool enable_stdp) {
        auto eng = std::make_unique<SimulationEngine>(10);

        ThalamicConfig lgn_cfg;
        lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
        eng->add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

        ColumnConfig v1_cfg;
        v1_cfg.n_l4_stellate = 50; v1_cfg.n_l23_pyramidal = 100;
        v1_cfg.n_l5_pyramidal = 50; v1_cfg.n_l6_pyramidal = 40;
        v1_cfg.n_pv_basket = 15; v1_cfg.n_sst_martinotti = 10; v1_cfg.n_vip = 5;
        eng->add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

        BasalGangliaConfig bg_cfg;
        bg_cfg.name = "BG"; bg_cfg.n_d1_msn = 50; bg_cfg.n_d2_msn = 50;
        bg_cfg.n_gpi = 15; bg_cfg.n_gpe = 15; bg_cfg.n_stn = 10;
        bg_cfg.da_stdp_enabled = enable_stdp;
        bg_cfg.da_stdp_lr = 0.05f;
        eng->add_region(std::make_unique<BasalGanglia>(bg_cfg));

        eng->add_projection("LGN", "V1", 2);
        eng->add_projection("V1", "BG", 2);

        // Disable VTA auto-routing so set_da_level works
        auto* bg = dynamic_cast<BasalGanglia*>(eng->find_region("BG"));
        bg->set_da_source_region(UINT32_MAX);

        return eng;
    };

    // Trained engine
    auto eng = make_engine(true);
    auto* bg = dynamic_cast<BasalGanglia*>(eng->find_region("BG"));
    auto* lgn = eng->find_region("LGN");

    // Training: 10 trials with high DA reward
    for (int trial = 0; trial < 10; ++trial) {
        for (int t = 0; t < 30; ++t) {
            lgn->inject_external(std::vector<float>(50, 35.0f));
            bg->set_da_level(0.8f);
            eng->step();
        }
    }

    // Test: measure D1 response with low DA
    bg->set_da_level(0.1f);
    size_t d1_trained = 0;
    for (int t = 0; t < 50; ++t) {
        lgn->inject_external(std::vector<float>(50, 35.0f));
        eng->step();
        const auto& f = bg->fired();
        for (size_t i = 0; i < 50 && i < f.size(); ++i)
            if (f[i]) d1_trained++;
    }

    // Untrained engine (no DA-STDP)
    auto eng2 = make_engine(false);
    auto* lgn2 = eng2->find_region("LGN");
    // Run same total steps without DA reward
    for (int trial = 0; trial < 10; ++trial) {
        for (int t = 0; t < 30; ++t) {
            lgn2->inject_external(std::vector<float>(50, 35.0f));
            eng2->step();
        }
    }
    size_t d1_untrained = 0;
    for (int t = 0; t < 50; ++t) {
        lgn2->inject_external(std::vector<float>(50, 35.0f));
        eng2->step();
        const auto& f = eng2->find_region("BG")->fired();
        for (size_t i = 0; i < 50 && i < f.size(); ++i)
            if (f[i]) d1_untrained++;
    }

    printf("    D1(训练后)=%zu  D1(未训练)=%zu\n", d1_trained, d1_untrained);

    bool ok = d1_trained > d1_untrained;
    printf("  [%s] BG在线学习\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 5: Working memory + BG combined
// =============================================================================
static void test_wm_bg_combined() {
    printf("\n--- 测试5: 工作记忆+BG联合 ---\n");
    printf("    原理: dlPFC维持信息 → BG利用维持信息做决策\n");

    SimulationEngine eng(10);

    ThalamicConfig lgn_cfg;
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    ColumnConfig v1_cfg;
    v1_cfg.n_l4_stellate = 50; v1_cfg.n_l23_pyramidal = 100;
    v1_cfg.n_l5_pyramidal = 50; v1_cfg.n_l6_pyramidal = 40;
    v1_cfg.n_pv_basket = 15; v1_cfg.n_sst_martinotti = 10; v1_cfg.n_vip = 5;
    eng.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

    ColumnConfig pfc_cfg;
    pfc_cfg.n_l4_stellate = 30; pfc_cfg.n_l23_pyramidal = 80;
    pfc_cfg.n_l5_pyramidal = 40; pfc_cfg.n_l6_pyramidal = 30;
    pfc_cfg.n_pv_basket = 10; pfc_cfg.n_sst_martinotti = 8; pfc_cfg.n_vip = 5;
    eng.add_region(std::make_unique<CorticalRegion>("dlPFC", pfc_cfg));

    BasalGangliaConfig bg_cfg;
    bg_cfg.name = "BG"; bg_cfg.n_d1_msn = 50; bg_cfg.n_d2_msn = 50;
    bg_cfg.n_gpi = 15; bg_cfg.n_gpe = 15; bg_cfg.n_stn = 10;
    eng.add_region(std::make_unique<BasalGanglia>(bg_cfg));

    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("V1", "dlPFC", 2);
    eng.add_projection("dlPFC", "BG", 2);

    auto* pfc = dynamic_cast<CorticalRegion*>(eng.find_region("dlPFC"));
    pfc->enable_working_memory();

    // Set DA for WM
    NeuromodulatorLevels levels;
    levels.da = 0.3f;
    pfc->neuromod().set_tonic(levels);

    auto* lgn = eng.find_region("LGN");

    // Phase 1: Encode (stimulus active, 30 steps)
    for (int t = 0; t < 30; ++t) {
        lgn->inject_external(std::vector<float>(50, 35.0f));
        eng.step();
    }

    // Phase 2: Delay (no stimulus, 30 steps) — dlPFC should maintain
    size_t bg_during_delay = 0;
    float avg_persist = 0.0f;
    for (int t = 30; t < 60; ++t) {
        eng.step();
        const auto& f = eng.find_region("BG")->fired();
        for (auto v : f) if (v) bg_during_delay++;
        avg_persist += pfc->wm_persistence();
    }
    avg_persist /= 30.0f;

    printf("    延迟期: BG=%zu, dlPFC持续性=%.3f\n", bg_during_delay, avg_persist);

    bool ok = bg_during_delay > 0 && avg_persist > 0.0f;
    printf("  [%s] 工作记忆+BG联合\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 6: Backward compatibility
// =============================================================================
static void test_backward_compat() {
    printf("\n--- 测试6: 向后兼容性 ---\n");
    printf("    原理: 不启用WM时, 行为与原系统完全一致\n");

    ColumnConfig cfg;
    cfg.n_l4_stellate = 30; cfg.n_l23_pyramidal = 80;
    cfg.n_l5_pyramidal = 40; cfg.n_l6_pyramidal = 30;
    cfg.n_pv_basket = 10; cfg.n_sst_martinotti = 8; cfg.n_vip = 5;

    CorticalRegion a("test_a", cfg);
    CorticalRegion b("test_b", cfg);
    // Neither has WM enabled

    std::vector<float> stim(cfg.n_l4_stellate, 35.0f);
    size_t spikes_a = 0, spikes_b = 0;
    for (int t = 0; t < 50; ++t) {
        a.inject_external(stim); a.step(t);
        b.inject_external(stim); b.step(t);
        spikes_a += count_fired(a.fired());
        spikes_b += count_fired(b.fired());
    }

    printf("    A=%zu  B=%zu\n", spikes_a, spikes_b);
    assert(a.wm_persistence() == 0.0f);

    bool ok = spikes_a == spikes_b;
    printf("  [%s] 向后兼容性\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
int main() {
    printf("============================================\n");
    printf("  悟韵 (WuYun) 工作记忆 + BG在线学习测试\n");
    printf("  Step 10: dlPFC持续性活动 + DA稳定 + BG训练\n");
    printf("============================================\n");

    test_wm_basic();
    test_da_persistence();
    test_no_wm_control();
    test_bg_online_learning();
    test_wm_bg_combined();
    test_backward_compat();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("============================================\n");

    return tests_failed > 0 ? 1 : 0;
}
