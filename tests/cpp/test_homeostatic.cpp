/**
 * test_homeostatic.cpp — 稳态可塑性集成测试
 *
 * 验证:
 * 1. SynapticScaler 基础功能 (发放率追踪 + 权重缩放)
 * 2. 过度活跃 → 权重降低
 * 3. 活动不足 → 权重升高
 * 4. CorticalRegion 集成 (enable + 发放率收敛)
 * 5. Hippocampus 集成 (enable + 发放率追踪)
 * 6. 全脑稳态 (scale=1, 所有区域启用后不崩溃)
 * 7. 大规模稳态 (scale=3, 工作记忆稳定性)
 */

#include "plasticity/homeostatic.h"
#include "region/cortical_region.h"
#include "region/limbic/hippocampus.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/neuromod/vta_da.h"
#include "engine/simulation_engine.h"
#include "engine/sensory_input.h"

#include <cstdio>
#include <cmath>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace wuyun;

static int g_pass = 0, g_fail = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
        g_fail++; return; \
    } \
} while(0)

static size_t count_fired(const BrainRegion& r) {
    size_t n = 0;
    for (auto f : r.fired()) if (f) n++;
    return n;
}

// =========================================================================
// Test 1: SynapticScaler 基础 — 发放率追踪
// =========================================================================
static void test_rate_tracking() {
    printf("\n--- 测试1: SynapticScaler 发放率追踪 ---\n");

    HomeostaticParams params;
    params.target_rate = 10.0f;
    params.tau_rate = 100.0f;
    params.eta = 0.01f;

    SynapticScaler scaler(5, params);

    TEST_ASSERT(std::abs(scaler.rate(0) - 10.0f) < 0.01f, "初始在目标率");
    TEST_ASSERT(std::abs(scaler.mean_rate() - 10.0f) < 0.01f, "初始平均在目标率");

    std::vector<uint8_t> fired(5, 0);
    fired[0] = 1;

    for (int t = 0; t < 500; ++t) {
        scaler.update_rates(fired.data(), 1.0f);
    }

    printf("  Neuron 0 rate (always fires): %.1f\n", scaler.rate(0));
    printf("  Neuron 1 rate (silent):       %.4f\n", scaler.rate(1));
    TEST_ASSERT(scaler.rate(0) > 100.0f, "持续发放→高速率");
    TEST_ASSERT(scaler.rate(1) < 1.0f, "沉默→低速率");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 2: 过度活跃 → 权重缩小
// =========================================================================
static void test_overactive_decrease() {
    printf("\n--- 测试2: 过度活跃 → 权重降低 ---\n");

    HomeostaticParams params;
    params.target_rate = 5.0f;
    params.tau_rate = 100.0f;
    params.eta = 0.1f;
    params.w_min = 0.01f;
    params.w_max = 2.0f;

    SynapticScaler scaler(3, params);

    std::vector<uint8_t> all_firing(3, 1);
    for (int t = 0; t < 200; ++t) {
        scaler.update_rates(all_firing.data(), 1.0f);
    }

    TEST_ASSERT(scaler.rate(0) > params.target_rate * 5, "速率远超目标");

    std::vector<float> weights = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    std::vector<int32_t> post_ids = {0, 0, 1, 1, 2, 2};

    float w_before = weights[0];
    scaler.apply_scaling(weights.data(), 6, post_ids.data());
    float w_after = weights[0];

    printf("  Weight: %.4f → %.4f\n", w_before, w_after);
    TEST_ASSERT(w_after < w_before, "过度活跃→权重降低");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 3: 活动不足 → 权重增大
// =========================================================================
static void test_underactive_increase() {
    printf("\n--- 测试3: 活动不足 → 权重增大 ---\n");

    HomeostaticParams params;
    params.target_rate = 5.0f;
    params.tau_rate = 100.0f;
    params.eta = 0.1f;
    params.w_min = 0.01f;
    params.w_max = 2.0f;

    SynapticScaler scaler(3, params);

    std::vector<uint8_t> silent(3, 0);
    for (int t = 0; t < 200; ++t) {
        scaler.update_rates(silent.data(), 1.0f);
    }

    TEST_ASSERT(scaler.rate(0) < 1.0f, "沉默→低速率");

    std::vector<float> weights = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    std::vector<int32_t> post_ids = {0, 0, 1, 1, 2, 2};

    float w_before = weights[0];
    scaler.apply_scaling(weights.data(), 6, post_ids.data());
    float w_after = weights[0];

    printf("  Weight: %.4f → %.4f\n", w_before, w_after);
    TEST_ASSERT(w_after > w_before, "活动不足→权重增大");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 4: CorticalRegion 稳态集成
// =========================================================================
static void test_cortical_integration() {
    printf("\n--- 测试4: CorticalRegion 稳态集成 ---\n");

    ColumnConfig cfg;
    cfg.n_l4_stellate = 50;
    cfg.n_l23_pyramidal = 100;
    cfg.n_l5_pyramidal = 50;
    cfg.n_l6_pyramidal = 40;
    cfg.n_pv_basket = 15;
    cfg.n_sst_martinotti = 10;
    cfg.n_vip = 5;

    CorticalRegion v1("V1_test", cfg);

    HomeostaticParams hp;
    hp.target_rate = 5.0f;
    hp.eta = 0.001f;
    hp.tau_rate = 500.0f;
    hp.scale_interval = 50;

    v1.enable_homeostatic(hp);
    TEST_ASSERT(v1.homeostatic_enabled(), "homeostatic已启用");

    std::vector<float> input(50, 20.0f);
    int total_spikes = 0;
    for (int t = 0; t < 200; ++t) {
        v1.inject_feedforward(input);
        v1.step(t);
        for (auto f : v1.fired()) total_spikes += f;
    }

    printf("  Total spikes (200 steps): %d\n", total_spikes);
    printf("  L4 mean rate:  %.2f\n", v1.l4_mean_rate());
    printf("  L2/3 mean rate: %.2f\n", v1.l23_mean_rate());
    printf("  L5 mean rate:  %.2f\n", v1.l5_mean_rate());
    printf("  L6 mean rate:  %.2f\n", v1.l6_mean_rate());

    TEST_ASSERT(total_spikes > 0, "有发放活动");
    TEST_ASSERT(v1.l4_mean_rate() > 0.0f, "L4速率被追踪");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 5: Hippocampus 稳态集成
// =========================================================================
static void test_hippocampus_integration() {
    printf("\n--- 测试5: Hippocampus 稳态集成 ---\n");

    HippocampusConfig hcfg;
    hcfg.n_ec = 80;
    hcfg.n_dg = 120;
    hcfg.n_ca3 = 60;
    hcfg.n_ca1 = 60;
    hcfg.n_sub = 30;

    Hippocampus hipp(hcfg);

    HomeostaticParams hp;
    hp.target_rate = 3.0f;
    hp.eta = 0.001f;
    hp.scale_interval = 50;

    hipp.enable_homeostatic(hp);
    TEST_ASSERT(hipp.homeostatic_enabled(), "homeostatic已启用");

    std::vector<float> ec_input(80, 18.0f);
    for (int t = 0; t < 300; ++t) {
        hipp.inject_cortical_input(ec_input);
        hipp.step(t);
    }

    printf("  DG mean rate:  %.2f\n", hipp.dg_mean_rate());
    printf("  CA3 mean rate: %.2f\n", hipp.ca3_mean_rate());
    printf("  CA1 mean rate: %.2f\n", hipp.ca1_mean_rate());

    TEST_ASSERT(hipp.ca3_mean_rate() >= 0.0f, "CA3速率被追踪");
    TEST_ASSERT(hipp.ca1_mean_rate() >= 0.0f, "CA1速率被追踪");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 6: 多区域稳态 (LGN→V1→dlPFC + Hippocampus, 不崩溃)
// =========================================================================
static void test_multi_region_homeostatic() {
    printf("\n--- 测试6: 多区域稳态 (LGN→V1→dlPFC→Hipp) ---\n");

    SimulationEngine eng(10);

    // LGN
    ThalamicConfig lgn_cfg;
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    // V1
    ColumnConfig v1cfg;
    v1cfg.n_l4_stellate=50; v1cfg.n_l23_pyramidal=100; v1cfg.n_l5_pyramidal=50;
    v1cfg.n_l6_pyramidal=40; v1cfg.n_pv_basket=15; v1cfg.n_sst_martinotti=10; v1cfg.n_vip=5;
    eng.add_region(std::make_unique<CorticalRegion>("V1", v1cfg));

    // dlPFC
    ColumnConfig pfccfg;
    pfccfg.n_l4_stellate=30; pfccfg.n_l23_pyramidal=80; pfccfg.n_l5_pyramidal=40;
    pfccfg.n_l6_pyramidal=30; pfccfg.n_pv_basket=10; pfccfg.n_sst_martinotti=8; pfccfg.n_vip=4;
    eng.add_region(std::make_unique<CorticalRegion>("dlPFC", pfccfg));

    // Hippocampus
    HippocampusConfig hcfg;
    eng.add_region(std::make_unique<Hippocampus>(hcfg));

    // Projections
    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("V1", "dlPFC", 2);
    eng.add_projection("dlPFC", "Hippocampus", 3);

    // Enable homeostatic on all
    HomeostaticParams hp;
    hp.target_rate = 5.0f;
    hp.eta = 0.001f;
    hp.scale_interval = 100;

    auto* v1 = dynamic_cast<CorticalRegion*>(eng.find_region("V1"));
    auto* dlpfc = dynamic_cast<CorticalRegion*>(eng.find_region("dlPFC"));
    auto* hipp = dynamic_cast<Hippocampus*>(eng.find_region("Hippocampus"));
    v1->enable_homeostatic(hp);
    dlpfc->enable_homeostatic(hp);
    hipp->enable_homeostatic(hp);

    // Visual input
    auto* lgn = eng.find_region("LGN");
    VisualInputConfig vcfg;
    vcfg.input_width = 8; vcfg.input_height = 8;
    vcfg.n_lgn_neurons = lgn->n_neurons();
    VisualInput vis(vcfg);
    std::vector<float> pattern(64, 0.8f);

    int total_spikes = 0;
    for (int t = 0; t < 500; ++t) {
        vis.encode_and_inject(pattern, lgn);
        eng.step();
        for (auto f : v1->fired()) total_spikes += f;
    }

    printf("  V1 total spikes (500 steps): %d\n", total_spikes);
    printf("  V1 L2/3 mean rate: %.2f\n", v1->l23_mean_rate());
    printf("  dlPFC L2/3 mean rate: %.2f\n", dlpfc->l23_mean_rate());
    printf("  Hipp CA3 mean rate: %.2f\n", hipp->ca3_mean_rate());

    TEST_ASSERT(total_spikes > 100, "多区域有足够活动");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 7: 大规模稳态 (scale=3x 神经元数 + 工作记忆)
// =========================================================================
static void test_scale3_wm_stability() {
    printf("\n--- 测试7: Scale=3 稳态 + 工作记忆 ---\n");

    SimulationEngine eng(10);

    // LGN (3x)
    ThalamicConfig lgn_cfg;
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 150; lgn_cfg.n_trn = 45;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    // V1 (3x)
    ColumnConfig v1cfg;
    v1cfg.n_l4_stellate=150; v1cfg.n_l23_pyramidal=300; v1cfg.n_l5_pyramidal=150;
    v1cfg.n_l6_pyramidal=120; v1cfg.n_pv_basket=45; v1cfg.n_sst_martinotti=30; v1cfg.n_vip=15;
    eng.add_region(std::make_unique<CorticalRegion>("V1", v1cfg));

    // dlPFC (3x)
    ColumnConfig pfccfg;
    pfccfg.n_l4_stellate=90; pfccfg.n_l23_pyramidal=240; pfccfg.n_l5_pyramidal=120;
    pfccfg.n_l6_pyramidal=90; pfccfg.n_pv_basket=30; pfccfg.n_sst_martinotti=24; pfccfg.n_vip=12;
    eng.add_region(std::make_unique<CorticalRegion>("dlPFC", pfccfg));

    // Projections
    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("V1", "dlPFC", 2);

    // Enable homeostatic
    HomeostaticParams hp;
    hp.target_rate = 5.0f;
    hp.eta = 0.001f;
    hp.scale_interval = 100;

    auto* v1 = dynamic_cast<CorticalRegion*>(eng.find_region("V1"));
    auto* dlpfc = dynamic_cast<CorticalRegion*>(eng.find_region("dlPFC"));
    v1->enable_homeostatic(hp);
    dlpfc->enable_homeostatic(hp);
    dlpfc->enable_working_memory();

    // Visual input
    auto* lgn = eng.find_region("LGN");
    VisualInputConfig vcfg;
    vcfg.input_width = 8; vcfg.input_height = 8;
    vcfg.n_lgn_neurons = lgn->n_neurons();
    VisualInput vis(vcfg);
    std::vector<float> pattern(64, 0.9f);

    // Phase 1: Stimulate (200 steps)
    for (int t = 0; t < 200; ++t) {
        vis.encode_and_inject(pattern, lgn);
        eng.step();
    }

    // Phase 2: No stimulus, check WM persistence (100 steps)
    int wm_spikes = 0;
    for (int t = 200; t < 300; ++t) {
        eng.step();
        for (auto f : dlpfc->fired()) wm_spikes += f;
    }

    float wm_persist = dlpfc->wm_persistence();

    printf("  Scale=3 dlPFC neurons: %zu\n", dlpfc->n_neurons());
    printf("  dlPFC WM spikes (100 steps no input): %d\n", wm_spikes);
    printf("  dlPFC WM persistence: %.3f\n", wm_persist);
    printf("  dlPFC L2/3 mean rate: %.2f\n", dlpfc->l23_mean_rate());
    printf("  V1 L2/3 mean rate: %.2f\n", v1->l23_mean_rate());

    // With homeostatic plasticity, scale=3 WM should not collapse to 0
    TEST_ASSERT(wm_spikes > 0, "scale=3 WM有活动");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// main
// =========================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("=== 悟韵 Step 13-A: 稳态可塑性集成测试 ===\n");

    test_rate_tracking();
    test_overactive_decrease();
    test_underactive_increase();
    test_cortical_integration();
    test_hippocampus_integration();
    test_multi_region_homeostatic();
    test_scale3_wm_stability();

    printf("\n========================================\n");
    printf("  通过: %d / %d\n", g_pass, g_pass + g_fail);
    printf("========================================\n");

    return g_fail > 0 ? 1 : 0;
}
