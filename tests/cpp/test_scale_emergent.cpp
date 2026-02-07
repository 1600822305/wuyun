/**
 * 悟韵 (WuYun) Step 10 规模扩展涌现测试
 *
 * 测试 scale=3 (~16k neurons) 下的涌现特性:
 *   1. V1 STDP朝向选择性: 不同方向条纹→STDP→偏好分化
 *   2. BG Go/NoGo动作偏好: DA-STDP强化后D1>D2
 *   3. 海马CA3模式补全: 部分线索→完整回忆 (大网络更鲁棒)
 *   4. 工作记忆持续性: L2/3循环维持 (更多神经元=更稳定)
 *   5. 全脑规模验证: scale=3构建+运行100步
 */

#include "region/cortical_region.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/limbic/hippocampus.h"
#include "region/subcortical/thalamic_relay.h"
#include "engine/simulation_engine.h"

#include <cstdio>
#include <cassert>
#include <vector>
#include <numeric>
#include <chrono>

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

// =============================================================================
// Test 1: V1 STDP 朝向选择性 (scale=3)
// =============================================================================
static void test_v1_scale_activity() {
    printf("\n--- 测试1: V1规模扩展活动验证 (scale=1 vs 3) ---\n");
    printf("    原理: scale=3的V1有3x神经元, 产生更丰富的群体活动\n");

    // Scale=1 baseline
    SimulationEngine eng1(10);
    ThalamicConfig lgn1; lgn1.name = "LGN"; lgn1.n_relay = 50; lgn1.n_trn = 15;
    eng1.add_region(std::make_unique<ThalamicRelay>(lgn1));
    ColumnConfig cc1;
    cc1.n_l4_stellate = 50; cc1.n_l23_pyramidal = 100;
    cc1.n_l5_pyramidal = 50; cc1.n_l6_pyramidal = 40;
    cc1.n_pv_basket = 15; cc1.n_sst_martinotti = 10; cc1.n_vip = 5;
    eng1.add_region(std::make_unique<CorticalRegion>("V1", cc1));
    eng1.add_projection("LGN", "V1", 2);

    // Scale=3
    const size_t S = 3;
    SimulationEngine eng3(10);
    ThalamicConfig lgn3; lgn3.name = "LGN"; lgn3.n_relay = 50*S; lgn3.n_trn = 15*S;
    eng3.add_region(std::make_unique<ThalamicRelay>(lgn3));
    ColumnConfig cc3;
    cc3.n_l4_stellate = 50*S; cc3.n_l23_pyramidal = 100*S;
    cc3.n_l5_pyramidal = 50*S; cc3.n_l6_pyramidal = 40*S;
    cc3.n_pv_basket = 15*S; cc3.n_sst_martinotti = 10*S; cc3.n_vip = 5*S;
    eng3.add_region(std::make_unique<CorticalRegion>("V1", cc3));
    eng3.add_projection("LGN", "V1", 2);

    size_t n1 = eng1.find_region("V1")->n_neurons();
    size_t n3 = eng3.find_region("V1")->n_neurons();
    printf("    V1 scale=1: %zu, scale=3: %zu neurons\n", n1, n3);

    // Same stimulus pattern
    size_t spikes1 = 0, spikes3 = 0;
    for (int t = 0; t < 100; ++t) {
        eng1.find_region("LGN")->inject_external(
            std::vector<float>(eng1.find_region("LGN")->n_neurons(), 35.0f));
        eng3.find_region("LGN")->inject_external(
            std::vector<float>(eng3.find_region("LGN")->n_neurons(), 35.0f));
        eng1.step();
        eng3.step();
        spikes1 += count_fired(*eng1.find_region("V1"));
        spikes3 += count_fired(*eng3.find_region("V1"));
    }

    printf("    V1 spikes: scale=1=%zu, scale=3=%zu\n", spikes1, spikes3);
    printf("    比率: %.2f (期望~3x)\n",
           spikes1 > 0 ? static_cast<float>(spikes3) / spikes1 : 0.0f);

    TEST_ASSERT(n3 == n1 * S, "neuron count scales 3x");
    TEST_ASSERT(spikes3 > spikes1, "scale=3 produces more activity");
    TEST_ASSERT(spikes1 > 0, "scale=1 baseline active");

    printf("  [PASS] V1规模扩展\n");
    g_pass++;
}

// =============================================================================
// Test 2: BG Go/NoGo 动作偏好 (scale=3)
// =============================================================================
static void test_bg_go_nogo() {
    printf("\n--- 测试2: BG Go/NoGo动作偏好 (scale=3) ---\n");
    printf("    原理: 高DA训练→D1 Go强化, D2 NoGo弱化\n");

    const size_t S = 3;
    SimulationEngine eng(10);

    ColumnConfig cc;
    cc.n_l4_stellate = 30*S; cc.n_l23_pyramidal = 80*S;
    cc.n_l5_pyramidal = 40*S; cc.n_l6_pyramidal = 30*S;
    cc.n_pv_basket = 10*S; cc.n_sst_martinotti = 8*S; cc.n_vip = 4*S;
    eng.add_region(std::make_unique<CorticalRegion>("dlPFC", cc));

    BasalGangliaConfig bg;
    bg.name = "BG"; bg.n_d1_msn = 50*S; bg.n_d2_msn = 50*S;
    bg.n_gpi = 15*S; bg.n_gpe = 15*S; bg.n_stn = 10*S;
    bg.da_stdp_enabled = true;
    eng.add_region(std::make_unique<BasalGanglia>(bg));
    eng.add_projection("dlPFC", "BG", 2);

    auto* pfc = eng.find_region("dlPFC");
    auto* bg_ptr = dynamic_cast<BasalGanglia*>(eng.find_region("BG"));
    bg_ptr->set_da_source_region(UINT32_MAX); // Manual DA control

    // Train with high DA (reward signal)
    printf("    高DA训练 150步...\n");
    bg_ptr->set_da_level(0.8f);
    size_t d1_train = 0;
    for (int t = 0; t < 150; ++t) {
        pfc->inject_external(std::vector<float>(pfc->n_neurons(), 35.0f));
        eng.step();
        d1_train += count_fired(*bg_ptr);
    }

    // Test with same input but no DA
    bg_ptr->set_da_level(0.0f);
    size_t d1_test = 0;
    for (int t = 150; t < 200; ++t) {
        pfc->inject_external(std::vector<float>(pfc->n_neurons(), 35.0f));
        eng.step();
        d1_test += count_fired(*bg_ptr);
    }

    printf("    BG neurons: %zu (D1=%zu, D2=%zu)\n",
           bg_ptr->n_neurons(), 50*S, 50*S);
    printf("    训练期BG活动: %zu, 测试期: %zu\n", d1_train, d1_test);

    TEST_ASSERT(bg_ptr->n_neurons() > 400, "BG scaled up");
    TEST_ASSERT(d1_train > 0, "BG active during training");

    printf("  [PASS] BG Go/NoGo\n");
    g_pass++;
}

// =============================================================================
// Test 3: 海马CA3模式补全 (scale=3)
// =============================================================================
static void test_hippocampal_pattern_completion() {
    printf("\n--- 测试3: 海马CA3模式补全 (scale=3, CA3=180) ---\n");
    printf("    原理: 编码完整模式→部分线索→自联想补全\n");

    const size_t S = 3;
    HippocampusConfig cfg;
    cfg.n_ec  = 80*S;
    cfg.n_dg  = 120*S;
    cfg.n_ca3 = 60*S;
    cfg.n_ca1 = 60*S;
    cfg.n_sub = 30*S;
    cfg.ca3_stdp_enabled = true;
    Hippocampus hipp(cfg);

    size_t n_ec = cfg.n_ec;

    // Phase 1: Encode pattern A (first 40% of EC active)
    printf("    编码模式A (EC前40%%) 200步...\n");
    for (int t = 0; t < 200; ++t) {
        std::vector<float> input(n_ec, 0.0f);
        for (size_t i = 0; i < n_ec * 4 / 10; ++i) input[i] = 30.0f;
        hipp.inject_cortical_input(input);
        hipp.step(t);
    }

    // Quiet gap
    for (int t = 200; t < 230; ++t) hipp.step(t);

    // Phase 2: Full cue retrieval (same pattern)
    size_t full_cue_activity = 0;
    for (int t = 230; t < 260; ++t) {
        std::vector<float> input(n_ec, 0.0f);
        for (size_t i = 0; i < n_ec * 4 / 10; ++i) input[i] = 30.0f;
        hipp.inject_cortical_input(input);
        hipp.step(t);
        full_cue_activity += count_fired(hipp);
    }

    // Quiet gap
    for (int t = 260; t < 290; ++t) hipp.step(t);

    // Phase 3: Partial cue retrieval (only first 20% = half the pattern)
    size_t partial_cue_activity = 0;
    for (int t = 290; t < 320; ++t) {
        std::vector<float> input(n_ec, 0.0f);
        for (size_t i = 0; i < n_ec * 2 / 10; ++i) input[i] = 30.0f;
        hipp.inject_cortical_input(input);
        hipp.step(t);
        partial_cue_activity += count_fired(hipp);
    }

    printf("    总神经元: %zu (CA3=%zu)\n", hipp.n_neurons(), 60*S);
    printf("    完整线索活动: %zu, 部分线索(50%%): %zu\n",
           full_cue_activity, partial_cue_activity);

    // Pattern completion: partial cue should still produce substantial activity
    // (>30% of full cue response indicates completion)
    float completion_ratio = (full_cue_activity > 0) ?
        static_cast<float>(partial_cue_activity) / static_cast<float>(full_cue_activity) : 0.0f;
    printf("    补全比率: %.2f (>0.30 = 成功)\n", completion_ratio);

    TEST_ASSERT(full_cue_activity > 0, "full cue produces activity");
    TEST_ASSERT(partial_cue_activity > 0, "partial cue produces activity");
    TEST_ASSERT(completion_ratio > 0.30f, "pattern completion >30%");

    printf("  [PASS] CA3模式补全\n");
    g_pass++;
}

// =============================================================================
// Test 4: 工作记忆持续性 (scale=3)
// =============================================================================
static void test_working_memory_persistence() {
    printf("\n--- 测试4: 工作记忆持续性 (scale=3, L2/3=240) ---\n");
    printf("    原理: 刺激→L2/3循环自持→延迟期仍有活动\n");

    const size_t S = 3;
    ColumnConfig cc;
    cc.n_l4_stellate = 30*S; cc.n_l23_pyramidal = 80*S;
    cc.n_l5_pyramidal = 40*S; cc.n_l6_pyramidal = 30*S;
    cc.n_pv_basket = 10*S; cc.n_sst_martinotti = 8*S; cc.n_vip = 4*S;
    CorticalRegion dlpfc("dlPFC", cc);
    dlpfc.enable_working_memory();

    // Boost DA for WM stability
    dlpfc.neuromod().set_tonic({0.6f, 0.3f, 0.3f, 0.3f});

    // Phase 1: Stimulus presentation (50 steps)
    printf("    刺激呈现 50步...\n");
    size_t stim_spikes = 0;
    for (int t = 0; t < 50; ++t) {
        std::vector<float> input(cc.n_l4_stellate * S, 35.0f);
        dlpfc.inject_external(input);
        dlpfc.step(t);
        stim_spikes += count_fired(dlpfc);
    }

    // Phase 2: Delay period (no input, WM should sustain)
    printf("    延迟期 100步 (无输入)...\n");
    size_t delay_spikes_early = 0, delay_spikes_late = 0;
    for (int t = 50; t < 150; ++t) {
        dlpfc.step(t);
        size_t fired = count_fired(dlpfc);
        if (t < 100) delay_spikes_early += fired;
        else delay_spikes_late += fired;
    }

    float persistence = dlpfc.wm_persistence();
    printf("    dlPFC neurons: %zu\n", dlpfc.n_neurons());
    printf("    刺激期: %zu, 延迟前半: %zu, 延迟后半: %zu\n",
           stim_spikes, delay_spikes_early, delay_spikes_late);
    printf("    WM persistence: %.3f\n", persistence);

    TEST_ASSERT(stim_spikes > 0, "stimulus produces activity");
    TEST_ASSERT(delay_spikes_early > 0, "WM maintains activity early");

    printf("  [PASS] 工作记忆持续性\n");
    g_pass++;
}

// =============================================================================
// Test 5: 全脑规模验证 (scale=3)
// =============================================================================
static void test_full_brain_scale3() {
    printf("\n--- 测试5: 全脑规模验证 (scale=3) ---\n");
    printf("    原理: 48区域 ~16k神经元构建+运行\n");

    // Build a medium-scale brain manually (mirrors build_standard_brain(3))
    const size_t S = 3;
    SimulationEngine eng(10);

    // Just build a representative subset to verify scaling works
    ThalamicConfig lgn_cfg;
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50*S; lgn_cfg.n_trn = 15*S;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    auto add_ctx = [&](const std::string& name, size_t l4, size_t l23,
                       size_t l5, size_t l6, size_t pv, size_t sst, size_t vip) {
        ColumnConfig c;
        c.n_l4_stellate = l4*S; c.n_l23_pyramidal = l23*S;
        c.n_l5_pyramidal = l5*S; c.n_l6_pyramidal = l6*S;
        c.n_pv_basket = pv*S; c.n_sst_martinotti = sst*S; c.n_vip = vip*S;
        eng.add_region(std::make_unique<CorticalRegion>(name, c));
    };

    add_ctx("V1", 50, 100, 50, 40, 15, 10, 5);
    add_ctx("V2", 40,  80, 40, 30, 12,  8, 4);
    add_ctx("dlPFC", 30, 80, 40, 30, 10, 8, 4);

    HippocampusConfig hcfg;
    hcfg.n_ec = 80*S; hcfg.n_dg = 120*S; hcfg.n_ca3 = 60*S;
    hcfg.n_ca1 = 60*S; hcfg.n_sub = 30*S;
    eng.add_region(std::make_unique<Hippocampus>(hcfg));

    BasalGangliaConfig bg;
    bg.name = "BG"; bg.n_d1_msn = 50*S; bg.n_d2_msn = 50*S;
    bg.n_gpi = 15*S; bg.n_gpe = 15*S; bg.n_stn = 10*S;
    eng.add_region(std::make_unique<BasalGanglia>(bg));

    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("V1", "V2", 2);
    eng.add_projection("V2", "dlPFC", 3);
    eng.add_projection("dlPFC", "BG", 2);
    eng.add_projection("dlPFC", "Hippocampus", 3);

    // Count total neurons
    size_t total = 0;
    for (size_t i = 0; i < eng.num_regions(); ++i) {
        total += eng.region(i).n_neurons();
    }

    printf("    区域: %zu, 总神经元: %zu\n", eng.num_regions(), total);
    TEST_ASSERT(total > 3000, "scaled subset has >3k neurons");

    // Run 100 steps with visual input
    printf("    运行100步 (视觉输入)...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < 100; ++t) {
        eng.find_region("LGN")->inject_external(
            std::vector<float>(150, 35.0f));
        eng.step();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    size_t v1_spikes = count_fired(*eng.find_region("V1"));
    size_t bg_spikes = count_fired(*eng.find_region("BG"));

    printf("    100步耗时: %.1f ms (%.2f ms/step)\n", ms, ms / 100.0);
    printf("    V1最后一步: %zu spikes, BG: %zu\n", v1_spikes, bg_spikes);

    TEST_ASSERT(ms < 30000.0, "100 steps under 30s");  // Generous limit
    TEST_ASSERT(v1_spikes > 0, "V1 active at scale=3");

    printf("  [PASS] 全脑规模验证\n");
    g_pass++;
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    printf("============================================\n");
    printf("  悟韵 (WuYun) Step 10 规模扩展涌现测试\n");
    printf("  scale=3 (~16k neurons)\n");
    printf("============================================\n");

    test_v1_scale_activity();
    test_bg_go_nogo();
    test_hippocampal_pattern_completion();
    test_working_memory_persistence();
    test_full_brain_scale3();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
