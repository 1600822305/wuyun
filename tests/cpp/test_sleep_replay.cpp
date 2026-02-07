/**
 * 悟韵 (WuYun) Step 8 睡眠/海马重放测试
 *
 * 测试内容:
 *   1. SWR基础生成: 睡眠模式下CA3噪声→SWR事件
 *   2. SWR不应期: 连续SWR间隔受refractory限制
 *   3. 清醒无SWR: 非睡眠模式不生成SWR
 *   4. 皮层慢波: 睡眠模式下up/down状态交替
 *   5. 慢波抑制: down state显著减少皮层活动
 *   6. 编码→重放: 先编码模式→睡眠→SWR重放→CA1活动
 *   7. 全系统集成: 48区域脑 + 睡眠模式 + SWR
 */

#include "region/limbic/hippocampus.h"
#include "region/cortical_region.h"
#include "region/limbic/hypothalamus.h"
#include "engine/simulation_engine.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/neuromod/vta_da.h"
#include "region/neuromod/lc_ne.h"
#include "region/neuromod/drn_5ht.h"
#include "region/neuromod/nbm_ach.h"
#include "region/limbic/amygdala.h"
#include "region/limbic/septal_nucleus.h"
#include "region/limbic/mammillary_body.h"
#include "region/subcortical/cerebellum.h"
#include "engine/global_workspace.h"

#include <cstdio>
#include <cassert>

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
// Test 1: SWR基础生成
// =============================================================================
static void test_swr_basic() {
    printf("\n--- 测试1: SWR基础生成 ---\n");
    printf("    原理: 睡眠模式→CA3随机噪声→自联想补全→SWR burst\n");

    HippocampusConfig cfg;
    cfg.swr_noise_amp = 35.0f;      // Strong noise (place cells need ~15-20 to fire)
    cfg.swr_ca3_threshold = 0.10f;  // Lower threshold for easier triggering
    cfg.swr_duration = 5;
    cfg.swr_refractory = 20;
    Hippocampus hipp(cfg);

    // First encode some patterns (so CA3 recurrent weights are non-trivial)
    for (int t = 0; t < 100; ++t) {
        std::vector<float> input(cfg.n_ec, 0.0f);
        for (size_t i = 0; i < 30; ++i) input[i] = 30.0f;
        hipp.inject_cortical_input(input);
        hipp.step(t);
    }

    // Enable sleep replay
    hipp.enable_sleep_replay();
    TEST_ASSERT(hipp.sleep_replay_enabled(), "sleep replay enabled");
    TEST_ASSERT(hipp.swr_count() == 0, "no SWR before sleep run");

    // Run sleep for a while
    for (int t = 100; t < 500; ++t) {
        hipp.step(t);
    }

    uint32_t swrs = hipp.swr_count();
    printf("    SWR events: %u (in 400 steps)\n", swrs);
    TEST_ASSERT(swrs > 0, "SWR generated during sleep");

    printf("  [PASS] SWR基础生成\n");
    g_pass++;
}

// =============================================================================
// Test 2: SWR不应期
// =============================================================================
static void test_swr_refractory() {
    printf("\n--- 测试2: SWR不应期 ---\n");
    printf("    原理: 连续SWR间隔 >= refractory期\n");

    HippocampusConfig cfg;
    cfg.swr_noise_amp = 35.0f;
    cfg.swr_ca3_threshold = 0.08f;
    cfg.swr_duration = 5;
    cfg.swr_refractory = 50;  // Long refractory for clear testing
    Hippocampus hipp(cfg);

    // Encode patterns
    for (int t = 0; t < 100; ++t) {
        std::vector<float> input(cfg.n_ec, 20.0f);
        hipp.inject_cortical_input(input);
        hipp.step(t);
    }

    hipp.enable_sleep_replay();

    // Run 500 steps: max theoretical SWRs = 500 / (5+50) ≈ 9
    for (int t = 100; t < 600; ++t) {
        hipp.step(t);
    }

    uint32_t swrs = hipp.swr_count();
    uint32_t max_possible = 500 / (cfg.swr_duration + cfg.swr_refractory);
    printf("    SWR: %u (max possible ~%u with refractory=%zu)\n",
           swrs, max_possible + 1, cfg.swr_refractory);

    // Should be limited by refractory
    TEST_ASSERT(swrs <= max_possible + 2, "SWR count limited by refractory");

    printf("  [PASS] SWR不应期\n");
    g_pass++;
}

// =============================================================================
// Test 3: 清醒无SWR
// =============================================================================
static void test_no_swr_awake() {
    printf("\n--- 测试3: 清醒无SWR ---\n");
    printf("    原理: 未启用sleep_replay → 不生成SWR\n");

    HippocampusConfig cfg;
    Hippocampus hipp(cfg);

    // Run without enabling sleep replay
    for (int t = 0; t < 300; ++t) {
        std::vector<float> input(cfg.n_ec, 20.0f);
        hipp.inject_cortical_input(input);
        hipp.step(t);
    }

    TEST_ASSERT(hipp.swr_count() == 0, "no SWR when awake");
    TEST_ASSERT(!hipp.is_swr(), "not in SWR");

    printf("    SWR count: %u (expected 0)\n", hipp.swr_count());
    printf("  [PASS] 清醒无SWR\n");
    g_pass++;
}

// =============================================================================
// Test 4: 皮层慢波up/down状态
// =============================================================================
static void test_cortical_slow_wave() {
    printf("\n--- 测试4: 皮层慢波up/down状态 ---\n");
    printf("    原理: 睡眠模式→~1Hz up/down交替\n");

    ColumnConfig cc;
    cc.n_l4_stellate = 30; cc.n_l23_pyramidal = 60;
    cc.n_l5_pyramidal = 30; cc.n_l6_pyramidal = 20;
    cc.n_pv_basket = 8; cc.n_sst_martinotti = 5; cc.n_vip = 2;
    CorticalRegion ctx("V1", cc);

    ctx.set_sleep_mode(true);
    TEST_ASSERT(ctx.is_sleep_mode(), "sleep mode on");

    // Track up/down transitions
    int up_count = 0, down_count = 0;
    bool prev_up = ctx.is_up_state();

    for (int t = 0; t < 2000; ++t) {
        ctx.step(t);
        bool now_up = ctx.is_up_state();
        if (now_up && !prev_up) up_count++;
        if (!now_up && prev_up) down_count++;
        prev_up = now_up;
    }

    printf("    Up→Down transitions: %d, Down→Up: %d\n", down_count, up_count);
    printf("    Phase after 2000 steps: %.3f\n", ctx.slow_wave_phase());

    // ~1Hz at SLOW_WAVE_FREQ=0.001 → ~2 full cycles in 2000 steps
    TEST_ASSERT(up_count >= 1 && up_count <= 4, "reasonable oscillation frequency");
    TEST_ASSERT(down_count >= 1, "down transitions occur");

    printf("  [PASS] 皮层慢波\n");
    g_pass++;
}

// =============================================================================
// Test 5: Down state抑制皮层活动
// =============================================================================
static void test_down_state_suppression() {
    printf("\n--- 测试5: Down state抑制皮层活动 ---\n");
    printf("    原理: down state注入抑制电流→显著减少发放\n");

    ColumnConfig cc;
    cc.n_l4_stellate = 50; cc.n_l23_pyramidal = 100;
    cc.n_l5_pyramidal = 50; cc.n_l6_pyramidal = 40;
    cc.n_pv_basket = 15; cc.n_sst_martinotti = 10; cc.n_vip = 5;

    // Awake control
    SimulationEngine eng1(10);
    ThalamicConfig tc; tc.name = "LGN"; tc.n_relay = 50; tc.n_trn = 15;
    eng1.add_region(std::make_unique<ThalamicRelay>(tc));
    eng1.add_region(std::make_unique<CorticalRegion>("V1", cc));
    eng1.add_projection("LGN", "V1", 2);

    size_t awake_spikes = 0;
    for (int t = 0; t < 500; ++t) {
        eng1.find_region("LGN")->inject_external(std::vector<float>(50, 30.0f));
        eng1.step();
        awake_spikes += count_fired(*eng1.find_region("V1"));
    }

    // Sleep with same input
    SimulationEngine eng2(10);
    eng2.add_region(std::make_unique<ThalamicRelay>(tc));
    eng2.add_region(std::make_unique<CorticalRegion>("V1", cc));
    eng2.add_projection("LGN", "V1", 2);

    auto* v1_sleep = dynamic_cast<CorticalRegion*>(eng2.find_region("V1"));
    v1_sleep->set_sleep_mode(true);

    size_t sleep_spikes = 0;
    size_t up_spikes = 0, down_spikes = 0;
    for (int t = 0; t < 500; ++t) {
        eng2.find_region("LGN")->inject_external(std::vector<float>(50, 30.0f));
        eng2.step();
        size_t s = count_fired(*eng2.find_region("V1"));
        sleep_spikes += s;
        if (v1_sleep->is_up_state()) up_spikes += s;
        else down_spikes += s;
    }

    printf("    Awake spikes: %zu\n", awake_spikes);
    printf("    Sleep spikes: %zu (up=%zu, down=%zu)\n",
           sleep_spikes, up_spikes, down_spikes);

    TEST_ASSERT(sleep_spikes < awake_spikes, "sleep reduces total activity");
    TEST_ASSERT(up_spikes > down_spikes, "more activity during up state");

    printf("  [PASS] Down state抑制\n");
    g_pass++;
}

// =============================================================================
// Test 6: 编码→睡眠→重放
// =============================================================================
static void test_encode_replay() {
    printf("\n--- 测试6: 编码→睡眠→SWR重放 ---\n");
    printf("    原理: 清醒编码→STDP存储→睡眠SWR→CA3补全→CA1 burst\n");

    HippocampusConfig cfg;
    cfg.ca3_stdp_enabled = true;
    cfg.swr_noise_amp = 35.0f;
    cfg.swr_ca3_threshold = 0.10f;
    cfg.swr_duration = 5;
    cfg.swr_refractory = 20;
    Hippocampus hipp(cfg);

    // Phase 1: Encode a strong pattern (repeated presentation)
    size_t encode_ca1 = 0;
    for (int t = 0; t < 200; ++t) {
        std::vector<float> input(cfg.n_ec, 0.0f);
        // Strong pattern: first 30 EC neurons
        for (size_t i = 0; i < 30; ++i) input[i] = 30.0f;
        hipp.inject_cortical_input(input);
        hipp.step(t);
        encode_ca1 += count_fired(hipp);
    }
    printf("    编码期CA1活动: %zu\n", encode_ca1);

    // Phase 2: Quiet gap (no input)
    for (int t = 200; t < 250; ++t) {
        hipp.step(t);
    }

    // Phase 3: Sleep replay
    hipp.enable_sleep_replay();
    size_t replay_ca1 = 0;
    size_t swr_ca1 = 0;
    for (int t = 250; t < 650; ++t) {
        hipp.step(t);
        size_t fired = count_fired(hipp);
        replay_ca1 += fired;
        if (hipp.is_swr()) swr_ca1 += fired;
    }

    uint32_t swrs = hipp.swr_count();
    float replay_str = hipp.last_replay_strength();
    printf("    睡眠SWR: %u次, 最后replay强度: %.2f\n", swrs, replay_str);
    printf("    重放期总活动: %zu (SWR期: %zu)\n", replay_ca1, swr_ca1);

    TEST_ASSERT(swrs > 0, "SWR generated after encoding");
    TEST_ASSERT(replay_ca1 > 0, "replay produces activity");

    printf("  [PASS] 编码→重放\n");
    g_pass++;
}

// =============================================================================
// Test 7: 全系统集成 (48区域 + 睡眠)
// =============================================================================
static void test_full_system_sleep() {
    printf("\n--- 测试7: 多区域集成睡眠 ---\n");
    printf("    原理: LGN→V1→dlPFC + Hipp + Hypo 联合睡眠\n");

    SimulationEngine eng(10);

    // Build minimal sleep-capable brain
    ThalamicConfig tc; tc.name = "LGN"; tc.n_relay = 50; tc.n_trn = 15;
    eng.add_region(std::make_unique<ThalamicRelay>(tc));

    ColumnConfig cc;
    cc.n_l4_stellate = 50; cc.n_l23_pyramidal = 100;
    cc.n_l5_pyramidal = 50; cc.n_l6_pyramidal = 40;
    cc.n_pv_basket = 15; cc.n_sst_martinotti = 10; cc.n_vip = 5;
    eng.add_region(std::make_unique<CorticalRegion>("V1", cc));

    ColumnConfig cc2;
    cc2.n_l4_stellate = 30; cc2.n_l23_pyramidal = 80;
    cc2.n_l5_pyramidal = 40; cc2.n_l6_pyramidal = 30;
    cc2.n_pv_basket = 10; cc2.n_sst_martinotti = 8; cc2.n_vip = 4;
    eng.add_region(std::make_unique<CorticalRegion>("dlPFC", cc2));

    HippocampusConfig hcfg;
    hcfg.swr_noise_amp = 15.0f;
    hcfg.swr_ca3_threshold = 0.10f;
    eng.add_region(std::make_unique<Hippocampus>(hcfg));
    eng.add_region(std::make_unique<Hypothalamus>(HypothalamusConfig{}));

    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("V1", "dlPFC", 2);
    eng.add_projection("dlPFC", "Hippocampus", 3);

    auto* v1    = dynamic_cast<CorticalRegion*>(eng.find_region("V1"));
    auto* dlpfc = dynamic_cast<CorticalRegion*>(eng.find_region("dlPFC"));
    auto* hipp  = dynamic_cast<Hippocampus*>(eng.find_region("Hippocampus"));
    auto* hypo  = dynamic_cast<Hypothalamus*>(eng.find_region("Hypothalamus"));
    TEST_ASSERT(v1 && dlpfc && hipp && hypo, "regions found");

    // Phase 1: Awake encoding (100 steps)
    printf("    Phase 1: 清醒编码...\n");
    size_t awake_v1 = 0;
    for (int t = 0; t < 100; ++t) {
        eng.find_region("LGN")->inject_external(std::vector<float>(50, 35.0f));
        eng.step();
        awake_v1 += count_fired(*v1);
    }
    printf("    V1 awake: %zu spikes\n", awake_v1);

    // Phase 2: Enter sleep mode
    printf("    Phase 2: 进入睡眠...\n");
    hypo->set_sleep_pressure(0.9f);
    hipp->enable_sleep_replay();
    v1->set_sleep_mode(true);
    dlpfc->set_sleep_mode(true);

    size_t sleep_v1 = 0;
    for (int t = 100; t < 400; ++t) {
        eng.step();
        sleep_v1 += count_fired(*v1);
    }

    uint32_t swrs = hipp->swr_count();
    float wake = hypo->wake_level();
    printf("    V1 sleep: %zu spikes (vs awake %zu)\n", sleep_v1, awake_v1);
    printf("    Hypothalamus wake: %.2f, SWR: %u\n", wake, swrs);
    printf("    Regions: %zu\n", eng.num_regions());

    TEST_ASSERT(sleep_v1 < awake_v1, "sleep reduces cortical activity");
    TEST_ASSERT(eng.num_regions() == 5, "5 regions in test brain");

    printf("  [PASS] 多区域集成睡眠\n");
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
    printf("  悟韵 (WuYun) Step 8 睡眠/海马重放测试\n");
    printf("  SWR生成 + 皮层慢波 + 记忆重放\n");
    printf("============================================\n");

    test_swr_basic();
    test_swr_refractory();
    test_no_swr_awake();
    test_cortical_slow_wave();
    test_down_state_suppression();
    test_encode_replay();
    test_full_system_sleep();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
