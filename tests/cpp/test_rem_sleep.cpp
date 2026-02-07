/**
 * 悟韵 (WuYun) Step 11 REM睡眠 + 梦境测试
 *
 * 测试内容:
 *   1. SleepCycleManager NREM→REM→NREM 状态转换
 *   2. REM周期增长 (后半夜REM变长)
 *   3. PGO波生成 (梦境视觉激活)
 *   4. CorticalRegion REM模式 (去同步化 + 运动弛缓)
 *   5. Hippocampus REM theta (6Hz振荡 + 创造性重组)
 *   6. 完整睡眠周期: NREM(SWR) → REM(theta) → NREM交替
 *   7. 全脑NREM→REM: 皮层从慢波切换到去同步化
 */

#include "engine/sleep_cycle.h"
#include "region/cortical_region.h"
#include "region/limbic/hippocampus.h"
#include "region/subcortical/thalamic_relay.h"
#include "engine/simulation_engine.h"

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

// =============================================================================
// Test 1: SleepCycleManager 基础状态转换
// =============================================================================
static void test_sleep_cycle_basics() {
    printf("\n--- 测试1: SleepCycleManager 基础状态转换 ---\n");

    SleepCycleConfig cfg;
    cfg.nrem_duration = 100;
    cfg.rem_duration = 50;
    cfg.rem_growth = 0;  // No growth for this test
    cfg.min_nrem_duration = 50;
    SleepCycleManager mgr(cfg);

    TEST_ASSERT(mgr.stage() == SleepStage::AWAKE, "starts awake");
    TEST_ASSERT(!mgr.is_sleeping(), "not sleeping initially");

    // Enter sleep
    mgr.enter_sleep();
    TEST_ASSERT(mgr.stage() == SleepStage::NREM, "enters NREM");
    TEST_ASSERT(mgr.is_nrem(), "is_nrem true");
    TEST_ASSERT(!mgr.is_rem(), "is_rem false");
    TEST_ASSERT(mgr.cycle_count() == 0, "cycle 0");

    // Run through NREM
    for (int i = 0; i < 100; ++i) mgr.step();
    TEST_ASSERT(mgr.stage() == SleepStage::REM, "transitions to REM after NREM");
    TEST_ASSERT(mgr.is_rem(), "is_rem true");
    printf("    NREM→REM transition at step 100 [OK]\n");

    // Run through REM
    for (int i = 0; i < 50; ++i) mgr.step();
    TEST_ASSERT(mgr.stage() == SleepStage::NREM, "back to NREM after REM");
    TEST_ASSERT(mgr.cycle_count() == 1, "cycle incremented to 1");
    printf("    REM→NREM cycle 1 at step 150 [OK]\n");

    // Wake up
    mgr.wake_up();
    TEST_ASSERT(mgr.stage() == SleepStage::AWAKE, "wakes up");
    TEST_ASSERT(!mgr.is_sleeping(), "not sleeping after wake");
    printf("    Wake up [OK]\n");

    printf("  [PASS] SleepCycleManager基础\n");
    g_pass++;
}

// =============================================================================
// Test 2: REM周期增长
// =============================================================================
static void test_rem_growth() {
    printf("\n--- 测试2: REM周期增长 (后半夜REM变长) ---\n");

    SleepCycleConfig cfg;
    cfg.nrem_duration = 200;
    cfg.rem_duration = 50;
    cfg.rem_growth = 30;       // Each cycle: REM +30 steps
    cfg.max_rem_duration = 200;
    cfg.nrem_growth = 20;      // Each cycle: NREM -20 steps
    cfg.min_nrem_duration = 100;
    SleepCycleManager mgr(cfg);

    mgr.enter_sleep();

    // Cycle 0: NREM=200, REM=50
    TEST_ASSERT(mgr.current_nrem_duration() == 200, "cycle0 NREM=200");
    TEST_ASSERT(mgr.current_rem_duration() == 50, "cycle0 REM=50");

    // Run to cycle 1
    for (int i = 0; i < 250; ++i) mgr.step();
    TEST_ASSERT(mgr.cycle_count() == 1, "reached cycle 1");
    // Cycle 1: NREM=200-20=180, REM=50+30=80
    TEST_ASSERT(mgr.current_nrem_duration() == 180, "cycle1 NREM=180");
    TEST_ASSERT(mgr.current_rem_duration() == 80, "cycle1 REM=80");
    printf("    Cycle1: NREM=180, REM=80 [OK]\n");

    // Run to cycle 2
    for (int i = 0; i < 260; ++i) mgr.step();
    TEST_ASSERT(mgr.cycle_count() == 2, "reached cycle 2");
    // Cycle 2: NREM=200-40=160, REM=50+60=110
    TEST_ASSERT(mgr.current_nrem_duration() == 160, "cycle2 NREM=160");
    TEST_ASSERT(mgr.current_rem_duration() == 110, "cycle2 REM=110");
    printf("    Cycle2: NREM=160, REM=110 [OK]\n");

    printf("  [PASS] REM周期增长\n");
    g_pass++;
}

// =============================================================================
// Test 3: PGO波生成
// =============================================================================
static void test_pgo_waves() {
    printf("\n--- 测试3: PGO波生成 (REM期间) ---\n");

    SleepCycleConfig cfg;
    cfg.nrem_duration = 10;
    cfg.rem_duration = 500;
    cfg.rem_pgo_prob = 0.05f;  // 5% per step
    cfg.min_nrem_duration = 5;
    SleepCycleManager mgr(cfg);

    mgr.enter_sleep();
    // Skip NREM
    for (int i = 0; i < 10; ++i) mgr.step();
    TEST_ASSERT(mgr.is_rem(), "in REM");

    // Count PGO events over 500 REM steps
    int pgo_count = 0;
    for (int i = 0; i < 500; ++i) {
        mgr.step();
        if (mgr.pgo_active()) pgo_count++;
    }

    float pgo_rate = static_cast<float>(pgo_count) / 500.0f;
    printf("    PGO events: %d/500 (rate=%.3f, expected~0.05)\n", pgo_count, pgo_rate);

    TEST_ASSERT(pgo_count > 5, "PGO events occur (>5)");
    TEST_ASSERT(pgo_count < 100, "PGO not too frequent (<100)");

    printf("  [PASS] PGO波生成\n");
    g_pass++;
}

// =============================================================================
// Test 4: CorticalRegion REM模式
// =============================================================================
static void test_cortical_rem() {
    printf("\n--- 测试4: CorticalRegion REM模式 ---\n");
    printf("    原理: REM=去同步化噪声 + 运动弛缓(M1)\n");

    ColumnConfig cc;
    cc.n_l4_stellate = 30; cc.n_l23_pyramidal = 60;
    cc.n_l5_pyramidal = 30; cc.n_l6_pyramidal = 20;
    cc.n_pv_basket = 8; cc.n_sst_martinotti = 5; cc.n_vip = 3;

    CorticalRegion v1("V1", cc);
    CorticalRegion m1("M1", cc);

    // Baseline: no sleep
    size_t awake_v1 = 0;
    for (int t = 0; t < 50; ++t) {
        v1.step(t); m1.step(t);
        awake_v1 += count_fired(v1);
    }

    // REM mode: V1 should have desynchronized activity
    v1.set_rem_mode(true);
    m1.set_rem_mode(true);
    m1.set_motor_atonia(true);

    TEST_ASSERT(v1.is_rem_mode(), "V1 in REM");
    TEST_ASSERT(!v1.is_sleep_mode(), "V1 not in NREM");
    TEST_ASSERT(m1.is_motor_atonia(), "M1 has atonia");

    size_t rem_v1 = 0, rem_m1 = 0;
    for (int t = 50; t < 150; ++t) {
        v1.step(t); m1.step(t);
        rem_v1 += count_fired(v1);
        rem_m1 += count_fired(m1);
    }

    // PGO wave injection (simulating dream imagery)
    size_t pgo_v1 = 0;
    for (int t = 150; t < 200; ++t) {
        v1.inject_pgo_wave(25.0f);
        v1.step(t);
        pgo_v1 += count_fired(v1);
    }

    printf("    V1 awake(50步): %zu, REM(100步): %zu, PGO(50步): %zu\n",
           awake_v1, rem_v1, pgo_v1);
    printf("    M1 REM+atonia(100步): %zu\n", rem_m1);

    TEST_ASSERT(rem_v1 > 0 || pgo_v1 > 0, "V1 active during REM or PGO");

    printf("  [PASS] CorticalRegion REM模式\n");
    g_pass++;
}

// =============================================================================
// Test 5: Hippocampus REM theta
// =============================================================================
static void test_hippocampal_rem_theta() {
    printf("\n--- 测试5: Hippocampus REM theta ---\n");
    printf("    原理: REM期间theta振荡 + 创造性重组\n");

    HippocampusConfig cfg;
    Hippocampus hipp(cfg);

    // Phase 1: Encode a pattern during "wakefulness"
    for (int t = 0; t < 100; ++t) {
        std::vector<float> input(cfg.n_ec, 0.0f);
        for (size_t i = 0; i < cfg.n_ec / 2; ++i) input[i] = 25.0f;
        hipp.inject_cortical_input(input);
        hipp.step(t);
    }

    // Phase 2: Enable REM theta
    hipp.enable_rem_theta();
    TEST_ASSERT(hipp.rem_theta_enabled(), "REM theta enabled");
    TEST_ASSERT(!hipp.sleep_replay_enabled(), "SWR disabled (mutual exclusion)");

    size_t rem_activity = 0;
    float max_theta_phase = 0.0f;
    for (int t = 100; t < 400; ++t) {
        hipp.step(t);
        rem_activity += count_fired(hipp);
        float phase = hipp.rem_theta_phase();
        if (phase > max_theta_phase) max_theta_phase = phase;
    }

    uint32_t recomb = hipp.rem_recombination_count();
    printf("    REM activity (300步): %zu\n", rem_activity);
    printf("    Theta phase max: %.3f\n", max_theta_phase);
    printf("    Creative recombination events: %u\n", recomb);

    TEST_ASSERT(rem_activity > 0, "hippocampus active during REM theta");
    TEST_ASSERT(max_theta_phase > 0.1f, "theta oscillation advancing");
    TEST_ASSERT(recomb > 0, "creative recombination occurred");

    // Disable
    hipp.disable_rem_theta();
    TEST_ASSERT(!hipp.rem_theta_enabled(), "REM theta disabled");

    printf("  [PASS] Hippocampus REM theta\n");
    g_pass++;
}

// =============================================================================
// Test 6: 完整NREM→REM→NREM睡眠周期
// =============================================================================
static void test_full_sleep_cycle() {
    printf("\n--- 测试6: 完整NREM→REM→NREM睡眠周期 ---\n");

    HippocampusConfig hcfg;
    Hippocampus hipp(hcfg);

    ColumnConfig cc;
    cc.n_l4_stellate = 30; cc.n_l23_pyramidal = 60;
    cc.n_l5_pyramidal = 30; cc.n_l6_pyramidal = 20;
    cc.n_pv_basket = 8; cc.n_sst_martinotti = 5; cc.n_vip = 3;
    CorticalRegion v1("V1", cc);

    SleepCycleConfig scfg;
    scfg.nrem_duration = 150;
    scfg.rem_duration = 100;
    scfg.rem_growth = 0;
    scfg.min_nrem_duration = 50;
    SleepCycleManager sleep(scfg);

    // Phase 1: Awake encoding (50 steps)
    size_t awake_hipp = 0;
    for (int t = 0; t < 50; ++t) {
        std::vector<float> input(hcfg.n_ec, 20.0f);
        hipp.inject_cortical_input(input);
        hipp.step(t); v1.step(t);
        awake_hipp += count_fired(hipp);
    }

    // Phase 2: Enter sleep
    sleep.enter_sleep();
    size_t nrem_swr = 0, rem_activity = 0;
    uint32_t swr_count = 0, recomb_count = 0;
    int nrem_steps = 0, rem_steps = 0;

    for (int t = 50; t < 400; ++t) {
        SleepStage prev = sleep.stage();
        sleep.step();
        SleepStage curr = sleep.stage();

        // Handle stage transitions
        if (prev != curr) {
            if (curr == SleepStage::REM) {
                // NREM → REM
                hipp.disable_sleep_replay();
                hipp.enable_rem_theta();
                v1.set_rem_mode(true);
            } else if (curr == SleepStage::NREM) {
                // REM → NREM
                hipp.disable_rem_theta();
                hipp.enable_sleep_replay();
                v1.set_sleep_mode(true);
            }
        }

        // Apply PGO during REM
        if (sleep.is_rem() && sleep.pgo_active()) {
            v1.inject_pgo_wave(scfg.rem_pgo_amplitude);
        }

        hipp.step(t); v1.step(t);

        if (sleep.is_nrem()) {
            nrem_swr += count_fired(hipp);
            nrem_steps++;
        } else if (sleep.is_rem()) {
            rem_activity += count_fired(hipp);
            rem_steps++;
        }
    }

    swr_count = hipp.swr_count();
    recomb_count = hipp.rem_recombination_count();

    printf("    Awake: hipp=%zu (50步)\n", awake_hipp);
    printf("    NREM: %d步, SWR=%u\n", nrem_steps, swr_count);
    printf("    REM: %d步, recomb=%u\n", rem_steps, recomb_count);
    printf("    Cycles: %u\n", sleep.cycle_count());

    TEST_ASSERT(nrem_steps > 0, "NREM steps occurred");
    TEST_ASSERT(rem_steps > 0, "REM steps occurred");
    TEST_ASSERT(sleep.cycle_count() >= 1, "at least 1 complete cycle");
    TEST_ASSERT(nrem_steps + rem_steps > 100, "sufficient sleep steps");

    printf("  [PASS] 完整睡眠周期\n");
    g_pass++;
}

// =============================================================================
// Test 7: 全脑NREM→REM切换
// =============================================================================
static void test_full_brain_nrem_rem() {
    printf("\n--- 测试7: 全脑NREM→REM切换 ---\n");

    SimulationEngine eng(10);
    ThalamicConfig lgn_cfg;
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    ColumnConfig cc;
    cc.n_l4_stellate = 50; cc.n_l23_pyramidal = 100;
    cc.n_l5_pyramidal = 50; cc.n_l6_pyramidal = 40;
    cc.n_pv_basket = 15; cc.n_sst_martinotti = 10; cc.n_vip = 5;
    eng.add_region(std::make_unique<CorticalRegion>("V1", cc));
    eng.add_projection("LGN", "V1", 2);

    auto* v1 = dynamic_cast<CorticalRegion*>(eng.find_region("V1"));

    // NREM phase (1200 steps to allow slow wave cycling at 0.001 freq)
    v1->set_sleep_mode(true);
    size_t nrem_spikes = 0;
    int up_count = 0, down_count = 0;
    for (int t = 0; t < 1200; ++t) {
        eng.step();
        nrem_spikes += count_fired(*v1);
        if (v1->is_up_state()) up_count++;
        else down_count++;
    }

    // Switch to REM
    v1->set_rem_mode(true);
    TEST_ASSERT(!v1->is_sleep_mode(), "NREM off after REM on");
    TEST_ASSERT(v1->is_rem_mode(), "REM mode on");

    size_t rem_spikes = 0;
    for (int t = 1200; t < 1400; ++t) {
        eng.step();
        rem_spikes += count_fired(*v1);
    }

    printf("    NREM: %zu spikes (1200步), up=%d, down=%d\n", nrem_spikes, up_count, down_count);
    printf("    REM:  %zu spikes (200步, desynchronized)\n", rem_spikes);

    TEST_ASSERT(rem_spikes > 0, "REM produces activity");
    TEST_ASSERT(up_count > 0 && down_count > 0, "NREM has up/down alternation");

    printf("  [PASS] 全脑NREM→REM切换\n");
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
    printf("  悟韵 (WuYun) Step 11 REM睡眠+梦境测试\n");
    printf("============================================\n");

    test_sleep_cycle_basics();
    test_rem_growth();
    test_pgo_waves();
    test_cortical_rem();
    test_hippocampal_rem_theta();
    test_full_sleep_cycle();
    test_full_brain_nrem_rem();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
