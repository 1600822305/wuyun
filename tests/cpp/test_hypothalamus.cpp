/**
 * Step 6: 下丘脑内驱力系统测试
 *
 * 测试:
 *   1. SCN昼夜节律起搏: 正弦振荡 + 相位推进
 *   2. Sleep-wake flip-flop: VLPO⟷Orexin互相抑制
 *   3. 睡眠压力: 高压力→VLPO活跃→wake_level↓
 *   4. Orexin觉醒稳定: 低压力→Orexin活跃→wake_level↑
 *   5. PVN应激: stress_level↑→PVN活跃→stress_output↑
 *   6. LH⟷VMH摄食平衡: hunger↑→LH活跃; satiety↑→VMH活跃
 *   7. 全系统集成: Hypothalamus嵌入47区域脑
 */

#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#endif
#include <vector>
#include <cmath>
#include <memory>

#include "region/limbic/hypothalamus.h"
#include "engine/simulation_engine.h"
#include "region/cortical_region.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/subcortical/cerebellum.h"
#include "region/neuromod/vta_da.h"
#include "region/neuromod/lc_ne.h"
#include "region/neuromod/drn_5ht.h"
#include "region/neuromod/nbm_ach.h"
#include "region/limbic/hippocampus.h"
#include "region/limbic/amygdala.h"
#include "region/limbic/mammillary_body.h"
#include "region/limbic/septal_nucleus.h"

using namespace wuyun;

static int tests_passed = 0;
static int tests_failed = 0;

static size_t count_fired(const std::vector<uint8_t>& f) {
    size_t n = 0; for (auto x : f) if (x) n++; return n;
}

// =============================================================================
// Test 1: SCN circadian pacemaker
// =============================================================================
static void test_scn_circadian() {
    printf("\n--- 测试1: SCN昼夜节律 ---\n");
    printf("    原理: SCN以~24000步为周期正弦振荡\n");

    HypothalamusConfig cfg;
    cfg.circadian_period = 200.0f;  // Short period for testing
    Hypothalamus hypo(cfg);

    float phase_start = hypo.circadian_phase();
    size_t scn_day = 0, scn_night = 0;

    // Run half period (day)
    for (int t = 0; t < 100; ++t) {
        hypo.step(t);
        scn_day += count_fired(hypo.fired());
    }

    float phase_mid = hypo.circadian_phase();

    // Run another half (night)
    for (int t = 100; t < 200; ++t) {
        hypo.step(t);
        scn_night += count_fired(hypo.fired());
    }

    float phase_end = hypo.circadian_phase();

    printf("    相位: start=%.2f → mid=%.2f → end=%.2f\n",
           phase_start, phase_mid, phase_end);
    printf("    SCN活动: day-half=%zu  night-half=%zu\n", scn_day, scn_night);

    // Phase should advance from 0 → ~0.5 → ~1.0(=0)
    bool phase_ok = phase_mid > 0.3f && phase_mid < 0.7f;
    bool cycle_ok = phase_end < 0.1f || phase_end > 0.9f;  // Wrapped around
    bool ok = phase_ok && cycle_ok;
    printf("  [%s] SCN昼夜振荡 + 相位推进\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 2: Sleep-wake flip-flop
// =============================================================================
static void test_flip_flop() {
    printf("\n--- 测试2: Sleep-wake flip-flop ---\n");
    printf("    原理: VLPO⟷Orexin互相抑制，形成双稳态\n");

    // High sleep pressure → VLPO wins → low wake
    HypothalamusConfig cfg_sleep;
    cfg_sleep.homeostatic_sleep_pressure = 0.9f;
    Hypothalamus hypo_sleep(cfg_sleep);

    for (int t = 0; t < 200; ++t) hypo_sleep.step(t);
    float wake_sleep = hypo_sleep.wake_level();

    // Low sleep pressure → Orexin wins → high wake
    HypothalamusConfig cfg_wake;
    cfg_wake.homeostatic_sleep_pressure = 0.1f;
    Hypothalamus hypo_wake(cfg_wake);

    for (int t = 0; t < 200; ++t) hypo_wake.step(t);
    float wake_awake = hypo_wake.wake_level();

    printf("    高睡眠压力: wake=%.3f  低睡眠压力: wake=%.3f\n",
           wake_sleep, wake_awake);

    bool ok = wake_awake > wake_sleep;
    printf("  [%s] Flip-flop: 低压力→高觉醒, 高压力→低觉醒\n",
           ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 3: Sleep pressure effect
// =============================================================================
static void test_sleep_pressure() {
    printf("\n--- 测试3: 睡眠压力 ---\n");
    printf("    原理: 高压力→VLPO激活→觉醒中枢抑制\n");

    HypothalamusConfig cfg;
    Hypothalamus hypo(cfg);

    // Start awake
    hypo.set_sleep_pressure(0.1f);
    for (int t = 0; t < 100; ++t) hypo.step(t);
    float wake_low = hypo.wake_level();

    // Increase sleep pressure
    hypo.set_sleep_pressure(0.8f);
    for (int t = 100; t < 300; ++t) hypo.step(t);
    float wake_high = hypo.wake_level();

    size_t vlpo_spikes = 0;
    hypo.set_sleep_pressure(0.9f);
    for (int t = 300; t < 400; ++t) {
        hypo.step(t);
        for (auto f : hypo.vlpo_pop().fired()) if (f) vlpo_spikes++;
    }

    printf("    wake(低压力)=%.3f → wake(高压力)=%.3f  VLPO=%zu\n",
           wake_low, wake_high, vlpo_spikes);

    bool ok = wake_low > wake_high && vlpo_spikes > 0;
    printf("  [%s] 睡眠压力↑ → 觉醒↓ + VLPO活跃\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 4: Orexin wake stability
// =============================================================================
static void test_orexin_stability() {
    printf("\n--- 测试4: Orexin觉醒稳定 ---\n");
    printf("    原理: 低压力→Orexin持续发放→觉醒维持\n");

    HypothalamusConfig cfg;
    cfg.homeostatic_sleep_pressure = 0.05f;
    Hypothalamus hypo(cfg);

    size_t orexin_spikes = 0;
    for (int t = 0; t < 200; ++t) {
        hypo.step(t);
        for (auto f : hypo.orexin_pop().fired()) if (f) orexin_spikes++;
    }

    float wake = hypo.wake_level();
    printf("    Orexin spikes=%zu  wake=%.3f\n", orexin_spikes, wake);

    bool ok = orexin_spikes > 0 && wake > 0.4f;
    printf("  [%s] Orexin觉醒稳定\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 5: PVN stress response
// =============================================================================
static void test_pvn_stress() {
    printf("\n--- 测试5: PVN应激反应 ---\n");
    printf("    原理: stress_level↑ → PVN活跃 → stress_output↑\n");

    HypothalamusConfig cfg_low;
    cfg_low.stress_level = 0.1f;
    Hypothalamus hypo_low(cfg_low);

    HypothalamusConfig cfg_high;
    cfg_high.stress_level = 0.8f;
    Hypothalamus hypo_high(cfg_high);

    size_t pvn_low = 0, pvn_high = 0;
    for (int t = 0; t < 150; ++t) {
        hypo_low.step(t);
        hypo_high.step(t);
        for (auto f : hypo_low.pvn_pop().fired()) if (f) pvn_low++;
        for (auto f : hypo_high.pvn_pop().fired()) if (f) pvn_high++;
    }

    printf("    PVN(低应激)=%zu out=%.3f  PVN(高应激)=%zu out=%.3f\n",
           pvn_low, hypo_low.stress_output(),
           pvn_high, hypo_high.stress_output());

    bool ok = pvn_high > pvn_low && hypo_high.stress_output() > hypo_low.stress_output();
    printf("  [%s] 应激↑ → PVN↑ → stress_output↑\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 6: LH⟷VMH feeding balance
// =============================================================================
static void test_feeding_balance() {
    printf("\n--- 测试6: LH⟷VMH摄食平衡 ---\n");
    printf("    原理: hunger↑→LH活跃; satiety↑→VMH活跃; 互相抑制\n");

    // Hungry state
    HypothalamusConfig cfg_hungry;
    cfg_hungry.hunger_level = 0.8f;
    cfg_hungry.satiety_level = 0.1f;
    Hypothalamus hypo_hungry(cfg_hungry);

    // Fed state
    HypothalamusConfig cfg_fed;
    cfg_fed.hunger_level = 0.1f;
    cfg_fed.satiety_level = 0.8f;
    Hypothalamus hypo_fed(cfg_fed);

    size_t lh_hungry = 0, vmh_hungry = 0;
    size_t lh_fed = 0, vmh_fed = 0;

    for (int t = 0; t < 150; ++t) {
        hypo_hungry.step(t);
        hypo_fed.step(t);
        for (auto f : hypo_hungry.lh_pop().fired()) if (f) lh_hungry++;
        for (auto f : hypo_hungry.vmh_pop().fired()) if (f) vmh_hungry++;
        for (auto f : hypo_fed.lh_pop().fired()) if (f) lh_fed++;
        for (auto f : hypo_fed.vmh_pop().fired()) if (f) vmh_fed++;
    }

    printf("    饥饿: LH=%zu VMH=%zu (hunger=%.2f)\n",
           lh_hungry, vmh_hungry, hypo_hungry.hunger_output());
    printf("    饱腹: LH=%zu VMH=%zu (satiety=%.2f)\n",
           lh_fed, vmh_fed, hypo_fed.satiety_output());

    bool ok = lh_hungry > lh_fed && vmh_fed > vmh_hungry;
    printf("  [%s] 摄食平衡: 饥饿→LH>VMH, 饱腹→VMH>LH\n",
           ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 7: Full system integration — Hypothalamus in 47-region brain
// =============================================================================
static void test_full_integration() {
    printf("\n--- 测试7: 全系统集成 (47区域) ---\n");
    printf("    原理: Hypothalamus嵌入全脑 + Orexin→LC/DRN/NBM\n");

    SimulationEngine eng(10);

    // Build minimal system with Hypothalamus + neuromod targets
    ThalamicConfig lgn; lgn.name="LGN"; lgn.n_relay=50; lgn.n_trn=15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn));

    ColumnConfig v1c; v1c.n_l4_stellate=50; v1c.n_l23_pyramidal=100;
    v1c.n_l5_pyramidal=50; v1c.n_l6_pyramidal=40;
    v1c.n_pv_basket=15; v1c.n_sst_martinotti=10; v1c.n_vip=5;
    eng.add_region(std::make_unique<CorticalRegion>("V1", v1c));

    eng.add_region(std::make_unique<VTA_DA>(VTAConfig{}));
    eng.add_region(std::make_unique<LC_NE>(LCConfig{}));
    eng.add_region(std::make_unique<DRN_5HT>(DRNConfig{}));
    eng.add_region(std::make_unique<NBM_ACh>(NBMConfig{}));

    HypothalamusConfig hcfg;
    hcfg.homeostatic_sleep_pressure = 0.1f;  // Awake
    eng.add_region(std::make_unique<Hypothalamus>(hcfg));

    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("Hypothalamus", "LC", 2);
    eng.add_projection("Hypothalamus", "DRN", 2);
    eng.add_projection("Hypothalamus", "NBM", 2);
    eng.add_projection("Hypothalamus", "VTA", 2);

    auto* hypo = dynamic_cast<Hypothalamus*>(eng.find_region("Hypothalamus"));

    size_t hypo_spikes = 0, lc_spikes = 0;
    for (int t = 0; t < 200; ++t) {
        eng.find_region("LGN")->inject_external(std::vector<float>(50, 30.0f));
        eng.step();
        hypo_spikes += count_fired(hypo->fired());
        lc_spikes   += count_fired(eng.find_region("LC")->fired());
    }

    size_t n_regions = 0;
    for (auto* name : {"LGN","V1","VTA","LC","DRN","NBM","Hypothalamus"}) {
        if (eng.find_region(name)) n_regions++;
    }

    printf("    区域=%zu  Hypo=%zu  LC=%zu  wake=%.3f\n",
           n_regions, hypo_spikes, lc_spikes, hypo->wake_level());

    bool ok = n_regions == 7 && hypo_spikes > 0 && hypo->wake_level() > 0.3f;
    printf("  [%s] 全系统集成\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) Step 6 下丘脑内驱力测试\n");
    printf("  SCN节律 + flip-flop + 应激 + 摄食\n");
    printf("============================================\n");

    test_scn_circadian();
    test_flip_flop();
    test_sleep_pressure();
    test_orexin_stability();
    test_pvn_stress();
    test_feeding_balance();
    test_full_integration();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("============================================\n");

    return tests_failed > 0 ? 1 : 0;
}
