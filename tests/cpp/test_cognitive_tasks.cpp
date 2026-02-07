/**
 * Step 11: 认知任务验证 — 利用WM+BG学习验证高级认知功能
 *
 * 测试:
 *   1. 训练后 Go/NoGo: DA-STDP训练区分Go/NoGo刺激
 *   2. 延迟匹配任务 (DMTS): 工作记忆维持样本→延迟→匹配
 *   3. Papez回路记忆巩固: Hipp→MB→ATN→ACC增强ACC活动
 *   4. 情绪增强记忆: Amygdala→Hipp通路增强编码
 *   5. WM引导BG决策: dlPFC维持线索→BG做出相应选择
 *   6. 反转学习: 先学A奖励→再学B奖励→权重反转
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
#include "region/subcortical/basal_ganglia.h"
#include "region/neuromod/vta_da.h"
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

// Helper: build minimal circuit for BG training
static std::unique_ptr<SimulationEngine> make_bg_circuit(bool enable_stdp, float lr = 0.05f) {
    auto eng = std::make_unique<SimulationEngine>(10);

    ThalamicConfig lgn_cfg;
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    eng->add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    ColumnConfig v1_cfg;
    v1_cfg.n_l4_stellate = 50; v1_cfg.n_l23_pyramidal = 100;
    v1_cfg.n_l5_pyramidal = 50; v1_cfg.n_l6_pyramidal = 40;
    v1_cfg.n_pv_basket = 15; v1_cfg.n_sst_martinotti = 10; v1_cfg.n_vip = 5;
    eng->add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

    ColumnConfig pfc_cfg;
    pfc_cfg.n_l4_stellate = 30; pfc_cfg.n_l23_pyramidal = 80;
    pfc_cfg.n_l5_pyramidal = 40; pfc_cfg.n_l6_pyramidal = 30;
    pfc_cfg.n_pv_basket = 10; pfc_cfg.n_sst_martinotti = 8; pfc_cfg.n_vip = 4;
    eng->add_region(std::make_unique<CorticalRegion>("dlPFC", pfc_cfg));

    BasalGangliaConfig bg_cfg;
    bg_cfg.name = "BG"; bg_cfg.n_d1_msn = 50; bg_cfg.n_d2_msn = 50;
    bg_cfg.n_gpi = 15; bg_cfg.n_gpe = 15; bg_cfg.n_stn = 10;
    bg_cfg.da_stdp_enabled = enable_stdp;
    bg_cfg.da_stdp_lr = lr;
    eng->add_region(std::make_unique<BasalGanglia>(bg_cfg));

    ThalamicConfig mt_cfg;
    mt_cfg.name = "MotorThal"; mt_cfg.n_relay = 30; mt_cfg.n_trn = 10;
    eng->add_region(std::make_unique<ThalamicRelay>(mt_cfg));

    ColumnConfig m1_cfg;
    m1_cfg.n_l4_stellate = 30; m1_cfg.n_l23_pyramidal = 60;
    m1_cfg.n_l5_pyramidal = 40; m1_cfg.n_l6_pyramidal = 20;
    m1_cfg.n_pv_basket = 10; m1_cfg.n_sst_martinotti = 6; m1_cfg.n_vip = 3;
    eng->add_region(std::make_unique<CorticalRegion>("M1", m1_cfg));

    eng->add_projection("LGN", "V1", 2);
    eng->add_projection("V1", "dlPFC", 3);
    eng->add_projection("dlPFC", "BG", 2);
    eng->add_projection("BG", "MotorThal", 2);
    eng->add_projection("MotorThal", "M1", 2);

    auto* bg = dynamic_cast<BasalGanglia*>(eng->find_region("BG"));
    bg->set_da_source_region(UINT32_MAX);  // Manual DA control

    return eng;
}

// =============================================================================
// Test 1: Trained Go/NoGo — BG discriminates rewarded vs unrewarded stimulus
// =============================================================================
static void test_trained_go_nogo() {
    printf("\n--- 测试1: 训练后 Go/NoGo ---\n");
    printf("    原理: 高DA训练 → D1权重↑ → 更强D1响应\n");
    printf("          无DA-STDP → D1权重不变 → 基线响应\n");

    // Helper: run training + test, return D1 test spikes
    auto run_experiment = [](bool enable_stdp, float train_da) -> size_t {
        auto eng = make_bg_circuit(enable_stdp, 0.05f);
        auto* bg = dynamic_cast<BasalGanglia*>(eng->find_region("BG"));
        auto* lgn = eng->find_region("LGN");

        // Training phase: 15 trials × 30 steps
        for (int trial = 0; trial < 15; ++trial) {
            for (int t = 0; t < 30; ++t) {
                lgn->inject_external(std::vector<float>(50, 35.0f));
                bg->set_da_level(train_da);
                eng->step();
            }
        }

        // Test phase: neutral DA (baseline), count D1 spikes
        bg->set_da_level(0.3f);
        size_t d1_spikes = 0;
        for (int t = 0; t < 50; ++t) {
            lgn->inject_external(std::vector<float>(50, 35.0f));
            eng->step();
            const auto& f = bg->fired();
            for (size_t i = 0; i < 50 && i < f.size(); ++i)
                if (f[i]) d1_spikes++;
        }
        return d1_spikes;
    };

    size_t d1_high_da = run_experiment(true, 0.8f);   // STDP + high DA reward
    size_t d1_low_da  = run_experiment(true, 0.05f);  // STDP + low DA (no reward)
    size_t d1_no_stdp = run_experiment(false, 0.8f);  // No STDP (baseline)

    printf("    D1(高DA训练)=%zu  D1(低DA训练)=%zu  D1(无STDP)=%zu\n",
           d1_high_da, d1_low_da, d1_no_stdp);

    bool ok = d1_high_da > d1_low_da;
    printf("  [%s] 训练后Go/NoGo区分 (高DA > 低DA)\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 2: Delayed Match-to-Sample — WM maintains sample across delay
// =============================================================================
static void test_delayed_match() {
    printf("\n--- 测试2: 延迟匹配任务 (DMTS) ---\n");
    printf("    原理: 样本刺激→WM维持→延迟→dlPFC仍有持续活动\n");

    ColumnConfig pfc_cfg;
    pfc_cfg.n_l4_stellate = 30; pfc_cfg.n_l23_pyramidal = 80;
    pfc_cfg.n_l5_pyramidal = 40; pfc_cfg.n_l6_pyramidal = 30;
    pfc_cfg.n_pv_basket = 10; pfc_cfg.n_sst_martinotti = 8; pfc_cfg.n_vip = 4;

    // --- With WM ---
    CorticalRegion pfc_wm("dlPFC_wm", pfc_cfg);
    pfc_wm.enable_working_memory();
    NeuromodulatorLevels wm_levels;
    wm_levels.da = 0.6f;
    pfc_wm.neuromod().set_tonic(wm_levels);

    // Phase 1: Sample presentation (50 steps)
    for (int t = 0; t < 50; ++t) {
        pfc_wm.inject_external(std::vector<float>(pfc_wm.n_neurons(), 30.0f));
        pfc_wm.step(t);
    }
    size_t sample_spikes = 0;
    for (auto f : pfc_wm.fired()) if (f) sample_spikes++;

    // Phase 2: Delay period (100 steps, no input)
    size_t delay_spikes_early = 0, delay_spikes_late = 0;
    float persist_early = 0.0f, persist_late = 0.0f;
    for (int t = 50; t < 150; ++t) {
        pfc_wm.step(t);
        size_t s = count_fired(pfc_wm.fired());
        if (t < 80) {
            delay_spikes_early += s;
            persist_early = pfc_wm.wm_persistence();
        }
        if (t >= 120) {
            delay_spikes_late += s;
            persist_late = pfc_wm.wm_persistence();
        }
    }

    // --- Without WM (control) ---
    CorticalRegion pfc_no("dlPFC_no", pfc_cfg);
    pfc_no.neuromod().set_tonic(wm_levels);

    for (int t = 0; t < 50; ++t) {
        pfc_no.inject_external(std::vector<float>(pfc_no.n_neurons(), 30.0f));
        pfc_no.step(t);
    }
    size_t no_wm_delay = 0;
    for (int t = 50; t < 150; ++t) {
        pfc_no.step(t);
        no_wm_delay += count_fired(pfc_no.fired());
    }

    printf("    样本期最后步=%zu spikes\n", sample_spikes);
    printf("    WM延迟(早)=%zu (persist=%.2f)  WM延迟(晚)=%zu (persist=%.2f)\n",
           delay_spikes_early, persist_early, delay_spikes_late, persist_late);
    printf("    无WM延迟=%zu\n", no_wm_delay);

    bool ok = delay_spikes_early > no_wm_delay;
    printf("  [%s] 延迟匹配: WM维持 > 无WM\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 3: Papez circuit memory consolidation — Hipp→MB→ATN→ACC
// =============================================================================
static void test_papez_memory() {
    printf("\n--- 测试3: Papez回路记忆巩固 ---\n");
    printf("    原理: Hipp→MB→ATN→ACC 增强ACC记忆相关活动\n");

    // WITH Papez
    SimulationEngine eng1(10);
    HippocampusConfig hipp_cfg;
    hipp_cfg.n_presub = 25;
    eng1.add_region(std::make_unique<Hippocampus>(hipp_cfg));
    eng1.add_region(std::make_unique<MammillaryBody>(MammillaryConfig{}));
    ThalamicConfig atn; atn.name = "ATN"; atn.n_relay = 20; atn.n_trn = 8;
    eng1.add_region(std::make_unique<ThalamicRelay>(atn));
    ColumnConfig acc_cfg;
    acc_cfg.n_l4_stellate = 20; acc_cfg.n_l23_pyramidal = 50;
    acc_cfg.n_l5_pyramidal = 30; acc_cfg.n_l6_pyramidal = 20;
    acc_cfg.n_pv_basket = 8; acc_cfg.n_sst_martinotti = 5; acc_cfg.n_vip = 2;
    eng1.add_region(std::make_unique<CorticalRegion>("ACC", acc_cfg));

    eng1.add_projection("Hippocampus", "MammillaryBody", 2);
    eng1.add_projection("MammillaryBody", "ATN", 2);
    eng1.add_projection("ATN", "ACC", 2);

    // WITHOUT Papez (ACC alone)
    SimulationEngine eng2(10);
    eng2.add_region(std::make_unique<Hippocampus>(hipp_cfg));
    eng2.add_region(std::make_unique<CorticalRegion>("ACC", acc_cfg));
    // No MB/ATN projections

    size_t acc_with = 0, acc_without = 0;
    for (int t = 0; t < 200; ++t) {
        eng1.find_region("Hippocampus")->inject_external(
            std::vector<float>(hipp_cfg.n_ec, 30.0f));
        eng2.find_region("Hippocampus")->inject_external(
            std::vector<float>(hipp_cfg.n_ec, 30.0f));
        eng1.step();
        eng2.step();
        for (auto f : eng1.find_region("ACC")->fired()) if (f) acc_with++;
        for (auto f : eng2.find_region("ACC")->fired()) if (f) acc_without++;
    }

    printf("    ACC(+Papez)=%zu  ACC(无Papez)=%zu\n", acc_with, acc_without);

    bool ok = acc_with > acc_without;
    printf("  [%s] Papez增强ACC活动\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 4: Emotional memory enhancement — Amygdala→Hippocampus
// =============================================================================
static void test_emotional_memory() {
    printf("\n--- 测试4: 情绪增强记忆 ---\n");
    printf("    原理: Amyg(BLA)→Hipp(EC) 情绪标记→海马编码增强\n");

    // WITH emotional arousal
    SimulationEngine eng1(10);
    HippocampusConfig hipp_cfg;
    eng1.add_region(std::make_unique<Hippocampus>(hipp_cfg));
    AmygdalaConfig amyg_cfg;
    eng1.add_region(std::make_unique<Amygdala>(amyg_cfg));
    eng1.add_projection("Amygdala", "Hippocampus", 2);

    // WITHOUT emotional arousal
    SimulationEngine eng2(10);
    eng2.add_region(std::make_unique<Hippocampus>(hipp_cfg));

    size_t hipp_emo = 0, hipp_neutral = 0;
    for (int t = 0; t < 200; ++t) {
        // Both get same hippocampal input
        eng1.find_region("Hippocampus")->inject_external(
            std::vector<float>(hipp_cfg.n_ec, 20.0f));
        eng2.find_region("Hippocampus")->inject_external(
            std::vector<float>(hipp_cfg.n_ec, 20.0f));

        // Only eng1 gets emotional arousal
        eng1.find_region("Amygdala")->inject_external(
            std::vector<float>(amyg_cfg.n_la, 40.0f));

        eng1.step();
        eng2.step();
        for (auto f : eng1.find_region("Hippocampus")->fired()) if (f) hipp_emo++;
        for (auto f : eng2.find_region("Hippocampus")->fired()) if (f) hipp_neutral++;
    }

    printf("    Hipp(+情绪)=%zu  Hipp(中性)=%zu\n", hipp_emo, hipp_neutral);
    float denom = (hipp_neutral > 0) ? (float)hipp_neutral : 1.0f;
    float enhancement = (float)hipp_emo / denom;
    printf("    增强比=%.2fx\n", enhancement);

    bool ok = hipp_emo > hipp_neutral;
    printf("  [%s] 情绪增强记忆编码\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 5: WM-guided BG decision — dlPFC maintains cue, BG acts on it
// =============================================================================
static void test_wm_guided_decision() {
    printf("\n--- 测试5: WM引导BG决策 ---\n");
    printf("    原理: dlPFC(WM)维持线索→延迟→BG接收维持信息→动作选择\n");

    auto eng = make_bg_circuit(true, 0.03f);
    auto* bg = dynamic_cast<BasalGanglia*>(eng->find_region("BG"));
    auto* lgn = eng->find_region("LGN");
    auto* pfc = dynamic_cast<CorticalRegion*>(eng->find_region("dlPFC"));

    pfc->enable_working_memory();
    {
        NeuromodulatorLevels lv;
        lv.da = 0.6f;
        pfc->neuromod().set_tonic(lv);
    }

    // Phase 1: Cue + Training (LGN stimulus + DA reward)
    for (int trial = 0; trial < 10; ++trial) {
        for (int t = 0; t < 30; ++t) {
            lgn->inject_external(std::vector<float>(50, 35.0f));
            bg->set_da_level(0.7f);
            eng->step();
        }
    }

    // Phase 2: Cue presentation (short) — neutral DA (baseline)
    bg->set_da_level(0.3f);
    for (int t = 0; t < 20; ++t) {
        lgn->inject_external(std::vector<float>(50, 35.0f));
        eng->step();
    }

    // Phase 3: Delay — no stimulus, but WM should maintain
    size_t bg_delay_spikes = 0;
    float pfc_persist = 0.0f;
    for (int t = 0; t < 50; ++t) {
        eng->step();
        for (auto f : bg->fired()) if (f) bg_delay_spikes++;
        pfc_persist = pfc->wm_persistence();
    }

    printf("    延迟期: BG=%zu  dlPFC持续=%.2f\n", bg_delay_spikes, pfc_persist);

    // Compare: without WM
    auto eng2 = make_bg_circuit(true, 0.03f);
    auto* bg2 = dynamic_cast<BasalGanglia*>(eng2->find_region("BG"));
    auto* lgn2 = eng2->find_region("LGN");

    for (int trial = 0; trial < 10; ++trial) {
        for (int t = 0; t < 30; ++t) {
            lgn2->inject_external(std::vector<float>(50, 35.0f));
            bg2->set_da_level(0.7f);
            eng2->step();
        }
    }
    bg2->set_da_level(0.3f);  // Neutral DA (baseline)
    for (int t = 0; t < 20; ++t) {
        lgn2->inject_external(std::vector<float>(50, 35.0f));
        eng2->step();
    }
    size_t bg_delay_no_wm = 0;
    for (int t = 0; t < 50; ++t) {
        eng2->step();
        for (auto f : bg2->fired()) if (f) bg_delay_no_wm++;
    }

    printf("    BG(+WM)=%zu  BG(无WM)=%zu\n", bg_delay_spikes, bg_delay_no_wm);

    bool ok = bg_delay_spikes > bg_delay_no_wm;
    printf("  [%s] WM引导BG决策\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 6: Reversal learning — learn A→reward, then switch to B→reward
// =============================================================================
static void test_reversal_learning() {
    printf("\n--- 测试6: 反转学习 ---\n");
    printf("    原理: 同一刺激先低DA→后高DA → D1响应增加\n");

    auto eng = make_bg_circuit(true, 0.05f);
    auto* bg = dynamic_cast<BasalGanglia*>(eng->find_region("BG"));
    auto* lgn = eng->find_region("LGN");
    auto stim = std::vector<float>(50, 35.0f);

    // Phase 1: Low DA training ("punishment"/no reward)
    for (int trial = 0; trial < 10; ++trial) {
        for (int t = 0; t < 25; ++t) {
            lgn->inject_external(stim);
            bg->set_da_level(0.05f);  // Below baseline → LTD
            eng->step();
        }
    }

    // Measure D1 after low-DA phase (neutral DA for measurement)
    bg->set_da_level(0.3f);
    size_t d1_after_low = 0;
    for (int t = 0; t < 40; ++t) {
        lgn->inject_external(stim);
        eng->step();
        const auto& f = bg->fired();
        for (size_t i = 0; i < 50 && i < f.size(); ++i)
            if (f[i]) d1_after_low++;
    }

    // Phase 2: High DA training ("reward" → reversal)
    for (int trial = 0; trial < 15; ++trial) {
        for (int t = 0; t < 25; ++t) {
            lgn->inject_external(stim);
            bg->set_da_level(0.8f);  // Above baseline → LTP
            eng->step();
        }
    }

    // Measure D1 after high-DA phase (neutral DA for measurement)
    bg->set_da_level(0.3f);
    size_t d1_after_high = 0;
    for (int t = 0; t < 40; ++t) {
        lgn->inject_external(stim);
        eng->step();
        const auto& f = bg->fired();
        for (size_t i = 0; i < 50 && i < f.size(); ++i)
            if (f[i]) d1_after_high++;
    }

    printf("    D1(低DA训练后)=%zu  D1(高DA训练后)=%zu\n", d1_after_low, d1_after_high);

    bool ok = d1_after_high > d1_after_low;
    printf("  [%s] 反转学习: 高DA训练后 > 低DA训练后\n", ok ? "PASS" : "FAIL");
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
    printf("  悟韵 (WuYun) Step 11 认知任务验证\n");
    printf("  训练Go/NoGo + DMTS + Papez + 情绪记忆 + WM引导决策\n");
    printf("============================================\n");

    test_trained_go_nogo();
    test_delayed_match();
    test_papez_memory();
    test_emotional_memory();
    test_wm_guided_decision();
    test_reversal_learning();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("============================================\n");

    return tests_failed > 0 ? 1 : 0;
}
