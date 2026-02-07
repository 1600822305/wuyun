/**
 * Step 4 补全测试: 隔核 + 乳头体 + Papez回路 + 前下托/HATA + 杏仁核扩展
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
#include "region/limbic/septal_nucleus.h"
#include "region/limbic/mammillary_body.h"
#include "region/limbic/hippocampus.h"
#include "region/limbic/amygdala.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/cortical_region.h"
#include "region/neuromod/vta_da.h"

using namespace wuyun;

static int tests_passed = 0;
static int tests_failed = 0;

// =============================================================================
// Test 1: SeptalNucleus theta pacemaker
// =============================================================================
static void test_septal_theta() {
    printf("\n--- 测试1: 隔核 theta 起搏 ---\n");
    printf("    原理: GABA神经元以theta频率(~6.7Hz)节律发放\n");

    SeptalConfig cfg;
    cfg.theta_period = 150.0f;  // ~6.7 Hz
    SeptalNucleus sep(cfg);

    // Run 500 steps, count GABA spikes in burst vs non-burst phases
    size_t burst_spikes = 0;
    size_t silent_spikes = 0;
    for (int t = 0; t < 500; ++t) {
        sep.step(t, 1.0f);
        float phase = sep.theta_phase();
        const auto& f = sep.fired();
        size_t n_ach = cfg.n_ach;
        // GABA neurons are after ACh in fired array
        for (size_t i = n_ach; i < f.size(); ++i) {
            if (f[i]) {
                if (phase < 0.2f) burst_spikes++;
                else silent_spikes++;
            }
        }
    }

    float ach = sep.ach_output();
    printf("    GABA burst期=%zu  silent期=%zu  ACh输出=%.3f\n",
           burst_spikes, silent_spikes, ach);

    bool ok = burst_spikes > silent_spikes;
    printf("  [%s] 隔核theta起搏\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 2: MammillaryBody relay
// =============================================================================
static void test_mammillary_body() {
    printf("\n--- 测试2: 乳头体中继 ---\n");
    printf("    原理: 外部输入→内侧核→外侧核 信号传播\n");

    MammillaryConfig cfg;
    MammillaryBody mb(cfg);

    // Inject input and check propagation
    size_t medial_spikes = 0;
    size_t lateral_spikes = 0;
    for (int t = 0; t < 100; ++t) {
        mb.inject_external(std::vector<float>(cfg.n_medial, 30.0f));
        mb.step(t, 1.0f);
        for (size_t i = 0; i < mb.medial().size(); ++i)
            if (mb.medial().fired()[i]) medial_spikes++;
        for (size_t i = 0; i < mb.lateral().size(); ++i)
            if (mb.lateral().fired()[i]) lateral_spikes++;
    }

    printf("    内侧核=%zu  外侧核=%zu\n", medial_spikes, lateral_spikes);

    bool ok = medial_spikes > 0 && lateral_spikes > 0;
    printf("  [%s] 乳头体中继\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 3: Hippocampus Presubiculum + HATA
// =============================================================================
static void test_hipp_presub_hata() {
    printf("\n--- 测试3: 前下托 + HATA 扩展 ---\n");
    printf("    原理: CA1→Presub→EC反馈 + CA1→HATA(过渡区)\n");

    HippocampusConfig cfg;
    cfg.n_presub = 25;
    cfg.n_hata   = 15;
    Hippocampus hipp(cfg);

    // Inject input to EC and run
    size_t presub_spikes = 0;
    size_t hata_spikes = 0;
    size_t ca1_spikes = 0;
    for (int t = 0; t < 200; ++t) {
        hipp.inject_cortical_input(std::vector<float>(cfg.n_ec, 30.0f));
        hipp.step(t, 1.0f);
        for (size_t i = 0; i < hipp.ca1().size(); ++i)
            if (hipp.ca1().fired()[i]) ca1_spikes++;
        for (size_t i = 0; i < hipp.presub().size(); ++i)
            if (hipp.presub().fired()[i]) presub_spikes++;
        for (size_t i = 0; i < hipp.hata().size(); ++i)
            if (hipp.hata().fired()[i]) hata_spikes++;
    }

    printf("    CA1=%zu  Presub=%zu  HATA=%zu\n", ca1_spikes, presub_spikes, hata_spikes);
    printf("    has_presub=%d  has_hata=%d\n", hipp.has_presub(), hipp.has_hata());

    bool ok = presub_spikes > 0 && hata_spikes > 0;
    printf("  [%s] 前下托+HATA\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 4: Hippocampus backward compat (no presub/hata)
// =============================================================================
static void test_hipp_backward_compat() {
    printf("\n--- 测试4: Hippocampus向后兼容 ---\n");
    printf("    原理: 默认config(presub=0,hata=0)行为不变\n");

    HippocampusConfig cfg;  // Default: n_presub=0, n_hata=0
    Hippocampus hipp(cfg);

    size_t total_spikes = 0;
    for (int t = 0; t < 100; ++t) {
        hipp.inject_cortical_input(std::vector<float>(cfg.n_ec, 25.0f));
        hipp.step(t, 1.0f);
        for (auto f : hipp.fired()) if (f) total_spikes++;
    }

    printf("    总发放=%zu  has_presub=%d  has_hata=%d  n_neurons=%zu\n",
           total_spikes, hipp.has_presub(), hipp.has_hata(), hipp.n_neurons());

    // Default neuron count = 505 (same as before)
    bool ok = hipp.n_neurons() == 505 && !hipp.has_presub() && !hipp.has_hata();
    printf("  [%s] Hippocampus向后兼容\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 5: Amygdala MeA/CoA/AB expansion
// =============================================================================
static void test_amygdala_expansion() {
    printf("\n--- 测试5: 杏仁核扩展 MeA/CoA/AB ---\n");
    printf("    原理: La→MeA→CeA, La→CoA, BLA→AB→CeA\n");

    AmygdalaConfig cfg;
    cfg.n_mea = 20;
    cfg.n_coa = 15;
    cfg.n_ab  = 20;
    Amygdala amyg(cfg);

    size_t mea_spikes = 0, coa_spikes = 0, ab_spikes = 0, cea_spikes = 0;
    for (int t = 0; t < 200; ++t) {
        amyg.inject_sensory(std::vector<float>(cfg.n_la, 30.0f));
        amyg.step(t, 1.0f);
        for (size_t i = 0; i < amyg.mea().size(); ++i)
            if (amyg.mea().fired()[i]) mea_spikes++;
        for (size_t i = 0; i < amyg.coa().size(); ++i)
            if (amyg.coa().fired()[i]) coa_spikes++;
        for (size_t i = 0; i < amyg.ab().size(); ++i)
            if (amyg.ab().fired()[i]) ab_spikes++;
        for (size_t i = 0; i < amyg.cea().size(); ++i)
            if (amyg.cea().fired()[i]) cea_spikes++;
    }

    printf("    MeA=%zu  CoA=%zu  AB=%zu  CeA=%zu\n",
           mea_spikes, coa_spikes, ab_spikes, cea_spikes);

    bool ok = mea_spikes > 0 && coa_spikes > 0 && ab_spikes > 0 && cea_spikes > 0;
    printf("  [%s] 杏仁核MeA/CoA/AB\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 6: Amygdala backward compat
// =============================================================================
static void test_amygdala_backward_compat() {
    printf("\n--- 测试6: Amygdala向后兼容 ---\n");
    printf("    原理: 默认config(mea=coa=ab=0)行为不变\n");

    AmygdalaConfig cfg;  // Default: n_mea=n_coa=n_ab=0
    Amygdala amyg(cfg);

    printf("    n_neurons=%zu  has_mea=%d  has_coa=%d  has_ab=%d\n",
           amyg.n_neurons(), amyg.has_mea(), amyg.has_coa(), amyg.has_ab());

    // Default: 180 neurons (50+80+30+20)
    bool ok = amyg.n_neurons() == 180 && !amyg.has_mea() && !amyg.has_coa() && !amyg.has_ab();
    printf("  [%s] Amygdala向后兼容\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 7: Papez circuit — Hipp→MB→ATN→ACC
// =============================================================================
static void test_papez_circuit() {
    printf("\n--- 测试7: Papez回路 ---\n");
    printf("    原理: Hipp(Sub)→乳头体→丘脑前核(ATN)→ACC\n");

    SimulationEngine eng(10);

    HippocampusConfig hipp_cfg;
    hipp_cfg.n_presub = 25;
    eng.add_region(std::make_unique<Hippocampus>(hipp_cfg));

    eng.add_region(std::make_unique<MammillaryBody>(MammillaryConfig{}));

    ThalamicConfig atn_cfg;
    atn_cfg.name = "ATN"; atn_cfg.n_relay = 20; atn_cfg.n_trn = 8;
    eng.add_region(std::make_unique<ThalamicRelay>(atn_cfg));

    ColumnConfig acc_cfg;
    acc_cfg.n_l4_stellate = 20; acc_cfg.n_l23_pyramidal = 50;
    acc_cfg.n_l5_pyramidal = 30; acc_cfg.n_l6_pyramidal = 20;
    acc_cfg.n_pv_basket = 8; acc_cfg.n_sst_martinotti = 5; acc_cfg.n_vip = 2;
    eng.add_region(std::make_unique<CorticalRegion>("ACC", acc_cfg));

    eng.add_projection("Hippocampus", "MammillaryBody", 2);
    eng.add_projection("MammillaryBody", "ATN", 2);
    eng.add_projection("ATN", "ACC", 2);

    auto* hipp = eng.find_region("Hippocampus");

    // Stimulate hippocampus and check signal reaches ACC
    size_t mb_spikes = 0, atn_spikes = 0, acc_spikes = 0;
    for (int t = 0; t < 200; ++t) {
        hipp->inject_external(std::vector<float>(hipp_cfg.n_ec, 30.0f));
        eng.step();
        for (auto f : eng.find_region("MammillaryBody")->fired()) if (f) mb_spikes++;
        for (auto f : eng.find_region("ATN")->fired()) if (f) atn_spikes++;
        for (auto f : eng.find_region("ACC")->fired()) if (f) acc_spikes++;
    }

    printf("    Hipp→MB=%zu  MB→ATN=%zu  ATN→ACC=%zu\n",
           mb_spikes, atn_spikes, acc_spikes);

    bool ok = mb_spikes > 0 && atn_spikes > 0 && acc_spikes > 0;
    printf("  [%s] Papez回路\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 8: Septal→Hippocampus theta modulation
// =============================================================================
static void test_septal_hipp_modulation() {
    printf("\n--- 测试8: 隔核→海马 theta调制 ---\n");
    printf("    原理: 隔核GABA节律→海马basket→theta震荡\n");

    SimulationEngine eng(10);

    eng.add_region(std::make_unique<SeptalNucleus>(SeptalConfig{}));

    HippocampusConfig hipp_cfg;
    eng.add_region(std::make_unique<Hippocampus>(hipp_cfg));

    eng.add_projection("SeptalNucleus", "Hippocampus", 1);

    auto* hipp = eng.find_region("Hippocampus");

    // Run with septal input + hippocampal input
    size_t hipp_spikes = 0;
    for (int t = 0; t < 300; ++t) {
        hipp->inject_external(std::vector<float>(hipp_cfg.n_ec, 20.0f));
        eng.step();
        for (auto f : hipp->fired()) if (f) hipp_spikes++;
    }

    // Compare with no septal
    SimulationEngine eng2(10);
    eng2.add_region(std::make_unique<Hippocampus>(hipp_cfg));
    auto* hipp2 = eng2.find_region("Hippocampus");

    size_t hipp_spikes_no_sep = 0;
    for (int t = 0; t < 300; ++t) {
        hipp2->inject_external(std::vector<float>(hipp_cfg.n_ec, 20.0f));
        eng2.step();
        for (auto f : hipp2->fired()) if (f) hipp_spikes_no_sep++;
    }

    printf("    Hipp(+Septal)=%zu  Hipp(无Septal)=%zu\n",
           hipp_spikes, hipp_spikes_no_sep);

    // Septal modulation should change hippocampal activity
    bool ok = hipp_spikes != hipp_spikes_no_sep;
    printf("  [%s] 隔核→海马调制\n", ok ? "PASS" : "FAIL");
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
    printf("  悟韵 (WuYun) Step 4 补全测试\n");
    printf("  隔核 + 乳头体 + Papez + 前下托/HATA + 杏仁核扩展\n");
    printf("============================================\n");

    test_septal_theta();
    test_mammillary_body();
    test_hipp_presub_hata();
    test_hipp_backward_compat();
    test_amygdala_expansion();
    test_amygdala_backward_compat();
    test_papez_circuit();
    test_septal_hipp_modulation();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("============================================\n");

    return tests_failed > 0 ? 1 : 0;
}
