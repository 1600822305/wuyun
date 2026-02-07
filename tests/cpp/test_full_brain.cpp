/**
 * Step 5 完整: 全脑系统测试
 *
 * 测试:
 *   1. 全系统构建: 所有区域+投射正确实例化
 *   2. 感觉通路: VPL→S1→S2, MGN→A1, 化学感觉
 *   3. 运动层级: dlPFC→SMA/PMC→M1 + BG→VA→PMC
 *   4. 语言回路: A1→Wernicke→Broca→PMC (弓状束)
 *   5. 默认模式网络: PCC↔vmPFC, TPJ↔PCC
 *   6. 丘脑核团: Pulvinar视觉注意, MD↔PFC, CeM/ILN觉醒
 *   7. 向后兼容: 原有通路信号传播不变
 */

#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#endif
#include <vector>
#include <string>
#include <memory>

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

// Helper: build the full brain (mirrors build_standard_brain)
static SimulationEngine build_full_brain() {
    SimulationEngine eng(10);

    ThalamicConfig lgn_cfg;
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    auto add_ctx = [&](const std::string& name, size_t l4, size_t l23,
                       size_t l5, size_t l6, size_t pv, size_t sst, size_t vip) {
        ColumnConfig c;
        c.n_l4_stellate = l4; c.n_l23_pyramidal = l23;
        c.n_l5_pyramidal = l5; c.n_l6_pyramidal = l6;
        c.n_pv_basket = pv; c.n_sst_martinotti = sst; c.n_vip = vip;
        eng.add_region(std::make_unique<CorticalRegion>(name, c));
    };

    add_ctx("V1",50,100,50,40,15,10,5); add_ctx("V2",40,80,40,30,12,8,4);
    add_ctx("V4",30,60,30,25,10,6,3);   add_ctx("IT",20,50,25,20,8,5,2);
    add_ctx("MT",35,70,35,25,10,7,3);   add_ctx("PPC",30,65,35,25,10,6,3);
    add_ctx("S1",40,80,40,30,12,8,4);   add_ctx("S2",25,50,25,20,8,5,2);
    add_ctx("A1",35,70,35,25,10,7,3);
    add_ctx("Gustatory",15,35,18,12,5,3,2); add_ctx("Piriform",15,35,18,12,5,3,2);
    add_ctx("OFC",25,60,30,20,8,5,3);   add_ctx("vmPFC",20,55,30,20,8,5,2);
    add_ctx("ACC",20,50,30,20,8,5,2);   add_ctx("dlPFC",30,80,40,30,10,8,4);
    add_ctx("FEF",20,45,25,18,7,4,2);
    add_ctx("PMC",25,55,35,20,8,5,3);   add_ctx("SMA",20,45,30,18,7,4,2);
    add_ctx("M1",30,60,40,20,10,6,3);
    add_ctx("PCC",18,45,25,18,6,4,2);   add_ctx("Insula",20,50,25,18,8,5,2);
    add_ctx("TPJ",20,50,25,18,7,5,2);   add_ctx("Broca",20,50,30,20,8,5,2);
    add_ctx("Wernicke",18,45,25,18,7,4,2);

    BasalGangliaConfig bg;
    bg.name="BG"; bg.n_d1_msn=50; bg.n_d2_msn=50; bg.n_gpi=15; bg.n_gpe=15; bg.n_stn=10;
    eng.add_region(std::make_unique<BasalGanglia>(bg));

    auto add_thal = [&](const std::string& name, size_t relay, size_t trn) {
        ThalamicConfig tc; tc.name=name; tc.n_relay=relay; tc.n_trn=trn;
        eng.add_region(std::make_unique<ThalamicRelay>(tc));
    };
    add_thal("MotorThal",30,10); add_thal("VPL",25,8); add_thal("MGN",20,6);
    add_thal("MD",25,8); add_thal("VA",20,6); add_thal("LP",18,6);
    add_thal("LD",15,5); add_thal("Pulvinar",30,10); add_thal("CeM",15,5);
    add_thal("ILN",12,4); add_thal("ATN",20,8);

    eng.add_region(std::make_unique<VTA_DA>(VTAConfig{}));
    HippocampusConfig hc; hc.n_presub=25; hc.n_hata=15;
    eng.add_region(std::make_unique<Hippocampus>(hc));
    AmygdalaConfig ac; ac.n_mea=20; ac.n_coa=15; ac.n_ab=20;
    eng.add_region(std::make_unique<Amygdala>(ac));
    eng.add_region(std::make_unique<Cerebellum>(CerebellumConfig{}));
    eng.add_region(std::make_unique<LC_NE>(LCConfig{}));
    eng.add_region(std::make_unique<DRN_5HT>(DRNConfig{}));
    eng.add_region(std::make_unique<NBM_ACh>(NBMConfig{}));
    eng.add_region(std::make_unique<SeptalNucleus>(SeptalConfig{}));
    eng.add_region(std::make_unique<MammillaryBody>(MammillaryConfig{}));

    // Visual
    eng.add_projection("LGN","V1",2); eng.add_projection("V1","V2",2);
    eng.add_projection("V2","V4",2); eng.add_projection("V4","IT",2);
    eng.add_projection("V2","V1",3); eng.add_projection("V4","V2",3);
    eng.add_projection("IT","V4",3);
    eng.add_projection("V1","MT",2); eng.add_projection("V2","MT",2);
    eng.add_projection("MT","PPC",2); eng.add_projection("PPC","MT",3);
    eng.add_projection("PPC","IT",3); eng.add_projection("IT","PPC",3);
    eng.add_projection("MT","FEF",2); eng.add_projection("FEF","V4",3);
    eng.add_projection("FEF","MT",3);
    // Pulvinar
    eng.add_projection("V1","Pulvinar",2); eng.add_projection("Pulvinar","V2",2);
    eng.add_projection("Pulvinar","V4",2); eng.add_projection("Pulvinar","MT",2);
    eng.add_projection("Pulvinar","PPC",2); eng.add_projection("FEF","Pulvinar",2);
    // Somatosensory
    eng.add_projection("VPL","S1",2); eng.add_projection("S1","S2",2);
    eng.add_projection("S2","S1",3); eng.add_projection("S1","M1",2);
    eng.add_projection("S2","PPC",2); eng.add_projection("S1","Insula",2);
    // Auditory
    eng.add_projection("MGN","A1",2); eng.add_projection("A1","Wernicke",2);
    eng.add_projection("A1","TPJ",2);
    // Chemical
    eng.add_projection("Gustatory","Insula",2); eng.add_projection("Gustatory","OFC",2);
    eng.add_projection("Piriform","Amygdala",2); eng.add_projection("Piriform","OFC",2);
    eng.add_projection("Piriform","Hippocampus",2);
    // Prefrontal
    eng.add_projection("IT","OFC",3); eng.add_projection("OFC","vmPFC",2);
    eng.add_projection("vmPFC","BG",2); eng.add_projection("vmPFC","Amygdala",3);
    eng.add_projection("ACC","dlPFC",2); eng.add_projection("ACC","LC",2);
    eng.add_projection("dlPFC","ACC",2); eng.add_projection("IT","dlPFC",3);
    eng.add_projection("PPC","dlPFC",3); eng.add_projection("dlPFC","FEF",2);
    eng.add_projection("Insula","ACC",2); eng.add_projection("Insula","Amygdala",2);
    eng.add_projection("OFC","Insula",2);
    // MD
    eng.add_projection("MD","dlPFC",2); eng.add_projection("MD","OFC",2);
    eng.add_projection("MD","ACC",2); eng.add_projection("dlPFC","MD",3);
    // Motor
    eng.add_projection("PPC","PMC",2); eng.add_projection("dlPFC","PMC",2);
    eng.add_projection("PMC","M1",2); eng.add_projection("SMA","M1",2);
    eng.add_projection("SMA","PMC",2); eng.add_projection("dlPFC","SMA",2);
    eng.add_projection("BG","VA",2); eng.add_projection("VA","PMC",2);
    eng.add_projection("VA","SMA",2); eng.add_projection("dlPFC","BG",2);
    eng.add_projection("BG","MotorThal",2); eng.add_projection("MotorThal","M1",2);
    eng.add_projection("M1","Cerebellum",2); eng.add_projection("Cerebellum","MotorThal",2);
    eng.add_projection("PPC","M1",3);
    // Language
    eng.add_projection("Wernicke","Broca",2); eng.add_projection("Broca","PMC",2);
    eng.add_projection("Broca","dlPFC",2); eng.add_projection("Wernicke","TPJ",2);
    eng.add_projection("Wernicke","IT",3); eng.add_projection("dlPFC","Broca",2);
    // DMN
    eng.add_projection("PCC","vmPFC",2); eng.add_projection("vmPFC","PCC",2);
    eng.add_projection("PCC","Hippocampus",2); eng.add_projection("TPJ","PCC",2);
    eng.add_projection("PCC","TPJ",2); eng.add_projection("TPJ","dlPFC",2);
    // LP/LD
    eng.add_projection("LP","PPC",2); eng.add_projection("PPC","LP",3);
    eng.add_projection("LD","PCC",2); eng.add_projection("LD","Hippocampus",2);
    // CeM/ILN
    eng.add_projection("CeM","BG",2); eng.add_projection("CeM","ACC",2);
    eng.add_projection("ILN","dlPFC",2); eng.add_projection("ILN","ACC",2);
    eng.add_projection("ACC","CeM",2);
    // Limbic
    eng.add_projection("V1","Amygdala",2); eng.add_projection("dlPFC","Amygdala",2);
    eng.add_projection("Amygdala","OFC",2); eng.add_projection("dlPFC","Hippocampus",3);
    eng.add_projection("Hippocampus","dlPFC",3); eng.add_projection("Amygdala","VTA",2);
    eng.add_projection("Amygdala","Hippocampus",2); eng.add_projection("Amygdala","Insula",2);
    eng.add_projection("VTA","BG",1);
    // Papez
    eng.add_projection("Hippocampus","MammillaryBody",2);
    eng.add_projection("MammillaryBody","ATN",2); eng.add_projection("ATN","ACC",2);
    eng.add_projection("SeptalNucleus","Hippocampus",1);

    // Neuromod
    using NM = SimulationEngine::NeuromodType;
    eng.register_neuromod_source("VTA",NM::DA); eng.register_neuromod_source("LC",NM::NE);
    eng.register_neuromod_source("DRN",NM::SHT); eng.register_neuromod_source("NBM",NM::ACh);

    auto* bg_ptr = dynamic_cast<BasalGanglia*>(eng.find_region("BG"));
    auto* vta_ptr = eng.find_region("VTA");
    if (bg_ptr && vta_ptr) bg_ptr->set_da_source_region(vta_ptr->region_id());
    auto* amyg_ptr = dynamic_cast<Amygdala*>(eng.find_region("Amygdala"));
    auto* pfc_ptr = eng.find_region("dlPFC");
    if (amyg_ptr && pfc_ptr) amyg_ptr->set_pfc_source_region(pfc_ptr->region_id());

    return eng;
}

// =============================================================================
// Test 1: Full system build — all regions + projections
// =============================================================================
static void test_full_build() {
    printf("\n--- 测试1: 全系统构建 ---\n");
    auto eng = build_full_brain();

    size_t n_regions = 0;
    size_t total_neurons = 0;
    std::vector<std::string> names = {
        "LGN","V1","V2","V4","IT","MT","PPC",
        "S1","S2","A1","Gustatory","Piriform",
        "OFC","vmPFC","ACC","dlPFC","FEF",
        "PMC","SMA","M1",
        "PCC","Insula","TPJ","Broca","Wernicke",
        "BG","MotorThal","VPL","MGN","MD","VA","LP","LD",
        "Pulvinar","CeM","ILN","ATN",
        "VTA","Hippocampus","Amygdala","Cerebellum",
        "LC","DRN","NBM","SeptalNucleus","MammillaryBody"
    };

    for (auto& name : names) {
        auto* r = eng.find_region(name);
        if (r) {
            n_regions++;
            total_neurons += r->n_neurons();
        } else {
            printf("    [MISS] %s\n", name.c_str());
        }
    }

    printf("    区域: %zu/%zu  神经元: %zu\n", n_regions, names.size(), total_neurons);

    bool ok = n_regions == names.size() && total_neurons > 5000;
    printf("  [%s] 全系统构建 (%zu区域, %zu神经元)\n",
           ok ? "PASS" : "FAIL", n_regions, total_neurons);
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 2: Somatosensory pathway — VPL→S1→S2→PPC
// =============================================================================
static void test_somatosensory() {
    printf("\n--- 测试2: 体感通路 VPL→S1→S2→PPC ---\n");
    auto eng = build_full_brain();

    for (int t = 0; t < 150; ++t) {
        if (20 <= t && t < 120)
            eng.find_region("VPL")->inject_external(std::vector<float>(25, 35.0f));
        eng.step();
    }

    size_t s1 = 0, s2 = 0, ppc = 0;
    // Rerun to measure steady state
    for (int t = 0; t < 100; ++t) {
        eng.find_region("VPL")->inject_external(std::vector<float>(25, 35.0f));
        eng.step();
        s1  += count_fired(eng.find_region("S1")->fired());
        s2  += count_fired(eng.find_region("S2")->fired());
        ppc += count_fired(eng.find_region("PPC")->fired());
    }

    printf("    S1=%zu  S2=%zu  PPC=%zu\n", s1, s2, ppc);
    bool ok = s1 > 0 && s2 > 0;
    printf("  [%s] 体感通路\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 3: Auditory → Language — MGN→A1→Wernicke→Broca
// =============================================================================
static void test_auditory_language() {
    printf("\n--- 测试3: 听觉→语言 MGN→A1→Wernicke→Broca ---\n");
    auto eng = build_full_brain();

    for (int t = 0; t < 100; ++t)
        eng.step();  // warmup

    size_t a1 = 0, wer = 0, bro = 0;
    for (int t = 0; t < 150; ++t) {
        eng.find_region("MGN")->inject_external(std::vector<float>(20, 35.0f));
        eng.step();
        a1  += count_fired(eng.find_region("A1")->fired());
        wer += count_fired(eng.find_region("Wernicke")->fired());
        bro += count_fired(eng.find_region("Broca")->fired());
    }

    printf("    A1=%zu  Wernicke=%zu  Broca=%zu\n", a1, wer, bro);
    bool ok = a1 > 0 && wer > 0;
    printf("  [%s] 听觉→语言通路\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 4: Motor hierarchy — dlPFC→SMA/PMC→M1
// =============================================================================
static void test_motor_hierarchy() {
    printf("\n--- 测试4: 运动层级 dlPFC→SMA/PMC→M1 ---\n");
    auto eng = build_full_brain();

    for (int t = 0; t < 50; ++t) eng.step();

    size_t pmc = 0, sma = 0, m1 = 0;
    for (int t = 0; t < 150; ++t) {
        eng.find_region("dlPFC")->inject_external(
            std::vector<float>(30, 35.0f));
        eng.step();
        pmc += count_fired(eng.find_region("PMC")->fired());
        sma += count_fired(eng.find_region("SMA")->fired());
        m1  += count_fired(eng.find_region("M1")->fired());
    }

    printf("    PMC=%zu  SMA=%zu  M1=%zu\n", pmc, sma, m1);
    bool ok = pmc > 0 && m1 > 0;
    printf("  [%s] 运动层级\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 5: Default mode network — PCC↔vmPFC + TPJ
// =============================================================================
static void test_dmn() {
    printf("\n--- 测试5: 默认模式网络 PCC↔vmPFC + TPJ ---\n");
    auto eng = build_full_brain();

    for (int t = 0; t < 50; ++t) eng.step();

    size_t pcc = 0, vmpfc = 0, tpj = 0;
    for (int t = 0; t < 150; ++t) {
        eng.find_region("PCC")->inject_external(
            std::vector<float>(18, 30.0f));
        eng.step();
        pcc   += count_fired(eng.find_region("PCC")->fired());
        vmpfc += count_fired(eng.find_region("vmPFC")->fired());
        tpj   += count_fired(eng.find_region("TPJ")->fired());
    }

    printf("    PCC=%zu  vmPFC=%zu  TPJ=%zu\n", pcc, vmpfc, tpj);
    bool ok = pcc > 0 && (vmpfc > 0 || tpj > 0);
    printf("  [%s] 默认模式网络\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 6: Pulvinar attention hub — V1→Pulvinar→V2/V4/MT/PPC
// =============================================================================
static void test_pulvinar() {
    printf("\n--- 测试6: Pulvinar视觉注意枢纽 ---\n");
    auto eng = build_full_brain();

    for (int t = 0; t < 50; ++t) eng.step();

    size_t pulv = 0, v2 = 0, v4 = 0;
    for (int t = 0; t < 150; ++t) {
        eng.find_region("LGN")->inject_external(std::vector<float>(50, 35.0f));
        eng.step();
        pulv += count_fired(eng.find_region("Pulvinar")->fired());
        v2   += count_fired(eng.find_region("V2")->fired());
        v4   += count_fired(eng.find_region("V4")->fired());
    }

    printf("    Pulvinar=%zu  V2=%zu  V4=%zu\n", pulv, v2, v4);
    bool ok = pulv > 0 && v2 > 0;
    printf("  [%s] Pulvinar视觉枢纽\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 7: MD↔PFC reciprocal — MD→dlPFC/OFC/ACC
// =============================================================================
static void test_md_pfc() {
    printf("\n--- 测试7: MD↔PFC双向投射 ---\n");
    auto eng = build_full_brain();

    for (int t = 0; t < 50; ++t) eng.step();

    size_t md = 0, dlpfc = 0, ofc = 0, acc = 0;
    for (int t = 0; t < 150; ++t) {
        eng.find_region("MD")->inject_external(std::vector<float>(25, 35.0f));
        eng.step();
        md    += count_fired(eng.find_region("MD")->fired());
        dlpfc += count_fired(eng.find_region("dlPFC")->fired());
        ofc   += count_fired(eng.find_region("OFC")->fired());
        acc   += count_fired(eng.find_region("ACC")->fired());
    }

    printf("    MD=%zu  dlPFC=%zu  OFC=%zu  ACC=%zu\n", md, dlpfc, ofc, acc);
    bool ok = md > 0 && (dlpfc > 0 || ofc > 0 || acc > 0);
    printf("  [%s] MD↔PFC双向\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 8: Visual pipeline backward compat — LGN→V1→V2→V4→IT→dlPFC→BG→M1
// =============================================================================
static void test_visual_pipeline() {
    printf("\n--- 测试8: 视觉→决策→运动 全链路 ---\n");
    auto eng = build_full_brain();

    for (int t = 0; t < 50; ++t) eng.step();

    size_t v1=0, it=0, dlpfc=0, bg=0, m1=0;
    for (int t = 0; t < 200; ++t) {
        eng.find_region("LGN")->inject_external(std::vector<float>(50, 35.0f));
        eng.step();
        v1    += count_fired(eng.find_region("V1")->fired());
        it    += count_fired(eng.find_region("IT")->fired());
        dlpfc += count_fired(eng.find_region("dlPFC")->fired());
        bg    += count_fired(eng.find_region("BG")->fired());
        m1    += count_fired(eng.find_region("M1")->fired());
    }

    printf("    V1=%zu  IT=%zu  dlPFC=%zu  BG=%zu  M1=%zu\n",
           v1, it, dlpfc, bg, m1);
    bool ok = v1 > 0 && it > 0 && dlpfc > 0 && m1 > 0;
    printf("  [%s] 视觉→决策→运动全链路\n", ok ? "PASS" : "FAIL");
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
    printf("  悟韵 (WuYun) Step 5 全脑系统测试\n");
    printf("  46区域 | ~90投射 | 全通路验证\n");
    printf("============================================\n");

    test_full_build();
    test_somatosensory();
    test_auditory_language();
    test_motor_hierarchy();
    test_dmn();
    test_pulvinar();
    test_md_pfc();
    test_visual_pipeline();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("============================================\n");

    return tests_failed > 0 ? 1 : 0;
}
