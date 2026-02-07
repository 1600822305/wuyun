/**
 * 悟韵 (WuYun) 决策皮层 + 背侧视觉通路 测试
 *
 * Step 5c: 决策皮层 OFC/vmPFC/ACC
 *   OFC  — 眶额皮层, 价值评估 (Amyg→OFC→vmPFC)
 *   vmPFC — 腹内侧前额叶, 情绪决策 (vmPFC→BG, vmPFC→Amyg)
 *   ACC  — 前扣带回, 冲突监控 (ACC→dlPFC, ACC→LC)
 *
 * Step 5d: 背侧视觉通路 (where pathway)
 *   MT   — 中颞区/V5, 运动方向感知
 *   PPC  — 后顶叶皮层, 空间注意/视觉运动整合
 *   双流: 腹侧(V1→V2→V4→IT, what) + 背侧(V1→V2→MT→PPC, where)
 */

#include "region/cortical_region.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/neuromod/vta_da.h"
#include "region/neuromod/lc_ne.h"
#include "region/neuromod/drn_5ht.h"
#include "region/neuromod/nbm_ach.h"
#include "region/limbic/hippocampus.h"
#include "region/limbic/amygdala.h"
#include "region/subcortical/cerebellum.h"
#include "engine/simulation_engine.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace wuyun;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("  [FAIL] %s\n", msg); \
        g_fail++; return; \
    } \
} while(0)

#define PASS(name) do { \
    printf("  [PASS] %s\n", name); \
    g_pass++; \
} while(0)

static size_t count_spikes(const BrainRegion& r) {
    size_t n = 0;
    for (size_t i = 0; i < r.n_neurons(); ++i)
        if (r.fired()[i]) n++;
    return n;
}

// =============================================================================
// 区域配置工厂
// =============================================================================
static ColumnConfig make_ofc_config() {
    ColumnConfig c;
    c.n_l4_stellate = 25; c.n_l23_pyramidal = 60;
    c.n_l5_pyramidal = 30; c.n_l6_pyramidal = 20;
    c.n_pv_basket = 8; c.n_sst_martinotti = 5; c.n_vip = 3;
    return c;  // 151n
}

static ColumnConfig make_vmpfc_config() {
    ColumnConfig c;
    c.n_l4_stellate = 20; c.n_l23_pyramidal = 55;
    c.n_l5_pyramidal = 30; c.n_l6_pyramidal = 20;
    c.n_pv_basket = 8; c.n_sst_martinotti = 5; c.n_vip = 2;
    return c;  // 140n
}

static ColumnConfig make_acc_config() {
    ColumnConfig c;
    c.n_l4_stellate = 20; c.n_l23_pyramidal = 50;
    c.n_l5_pyramidal = 30; c.n_l6_pyramidal = 20;
    c.n_pv_basket = 8; c.n_sst_martinotti = 5; c.n_vip = 2;
    return c;  // 135n
}

static ColumnConfig make_mt_config() {
    ColumnConfig c;
    c.n_l4_stellate = 35; c.n_l23_pyramidal = 70;
    c.n_l5_pyramidal = 35; c.n_l6_pyramidal = 25;
    c.n_pv_basket = 10; c.n_sst_martinotti = 7; c.n_vip = 3;
    return c;  // 185n
}

static ColumnConfig make_ppc_config() {
    ColumnConfig c;
    c.n_l4_stellate = 30; c.n_l23_pyramidal = 65;
    c.n_l5_pyramidal = 35; c.n_l6_pyramidal = 25;
    c.n_pv_basket = 10; c.n_sst_martinotti = 6; c.n_vip = 3;
    return c;  // 174n
}

// =============================================================================
// 测试1: 决策皮层构造 (5c)
// =============================================================================
void test_decision_cortex_construction() {
    printf("\n--- 测试1: 决策皮层 OFC/vmPFC/ACC 构造 ---\n");

    SimulationEngine engine(10);

    engine.add_region(std::make_unique<CorticalRegion>("OFC", make_ofc_config()));
    engine.add_region(std::make_unique<CorticalRegion>("vmPFC", make_vmpfc_config()));
    engine.add_region(std::make_unique<CorticalRegion>("ACC", make_acc_config()));

    auto* ofc   = engine.find_region("OFC");
    auto* vmpfc = engine.find_region("vmPFC");
    auto* acc   = engine.find_region("ACC");

    printf("    OFC=%zu  vmPFC=%zu  ACC=%zu  总=%zu\n",
           ofc->n_neurons(), vmpfc->n_neurons(), acc->n_neurons(),
           ofc->n_neurons() + vmpfc->n_neurons() + acc->n_neurons());

    CHECK(ofc->n_neurons() == 151, "OFC=151");
    CHECK(vmpfc->n_neurons() == 140, "vmPFC=140");
    CHECK(acc->n_neurons() == 135, "ACC=135");

    PASS("决策皮层构造");
}

// =============================================================================
// 测试2: 背侧视觉通路构造 (5d)
// =============================================================================
void test_dorsal_pathway_construction() {
    printf("\n--- 测试2: 背侧视觉 MT/PPC 构造 ---\n");

    SimulationEngine engine(10);

    engine.add_region(std::make_unique<CorticalRegion>("MT", make_mt_config()));
    engine.add_region(std::make_unique<CorticalRegion>("PPC", make_ppc_config()));

    auto* mt  = engine.find_region("MT");
    auto* ppc = engine.find_region("PPC");

    printf("    MT=%zu  PPC=%zu  总=%zu\n",
           mt->n_neurons(), ppc->n_neurons(),
           mt->n_neurons() + ppc->n_neurons());

    CHECK(mt->n_neurons() == 185, "MT=185");
    CHECK(ppc->n_neurons() == 174, "PPC=174");

    PASS("背侧视觉构造");
}

// =============================================================================
// 测试3: 决策通路信号传播 (IT→OFC→vmPFC→BG)
// =============================================================================
void test_decision_signal_flow() {
    printf("\n--- 测试3: 决策通路信号传播 ---\n");
    printf("    通路: 视觉→IT→OFC→vmPFC→BG (价值→决策→动作)\n");

    SimulationEngine engine(10);

    // Visual input
    auto lgn_cfg = ThalamicConfig{};
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    // Ventral stream to IT
    auto v1_cfg = ColumnConfig{};
    v1_cfg.n_l4_stellate = 50; v1_cfg.n_l23_pyramidal = 100;
    v1_cfg.n_l5_pyramidal = 50; v1_cfg.n_l6_pyramidal = 40;
    v1_cfg.n_pv_basket = 15; v1_cfg.n_sst_martinotti = 10; v1_cfg.n_vip = 5;
    engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

    ColumnConfig v4_cfg;
    v4_cfg.n_l4_stellate = 30; v4_cfg.n_l23_pyramidal = 60;
    v4_cfg.n_l5_pyramidal = 30; v4_cfg.n_l6_pyramidal = 25;
    v4_cfg.n_pv_basket = 10; v4_cfg.n_sst_martinotti = 6; v4_cfg.n_vip = 3;
    engine.add_region(std::make_unique<CorticalRegion>("V4", v4_cfg));

    ColumnConfig it_cfg;
    it_cfg.n_l4_stellate = 20; it_cfg.n_l23_pyramidal = 50;
    it_cfg.n_l5_pyramidal = 25; it_cfg.n_l6_pyramidal = 20;
    it_cfg.n_pv_basket = 8; it_cfg.n_sst_martinotti = 5; it_cfg.n_vip = 2;
    engine.add_region(std::make_unique<CorticalRegion>("IT", it_cfg));

    // Decision cortex
    engine.add_region(std::make_unique<CorticalRegion>("OFC", make_ofc_config()));
    engine.add_region(std::make_unique<CorticalRegion>("vmPFC", make_vmpfc_config()));

    // BG
    auto bg_cfg = BasalGangliaConfig{};
    bg_cfg.name = "BG";
    bg_cfg.n_d1_msn = 50; bg_cfg.n_d2_msn = 50;
    bg_cfg.n_gpi = 15; bg_cfg.n_gpe = 15; bg_cfg.n_stn = 10;
    engine.add_region(std::make_unique<BasalGanglia>(bg_cfg));

    // Projections: visual → decision → action
    engine.add_projection("LGN", "V1", 2);
    engine.add_projection("V1",  "V4", 2);
    engine.add_projection("V4",  "IT", 2);
    engine.add_projection("IT",  "OFC", 3);    // what → value
    engine.add_projection("OFC", "vmPFC", 2);  // value → decision
    engine.add_projection("vmPFC", "BG", 2);   // decision → action selection

    auto* lgn = dynamic_cast<ThalamicRelay*>(engine.find_region("LGN"));
    size_t sp_ofc = 0, sp_vmpfc = 0, sp_bg = 0;

    for (int t = 0; t < 200; ++t) {
        if (t < 50) {
            std::vector<float> vis(50, 35.0f);
            lgn->inject_external(vis);
        }
        engine.step();
        sp_ofc   += count_spikes(*engine.find_region("OFC"));
        sp_vmpfc += count_spikes(*engine.find_region("vmPFC"));
        sp_bg    += count_spikes(*engine.find_region("BG"));
    }

    printf("    IT→OFC=%zu  OFC→vmPFC=%zu  vmPFC→BG=%zu\n",
           sp_ofc, sp_vmpfc, sp_bg);

    CHECK(sp_ofc > 0,   "OFC应有活动 (IT→OFC)");
    CHECK(sp_vmpfc > 0, "vmPFC应有活动 (OFC→vmPFC)");
    CHECK(sp_bg > 0,    "BG应有活动 (vmPFC→BG)");

    PASS("决策通路信号传播");
}

// =============================================================================
// 测试4: 双流视觉 (what + where)
// =============================================================================
void test_dual_stream_vision() {
    printf("\n--- 测试4: 双流视觉架构 ---\n");
    printf("    腹侧(what): V1→V4→IT  背侧(where): V1→MT→PPC\n");

    SimulationEngine engine(10);

    // Shared early vision
    auto lgn_cfg = ThalamicConfig{};
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    auto v1_cfg = ColumnConfig{};
    v1_cfg.n_l4_stellate = 50; v1_cfg.n_l23_pyramidal = 100;
    v1_cfg.n_l5_pyramidal = 50; v1_cfg.n_l6_pyramidal = 40;
    v1_cfg.n_pv_basket = 15; v1_cfg.n_sst_martinotti = 10; v1_cfg.n_vip = 5;
    engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

    ColumnConfig v2_cfg;
    v2_cfg.n_l4_stellate = 40; v2_cfg.n_l23_pyramidal = 80;
    v2_cfg.n_l5_pyramidal = 40; v2_cfg.n_l6_pyramidal = 30;
    v2_cfg.n_pv_basket = 12; v2_cfg.n_sst_martinotti = 8; v2_cfg.n_vip = 4;
    engine.add_region(std::make_unique<CorticalRegion>("V2", v2_cfg));

    // Ventral (what)
    ColumnConfig v4_cfg;
    v4_cfg.n_l4_stellate = 30; v4_cfg.n_l23_pyramidal = 60;
    v4_cfg.n_l5_pyramidal = 30; v4_cfg.n_l6_pyramidal = 25;
    v4_cfg.n_pv_basket = 10; v4_cfg.n_sst_martinotti = 6; v4_cfg.n_vip = 3;
    engine.add_region(std::make_unique<CorticalRegion>("V4", v4_cfg));

    ColumnConfig it_cfg;
    it_cfg.n_l4_stellate = 20; it_cfg.n_l23_pyramidal = 50;
    it_cfg.n_l5_pyramidal = 25; it_cfg.n_l6_pyramidal = 20;
    it_cfg.n_pv_basket = 8; it_cfg.n_sst_martinotti = 5; it_cfg.n_vip = 2;
    engine.add_region(std::make_unique<CorticalRegion>("IT", it_cfg));

    // Dorsal (where)
    engine.add_region(std::make_unique<CorticalRegion>("MT", make_mt_config()));
    engine.add_region(std::make_unique<CorticalRegion>("PPC", make_ppc_config()));

    // Shared early projections
    engine.add_projection("LGN", "V1", 2);
    engine.add_projection("V1",  "V2", 2);

    // Ventral stream
    engine.add_projection("V2", "V4", 2);
    engine.add_projection("V4", "IT", 2);

    // Dorsal stream
    engine.add_projection("V1", "MT", 2);   // V1 直接→MT (快速运动)
    engine.add_projection("V2", "MT", 2);   // V2→MT
    engine.add_projection("MT", "PPC", 2);  // MT→PPC (空间整合)

    // Cross-stream (dorsal↔ventral interaction)
    engine.add_projection("PPC", "IT", 3);  // where→what (空间引导识别)

    auto* lgn = dynamic_cast<ThalamicRelay*>(engine.find_region("LGN"));
    size_t sp_it = 0, sp_mt = 0, sp_ppc = 0;

    for (int t = 0; t < 200; ++t) {
        if (t < 50) {
            std::vector<float> vis(50, 35.0f);
            lgn->inject_external(vis);
        }
        engine.step();
        sp_it  += count_spikes(*engine.find_region("IT"));
        sp_mt  += count_spikes(*engine.find_region("MT"));
        sp_ppc += count_spikes(*engine.find_region("PPC"));
    }

    printf("    腹侧: IT=%zu  背侧: MT=%zu → PPC=%zu\n", sp_it, sp_mt, sp_ppc);

    CHECK(sp_it > 0,  "IT应有活动 (腹侧what)");
    CHECK(sp_mt > 0,  "MT应有活动 (背侧运动)");
    CHECK(sp_ppc > 0, "PPC应有活动 (背侧空间)");

    PASS("双流视觉架构");
}

// =============================================================================
// 测试5: ACC冲突监控→NE唤醒
// =============================================================================
void test_acc_conflict_monitoring() {
    printf("\n--- 测试5: ACC冲突监控 ---\n");
    printf("    原理: 冲突→ACC→LC_NE→NE↑→全脑增益调制\n");

    SimulationEngine engine(10);

    engine.add_region(std::make_unique<CorticalRegion>("ACC", make_acc_config()));
    engine.add_region(std::make_unique<LC_NE>(LCConfig{}));

    engine.add_projection("ACC", "LC", 2);  // ACC → LC (冲突→唤醒)
    engine.register_neuromod_source("LC", SimulationEngine::NeuromodType::NE);

    auto* acc = dynamic_cast<CorticalRegion*>(engine.find_region("ACC"));
    auto* lc  = dynamic_cast<LC_NE*>(engine.find_region("LC"));

    // Phase 1: no conflict (quiet ACC)
    engine.run(50);
    float ne_baseline = lc->ne_output();

    // Phase 2: conflict (strong ACC input)
    for (int t = 0; t < 100; ++t) {
        std::vector<float> conflict(acc->n_neurons(), 25.0f);
        acc->inject_external(conflict);
        engine.step();
    }
    float ne_conflict = lc->ne_output();

    printf("    NE(基线)=%.3f  NE(冲突)=%.3f\n", ne_baseline, ne_conflict);

    CHECK(ne_conflict > ne_baseline, "冲突应提升NE (ACC→LC)");

    PASS("ACC冲突监控");
}

// =============================================================================
// 测试6: 完整21区域系统
// =============================================================================
void test_full_21_region_system() {
    printf("\n--- 测试6: 21区域全系统 ---\n");
    printf("    16区域 + OFC/vmPFC/ACC/MT/PPC = 21区域\n");

    SimulationEngine engine(10);

    // === LGN ===
    auto lgn_cfg = ThalamicConfig{};
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    // === Visual cortex ===
    auto v1_cfg = ColumnConfig{};
    v1_cfg.n_l4_stellate = 50; v1_cfg.n_l23_pyramidal = 100;
    v1_cfg.n_l5_pyramidal = 50; v1_cfg.n_l6_pyramidal = 40;
    v1_cfg.n_pv_basket = 15; v1_cfg.n_sst_martinotti = 10; v1_cfg.n_vip = 5;
    engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

    auto make_cortex = [](const std::string& name, size_t l4, size_t l23, size_t l5, size_t l6,
                          size_t pv, size_t sst, size_t vip) {
        ColumnConfig c;
        c.n_l4_stellate = l4; c.n_l23_pyramidal = l23;
        c.n_l5_pyramidal = l5; c.n_l6_pyramidal = l6;
        c.n_pv_basket = pv; c.n_sst_martinotti = sst; c.n_vip = vip;
        return std::make_unique<CorticalRegion>(name, c);
    };

    // Ventral stream
    engine.add_region(make_cortex("V2", 40, 80, 40, 30, 12, 8, 4));
    engine.add_region(make_cortex("V4", 30, 60, 30, 25, 10, 6, 3));
    engine.add_region(make_cortex("IT", 20, 50, 25, 20, 8, 5, 2));

    // Dorsal stream (5d)
    engine.add_region(std::make_unique<CorticalRegion>("MT", make_mt_config()));
    engine.add_region(std::make_unique<CorticalRegion>("PPC", make_ppc_config()));

    // Decision cortex (5c)
    engine.add_region(std::make_unique<CorticalRegion>("OFC", make_ofc_config()));
    engine.add_region(std::make_unique<CorticalRegion>("vmPFC", make_vmpfc_config()));
    engine.add_region(std::make_unique<CorticalRegion>("ACC", make_acc_config()));

    // Existing cortex
    engine.add_region(make_cortex("dlPFC", 30, 80, 40, 30, 10, 8, 4));
    engine.add_region(make_cortex("M1", 30, 60, 40, 20, 10, 6, 3));

    // === Subcortical ===
    auto bg_cfg = BasalGangliaConfig{};
    bg_cfg.name = "BG";
    bg_cfg.n_d1_msn = 50; bg_cfg.n_d2_msn = 50;
    bg_cfg.n_gpi = 15; bg_cfg.n_gpe = 15; bg_cfg.n_stn = 10;
    engine.add_region(std::make_unique<BasalGanglia>(bg_cfg));

    auto mthal_cfg = ThalamicConfig{};
    mthal_cfg.name = "MotorThal"; mthal_cfg.n_relay = 30; mthal_cfg.n_trn = 10;
    engine.add_region(std::make_unique<ThalamicRelay>(mthal_cfg));

    engine.add_region(std::make_unique<VTA_DA>(VTAConfig{}));
    engine.add_region(std::make_unique<Hippocampus>(HippocampusConfig{}));
    engine.add_region(std::make_unique<Amygdala>(AmygdalaConfig{}));
    engine.add_region(std::make_unique<Cerebellum>(CerebellumConfig{}));

    // === Neuromodulators ===
    engine.add_region(std::make_unique<LC_NE>(LCConfig{}));
    engine.add_region(std::make_unique<DRN_5HT>(DRNConfig{}));
    engine.add_region(std::make_unique<NBM_ACh>(NBMConfig{}));

    // === Projections ===
    // Visual: shared early
    engine.add_projection("LGN", "V1", 2);
    engine.add_projection("V1",  "V2", 2);

    // Ventral (what): V2→V4→IT→OFC
    engine.add_projection("V2", "V4", 2);
    engine.add_projection("V4", "IT", 2);
    engine.add_projection("V2", "V1", 3);  // feedback
    engine.add_projection("V4", "V2", 3);
    engine.add_projection("IT", "V4", 3);

    // Dorsal (where): V1/V2→MT→PPC
    engine.add_projection("V1", "MT", 2);
    engine.add_projection("V2", "MT", 2);
    engine.add_projection("MT", "PPC", 2);
    engine.add_projection("PPC", "MT", 3);  // feedback

    // Cross-stream
    engine.add_projection("PPC", "IT", 3);   // where→what
    engine.add_projection("IT",  "PPC", 3);  // what→where

    // Decision: IT→OFC→vmPFC→BG, ACC
    engine.add_projection("IT",    "OFC", 3);
    engine.add_projection("OFC",   "vmPFC", 2);
    engine.add_projection("vmPFC", "BG", 2);
    engine.add_projection("vmPFC", "Amygdala", 3);  // emotion regulation
    engine.add_projection("ACC",   "dlPFC", 2);     // conflict→control
    engine.add_projection("ACC",   "LC", 2);        // conflict→arousal
    engine.add_projection("dlPFC", "ACC", 2);       // control→monitoring

    // Existing pathways
    engine.add_projection("IT",    "dlPFC", 3);
    engine.add_projection("PPC",   "dlPFC", 3);  // spatial→decision
    engine.add_projection("PPC",   "M1", 3);     // visuomotor
    engine.add_projection("dlPFC", "BG", 2);
    engine.add_projection("BG",    "MotorThal", 2);
    engine.add_projection("MotorThal", "M1", 2);
    engine.add_projection("M1",    "Cerebellum", 2);
    engine.add_projection("Cerebellum", "MotorThal", 2);

    // Emotion/memory
    engine.add_projection("V1",          "Amygdala", 2);
    engine.add_projection("dlPFC",       "Amygdala", 2);
    engine.add_projection("Amygdala",    "OFC", 2);     // emotion→value
    engine.add_projection("dlPFC",       "Hippocampus", 3);
    engine.add_projection("Hippocampus", "dlPFC", 3);
    engine.add_projection("Amygdala",    "VTA", 2);
    engine.add_projection("Amygdala",    "Hippocampus", 2);
    engine.add_projection("VTA",         "BG", 1);

    // Neuromod registration
    using NM = SimulationEngine::NeuromodType;
    engine.register_neuromod_source("VTA", NM::DA);
    engine.register_neuromod_source("LC",  NM::NE);
    engine.register_neuromod_source("DRN", NM::SHT);
    engine.register_neuromod_source("NBM", NM::ACh);

    // Wire sources
    auto* bg = dynamic_cast<BasalGanglia*>(engine.find_region("BG"));
    auto* vta = engine.find_region("VTA");
    if (bg && vta) bg->set_da_source_region(vta->region_id());
    auto* amyg = dynamic_cast<Amygdala*>(engine.find_region("Amygdala"));
    auto* pfc = engine.find_region("dlPFC");
    if (amyg && pfc) amyg->set_pfc_source_region(pfc->region_id());

    // Count
    size_t total_neurons = 0;
    for (size_t i = 0; i < engine.num_regions(); ++i)
        total_neurons += engine.region(i).n_neurons();

    printf("    区域: %zu  神经元: %zu  投射: %zu\n",
           engine.num_regions(), total_neurons, engine.bus().num_projections());

    CHECK(engine.num_regions() == 21, "应有21个区域");

    // Run
    auto* lgn = dynamic_cast<ThalamicRelay*>(engine.find_region("LGN"));
    size_t sp_ofc = 0, sp_vmpfc = 0, sp_acc = 0, sp_mt = 0, sp_ppc = 0, sp_m1 = 0;

    for (int t = 0; t < 200; ++t) {
        if (t < 50) {
            std::vector<float> vis(50, 35.0f);
            lgn->inject_external(vis);
        }
        engine.step();
        sp_ofc   += count_spikes(*engine.find_region("OFC"));
        sp_vmpfc += count_spikes(*engine.find_region("vmPFC"));
        sp_acc   += count_spikes(*engine.find_region("ACC"));
        sp_mt    += count_spikes(*engine.find_region("MT"));
        sp_ppc   += count_spikes(*engine.find_region("PPC"));
        sp_m1    += count_spikes(*engine.find_region("M1"));
    }

    printf("    腹侧→决策: OFC=%zu vmPFC=%zu ACC=%zu\n", sp_ofc, sp_vmpfc, sp_acc);
    printf("    背侧→空间: MT=%zu PPC=%zu\n", sp_mt, sp_ppc);
    printf("    运动输出: M1=%zu\n", sp_m1);

    CHECK(sp_mt > 0,    "MT应有活动 (背侧通路)");
    CHECK(sp_ppc > 0,   "PPC应有活动 (空间通路)");
    CHECK(sp_ofc > 0,   "OFC应有活动 (价值评估)");
    CHECK(sp_m1 > 0,    "M1应有活动 (运动输出)");

    PASS("21区域全系统");
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) 决策皮层 + 背侧视觉 测试\n");
    printf("  Step 5c: OFC/vmPFC/ACC 价值决策\n");
    printf("  Step 5d: MT/PPC 背侧where通路\n");
    printf("============================================\n");

    test_decision_cortex_construction();
    test_dorsal_pathway_construction();
    test_decision_signal_flow();
    test_dual_stream_vision();
    test_acc_conflict_monitoring();
    test_full_21_region_system();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
