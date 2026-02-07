/**
 * 悟韵 (WuYun) 神经调质广播系统测试
 *
 * 测试 4 大调质系统:
 *   DA  (VTA)  → 奖励/学习率
 *   NE  (LC)   → 增益/警觉
 *   5-HT (DRN) → 折扣/耐心
 *   ACh (NBM)  → 注意力/学习模式
 *
 * 验证: 区域构造, 广播机制, 增益调制效应
 */

#include "region/neuromod/lc_ne.h"
#include "region/neuromod/drn_5ht.h"
#include "region/neuromod/nbm_ach.h"
#include "region/neuromod/vta_da.h"
#include "region/cortical_region.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/limbic/hippocampus.h"
#include "region/limbic/amygdala.h"
#include "engine/simulation_engine.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

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
// 测试1: 新区域构造验证
// =============================================================================
void test_region_construction() {
    printf("\n--- 测试1: 调质区域构造验证 ---\n");

    LCConfig lc_cfg;
    lc_cfg.n_ne_neurons = 15;
    LC_NE lc(lc_cfg);
    printf("    LC: %zu NE neurons, ne_output=%.2f\n", lc.n_neurons(), lc.ne_output());
    CHECK(lc.n_neurons() == 15, "LC 应有15个NE神经元");
    CHECK(lc.ne_output() > 0.0f, "LC 应有tonic NE输出");

    DRNConfig drn_cfg;
    drn_cfg.n_5ht_neurons = 20;
    DRN_5HT drn(drn_cfg);
    printf("    DRN: %zu 5-HT neurons, sht_output=%.2f\n", drn.n_neurons(), drn.sht_output());
    CHECK(drn.n_neurons() == 20, "DRN 应有20个5-HT神经元");
    CHECK(drn.sht_output() > 0.0f, "DRN 应有tonic 5-HT输出");

    NBMConfig nbm_cfg;
    nbm_cfg.n_ach_neurons = 15;
    NBM_ACh nbm(nbm_cfg);
    printf("    NBM: %zu ACh neurons, ach_output=%.2f\n", nbm.n_neurons(), nbm.ach_output());
    CHECK(nbm.n_neurons() == 15, "NBM 应有15个ACh神经元");
    CHECK(nbm.ach_output() > 0.0f, "NBM 应有tonic ACh输出");

    PASS("调质区域构造");
}

// =============================================================================
// 测试2: 广播机制验证
// =============================================================================
void test_broadcast_mechanism() {
    printf("\n--- 测试2: 广播机制验证 ---\n");
    printf("    原理: 源区域输出 → SimulationEngine收集 → 全局广播到所有区域\n");

    SimulationEngine engine(10);

    // 添加一个简单皮层 + 所有4个调质源
    auto v1_cfg = ColumnConfig{};
    v1_cfg.name = "V1";
    v1_cfg.n_l4_stellate = 30; v1_cfg.n_l23_pyramidal = 60;
    v1_cfg.n_l5_pyramidal = 30; v1_cfg.n_l6_pyramidal = 20;
    v1_cfg.n_pv_basket = 10; v1_cfg.n_sst_martinotti = 6; v1_cfg.n_vip = 3;
    engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

    auto vta_cfg = VTAConfig{};
    vta_cfg.n_da_neurons = 20;
    engine.add_region(std::make_unique<VTA_DA>(vta_cfg));

    auto lc_cfg = LCConfig{};
    lc_cfg.n_ne_neurons = 15;
    engine.add_region(std::make_unique<LC_NE>(lc_cfg));

    auto drn_cfg = DRNConfig{};
    drn_cfg.n_5ht_neurons = 20;
    engine.add_region(std::make_unique<DRN_5HT>(drn_cfg));

    auto nbm_cfg = NBMConfig{};
    nbm_cfg.n_ach_neurons = 15;
    engine.add_region(std::make_unique<NBM_ACh>(nbm_cfg));

    // 注册调质源
    using NM = SimulationEngine::NeuromodType;
    engine.register_neuromod_source("VTA", NM::DA);
    engine.register_neuromod_source("LC",  NM::NE);
    engine.register_neuromod_source("DRN", NM::SHT);
    engine.register_neuromod_source("NBM", NM::ACh);

    // 运行几步让调质系统稳定
    engine.run(10);

    auto levels = engine.global_neuromod();
    printf("    全局调质: DA=%.3f NE=%.3f 5-HT=%.3f ACh=%.3f\n",
           levels.da, levels.ne, levels.sht, levels.ach);

    // 所有调质应在合理范围 (0~1)
    CHECK(levels.da  >= 0.0f && levels.da  <= 1.0f, "DA 应在 0~1");
    CHECK(levels.ne  >= 0.0f && levels.ne  <= 1.0f, "NE 应在 0~1");
    CHECK(levels.sht >= 0.0f && levels.sht <= 1.0f, "5-HT 应在 0~1");
    CHECK(levels.ach >= 0.0f && levels.ach <= 1.0f, "ACh 应在 0~1");

    // 验证V1区域收到了广播的调质水平
    auto* v1 = engine.find_region("V1");
    auto v1_levels = v1->neuromod().current();
    CHECK(std::abs(v1_levels.ne - levels.ne) < 0.01f,
          "V1的NE水平应等于全局广播值");

    PASS("广播机制");
}

// =============================================================================
// 测试3: NE 增益调制 (直接设NE水平 → 皮层增益变化)
// =============================================================================
void test_ne_gain_modulation() {
    printf("\n--- 测试3: NE增益调制 ---\n");
    printf("    原理: NE↑→gain↑→PSP放大→相同输入更强响应\n");
    printf("    gain公式: gain = 0.5 + 1.5 * NE  (NE=0.1→0.65, NE=0.8→1.70)\n");

    auto run_with_ne = [](float ne_level) -> size_t {
        SimulationEngine engine(10);
        auto lgn_cfg = ThalamicConfig{};
        lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
        engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

        auto v1_cfg = ColumnConfig{};
        v1_cfg.name = "V1";
        v1_cfg.n_l4_stellate = 50; v1_cfg.n_l23_pyramidal = 100;
        v1_cfg.n_l5_pyramidal = 50; v1_cfg.n_l6_pyramidal = 40;
        v1_cfg.n_pv_basket = 15; v1_cfg.n_sst_martinotti = 10; v1_cfg.n_vip = 5;
        engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

        engine.add_projection("LGN", "V1", 2);

        auto* lgn = dynamic_cast<ThalamicRelay*>(engine.find_region("LGN"));
        auto* v1  = engine.find_region("V1");

        size_t total = 0;
        for (int t = 0; t < 100; ++t) {
            // Set NE level directly on V1
            NeuromodulatorLevels levels;
            levels.ne = ne_level;
            v1->neuromod().set_tonic(levels);

            if (t < 50) {
                std::vector<float> vis(50, 25.0f);
                lgn->inject_external(vis);
            }
            engine.step();
            total += count_spikes(*v1);
        }
        return total;
    };

    size_t v1_low  = run_with_ne(0.1f);  // gain = 0.65
    size_t v1_mid  = run_with_ne(0.5f);  // gain = 1.25
    size_t v1_high = run_with_ne(0.9f);  // gain = 1.85

    printf("    V1(NE=0.1, gain=0.65)=%zu\n", v1_low);
    printf("    V1(NE=0.5, gain=1.25)=%zu\n", v1_mid);
    printf("    V1(NE=0.9, gain=1.85)=%zu\n", v1_high);

    // Yerkes-Dodson inverted-U: moderate NE > low NE
    // Very high NE may DECREASE activity (PV inhibition also amplified)
    CHECK(v1_mid > v1_low, "适度NE应强于低NE (增益调制)");

    PASS("NE增益调制");
}

// =============================================================================
// 测试4: 调质源驱动验证 (外部输入→调质水平变化)
// =============================================================================
void test_neuromod_drive() {
    printf("\n--- 测试4: 调质源驱动验证 ---\n");
    printf("    原理: 外部输入→调质神经元发放→输出水平升高\n");

    // LC: arousal input → NE↑
    {
        LCConfig cfg;
        cfg.n_ne_neurons = 15;
        LC_NE lc(cfg);
        float ne_baseline = lc.ne_output();

        for (int t = 0; t < 50; ++t) {
            lc.inject_arousal(0.9f);
            lc.step(t);
        }
        float ne_aroused = lc.ne_output();
        printf("    LC: NE基线=%.3f  NE+应激=%.3f\n", ne_baseline, ne_aroused);
        CHECK(ne_aroused > ne_baseline, "应激输入应提高NE水平");
    }

    // DRN: wellbeing input → 5-HT↑
    {
        DRNConfig cfg;
        cfg.n_5ht_neurons = 20;
        DRN_5HT drn(cfg);
        float sht_baseline = drn.sht_output();

        for (int t = 0; t < 50; ++t) {
            drn.inject_wellbeing(0.8f);
            drn.step(t);
        }
        float sht_well = drn.sht_output();
        printf("    DRN: 5-HT基线=%.3f  5-HT+安康=%.3f\n", sht_baseline, sht_well);
        CHECK(sht_well > sht_baseline, "安康输入应提高5-HT水平");
    }

    // NBM: surprise input → ACh↑
    {
        NBMConfig cfg;
        cfg.n_ach_neurons = 15;
        NBM_ACh nbm(cfg);
        float ach_baseline = nbm.ach_output();

        for (int t = 0; t < 50; ++t) {
            nbm.inject_surprise(0.7f);
            nbm.step(t);
        }
        float ach_surprised = nbm.ach_output();
        printf("    NBM: ACh基线=%.3f  ACh+意外=%.3f\n", ach_baseline, ach_surprised);
        CHECK(ach_surprised > ach_baseline, "意外输入应提高ACh水平");
    }

    PASS("调质源驱动");
}

// =============================================================================
// 测试5: 完整12区域系统 (9区域 + 3调质源)
// =============================================================================
void test_full_12_region_system() {
    printf("\n--- 测试5: 12区域全系统 ---\n");
    printf("    9个功能区 + 3个调质源 (LC/DRN/NBM) + VTA已有\n");

    SimulationEngine engine(10);

    // 原有9区域
    auto lgn_cfg = ThalamicConfig{};
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    auto v1_cfg = ColumnConfig{};
    v1_cfg.name = "V1";
    v1_cfg.n_l4_stellate = 50; v1_cfg.n_l23_pyramidal = 100;
    v1_cfg.n_l5_pyramidal = 50; v1_cfg.n_l6_pyramidal = 40;
    v1_cfg.n_pv_basket = 15; v1_cfg.n_sst_martinotti = 10; v1_cfg.n_vip = 5;
    engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

    auto pfc_cfg = ColumnConfig{};
    pfc_cfg.name = "dlPFC";
    pfc_cfg.n_l4_stellate = 30; pfc_cfg.n_l23_pyramidal = 80;
    pfc_cfg.n_l5_pyramidal = 40; pfc_cfg.n_l6_pyramidal = 30;
    pfc_cfg.n_pv_basket = 10; pfc_cfg.n_sst_martinotti = 8; pfc_cfg.n_vip = 4;
    engine.add_region(std::make_unique<CorticalRegion>("dlPFC", pfc_cfg));

    auto bg_cfg = BasalGangliaConfig{};
    bg_cfg.name = "BG";
    bg_cfg.n_d1_msn = 50; bg_cfg.n_d2_msn = 50;
    bg_cfg.n_gpi = 15; bg_cfg.n_gpe = 15; bg_cfg.n_stn = 10;
    engine.add_region(std::make_unique<BasalGanglia>(bg_cfg));

    auto mthal_cfg = ThalamicConfig{};
    mthal_cfg.name = "MotorThal"; mthal_cfg.n_relay = 30; mthal_cfg.n_trn = 10;
    engine.add_region(std::make_unique<ThalamicRelay>(mthal_cfg));

    auto m1_cfg = ColumnConfig{};
    m1_cfg.name = "M1";
    m1_cfg.n_l4_stellate = 30; m1_cfg.n_l23_pyramidal = 60;
    m1_cfg.n_l5_pyramidal = 40; m1_cfg.n_l6_pyramidal = 20;
    m1_cfg.n_pv_basket = 10; m1_cfg.n_sst_martinotti = 6; m1_cfg.n_vip = 3;
    engine.add_region(std::make_unique<CorticalRegion>("M1", m1_cfg));

    auto vta_cfg = VTAConfig{};
    vta_cfg.name = "VTA"; vta_cfg.n_da_neurons = 20;
    engine.add_region(std::make_unique<VTA_DA>(vta_cfg));

    auto hipp_cfg = HippocampusConfig{};
    hipp_cfg.name = "Hippocampus";
    engine.add_region(std::make_unique<Hippocampus>(hipp_cfg));

    auto amyg_cfg = AmygdalaConfig{};
    amyg_cfg.name = "Amygdala";
    engine.add_region(std::make_unique<Amygdala>(amyg_cfg));

    // 3个新调质源
    auto lc_cfg = LCConfig{};
    lc_cfg.name = "LC"; lc_cfg.n_ne_neurons = 15;
    engine.add_region(std::make_unique<LC_NE>(lc_cfg));

    auto drn_cfg = DRNConfig{};
    drn_cfg.name = "DRN"; drn_cfg.n_5ht_neurons = 20;
    engine.add_region(std::make_unique<DRN_5HT>(drn_cfg));

    auto nbm_cfg = NBMConfig{};
    nbm_cfg.name = "NBM"; nbm_cfg.n_ach_neurons = 15;
    engine.add_region(std::make_unique<NBM_ACh>(nbm_cfg));

    // 原有13投射
    engine.add_projection("LGN", "V1", 2);
    engine.add_projection("V1", "dlPFC", 3);
    engine.add_projection("dlPFC", "V1", 3);
    engine.add_projection("dlPFC", "BG", 2);
    engine.add_projection("BG", "MotorThal", 2);
    engine.add_projection("MotorThal", "M1", 2);
    engine.add_projection("VTA", "BG", 1);
    engine.add_projection("V1", "Amygdala", 2);
    engine.add_projection("dlPFC", "Amygdala", 2);
    engine.add_projection("dlPFC", "Hippocampus", 3);
    engine.add_projection("Hippocampus", "dlPFC", 3);
    engine.add_projection("Amygdala", "VTA", 2);
    engine.add_projection("Amygdala", "Hippocampus", 2);

    // 新调质投射 (Amygdala CeA → LC arousal)
    engine.add_projection("Amygdala", "LC", 2);

    // 注册广播源
    using NM = SimulationEngine::NeuromodType;
    engine.register_neuromod_source("VTA", NM::DA);
    engine.register_neuromod_source("LC",  NM::NE);
    engine.register_neuromod_source("DRN", NM::SHT);
    engine.register_neuromod_source("NBM", NM::ACh);

    // DA source for BG
    auto* bg = dynamic_cast<BasalGanglia*>(engine.find_region("BG"));
    auto* vta = engine.find_region("VTA");
    if (bg && vta) bg->set_da_source_region(vta->region_id());

    auto* amyg = dynamic_cast<Amygdala*>(engine.find_region("Amygdala"));
    auto* pfc = engine.find_region("dlPFC");
    if (amyg && pfc) amyg->set_pfc_source_region(pfc->region_id());

    // 统计
    size_t total_neurons = 0;
    for (size_t i = 0; i < engine.num_regions(); ++i)
        total_neurons += engine.region(i).n_neurons();

    printf("    区域: %zu  神经元: %zu  投射: %zu\n",
           engine.num_regions(), total_neurons, engine.bus().num_projections());

    CHECK(engine.num_regions() == 12, "应有12个区域");
    CHECK(engine.bus().num_projections() == 14, "应有14条投射");

    // 沉默测试
    engine.run(20);
    auto levels = engine.global_neuromod();
    printf("    20步后调质: DA=%.3f NE=%.3f 5-HT=%.3f ACh=%.3f\n",
           levels.da, levels.ne, levels.sht, levels.ach);

    CHECK(levels.da  >= 0.0f && levels.da  <= 1.0f, "DA水平合理");
    CHECK(levels.ne  >= 0.0f && levels.ne  <= 1.0f, "NE水平合理");
    CHECK(levels.sht >= 0.0f && levels.sht <= 1.0f, "5-HT水平合理");
    CHECK(levels.ach >= 0.0f && levels.ach <= 1.0f, "ACh水平合理");

    PASS("12区域全系统");
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) 神经调质广播系统测试\n");
    printf("  DA(VTA) + NE(LC) + 5-HT(DRN) + ACh(NBM)\n");
    printf("============================================\n");

    test_region_construction();
    test_broadcast_mechanism();
    test_ne_gain_modulation();
    test_neuromod_drive();
    test_full_12_region_system();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
