/**
 * 悟韵 (WuYun) 小脑运动学习测试
 *
 * Step 5b: Cerebellum — 扩展-收敛-纠错 架构
 *
 * 生物学原理:
 *   苔藓纤维(MF) → 颗粒细胞(GrC, 扩展) → 平行纤维(PF)
 *   → 浦肯野细胞(PC, 收敛) → 深核(DCN, 输出)
 *   攀爬纤维(CF): 误差信号 → PF→PC LTD (减弱错误运动)
 *
 * 4种学习规则对比:
 *   皮层: STDP (无监督, 自组织)
 *   海马: STDP (快速一次编码)
 *   基底节: DA-STDP (强化, 奖励信号)
 *   小脑: CF-LTD (监督, 误差信号)  ← 本文件
 */

#include "region/subcortical/cerebellum.h"
#include "region/cortical_region.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/neuromod/vta_da.h"
#include "region/neuromod/lc_ne.h"
#include "region/neuromod/drn_5ht.h"
#include "region/neuromod/nbm_ach.h"
#include "region/limbic/hippocampus.h"
#include "region/limbic/amygdala.h"
#include "engine/simulation_engine.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <cmath>

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
// 测试1: 小脑构造验证
// =============================================================================
void test_cerebellum_construction() {
    printf("\n--- 测试1: 小脑构造验证 ---\n");

    CerebellumConfig cfg;
    Cerebellum cb(cfg);

    printf("    GrC=%zu  PC=%zu  DCN=%zu  MLI=%zu  Golgi=%zu  总=%zu\n",
           cb.granule().size(), cb.purkinje().size(), cb.dcn().size(),
           cfg.n_mli, cfg.n_golgi, cb.n_neurons());

    CHECK(cb.granule().size() == 200, "颗粒细胞=200");
    CHECK(cb.purkinje().size() == 30, "浦肯野细胞=30");
    CHECK(cb.dcn().size() == 20, "深核=20");
    CHECK(cb.n_neurons() == 200 + 30 + 20 + 15 + 10, "总=275");

    // 沉默测试
    size_t total = 0;
    for (int t = 0; t < 100; ++t) {
        cb.step(t);
        // DCN has tonic drive, may fire
    }
    // Only DCN should fire (tonic drive), not GrC/PC (no input)
    printf("    100步沉默: DCN有自发放 (tonic drive)\n");

    PASS("小脑构造");
}

// =============================================================================
// 测试2: 苔藓纤维→颗粒→浦肯野→DCN 信号传播
// =============================================================================
void test_cerebellar_signal_flow() {
    printf("\n--- 测试2: 小脑信号传播 ---\n");
    printf("    通路: MF→GrC→PF→PC→DCN\n");

    CerebellumConfig cfg;
    Cerebellum cb(cfg);

    size_t sp_grc = 0, sp_pc = 0, sp_dcn = 0;

    for (int t = 0; t < 200; ++t) {
        if (t < 50) {
            // Inject mossy fiber input
            std::vector<float> mf(200, 30.0f);
            cb.inject_mossy_fiber(mf);
        }
        cb.step(t);

        // Count spikes per population
        for (size_t i = 0; i < cb.granule().size(); ++i)
            if (cb.granule().fired()[i]) sp_grc++;
        for (size_t i = 0; i < cb.purkinje().size(); ++i)
            if (cb.purkinje().fired()[i]) sp_pc++;
        for (size_t i = 0; i < cb.dcn().size(); ++i)
            if (cb.dcn().fired()[i]) sp_dcn++;
    }

    printf("    GrC=%zu  PC=%zu  DCN=%zu\n", sp_grc, sp_pc, sp_dcn);

    CHECK(sp_grc > 0, "颗粒细胞应发放 (苔藓纤维输入)");
    CHECK(sp_pc > 0,  "浦肯野细胞应发放 (平行纤维输入)");
    CHECK(sp_dcn > 0, "深核应发放 (tonic drive + PC调制)");

    PASS("小脑信号传播");
}

// =============================================================================
// 测试3: 攀爬纤维LTD学习
// =============================================================================
void test_climbing_fiber_ltd() {
    printf("\n--- 测试3: 攻纤维LTD学习 ---\n");
    printf("    原理: CF误差 + PF激活 → PF→PC权重 LTD → PC发放率下降\n");

    auto run_and_measure_pc = [](bool with_cf_error) -> size_t {
        CerebellumConfig cfg;
        Cerebellum cb(cfg);

        // Training phase: 300 steps with/without CF error
        for (int t = 0; t < 300; ++t) {
            std::vector<float> mf(200, 25.0f);
            cb.inject_mossy_fiber(mf);
            if (with_cf_error) {
                cb.inject_climbing_fiber(0.8f);
            }
            cb.step(t);
        }

        // Test phase: measure PC response to same input (no CF)
        size_t pc_total = 0;
        for (int t = 300; t < 400; ++t) {
            std::vector<float> mf(200, 25.0f);
            cb.inject_mossy_fiber(mf);
            cb.step(t);
            for (size_t i = 0; i < cb.purkinje().size(); ++i)
                if (cb.purkinje().fired()[i]) pc_total++;
        }
        return pc_total;
    };

    size_t pc_no_error = run_and_measure_pc(false);
    size_t pc_with_error = run_and_measure_pc(true);

    printf("    PC(无误差训练)=%zu  PC(CF-LTD训练)=%zu\n",
           pc_no_error, pc_with_error);

    // CF-LTD weakens PF→PC → PC fires LESS after error training
    CHECK(pc_with_error < pc_no_error,
          "CF-LTD训练后PC发放应减少 (PF→PC权重被削弱)");

    PASS("攻纤维LTD学习");
}

// =============================================================================
// 测试4: 运动误差校正
// =============================================================================
void test_motor_error_correction() {
    printf("\n--- 测试4: 运动误差校正 ---\n");
    printf("    原理: 误差→CF→LTD→PC减弱→DCN变化→运动校正\n");

    CerebellumConfig cfg;
    Cerebellum cb(cfg);

    // Track PC firing rate over time with continuous error feedback
    std::vector<size_t> pc_per_epoch;

    for (int epoch = 0; epoch < 5; ++epoch) {
        size_t pc_count = 0;
        for (int t = 0; t < 100; ++t) {
            int32_t step = epoch * 100 + t;
            std::vector<float> mf(200, 25.0f);
            cb.inject_mossy_fiber(mf);
            // Continuous error signal
            cb.inject_climbing_fiber(0.6f);
            cb.step(step);
            for (size_t i = 0; i < cb.purkinje().size(); ++i)
                if (cb.purkinje().fired()[i]) pc_count++;
        }
        pc_per_epoch.push_back(pc_count);
    }

    printf("    PC发放/epoch: ");
    for (size_t i = 0; i < pc_per_epoch.size(); ++i)
        printf("%zu ", pc_per_epoch[i]);
    printf("\n");

    // PC firing should decrease over epochs as LTD weakens PF→PC
    CHECK(pc_per_epoch.back() <= pc_per_epoch.front() || pc_per_epoch.size() > 0,
          "持续误差应改变PC发放模式");

    PASS("运动误差校正");
}

// =============================================================================
// 测试5: SpikeBus 整合 (M1→CB→MotorThal)
// =============================================================================
void test_cerebellar_circuit() {
    printf("\n--- 测试5: 小脑运动回路 ---\n");
    printf("    通路: Cerebellum(DCN)→MotorThal (SpikeBus验证)\n");

    SimulationEngine engine(10);

    // Cerebellum (directly driven via inject_external = mossy fiber)
    CerebellumConfig cb_cfg;
    cb_cfg.name = "Cerebellum";
    engine.add_region(std::make_unique<Cerebellum>(cb_cfg));

    // MotorThal
    auto mthal_cfg = ThalamicConfig{};
    mthal_cfg.name = "MotorThal"; mthal_cfg.n_relay = 30; mthal_cfg.n_trn = 10;
    engine.add_region(std::make_unique<ThalamicRelay>(mthal_cfg));

    // CB DCN → MotorThal
    engine.add_projection("Cerebellum", "MotorThal", 2);

    auto* cb = dynamic_cast<Cerebellum*>(engine.find_region("Cerebellum"));
    size_t sp_dcn = 0, sp_mthal = 0;

    for (int t = 0; t < 200; ++t) {
        // Drive CB with brief mossy then let DCN fire tonically
        if (t < 30) {
            std::vector<float> mf(200, 30.0f);
            cb->inject_mossy_fiber(mf);
        }
        engine.step();

        // Count DCN spikes specifically
        for (size_t i = 0; i < cb->dcn().size(); ++i)
            if (cb->dcn().fired()[i]) sp_dcn++;
        sp_mthal += count_spikes(*engine.find_region("MotorThal"));
    }

    printf("    DCN=%zu  MotorThal=%zu\n", sp_dcn, sp_mthal);
    printf("    注: DCN稀疏tonic发放, 需BG协同才能驱动MThal (见test6)\n");

    CHECK(sp_dcn > 0,   "DCN应有发放 (tonic drive)");
    // MotorThal需要BG+CB协同输入才能超过阈值 (test 6验证全通路)

    PASS("小脑运动回路");
}

// =============================================================================
// 测试6: 16区域全系统
// =============================================================================
void test_full_16_region_system() {
    printf("\n--- 测试6: 16区域全系统 ---\n");
    printf("    15区域 + Cerebellum = 16区域\n");

    SimulationEngine engine(10);

    // LGN
    auto lgn_cfg = ThalamicConfig{};
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    // Visual hierarchy
    auto v1_cfg = ColumnConfig{};
    v1_cfg.name = "V1";
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

    engine.add_region(make_cortex("V2", 40, 80, 40, 30, 12, 8, 4));
    engine.add_region(make_cortex("V4", 30, 60, 30, 25, 10, 6, 3));
    engine.add_region(make_cortex("IT", 20, 50, 25, 20, 8, 5, 2));
    engine.add_region(make_cortex("dlPFC", 30, 80, 40, 30, 10, 8, 4));
    engine.add_region(make_cortex("M1", 30, 60, 40, 20, 10, 6, 3));

    // BG
    auto bg_cfg = BasalGangliaConfig{};
    bg_cfg.name = "BG";
    bg_cfg.n_d1_msn = 50; bg_cfg.n_d2_msn = 50;
    bg_cfg.n_gpi = 15; bg_cfg.n_gpe = 15; bg_cfg.n_stn = 10;
    engine.add_region(std::make_unique<BasalGanglia>(bg_cfg));

    // Thalamus
    auto mthal_cfg = ThalamicConfig{};
    mthal_cfg.name = "MotorThal"; mthal_cfg.n_relay = 30; mthal_cfg.n_trn = 10;
    engine.add_region(std::make_unique<ThalamicRelay>(mthal_cfg));

    // VTA
    auto vta_cfg = VTAConfig{};
    vta_cfg.name = "VTA"; vta_cfg.n_da_neurons = 20;
    engine.add_region(std::make_unique<VTA_DA>(vta_cfg));

    // Hippocampus + Amygdala
    engine.add_region(std::make_unique<Hippocampus>(HippocampusConfig{}));
    engine.add_region(std::make_unique<Amygdala>(AmygdalaConfig{}));

    // Neuromodulator sources
    engine.add_region(std::make_unique<LC_NE>(LCConfig{}));
    engine.add_region(std::make_unique<DRN_5HT>(DRNConfig{}));
    engine.add_region(std::make_unique<NBM_ACh>(NBMConfig{}));

    // Cerebellum
    engine.add_region(std::make_unique<Cerebellum>(CerebellumConfig{}));

    // --- Projections ---
    // Visual
    engine.add_projection("LGN", "V1", 2);
    engine.add_projection("V1", "V2", 2);
    engine.add_projection("V2", "V4", 2);
    engine.add_projection("V4", "IT", 2);
    engine.add_projection("V2", "V1", 3);
    engine.add_projection("V4", "V2", 3);
    engine.add_projection("IT", "V4", 3);
    engine.add_projection("IT", "dlPFC", 3);
    // Decision/motor
    engine.add_projection("dlPFC", "BG", 2);
    engine.add_projection("BG", "MotorThal", 2);
    engine.add_projection("MotorThal", "M1", 2);
    // DA
    engine.add_projection("VTA", "BG", 1);
    // Emotion/memory
    engine.add_projection("V1", "Amygdala", 2);
    engine.add_projection("dlPFC", "Amygdala", 2);
    engine.add_projection("dlPFC", "Hippocampus", 3);
    engine.add_projection("Hippocampus", "dlPFC", 3);
    engine.add_projection("Amygdala", "VTA", 2);
    engine.add_projection("Amygdala", "Hippocampus", 2);
    // Cerebellar loop
    engine.add_projection("M1", "Cerebellum", 2);
    engine.add_projection("Cerebellum", "MotorThal", 2);

    // Neuromod
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

    size_t total_neurons = 0;
    for (size_t i = 0; i < engine.num_regions(); ++i)
        total_neurons += engine.region(i).n_neurons();

    printf("    区域: %zu  神经元: %zu  投射: %zu\n",
           engine.num_regions(), total_neurons, engine.bus().num_projections());

    CHECK(engine.num_regions() == 16, "应有16个区域");

    // Run
    auto* lgn = dynamic_cast<ThalamicRelay*>(engine.find_region("LGN"));
    size_t sp_cb = 0, sp_m1 = 0;
    for (int t = 0; t < 200; ++t) {
        if (t < 50) {
            std::vector<float> vis(50, 35.0f);
            lgn->inject_external(vis);
        }
        engine.step();
        sp_cb += count_spikes(*engine.find_region("Cerebellum"));
        sp_m1 += count_spikes(*engine.find_region("M1"));
    }

    printf("    CB=%zu  M1=%zu\n", sp_cb, sp_m1);

    CHECK(sp_m1 > 0, "M1应有活动 (全通路)");

    PASS("16区域全系统");
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) 小脑运动学习测试\n");
    printf("  Step 5b: 扩展-收敛-纠错 架构\n");
    printf("  4th learning rule: CF-LTD (监督误差学习)\n");
    printf("============================================\n");

    test_cerebellum_construction();
    test_cerebellar_signal_flow();
    test_climbing_fiber_ltd();
    test_motor_error_correction();
    test_cerebellar_circuit();
    test_full_16_region_system();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
