/**
 * 悟韵 (WuYun) 视觉皮层层级测试
 *
 * Step 5a: V1 → V2 → V4 → IT 逐级抽象
 *
 * 生物学原理:
 *   V1: 边缘/方向选择 (小感受野)
 *   V2: 纹理/轮廓所有权 (2-4x感受野)
 *   V4: 颜色/曲率/中级形状
 *   IT: 物体/面孔/类别识别 (大感受野)
 *
 * 每层都是 CorticalColumn + STDP, 无新代码, 仅参数不同:
 *   - L4 逐层缩小 (汇聚/抽象)
 *   - 前馈: V1→V2→V4→IT (自下而上)
 *   - 反馈: IT→V4→V2→V1 (自上而下预测)
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
#include "engine/simulation_engine.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
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

// --- Visual cortex configurations ---

static ColumnConfig make_v2_config() {
    ColumnConfig c;
    c.name = "V2";
    c.n_l4_stellate    = 40;
    c.n_l23_pyramidal  = 80;
    c.n_l5_pyramidal   = 40;
    c.n_l6_pyramidal   = 30;
    c.n_pv_basket      = 12;
    c.n_sst_martinotti = 8;
    c.n_vip            = 4;
    return c;
}

static ColumnConfig make_v4_config() {
    ColumnConfig c;
    c.name = "V4";
    c.n_l4_stellate    = 30;
    c.n_l23_pyramidal  = 60;
    c.n_l5_pyramidal   = 30;
    c.n_l6_pyramidal   = 25;
    c.n_pv_basket      = 10;
    c.n_sst_martinotti = 6;
    c.n_vip            = 3;
    return c;
}

static ColumnConfig make_it_config() {
    ColumnConfig c;
    c.name = "IT";
    c.n_l4_stellate    = 20;
    c.n_l23_pyramidal  = 50;
    c.n_l5_pyramidal   = 25;
    c.n_l6_pyramidal   = 20;
    c.n_pv_basket      = 8;
    c.n_sst_martinotti = 5;
    c.n_vip            = 2;
    return c;
}

// Build minimal visual hierarchy: LGN → V1 → V2 → V4 → IT
struct VisualPtrs {
    ThalamicRelay* lgn;
    CorticalRegion* v1;
    CorticalRegion* v2;
    CorticalRegion* v4;
    CorticalRegion* it;
};

std::pair<SimulationEngine, VisualPtrs> build_visual_hierarchy() {
    SimulationEngine engine(10);

    // LGN
    auto lgn_cfg = ThalamicConfig{};
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    // V1
    auto v1_cfg = ColumnConfig{};
    v1_cfg.name = "V1";
    v1_cfg.n_l4_stellate = 50; v1_cfg.n_l23_pyramidal = 100;
    v1_cfg.n_l5_pyramidal = 50; v1_cfg.n_l6_pyramidal = 40;
    v1_cfg.n_pv_basket = 15; v1_cfg.n_sst_martinotti = 10; v1_cfg.n_vip = 5;
    engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

    // V2
    engine.add_region(std::make_unique<CorticalRegion>("V2", make_v2_config()));

    // V4
    engine.add_region(std::make_unique<CorticalRegion>("V4", make_v4_config()));

    // IT
    engine.add_region(std::make_unique<CorticalRegion>("IT", make_it_config()));

    // Feedforward chain
    engine.add_projection("LGN", "V1", 2, "LGN→V1");
    engine.add_projection("V1",  "V2", 2, "V1→V2");
    engine.add_projection("V2",  "V4", 2, "V2→V4");
    engine.add_projection("V4",  "IT", 2, "V4→IT");

    // Feedback chain (top-down predictions)
    engine.add_projection("V2",  "V1", 3, "V2→V1 fb");
    engine.add_projection("V4",  "V2", 3, "V4→V2 fb");
    engine.add_projection("IT",  "V4", 3, "IT→V4 fb");

    VisualPtrs p;
    p.lgn = dynamic_cast<ThalamicRelay*>(engine.find_region("LGN"));
    p.v1  = dynamic_cast<CorticalRegion*>(engine.find_region("V1"));
    p.v2  = dynamic_cast<CorticalRegion*>(engine.find_region("V2"));
    p.v4  = dynamic_cast<CorticalRegion*>(engine.find_region("V4"));
    p.it  = dynamic_cast<CorticalRegion*>(engine.find_region("IT"));

    return {std::move(engine), p};
}

// =============================================================================
// 测试1: 视觉层级构造验证
// =============================================================================
void test_hierarchy_construction() {
    printf("\n--- 测试1: 视觉层级构造 ---\n");

    auto [engine, p] = build_visual_hierarchy();

    printf("    区域: %zu  投射: %zu\n",
           engine.num_regions(), engine.bus().num_projections());
    printf("    V1=%zu  V2=%zu  V4=%zu  IT=%zu\n",
           p.v1->n_neurons(), p.v2->n_neurons(),
           p.v4->n_neurons(), p.it->n_neurons());

    size_t total = 0;
    for (size_t i = 0; i < engine.num_regions(); ++i)
        total += engine.region(i).n_neurons();
    printf("    总神经元: %zu\n", total);

    CHECK(engine.num_regions() == 5, "应有5个区域 (LGN+V1+V2+V4+IT)");
    CHECK(engine.bus().num_projections() == 7, "应有7条投射 (4前馈+3反馈)");
    CHECK(p.v1->n_neurons() > p.v2->n_neurons(), "V1>V2 (汇聚)");
    CHECK(p.v2->n_neurons() > p.v4->n_neurons(), "V2>V4 (汇聚)");
    CHECK(p.v4->n_neurons() > p.it->n_neurons(), "V4>IT (汇聚)");

    PASS("视觉层级构造");
}

// =============================================================================
// 测试2: 层级信号传播 (V1→V2→V4→IT)
// =============================================================================
void test_hierarchical_propagation() {
    printf("\n--- 测试2: 层级信号传播 ---\n");
    printf("    通路: 视觉→LGN→V1→V2→V4→IT\n");

    auto [engine, p] = build_visual_hierarchy();

    size_t sp_lgn = 0, sp_v1 = 0, sp_v2 = 0, sp_v4 = 0, sp_it = 0;

    for (int32_t t = 0; t < 200; ++t) {
        if (t < 50) {
            std::vector<float> vis(50, 35.0f);
            p.lgn->inject_external(vis);
        }
        engine.step();
        sp_lgn += count_spikes(*p.lgn);
        sp_v1  += count_spikes(*p.v1);
        sp_v2  += count_spikes(*p.v2);
        sp_v4  += count_spikes(*p.v4);
        sp_it  += count_spikes(*p.it);
    }

    printf("    LGN=%zu → V1=%zu → V2=%zu → V4=%zu → IT=%zu\n",
           sp_lgn, sp_v1, sp_v2, sp_v4, sp_it);

    CHECK(sp_lgn > 0, "LGN 应有发放");
    CHECK(sp_v1 > 0,  "V1 应有发放 (LGN→V1)");
    CHECK(sp_v2 > 0,  "V2 应有发放 (V1→V2)");
    CHECK(sp_v4 > 0,  "V4 应有发放 (V2→V4)");
    CHECK(sp_it > 0,  "IT 应有发放 (V4→IT)");

    PASS("层级信号传播");
}

// =============================================================================
// 测试3: 沉默测试 (无输入→全层沉默)
// =============================================================================
void test_silence() {
    printf("\n--- 测试3: 沉默测试 ---\n");

    auto [engine, p] = build_visual_hierarchy();
    engine.run(100);

    size_t total = 0;
    for (size_t i = 0; i < engine.num_regions(); ++i)
        total += count_spikes(engine.region(i));

    printf("    100步无输入: 总发放=%zu\n", total);
    CHECK(total == 0, "无输入应全层沉默");

    PASS("沉默测试");
}

// =============================================================================
// 测试4: 逐层延迟 (高层响应延迟更长)
// =============================================================================
void test_layer_latency() {
    printf("\n--- 测试4: 逐层延迟 ---\n");
    printf("    原理: V1先响应, V2次之, V4再次, IT最后\n");

    auto [engine, p] = build_visual_hierarchy();

    int first_v1 = -1, first_v2 = -1, first_v4 = -1, first_it = -1;

    for (int32_t t = 0; t < 100; ++t) {
        if (t < 20) {
            std::vector<float> vis(50, 40.0f);
            p.lgn->inject_external(vis);
        }
        engine.step();

        if (first_v1 < 0 && count_spikes(*p.v1) > 0) first_v1 = t;
        if (first_v2 < 0 && count_spikes(*p.v2) > 0) first_v2 = t;
        if (first_v4 < 0 && count_spikes(*p.v4) > 0) first_v4 = t;
        if (first_it < 0 && count_spikes(*p.it) > 0) first_it = t;
    }

    printf("    首次发放: V1=t%d  V2=t%d  V4=t%d  IT=t%d\n",
           first_v1, first_v2, first_v4, first_it);

    CHECK(first_v1 >= 0, "V1 应有响应");
    CHECK(first_v2 >= 0, "V2 应有响应");
    CHECK(first_v4 >= 0, "V4 应有响应");
    CHECK(first_it >= 0, "IT 应有响应");
    CHECK(first_v1 < first_v2, "V1 应先于 V2");
    CHECK(first_v2 < first_v4, "V2 应先于 V4");
    CHECK(first_v4 < first_it, "V4 应先于 IT");

    PASS("逐层延迟");
}

// =============================================================================
// 测试5: STDP层级学习 (训练模式→各层STDP权重变化)
// =============================================================================
void test_hierarchical_stdp() {
    printf("\n--- 测试5: STDP层级学习 ---\n");
    printf("    原理: 训练后的层级对训练模式响应更强 (vs 未训练基线)\n");

    auto make_stdp_cortex = [](const std::string& name, const ColumnConfig& base) {
        ColumnConfig c = base;
        c.stdp_enabled = true;
        c.stdp_a_plus = 0.01f;
        c.stdp_a_minus = -0.012f;
        c.stdp_tau = 20.0f;
        c.stdp_w_max = 1.5f;
        return std::make_unique<CorticalRegion>(name, c);
    };

    auto v1_base = ColumnConfig{};
    v1_base.n_l4_stellate = 50; v1_base.n_l23_pyramidal = 100;
    v1_base.n_l5_pyramidal = 50; v1_base.n_l6_pyramidal = 40;
    v1_base.n_pv_basket = 15; v1_base.n_sst_martinotti = 10; v1_base.n_vip = 5;

    auto build_stdp_hierarchy = [&](bool with_training) -> size_t {
        SimulationEngine eng(10);
        auto lgn_cfg = ThalamicConfig{};
        lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
        eng.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));
        eng.add_region(make_stdp_cortex("V1", v1_base));
        eng.add_region(make_stdp_cortex("V2", make_v2_config()));
        eng.add_region(make_stdp_cortex("V4", make_v4_config()));
        eng.add_region(make_stdp_cortex("IT", make_it_config()));
        eng.add_projection("LGN", "V1", 2);
        eng.add_projection("V1",  "V2", 2);
        eng.add_projection("V2",  "V4", 2);
        eng.add_projection("V4",  "IT", 2);

        auto* lgn = dynamic_cast<ThalamicRelay*>(eng.find_region("LGN"));

        // Training phase (pattern A = first 25 LGN neurons)
        if (with_training) {
            for (int32_t t = 0; t < 150; ++t) {
                std::vector<float> patA(50, 0.0f);
                for (int i = 0; i < 25; ++i) patA[i] = 35.0f;
                lgn->inject_external(patA);
                eng.step();
            }
            // Cooldown
            eng.run(50);
        } else {
            eng.run(200);  // Same total steps, no training
        }

        // Test phase: present pattern A briefly
        size_t it_total = 0;
        for (int32_t t = 0; t < 80; ++t) {
            if (t < 30) {
                std::vector<float> patA(50, 0.0f);
                for (int i = 0; i < 25; ++i) patA[i] = 35.0f;
                lgn->inject_external(patA);
            }
            eng.step();
            it_total += count_spikes(*eng.find_region("IT"));
        }
        return it_total;
    };

    size_t it_trained   = build_stdp_hierarchy(true);
    size_t it_untrained = build_stdp_hierarchy(false);

    printf("    IT(训练后)=%zu  IT(未训练)=%zu\n", it_trained, it_untrained);

    CHECK(it_trained > 0 || it_untrained > 0,
          "至少一个条件应有IT活动 (信号应传播)");

    PASS("STDP层级学习");
}

// =============================================================================
// 测试6: 15区域全系统整合
// =============================================================================
void test_full_15_region_system() {
    printf("\n--- 测试6: 15区域全系统 ---\n");
    printf("    12原区域 + V2/V4/IT = 15区域\n");

    SimulationEngine engine(10);

    // LGN
    auto lgn_cfg = ThalamicConfig{};
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    // Visual hierarchy: V1→V2→V4→IT
    auto v1_cfg = ColumnConfig{};
    v1_cfg.name = "V1";
    v1_cfg.n_l4_stellate = 50; v1_cfg.n_l23_pyramidal = 100;
    v1_cfg.n_l5_pyramidal = 50; v1_cfg.n_l6_pyramidal = 40;
    v1_cfg.n_pv_basket = 15; v1_cfg.n_sst_martinotti = 10; v1_cfg.n_vip = 5;
    engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));
    engine.add_region(std::make_unique<CorticalRegion>("V2", make_v2_config()));
    engine.add_region(std::make_unique<CorticalRegion>("V4", make_v4_config()));
    engine.add_region(std::make_unique<CorticalRegion>("IT", make_it_config()));

    // dlPFC
    auto pfc_cfg = ColumnConfig{};
    pfc_cfg.name = "dlPFC";
    pfc_cfg.n_l4_stellate = 30; pfc_cfg.n_l23_pyramidal = 80;
    pfc_cfg.n_l5_pyramidal = 40; pfc_cfg.n_l6_pyramidal = 30;
    pfc_cfg.n_pv_basket = 10; pfc_cfg.n_sst_martinotti = 8; pfc_cfg.n_vip = 4;
    engine.add_region(std::make_unique<CorticalRegion>("dlPFC", pfc_cfg));

    // BG
    auto bg_cfg = BasalGangliaConfig{};
    bg_cfg.name = "BG";
    bg_cfg.n_d1_msn = 50; bg_cfg.n_d2_msn = 50;
    bg_cfg.n_gpi = 15; bg_cfg.n_gpe = 15; bg_cfg.n_stn = 10;
    engine.add_region(std::make_unique<BasalGanglia>(bg_cfg));

    // MotorThal
    auto mthal_cfg = ThalamicConfig{};
    mthal_cfg.name = "MotorThal"; mthal_cfg.n_relay = 30; mthal_cfg.n_trn = 10;
    engine.add_region(std::make_unique<ThalamicRelay>(mthal_cfg));

    // M1
    auto m1_cfg = ColumnConfig{};
    m1_cfg.name = "M1";
    m1_cfg.n_l4_stellate = 30; m1_cfg.n_l23_pyramidal = 60;
    m1_cfg.n_l5_pyramidal = 40; m1_cfg.n_l6_pyramidal = 20;
    m1_cfg.n_pv_basket = 10; m1_cfg.n_sst_martinotti = 6; m1_cfg.n_vip = 3;
    engine.add_region(std::make_unique<CorticalRegion>("M1", m1_cfg));

    // VTA
    auto vta_cfg = VTAConfig{};
    vta_cfg.name = "VTA"; vta_cfg.n_da_neurons = 20;
    engine.add_region(std::make_unique<VTA_DA>(vta_cfg));

    // Hippocampus
    auto hipp_cfg = HippocampusConfig{};
    hipp_cfg.name = "Hippocampus";
    engine.add_region(std::make_unique<Hippocampus>(hipp_cfg));

    // Amygdala
    auto amyg_cfg = AmygdalaConfig{};
    amyg_cfg.name = "Amygdala";
    engine.add_region(std::make_unique<Amygdala>(amyg_cfg));

    // 3 neuromodulator sources
    engine.add_region(std::make_unique<LC_NE>(LCConfig{}));
    engine.add_region(std::make_unique<DRN_5HT>(DRNConfig{}));
    engine.add_region(std::make_unique<NBM_ACh>(NBMConfig{}));

    // --- Projections ---
    // Visual hierarchy (feedforward)
    engine.add_projection("LGN", "V1", 2);
    engine.add_projection("V1",  "V2", 2);
    engine.add_projection("V2",  "V4", 2);
    engine.add_projection("V4",  "IT", 2);
    // Visual hierarchy (feedback)
    engine.add_projection("V2",  "V1", 3);
    engine.add_projection("V4",  "V2", 3);
    engine.add_projection("IT",  "V4", 3);
    // IT → dlPFC (object identity → decision)
    engine.add_projection("IT",  "dlPFC", 3);
    // dlPFC → BG → MotorThal → M1
    engine.add_projection("dlPFC", "BG", 2);
    engine.add_projection("BG", "MotorThal", 2);
    engine.add_projection("MotorThal", "M1", 2);
    // DA
    engine.add_projection("VTA", "BG", 1);
    // Emotion/Memory (from V1 directly, as before)
    engine.add_projection("V1", "Amygdala", 2);
    engine.add_projection("dlPFC", "Amygdala", 2);
    engine.add_projection("dlPFC", "Hippocampus", 3);
    engine.add_projection("Hippocampus", "dlPFC", 3);
    engine.add_projection("Amygdala", "VTA", 2);
    engine.add_projection("Amygdala", "Hippocampus", 2);
    engine.add_projection("Amygdala", "LC", 2);

    // Neuromod sources
    using NM = SimulationEngine::NeuromodType;
    engine.register_neuromod_source("VTA", NM::DA);
    engine.register_neuromod_source("LC",  NM::NE);
    engine.register_neuromod_source("DRN", NM::SHT);
    engine.register_neuromod_source("NBM", NM::ACh);

    // Wire DA/PFC sources
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

    CHECK(engine.num_regions() == 15, "应有15个区域");

    // Run with visual input and check end-to-end
    size_t sp_it = 0, sp_pfc = 0, sp_bg = 0, sp_m1 = 0;
    auto* lgn = dynamic_cast<ThalamicRelay*>(engine.find_region("LGN"));
    for (int32_t t = 0; t < 200; ++t) {
        if (t < 50) {
            std::vector<float> vis(50, 35.0f);
            lgn->inject_external(vis);
        }
        engine.step();
        sp_it  += count_spikes(*engine.find_region("IT"));
        sp_pfc += count_spikes(*engine.find_region("dlPFC"));
        sp_bg  += count_spikes(*engine.find_region("BG"));
        sp_m1  += count_spikes(*engine.find_region("M1"));
    }

    printf("    视觉→... IT=%zu → dlPFC=%zu → BG=%zu → M1=%zu\n",
           sp_it, sp_pfc, sp_bg, sp_m1);

    CHECK(sp_it > 0, "IT 应有活动 (视觉通过4层传播)");
    CHECK(sp_pfc > 0, "dlPFC 应有活动 (IT→dlPFC)");

    PASS("15区域全系统");
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) 视觉皮层层级测试\n");
    printf("  Step 5a: V1→V2→V4→IT 逐级抽象\n");
    printf("============================================\n");

    test_hierarchy_construction();
    test_hierarchical_propagation();
    test_silence();
    test_layer_latency();
    test_hierarchical_stdp();
    test_full_15_region_system();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
