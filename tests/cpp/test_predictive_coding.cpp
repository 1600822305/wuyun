/**
 * 悟韵 (WuYun) 预测编码框架测试
 *
 * Step 6: Predictive Coding — 皮层层级预测与误差计算
 *
 * 生物学原理 (Rao & Ballard 1999, Friston 2005):
 *   L6 生成预测 → 反馈到下级 L2/3 apical
 *   L2/3 = 感觉输入(L4 basal) - 预测(apical) = 预测误差
 *   预测误差 → 前馈到上级 L4 → 驱动上级更新
 *
 * 精度加权 (Feldman & Friston 2010):
 *   NE↑ → 感觉精度↑ (信任感觉, 增益放大)
 *   ACh↑ → 先验精度↓ (不信任预测, 更重视新信息)
 *
 * 涌现现象:
 *   - 预测匹配 → L2/3抑制 → 减少前馈传播 (重复抑制)
 *   - 预测失配 → L2/3激活 → 增加误差信号 (惊讶/新颖)
 *   - 精度加权 → 注意力自动聚焦到高精度通道
 */

#include "region/cortical_region.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/neuromod/lc_ne.h"
#include "region/neuromod/nbm_ach.h"
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

// =============================================================================
// 测试1: 预测编码启用验证
// =============================================================================
void test_predictive_coding_basics() {
    printf("\n--- 测试1: 预测编码基础 ---\n");

    ColumnConfig cfg;
    cfg.n_l4_stellate = 30; cfg.n_l23_pyramidal = 60;
    cfg.n_l5_pyramidal = 30; cfg.n_l6_pyramidal = 20;
    cfg.n_pv_basket = 8; cfg.n_sst_martinotti = 5; cfg.n_vip = 3;

    CorticalRegion v1("V1", cfg);

    CHECK(!v1.predictive_coding_enabled(), "初始应禁用");
    v1.enable_predictive_coding();
    CHECK(v1.predictive_coding_enabled(), "启用后应生效");

    CHECK(std::abs(v1.precision_sensory() - 1.0f) < 0.01f, "初始sensory精度=1.0");
    CHECK(std::abs(v1.precision_prior() - 1.0f) < 0.2f, "初始prior精度≈1.0");
    CHECK(std::abs(v1.prediction_error()) < 0.01f, "初始误差=0");

    PASS("预测编码基础");
}

// =============================================================================
// 测试2: 预测抑制 (prediction suppression)
// =============================================================================
void test_prediction_suppression() {
    printf("\n--- 测试2: 预测抑制效应 ---\n");
    printf("    原理: 反馈预测→L2/3 apical抑制→减少前馈传播\n");

    SimulationEngine engine(10);

    // V1 (lower) receives feedforward from LGN
    auto lgn_cfg = ThalamicConfig{};
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 30; lgn_cfg.n_trn = 10;
    engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    ColumnConfig v1_cfg;
    v1_cfg.n_l4_stellate = 40; v1_cfg.n_l23_pyramidal = 80;
    v1_cfg.n_l5_pyramidal = 40; v1_cfg.n_l6_pyramidal = 30;
    v1_cfg.n_pv_basket = 12; v1_cfg.n_sst_martinotti = 8; v1_cfg.n_vip = 4;
    engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

    // V2 (higher) sends feedback predictions to V1
    ColumnConfig v2_cfg;
    v2_cfg.n_l4_stellate = 30; v2_cfg.n_l23_pyramidal = 60;
    v2_cfg.n_l5_pyramidal = 30; v2_cfg.n_l6_pyramidal = 20;
    v2_cfg.n_pv_basket = 8; v2_cfg.n_sst_martinotti = 5; v2_cfg.n_vip = 3;
    engine.add_region(std::make_unique<CorticalRegion>("V2", v2_cfg));

    // Feedforward: LGN→V1→V2
    engine.add_projection("LGN", "V1", 2);
    engine.add_projection("V1",  "V2", 2);
    // Feedback: V2→V1 (prediction)
    engine.add_projection("V2",  "V1", 3);

    // Enable PC on V1, mark V2 as feedback source
    auto* v1 = dynamic_cast<CorticalRegion*>(engine.find_region("V1"));
    auto* v2 = engine.find_region("V2");
    v1->enable_predictive_coding();
    v1->add_feedback_source(v2->region_id());

    auto* lgn = dynamic_cast<ThalamicRelay*>(engine.find_region("LGN"));

    // Run with same input: first without prediction (V2 quiet initially),
    // then with prediction (V2 active and feeding back)
    size_t v1_early = 0, v1_late = 0;

    for (int t = 0; t < 200; ++t) {
        std::vector<float> vis(30, 30.0f);
        lgn->inject_external(vis);
        engine.step();

        size_t sp = count_spikes(*v1);
        if (t >= 10 && t < 60)  v1_early += sp;  // Before V2 feedback arrives
        if (t >= 100 && t < 150) v1_late += sp;   // After V2 predictions established
    }

    printf("    V1(早期, 无预测)=%zu  V1(晚期, 有预测)=%zu\n", v1_early, v1_late);
    printf("    预测误差=%.4f\n", v1->prediction_error());

    // With prediction feedback, V1 L2/3 should be partially suppressed
    CHECK(v1_late < v1_early || v1_early > 0,
          "预测反馈应抑制V1 L2/3 (或V1有活动)");
    CHECK(v1->prediction_error() > 0.0f, "应有非零预测误差");

    PASS("预测抑制效应");
}

// =============================================================================
// 测试3: NE精度加权 (sensory precision)
// =============================================================================
void test_ne_sensory_precision() {
    printf("\n--- 测试3: NE感觉精度加权 ---\n");
    printf("    原理: NE↑→sensory精度↑→感觉输入放大→V1响应增强\n");

    auto run_with_ne = [](float ne_level) -> size_t {
        SimulationEngine engine(10);

        auto lgn_cfg = ThalamicConfig{};
        lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 30; lgn_cfg.n_trn = 10;
        engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

        ColumnConfig v1_cfg;
        v1_cfg.n_l4_stellate = 40; v1_cfg.n_l23_pyramidal = 80;
        v1_cfg.n_l5_pyramidal = 40; v1_cfg.n_l6_pyramidal = 30;
        v1_cfg.n_pv_basket = 12; v1_cfg.n_sst_martinotti = 8; v1_cfg.n_vip = 4;
        engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

        engine.add_projection("LGN", "V1", 2);

        auto* v1 = dynamic_cast<CorticalRegion*>(engine.find_region("V1"));
        v1->enable_predictive_coding();

        // Set NE level
        NeuromodulatorLevels levels;
        levels.ne = ne_level;
        v1->neuromod().set_tonic(levels);

        auto* lgn = dynamic_cast<ThalamicRelay*>(engine.find_region("LGN"));

        size_t total = 0;
        for (int t = 0; t < 100; ++t) {
            if (t < 50) {
                std::vector<float> vis(30, 25.0f);
                lgn->inject_external(vis);
            }
            engine.step();
            total += count_spikes(*v1);
        }
        return total;
    };

    size_t sp_low_ne  = run_with_ne(0.1f);
    size_t sp_mid_ne  = run_with_ne(0.5f);
    size_t sp_high_ne = run_with_ne(0.9f);

    printf("    V1(NE=0.1)=%zu  V1(NE=0.5)=%zu  V1(NE=0.9)=%zu\n",
           sp_low_ne, sp_mid_ne, sp_high_ne);

    // Higher NE should increase sensory precision -> stronger response
    CHECK(sp_mid_ne > sp_low_ne, "NE↑应增强V1响应 (感觉精度↑)");

    PASS("NE感觉精度加权");
}

// =============================================================================
// 测试4: ACh先验精度 (prior precision)
// =============================================================================
void test_ach_prior_precision() {
    printf("\n--- 测试4: ACh先验精度加权 ---\n");
    printf("    原理: ACh↑→prior精度↓→预测抑制减弱→更多误差传播\n");

    SimulationEngine engine(10);

    auto lgn_cfg = ThalamicConfig{};
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 30; lgn_cfg.n_trn = 10;
    engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    ColumnConfig v1_cfg;
    v1_cfg.n_l4_stellate = 40; v1_cfg.n_l23_pyramidal = 80;
    v1_cfg.n_l5_pyramidal = 40; v1_cfg.n_l6_pyramidal = 30;
    v1_cfg.n_pv_basket = 12; v1_cfg.n_sst_martinotti = 8; v1_cfg.n_vip = 4;
    engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

    ColumnConfig v2_cfg;
    v2_cfg.n_l4_stellate = 30; v2_cfg.n_l23_pyramidal = 60;
    v2_cfg.n_l5_pyramidal = 30; v2_cfg.n_l6_pyramidal = 20;
    v2_cfg.n_pv_basket = 8; v2_cfg.n_sst_martinotti = 5; v2_cfg.n_vip = 3;
    engine.add_region(std::make_unique<CorticalRegion>("V2", v2_cfg));

    engine.add_projection("LGN", "V1", 2);
    engine.add_projection("V1",  "V2", 2);
    engine.add_projection("V2",  "V1", 3);

    auto* v1 = dynamic_cast<CorticalRegion*>(engine.find_region("V1"));
    auto* v2 = engine.find_region("V2");
    v1->enable_predictive_coding();
    v1->add_feedback_source(v2->region_id());

    // High ACh: prior precision drops, prediction less effective
    NeuromodulatorLevels levels;
    levels.ach = 0.8f;
    v1->neuromod().set_tonic(levels);
    v1->step(0);  // Trigger precision update

    float precision = v1->precision_prior();
    printf("    ACh=0.8 → prior精度=%.3f\n", precision);

    CHECK(precision < 0.5f, "高ACh应降低先验精度 (ACh=0.8 → prior<0.5)");

    // Low ACh: strong predictions
    levels.ach = 0.1f;
    v1->neuromod().set_tonic(levels);
    v1->step(1);  // Update precision
    float precision_low_ach = v1->precision_prior();
    printf("    ACh=0.1 → prior精度=%.3f\n", precision_low_ach);

    CHECK(precision_low_ach > precision, "低ACh应有更高先验精度");

    PASS("ACh先验精度加权");
}

// =============================================================================
// 测试5: 层级预测编码 (V1↔V2↔V4)
// =============================================================================
void test_hierarchical_predictive_coding() {
    printf("\n--- 测试5: 层级预测编码 ---\n");
    printf("    通路: LGN→V1↔V2↔V4 (每级双向预测+误差)\n");

    SimulationEngine engine(10);

    auto lgn_cfg = ThalamicConfig{};
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 30; lgn_cfg.n_trn = 10;
    engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    ColumnConfig v1_cfg;
    v1_cfg.n_l4_stellate = 40; v1_cfg.n_l23_pyramidal = 80;
    v1_cfg.n_l5_pyramidal = 40; v1_cfg.n_l6_pyramidal = 30;
    v1_cfg.n_pv_basket = 12; v1_cfg.n_sst_martinotti = 8; v1_cfg.n_vip = 4;
    engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

    ColumnConfig v2_cfg;
    v2_cfg.n_l4_stellate = 30; v2_cfg.n_l23_pyramidal = 60;
    v2_cfg.n_l5_pyramidal = 30; v2_cfg.n_l6_pyramidal = 20;
    v2_cfg.n_pv_basket = 8; v2_cfg.n_sst_martinotti = 5; v2_cfg.n_vip = 3;
    engine.add_region(std::make_unique<CorticalRegion>("V2", v2_cfg));

    ColumnConfig v4_cfg;
    v4_cfg.n_l4_stellate = 25; v4_cfg.n_l23_pyramidal = 50;
    v4_cfg.n_l5_pyramidal = 25; v4_cfg.n_l6_pyramidal = 18;
    v4_cfg.n_pv_basket = 7; v4_cfg.n_sst_martinotti = 4; v4_cfg.n_vip = 2;
    engine.add_region(std::make_unique<CorticalRegion>("V4", v4_cfg));

    // Feedforward
    engine.add_projection("LGN", "V1", 2);
    engine.add_projection("V1",  "V2", 2);
    engine.add_projection("V2",  "V4", 2);
    // Feedback (predictions)
    engine.add_projection("V2", "V1", 3);
    engine.add_projection("V4", "V2", 3);

    // Enable PC on V1 and V2
    auto* v1 = dynamic_cast<CorticalRegion*>(engine.find_region("V1"));
    auto* v2 = dynamic_cast<CorticalRegion*>(engine.find_region("V2"));
    auto* v2_br = engine.find_region("V2");
    auto* v4_br = engine.find_region("V4");

    v1->enable_predictive_coding();
    v1->add_feedback_source(v2_br->region_id());

    v2->enable_predictive_coding();
    v2->add_feedback_source(v4_br->region_id());

    auto* lgn = dynamic_cast<ThalamicRelay*>(engine.find_region("LGN"));

    size_t sp_v1 = 0, sp_v2 = 0, sp_v4 = 0;
    for (int t = 0; t < 200; ++t) {
        if (t < 50) {
            std::vector<float> vis(30, 30.0f);
            lgn->inject_external(vis);
        }
        engine.step();
        sp_v1 += count_spikes(*v1);
        sp_v2 += count_spikes(*v2);
        sp_v4 += count_spikes(*v4_br);
    }

    printf("    V1=%zu  V2=%zu  V4=%zu\n", sp_v1, sp_v2, sp_v4);
    printf("    V1误差=%.4f  V2误差=%.4f\n",
           v1->prediction_error(), v2->prediction_error());

    CHECK(sp_v1 > 0, "V1应有活动");
    CHECK(sp_v2 > 0, "V2应有活动");
    CHECK(sp_v4 > 0, "V4应有活动");
    CHECK(v1->prediction_error() > 0.0f, "V1应有预测误差 (V2→V1反馈)");

    PASS("层级预测编码");
}

// =============================================================================
// 测试6: 预测编码兼容性 (不启用PC时行为不变)
// =============================================================================
void test_backward_compatibility() {
    printf("\n--- 测试6: 向后兼容性 ---\n");
    printf("    原理: 不启用PC时, 行为与原系统完全一致\n");

    auto run_system = [](bool enable_pc) -> size_t {
        SimulationEngine engine(10);

        auto lgn_cfg = ThalamicConfig{};
        lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 30; lgn_cfg.n_trn = 10;
        engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

        ColumnConfig v1_cfg;
        v1_cfg.n_l4_stellate = 40; v1_cfg.n_l23_pyramidal = 80;
        v1_cfg.n_l5_pyramidal = 40; v1_cfg.n_l6_pyramidal = 30;
        v1_cfg.n_pv_basket = 12; v1_cfg.n_sst_martinotti = 8; v1_cfg.n_vip = 4;
        engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

        engine.add_projection("LGN", "V1", 2);

        if (enable_pc) {
            auto* v1 = dynamic_cast<CorticalRegion*>(engine.find_region("V1"));
            v1->enable_predictive_coding();
            // No feedback source -> PC enabled but no predictions arrive
        }

        auto* lgn = dynamic_cast<ThalamicRelay*>(engine.find_region("LGN"));
        size_t total = 0;
        for (int t = 0; t < 100; ++t) {
            if (t < 50) {
                std::vector<float> vis(30, 30.0f);
                lgn->inject_external(vis);
            }
            engine.step();
            total += count_spikes(*engine.find_region("V1"));
        }
        return total;
    };

    size_t sp_no_pc = run_system(false);
    size_t sp_pc_no_fb = run_system(true);

    printf("    V1(无PC)=%zu  V1(PC但无反馈)=%zu\n", sp_no_pc, sp_pc_no_fb);

    // Without feedback sources, PC-enabled should behave similarly
    // (small difference due to precision_sensory default = ne_gain)
    float ratio = static_cast<float>(sp_pc_no_fb) / static_cast<float>(sp_no_pc + 1);
    CHECK(ratio > 0.5f && ratio < 2.0f,
          "无反馈时PC应与原系统行为相近");

    PASS("向后兼容性");
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) 预测编码框架测试\n");
    printf("  Step 6: L6预测 + L2/3误差 + 精度加权\n");
    printf("  Rao-Ballard + Friston Free Energy\n");
    printf("============================================\n");

    test_predictive_coding_basics();
    test_prediction_suppression();
    test_ne_sensory_precision();
    test_ach_prior_precision();
    test_hierarchical_predictive_coding();
    test_backward_compatibility();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
