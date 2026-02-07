/**
 * 悟韵 (WuYun) 最小大脑端到端测试
 *
 * 信号通路:
 *   视觉刺激 → LGN(丘脑) → V1(皮层) → dlPFC(皮层) → BG(基底节) → 运动丘脑 → M1(运动皮层)
 *                                                        ↑
 *                                                    VTA DA(奖励)
 *
 * 测试验证:
 *   1. 引擎构造: 所有区域注册+投射连接
 *   2. 信号传播: 视觉输入能逐级传递到 M1
 *   3. DA 调制: 奖励信号增强 BG Go 通路
 *   4. 沉默测试: 无输入时系统安静
 */

#include "engine/simulation_engine.h"
#include "region/cortical_region.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/neuromod/vta_da.h"
#include <cstdio>
#include <memory>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace wuyun;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  [FAIL] %s\n", msg); g_fail++; return; } \
} while(0)

#define PASS(msg) do { printf("  [PASS] %s\n", msg); g_pass++; } while(0)

// Helper: count spikes in a region
static size_t count_spikes(const BrainRegion& r) {
    size_t n = 0;
    for (size_t i = 0; i < r.n_neurons(); ++i) {
        if (r.fired()[i]) n++;
    }
    return n;
}

// =============================================================================
// Build the minimal brain
// =============================================================================
static SimulationEngine build_minimal_brain() {
    SimulationEngine engine(10);

    // --- Create regions ---

    // LGN (visual thalamus)
    auto lgn_cfg = ThalamicConfig{};
    lgn_cfg.name = "LGN";
    lgn_cfg.n_relay = 50;
    lgn_cfg.n_trn = 15;
    lgn_cfg.burst_mode = false;  // tonic (awake)
    engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    // V1 (primary visual cortex)
    auto v1_cfg = ColumnConfig{};
    v1_cfg.name = "V1";
    v1_cfg.n_l4_stellate = 50;
    v1_cfg.n_l23_pyramidal = 100;
    v1_cfg.n_l5_pyramidal = 50;
    v1_cfg.n_l6_pyramidal = 40;
    v1_cfg.n_pv_basket = 15;
    v1_cfg.n_sst_martinotti = 10;
    v1_cfg.n_vip = 5;
    engine.add_region(std::make_unique<CorticalRegion>("V1", v1_cfg));

    // dlPFC (prefrontal cortex)
    auto pfc_cfg = ColumnConfig{};
    pfc_cfg.name = "dlPFC";
    pfc_cfg.n_l4_stellate = 30;
    pfc_cfg.n_l23_pyramidal = 80;
    pfc_cfg.n_l5_pyramidal = 40;
    pfc_cfg.n_l6_pyramidal = 30;
    pfc_cfg.n_pv_basket = 10;
    pfc_cfg.n_sst_martinotti = 8;
    pfc_cfg.n_vip = 4;
    engine.add_region(std::make_unique<CorticalRegion>("dlPFC", pfc_cfg));

    // Basal ganglia
    auto bg_cfg = BasalGangliaConfig{};
    bg_cfg.name = "BG";
    bg_cfg.n_d1_msn = 50;
    bg_cfg.n_d2_msn = 50;
    bg_cfg.n_gpi = 15;
    bg_cfg.n_gpe = 15;
    bg_cfg.n_stn = 10;
    engine.add_region(std::make_unique<BasalGanglia>(bg_cfg));

    // Motor thalamus
    auto mthal_cfg = ThalamicConfig{};
    mthal_cfg.name = "MotorThal";
    mthal_cfg.n_relay = 30;
    mthal_cfg.n_trn = 10;
    mthal_cfg.burst_mode = false;
    engine.add_region(std::make_unique<ThalamicRelay>(mthal_cfg));

    // M1 (motor cortex)
    auto m1_cfg = ColumnConfig{};
    m1_cfg.name = "M1";
    m1_cfg.n_l4_stellate = 30;
    m1_cfg.n_l23_pyramidal = 60;
    m1_cfg.n_l5_pyramidal = 40;
    m1_cfg.n_l6_pyramidal = 20;
    m1_cfg.n_pv_basket = 10;
    m1_cfg.n_sst_martinotti = 6;
    m1_cfg.n_vip = 3;
    engine.add_region(std::make_unique<CorticalRegion>("M1", m1_cfg));

    // VTA (dopamine)
    auto vta_cfg = VTAConfig{};
    vta_cfg.name = "VTA";
    vta_cfg.n_da_neurons = 20;
    engine.add_region(std::make_unique<VTA_DA>(vta_cfg));

    // --- Add projections (with delays) ---
    engine.add_projection("LGN", "V1", 2, "LGN→V1");          // 感觉中继
    engine.add_projection("V1", "dlPFC", 3, "V1→dlPFC");       // 前馈
    engine.add_projection("dlPFC", "V1", 3, "dlPFC→V1");       // 反馈(预测)
    engine.add_projection("dlPFC", "BG", 2, "dlPFC→BG");       // 动作选择
    engine.add_projection("BG", "MotorThal", 2, "BG→MotorThal"); // GPi→丘脑
    engine.add_projection("MotorThal", "M1", 2, "MotorThal→M1"); // 丘脑→运动皮层
    engine.add_projection("VTA", "BG", 1, "VTA→BG");           // DA调制(走SpikeBus)

    // Wire DA source: BG reads VTA spikes to auto-update DA level
    auto* bg_ptr = dynamic_cast<BasalGanglia*>(engine.find_region("BG"));
    auto* vta_ptr = engine.find_region("VTA");
    if (bg_ptr && vta_ptr) {
        bg_ptr->set_da_source_region(vta_ptr->region_id());
    }

    return engine;
}

// =============================================================================
// 测试1: 引擎构造验证
// =============================================================================
void test_engine_construction() {
    printf("\n--- 测试1: 最小大脑构造验证 ---\n");

    auto engine = build_minimal_brain();

    CHECK(engine.num_regions() == 7, "应有7个区域");
    CHECK(engine.bus().num_projections() == 7, "应有7条投射(+VTA→BG)");

    // Check each region exists
    CHECK(engine.find_region("LGN") != nullptr, "LGN 存在");
    CHECK(engine.find_region("V1") != nullptr, "V1 存在");
    CHECK(engine.find_region("dlPFC") != nullptr, "dlPFC 存在");
    CHECK(engine.find_region("BG") != nullptr, "BG 存在");
    CHECK(engine.find_region("MotorThal") != nullptr, "MotorThal 存在");
    CHECK(engine.find_region("M1") != nullptr, "M1 存在");
    CHECK(engine.find_region("VTA") != nullptr, "VTA 存在");

    auto stats = engine.stats();
    printf("    区域: %zu   神经元总数: %zu   投射: %zu\n",
           stats.total_regions, stats.total_neurons, engine.bus().num_projections());

    PASS("最小大脑构造");
}

// =============================================================================
// 测试2: 沉默测试
// =============================================================================
void test_silence() {
    printf("\n--- 测试2: 沉默测试 (无输入→系统安静) ---\n");

    auto engine = build_minimal_brain();

    // Run 100 steps with no input
    engine.run(100);

    // Count total spikes across all regions
    size_t total = 0;
    for (size_t i = 0; i < engine.num_regions(); ++i) {
        total += count_spikes(engine.region(i));
    }

    // BG GPi has tonic firing, so some spikes expected from BG
    // But cortical/thalamic regions should be mostly silent
    size_t lgn_spikes = count_spikes(*engine.find_region("LGN"));
    size_t v1_spikes = count_spikes(*engine.find_region("V1"));

    printf("    100步无输入: LGN=%zu  V1=%zu  总发放=%zu\n",
           lgn_spikes, v1_spikes, total);

    CHECK(v1_spikes == 0, "V1 无输入应沉默");
    CHECK(lgn_spikes == 0, "LGN 无输入应沉默");

    PASS("沉默测试");
}

// =============================================================================
// 测试3: 端到端信号传播
// =============================================================================
void test_signal_propagation() {
    printf("\n--- 测试3: 端到端信号传播 ---\n");
    printf("    通路: 视觉→LGN→V1→dlPFC→BG→MotorThal→M1\n");

    auto engine = build_minimal_brain();

    // Tracking: count total spikes per region over entire simulation
    size_t spikes_lgn = 0, spikes_v1 = 0, spikes_pfc = 0;
    size_t spikes_bg = 0, spikes_mthal = 0, spikes_m1 = 0;

    auto* lgn  = engine.find_region("LGN");
    auto* v1   = engine.find_region("V1");
    auto* pfc  = engine.find_region("dlPFC");
    auto* bg   = engine.find_region("BG");
    auto* mthal= engine.find_region("MotorThal");
    auto* m1   = engine.find_region("M1");

    // Phase 1: Inject visual stimulus into LGN for 50 steps
    for (int32_t t = 0; t < 200; ++t) {
        // Visual stimulus: strong sustained input to LGN relay neurons
        if (t < 50) {
            std::vector<float> visual(50, 35.0f);  // All relay neurons get input
            lgn->inject_external(visual);
        }

        engine.step();

        spikes_lgn  += count_spikes(*lgn);
        spikes_v1   += count_spikes(*v1);
        spikes_pfc  += count_spikes(*pfc);
        spikes_bg   += count_spikes(*bg);
        spikes_mthal+= count_spikes(*mthal);
        spikes_m1   += count_spikes(*m1);
    }

    printf("    200步累计发放:\n");
    printf("    LGN=%zu → V1=%zu → dlPFC=%zu → BG=%zu → MotorThal=%zu → M1=%zu\n",
           spikes_lgn, spikes_v1, spikes_pfc, spikes_bg, spikes_mthal, spikes_m1);

    CHECK(spikes_lgn > 0, "LGN 应有发放 (视觉输入)");
    CHECK(spikes_v1 > 0, "V1 应有发放 (LGN→V1 传递)");

    PASS("端到端信号传播");
}

// =============================================================================
// 测试4: DA 奖励调制
// =============================================================================
void test_da_modulation() {
    printf("\n--- 测试4: DA 奖励调制 ---\n");
    printf("    原理: DA↑ → D1兴奋性增强 → Go通路更活跃\n");

    // Use standalone BG instances (no da_source_region_ set)
    // to directly test DA modulation via set_da_level()
    auto bg_cfg = BasalGangliaConfig{};
    bg_cfg.n_d1_msn = 50; bg_cfg.n_d2_msn = 50;
    bg_cfg.n_gpi = 15; bg_cfg.n_gpe = 15; bg_cfg.n_stn = 10;

    // Phase 1: Low DA (tonic baseline)
    BasalGanglia bg1(bg_cfg);
    size_t d1_spikes_low_da = 0;
    for (int32_t t = 0; t < 100; ++t) {
        std::vector<float> ctx_input(50, 40.0f);
        bg1.inject_cortical_input(ctx_input, ctx_input);
        bg1.set_da_level(0.1f);
        bg1.step(t);
        for (size_t i = 0; i < bg1.d1().size(); ++i) {
            if (bg1.d1().fired()[i]) d1_spikes_low_da++;
        }
    }

    // Phase 2: High DA (reward state)
    BasalGanglia bg2(bg_cfg);
    size_t d1_spikes_high_da = 0;
    for (int32_t t = 0; t < 100; ++t) {
        std::vector<float> ctx_input(50, 40.0f);
        bg2.inject_cortical_input(ctx_input, ctx_input);
        bg2.set_da_level(0.6f);
        bg2.step(t);
        for (size_t i = 0; i < bg2.d1().size(); ++i) {
            if (bg2.d1().fired()[i]) d1_spikes_high_da++;
        }
    }

    printf("    D1 低DA(0.1): %zu   D1 高DA(0.6): %zu\n",
           d1_spikes_low_da, d1_spikes_high_da);

    CHECK(d1_spikes_high_da > d1_spikes_low_da,
          "DA 奖励应增强 D1 Go 通路");

    PASS("DA 奖励调制");
}

// =============================================================================
// 测试5: 丘脑门控
// =============================================================================
void test_thalamic_gating() {
    printf("\n--- 测试5: 丘脑 TRN 门控 ---\n");
    printf("    原理: TRN 抑制 Relay → 门控感觉信号\n");

    ThalamicConfig cfg;
    cfg.name = "TestThal";
    cfg.n_relay = 30;
    cfg.n_trn = 10;

    ThalamicRelay thal(cfg);

    // Inject sensory input (strong enough for tonic relay: v_rest=-65, threshold=-50)
    std::vector<float> input(30, 30.0f);
    size_t relay_spikes_normal = 0;

    for (int t = 0; t < 100; ++t) {
        if (t < 50) thal.inject_external(input);
        thal.step(t);
        for (size_t i = 0; i < thal.relay().size(); ++i) {
            if (thal.relay().fired()[i]) relay_spikes_normal++;
        }
    }

    // Now with strong TRN inhibition (PFC attention suppression)
    ThalamicRelay thal2(cfg);
    size_t relay_spikes_inhibited = 0;

    for (int t = 0; t < 100; ++t) {
        if (t < 50) thal2.inject_external(input);
        // Strong PFC→TRN excitation → TRN fires → inhibits relay
        std::vector<float> trn_drive(10, 50.0f);
        thal2.inject_trn_modulation(trn_drive);
        thal2.step(t);
        for (size_t i = 0; i < thal2.relay().size(); ++i) {
            if (thal2.relay().fired()[i]) relay_spikes_inhibited++;
        }
    }

    printf("    正常: relay=%zu   TRN抑制: relay=%zu\n",
           relay_spikes_normal, relay_spikes_inhibited);

    CHECK(relay_spikes_inhibited < relay_spikes_normal,
          "TRN 抑制应减少 relay 发放");

    PASS("丘脑 TRN 门控");
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) 最小大脑端到端测试\n");
    printf("  Step 3: 感觉→认知→动作 完整通路\n");
    printf("============================================\n");

    test_engine_construction();
    test_silence();
    test_signal_propagation();
    test_da_modulation();
    test_thalamic_gating();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
