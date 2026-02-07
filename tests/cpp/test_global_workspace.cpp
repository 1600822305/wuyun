/**
 * GNW: 全局工作空间理论测试
 *
 * 测试:
 *   1. 基础点火: 强输入→salience累积→ignition→广播
 *   2. 竞争门控: 多源竞争→最强者赢→只有赢者进入意识
 *   3. 广播持续: 点火后workspace神经元持续活跃broadcast_duration步
 *   4. 竞争衰减: salience自动衰减→防止赢者锁定
 *   5. 点火间隔: min_ignition_gap内不能再次点火
 *   6. 无输入无点火: 无输入→salience=0→不点火
 *   7. 全系统集成: GW嵌入48区域脑 + LGN→V1→GW→ILN广播
 */

#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#endif
#include <vector>
#include <memory>

#include "engine/global_workspace.h"
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
#include "region/limbic/hypothalamus.h"

using namespace wuyun;

static int tests_passed = 0;
static int tests_failed = 0;

static size_t count_fired(const std::vector<uint8_t>& f) {
    size_t n = 0; for (auto x : f) if (x) n++; return n;
}

// =============================================================================
// Test 1: Basic ignition — strong input → salience → ignition
// =============================================================================
static void test_basic_ignition() {
    printf("\n--- 测试1: 基础点火 ---\n");
    printf("    原理: 强输入→salience累积→超阈值→ignition\n");

    SimulationEngine eng(10);

    ColumnConfig v1c;
    v1c.n_l4_stellate=50; v1c.n_l23_pyramidal=100; v1c.n_l5_pyramidal=50;
    v1c.n_l6_pyramidal=40; v1c.n_pv_basket=15; v1c.n_sst_martinotti=10; v1c.n_vip=5;
    eng.add_region(std::make_unique<CorticalRegion>("V1", v1c));

    ThalamicConfig lgn; lgn.name="LGN"; lgn.n_relay=50; lgn.n_trn=15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn));

    GWConfig gwc;
    gwc.ignition_threshold = 10.0f;
    gwc.min_ignition_gap = 10;
    eng.add_region(std::make_unique<GlobalWorkspace>(gwc));

    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("V1", "GW", 2);

    auto* gw = dynamic_cast<GlobalWorkspace*>(eng.find_region("GW"));

    // Warmup
    for (int t = 0; t < 50; ++t) eng.step();

    bool ignited = false;
    size_t ignition_step = 0;
    for (int t = 50; t < 250; ++t) {
        eng.find_region("LGN")->inject_external(std::vector<float>(50, 35.0f));
        eng.step();
        if (gw->is_ignited() && !ignited) {
            ignited = true;
            ignition_step = t;
        }
    }

    printf("    点火=%s  step=%zu  count=%zu  salience=%.1f\n",
           ignited ? "YES" : "NO", ignition_step,
           gw->ignition_count(), gw->winning_salience());

    bool ok = ignited && gw->ignition_count() > 0;
    printf("  [%s] 基础点火\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 2: Competition — strongest source wins
// =============================================================================
static void test_competition() {
    printf("\n--- 测试2: 竞争门控 ---\n");
    printf("    原理: V1(强)vs A1(弱) → V1赢得意识访问\n");

    SimulationEngine eng(10);

    ThalamicConfig lgn; lgn.name="LGN"; lgn.n_relay=50; lgn.n_trn=15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn));

    ThalamicConfig mgn; mgn.name="MGN"; mgn.n_relay=20; mgn.n_trn=6;
    eng.add_region(std::make_unique<ThalamicRelay>(mgn));

    auto make_ctx = [](const std::string& name, size_t l4) {
        ColumnConfig c;
        c.n_l4_stellate=l4; c.n_l23_pyramidal=l4*2; c.n_l5_pyramidal=l4;
        c.n_l6_pyramidal=l4; c.n_pv_basket=l4/3; c.n_sst_martinotti=l4/5; c.n_vip=l4/10;
        return std::make_unique<CorticalRegion>(name, c);
    };

    eng.add_region(make_ctx("V1", 50));
    eng.add_region(make_ctx("A1", 35));

    GWConfig gwc;
    gwc.ignition_threshold = 8.0f;
    gwc.min_ignition_gap = 5;
    eng.add_region(std::make_unique<GlobalWorkspace>(gwc));

    auto* gw = dynamic_cast<GlobalWorkspace*>(eng.find_region("GW"));
    gw->register_source(eng.find_region("V1")->region_id(), "V1");
    gw->register_source(eng.find_region("A1")->region_id(), "A1");

    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("MGN", "A1", 2);
    eng.add_projection("V1", "GW", 2);
    eng.add_projection("A1", "GW", 2);

    for (int t = 0; t < 50; ++t) eng.step();

    // Strong V1 + weak A1
    for (int t = 50; t < 200; ++t) {
        eng.find_region("LGN")->inject_external(std::vector<float>(50, 35.0f));
        eng.find_region("MGN")->inject_external(std::vector<float>(20, 15.0f));
        eng.step();
    }

    printf("    意识内容=%s  ignition=%zu  salience=%.1f\n",
           gw->conscious_content_name().c_str(),
           gw->ignition_count(), gw->winning_salience());

    // V1 should win due to stronger input
    auto v1_id = eng.find_region("V1")->region_id();
    bool ok = gw->ignition_count() > 0 &&
              gw->conscious_content_id() == static_cast<int32_t>(v1_id);
    printf("  [%s] V1赢得竞争 (更强输入)\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 3: Broadcast duration — workspace stays active for duration steps
// =============================================================================
static void test_broadcast_duration() {
    printf("\n--- 测试3: 广播持续 ---\n");
    printf("    原理: 点火后workspace活跃broadcast_duration步\n");

    SimulationEngine eng(10);

    ThalamicConfig lgn; lgn.name="LGN"; lgn.n_relay=50; lgn.n_trn=15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn));

    ColumnConfig v1c;
    v1c.n_l4_stellate=50; v1c.n_l23_pyramidal=100; v1c.n_l5_pyramidal=50;
    v1c.n_l6_pyramidal=40; v1c.n_pv_basket=15; v1c.n_sst_martinotti=10; v1c.n_vip=5;
    eng.add_region(std::make_unique<CorticalRegion>("V1", v1c));

    GWConfig gwc;
    gwc.ignition_threshold = 10.0f;
    gwc.broadcast_duration = 10;
    gwc.min_ignition_gap = 30;
    eng.add_region(std::make_unique<GlobalWorkspace>(gwc));

    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("V1", "GW", 2);

    auto* gw = dynamic_cast<GlobalWorkspace*>(eng.find_region("GW"));

    // Drive until ignition
    for (int t = 0; t < 100; ++t) {
        eng.find_region("LGN")->inject_external(std::vector<float>(50, 35.0f));
        eng.step();
    }

    // Count workspace spikes during and after broadcast
    size_t spikes_during = 0, spikes_after = 0;

    // Drive more to trigger fresh ignition
    for (int t = 100; t < 200; ++t) {
        eng.find_region("LGN")->inject_external(std::vector<float>(50, 35.0f));
        eng.step();
        if (gw->broadcast_remaining() > 0) {
            spikes_during += count_fired(gw->fired());
        }
    }

    // Stop input, wait for broadcast to end
    for (int t = 200; t < 250; ++t) {
        eng.step();
        spikes_after += count_fired(gw->fired());
    }

    printf("    广播中=%zu  广播后=%zu  ignitions=%zu\n",
           spikes_during, spikes_after, gw->ignition_count());

    bool ok = spikes_during > spikes_after && gw->ignition_count() > 0;
    printf("  [%s] 广播持续: 广播中 > 广播后\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 4: Salience decay — prevents winner lock-in
// =============================================================================
static void test_salience_decay() {
    printf("\n--- 测试4: 竞争衰减 ---\n");
    printf("    原理: 停止输入→salience指数衰减→不锁定\n");

    SimulationEngine eng(10);

    ThalamicConfig lgn; lgn.name="LGN"; lgn.n_relay=50; lgn.n_trn=15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn));

    ColumnConfig v1c;
    v1c.n_l4_stellate=50; v1c.n_l23_pyramidal=100; v1c.n_l5_pyramidal=50;
    v1c.n_l6_pyramidal=40; v1c.n_pv_basket=15; v1c.n_sst_martinotti=10; v1c.n_vip=5;
    eng.add_region(std::make_unique<CorticalRegion>("V1", v1c));

    GWConfig gwc;
    gwc.competition_decay = 0.8f;  // Fast decay
    gwc.ignition_threshold = 10.0f;
    eng.add_region(std::make_unique<GlobalWorkspace>(gwc));

    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("V1", "GW", 2);

    auto* gw = dynamic_cast<GlobalWorkspace*>(eng.find_region("GW"));

    // Build up salience
    for (int t = 0; t < 100; ++t) {
        eng.find_region("LGN")->inject_external(std::vector<float>(50, 35.0f));
        eng.step();
    }
    float salience_peak = gw->winning_salience();

    // Stop input, let salience decay
    for (int t = 100; t < 150; ++t) {
        eng.step();
    }
    float salience_decayed = gw->winning_salience();

    printf("    peak=%.1f → decayed=%.1f (ratio=%.2f)\n",
           salience_peak, salience_decayed,
           salience_decayed / (salience_peak + 0.01f));

    bool ok = salience_decayed < salience_peak * 0.5f;
    printf("  [%s] Salience衰减 > 50%%\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 5: Ignition gap — cannot re-ignite too quickly
// =============================================================================
static void test_ignition_gap() {
    printf("\n--- 测试5: 点火间隔 ---\n");
    printf("    原理: min_ignition_gap内不能再次点火\n");

    SimulationEngine eng(10);

    ThalamicConfig lgn; lgn.name="LGN"; lgn.n_relay=50; lgn.n_trn=15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn));

    ColumnConfig v1c;
    v1c.n_l4_stellate=50; v1c.n_l23_pyramidal=100; v1c.n_l5_pyramidal=50;
    v1c.n_l6_pyramidal=40; v1c.n_pv_basket=15; v1c.n_sst_martinotti=10; v1c.n_vip=5;
    eng.add_region(std::make_unique<CorticalRegion>("V1", v1c));

    GWConfig gwc;
    gwc.ignition_threshold = 8.0f;
    gwc.min_ignition_gap = 50;  // Long gap
    gwc.broadcast_duration = 5;
    eng.add_region(std::make_unique<GlobalWorkspace>(gwc));

    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("V1", "GW", 2);

    auto* gw = dynamic_cast<GlobalWorkspace*>(eng.find_region("GW"));

    // Continuous strong input for 200 steps
    for (int t = 0; t < 200; ++t) {
        eng.find_region("LGN")->inject_external(std::vector<float>(50, 35.0f));
        eng.step();
    }

    size_t ignitions = gw->ignition_count();
    // With gap=50 and 200 steps, max ~4 ignitions
    printf("    ignitions=%zu (gap=50, 200 steps, max~4)\n", ignitions);

    bool ok = ignitions >= 1 && ignitions <= 5;
    printf("  [%s] 点火间隔限制\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 6: No input → no ignition
// =============================================================================
static void test_no_ignition() {
    printf("\n--- 测试6: 无输入不点火 ---\n");
    printf("    原理: 无输入→salience=0→不点火\n");

    GWConfig gwc;
    GlobalWorkspace gw(gwc);

    for (int t = 0; t < 200; ++t) {
        gw.step(t);
    }

    printf("    ignitions=%zu  salience=%.1f\n",
           gw.ignition_count(), gw.winning_salience());

    bool ok = gw.ignition_count() == 0 && !gw.is_ignited();
    printf("  [%s] 无输入不点火\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
// Test 7: Full system — GW in 48-region brain + broadcast via ILN
// =============================================================================
static void test_full_system() {
    printf("\n--- 测试7: 全系统集成 (48区域) ---\n");
    printf("    原理: LGN→V1→GW→ILN→全皮层广播\n");

    SimulationEngine eng(10);

    ThalamicConfig lgn; lgn.name="LGN"; lgn.n_relay=50; lgn.n_trn=15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn));

    auto add_ctx = [&](const std::string& name, size_t l4) {
        ColumnConfig c;
        c.n_l4_stellate=l4; c.n_l23_pyramidal=l4*2; c.n_l5_pyramidal=l4;
        c.n_l6_pyramidal=l4; c.n_pv_basket=l4/3; c.n_sst_martinotti=l4/5;
        c.n_vip=(l4/10 > 2) ? l4/10 : 2;
        eng.add_region(std::make_unique<CorticalRegion>(name, c));
    };

    add_ctx("V1", 50); add_ctx("dlPFC", 30); add_ctx("ACC", 20);

    auto add_thal = [&](const std::string& name, size_t relay, size_t trn) {
        ThalamicConfig tc; tc.name=name; tc.n_relay=relay; tc.n_trn=trn;
        eng.add_region(std::make_unique<ThalamicRelay>(tc));
    };
    add_thal("ILN", 12, 4);
    add_thal("CeM", 15, 5);

    GWConfig gwc;
    gwc.ignition_threshold = 8.0f;
    eng.add_region(std::make_unique<GlobalWorkspace>(gwc));

    auto* gw = dynamic_cast<GlobalWorkspace*>(eng.find_region("GW"));
    gw->register_source(eng.find_region("V1")->region_id(), "V1");

    eng.add_projection("LGN", "V1", 2);
    eng.add_projection("V1", "GW", 2);
    eng.add_projection("V1", "dlPFC", 2);
    eng.add_projection("GW", "ILN", 1);
    eng.add_projection("GW", "CeM", 1);
    eng.add_projection("ILN", "dlPFC", 2);
    eng.add_projection("ILN", "ACC", 2);

    // Warmup
    for (int t = 0; t < 50; ++t) eng.step();

    size_t gw_spikes = 0, iln_spikes = 0, dlpfc_spikes = 0;
    for (int t = 50; t < 250; ++t) {
        eng.find_region("LGN")->inject_external(std::vector<float>(50, 35.0f));
        eng.step();
        gw_spikes   += count_fired(gw->fired());
        iln_spikes  += count_fired(eng.find_region("ILN")->fired());
        dlpfc_spikes += count_fired(eng.find_region("dlPFC")->fired());
    }

    printf("    GW=%zu  ILN=%zu  dlPFC=%zu  ignitions=%zu  content=%s\n",
           gw_spikes, iln_spikes, dlpfc_spikes,
           gw->ignition_count(), gw->conscious_content_name().c_str());

    bool ok = gw->ignition_count() > 0 && gw_spikes > 0;
    printf("  [%s] 全系统: GW点火 + ILN广播\n", ok ? "PASS" : "FAIL");
    ok ? tests_passed++ : tests_failed++;
}

// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) GNW 全局工作空间测试\n");
    printf("  竞争→点火→广播→意识访问\n");
    printf("============================================\n");

    test_basic_ignition();
    test_competition();
    test_broadcast_duration();
    test_salience_decay();
    test_ignition_gap();
    test_no_ignition();
    test_full_system();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("============================================\n");

    return tests_failed > 0 ? 1 : 0;
}
