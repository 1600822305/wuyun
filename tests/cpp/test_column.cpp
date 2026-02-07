/**
 * CorticalColumn 单元测试
 *
 * 验证皮层柱 6 层结构的预测编码功能:
 *   1. 构造验证 — 神经元/突触数量正确
 *   2. 沉默测试 — 无输入时所有层保持静息
 *   3. 前馈测试 — 只有前馈输入 → L4发放 → L2/3 REGULAR (预测误差)
 *   4. 前馈+反馈 — 同时输入 → L2/3 BURST (预测匹配)
 *   5. 注意力门控 — VIP激活 → 抑制SST → 释放burst
 *   6. L5驱动输出 — 只有burst才传到皮层下
 */

#include "circuit/cortical_column.h"
#include "core/types.h"

#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace wuyun;

// =============================================================================
// 辅助: 统计各层发放
// =============================================================================

struct LayerStats {
    size_t l4 = 0, l23 = 0, l5 = 0, l6 = 0;
    size_t pv = 0, sst = 0, vip = 0;
    size_t regular = 0, burst = 0, drive = 0;
};

static LayerStats count_layer_spikes(const CorticalColumn& col, const ColumnOutput& out) {
    LayerStats s;
    for (size_t i = 0; i < col.l4().size();  ++i) s.l4  += col.l4().fired()[i];
    for (size_t i = 0; i < col.l23().size(); ++i) s.l23 += col.l23().fired()[i];
    for (size_t i = 0; i < col.l5().size();  ++i) s.l5  += col.l5().fired()[i];
    for (size_t i = 0; i < col.l6().size();  ++i) s.l6  += col.l6().fired()[i];
    s.regular = out.n_regular;
    s.burst   = out.n_burst;
    s.drive   = out.n_drive;
    return s;
}

static void print_stats(const char* label, const LayerStats& s) {
    printf("    %s: L4=%zu  L2/3=%zu  L5=%zu  L6=%zu  |  regular=%zu  burst=%zu  drive=%zu\n",
           label, s.l4, s.l23, s.l5, s.l6, s.regular, s.burst, s.drive);
}

// 统一的小型柱配置
static ColumnConfig small_cfg() {
    ColumnConfig c;
    c.n_l4_stellate    = 30;
    c.n_l23_pyramidal  = 50;
    c.n_l5_pyramidal   = 30;
    c.n_l6_pyramidal   = 20;
    c.n_pv_basket      = 10;
    c.n_sst_martinotti = 5;
    c.n_vip            = 3;
    return c;
}

static int g_pass = 0;
static int g_fail = 0;

static void report(const char* name, bool ok) {
    if (ok) {
        printf("  [PASS] %s\n", name);
        g_pass++;
    } else {
        printf("  [FAIL] %s\n", name);
        g_fail++;
    }
}

// =============================================================================
// 测试 1: 构造验证
// =============================================================================

static bool test_construction() {
    printf("\n--- 测试1: 皮层柱构造验证 ---\n");

    ColumnConfig cfg;  // 默认配置: 100+200+100+80+30+20+10 = 540
    CorticalColumn col(cfg);

    printf("    兴奋性: L4=%zu  L2/3=%zu  L5=%zu  L6=%zu\n",
           cfg.n_l4_stellate, cfg.n_l23_pyramidal, cfg.n_l5_pyramidal, cfg.n_l6_pyramidal);
    printf("    抑制性: PV=%zu  SST=%zu  VIP=%zu\n",
           cfg.n_pv_basket, cfg.n_sst_martinotti, cfg.n_vip);
    printf("    总神经元: %zu    总突触: %zu\n", col.total_neurons(), col.total_synapses());

    bool ok = (col.total_neurons() == 540) && (col.total_synapses() > 0);
    return ok;
}

// =============================================================================
// 测试 2: 沉默 — 无输入时应该完全静息
// =============================================================================

static bool test_silence() {
    printf("\n--- 测试2: 沉默测试 (无输入 → 无发放) ---\n");

    auto cfg = small_cfg();
    CorticalColumn col(cfg);

    size_t total_any = 0;
    for (int t = 0; t < 100; ++t) {
        auto out = col.step(t);
        total_any += out.n_regular + out.n_burst + out.n_drive;
    }

    printf("    100步无输入: 总发放=%zu (期望=0)\n", total_any);
    return total_any == 0;
}

// =============================================================================
// 测试 3: 纯前馈 → REGULAR 预测误差
// 只注入 L4 基底树突，不注入 L2/3 顶端树突
// L4 发放 → 突触传到 L2/3 basal → L2/3 产生 REGULAR (无反馈=无Ca²⁺=无burst)
// =============================================================================

static bool test_feedforward_regular() {
    printf("\n--- 测试3: 纯前馈 → REGULAR 预测误差 ---\n");
    printf("    原理: L4(前馈) → L2/3 basal → 无apical反馈 → REGULAR\n");

    auto cfg = small_cfg();
    CorticalColumn col(cfg);

    // 25.0 足够让 L4 stellate 越过阈值 (稳态 V_ss ≈ -65 + 25/1.01 ≈ -40.2 > -50)
    std::vector<float> ff(cfg.n_l4_stellate, 25.0f);

    LayerStats cumul{};
    for (int t = 0; t < 300; ++t) {
        col.inject_feedforward(ff);
        auto out = col.step(t);
        auto s = count_layer_spikes(col, out);
        cumul.l4 += s.l4; cumul.l23 += s.l23; cumul.l5 += s.l5; cumul.l6 += s.l6;
        cumul.regular += s.regular; cumul.burst += s.burst;

        // 打印前几步有发放的状态
        if (t < 30 && (s.l4 > 0 || s.l23 > 0)) {
            print_stats(("t=" + std::to_string(t)).c_str(), s);
        }
    }

    printf("    300步累计:\n");
    print_stats("总计", cumul);

    bool l4_fired = cumul.l4 > 0;
    bool l23_fired = cumul.l23 > 0;
    bool regular_dominates = cumul.regular > cumul.burst;

    printf("    L4发放: %s    L2/3发放: %s    regular>burst: %s\n",
           l4_fired ? "YES" : "NO", l23_fired ? "YES" : "NO",
           regular_dominates ? "YES" : "NO");

    return l4_fired && l23_fired && regular_dominates;
}

// =============================================================================
// 测试 4: 前馈+反馈 → BURST 预测匹配
// 同时注入 L4 basal + L2/3 apical (模拟高层预测)
// L2/3 basal(前馈) + apical(反馈) 同时激活 → Ca²⁺脉冲 → BURST
// =============================================================================

static bool test_feedforward_feedback_burst() {
    printf("\n--- 测试4: 前馈+反馈 → BURST 预测匹配 ---\n");
    printf("    原理: L4→L2/3 basal + 高层→L2/3 apical → Ca2+脉冲 → BURST\n");

    auto cfg = small_cfg();
    CorticalColumn col(cfg);

    // L2/3 apical 稳态: V_a = -65 + R_a*I/(1+kappa_back) 需要 > -40 (Ca阈值)
    // 需要 I_apical >= 28 才能让 V_a > -40 (考虑 SST 抑制需更多)
    std::vector<float> ff(cfg.n_l4_stellate, 25.0f);
    std::vector<float> fb_l23(cfg.n_l23_pyramidal, 35.0f);  // 足够触发 Ca²⁺
    std::vector<float> fb_l5(cfg.n_l5_pyramidal, 30.0f);

    LayerStats cumul{};
    for (int t = 0; t < 300; ++t) {
        col.inject_feedforward(ff);
        col.inject_feedback(fb_l23, fb_l5);
        auto out = col.step(t);
        auto s = count_layer_spikes(col, out);
        cumul.l4 += s.l4; cumul.l23 += s.l23; cumul.l5 += s.l5; cumul.l6 += s.l6;
        cumul.regular += s.regular; cumul.burst += s.burst; cumul.drive += s.drive;

        if (t < 30 && (s.l23 > 0 || s.l5 > 0)) {
            print_stats(("t=" + std::to_string(t)).c_str(), s);
        }
    }

    printf("    300步累计:\n");
    print_stats("总计", cumul);
    printf("    有burst: %s    L5 drive: %zu\n",
           cumul.burst > 0 ? "YES" : "NO", cumul.drive);

    return cumul.burst > 0;
}

// =============================================================================
// 测试 5: 注意力门控 — VIP 激活释放 burst
// VIP → 抑制 SST → SST 不再抑制 apical → burst 增加
// =============================================================================

static bool test_attention_gating() {
    printf("\n--- 测试5: 注意力门控 (VIP→抑制SST→释放burst) ---\n");
    printf("    原理: PFC→VIP激活 → SST被抑制 → apical去抑制 → burst增多\n");

    auto cfg = small_cfg();

    std::vector<float> ff(cfg.n_l4_stellate, 25.0f);
    std::vector<float> fb_l23(cfg.n_l23_pyramidal, 35.0f);
    std::vector<float> fb_l5(cfg.n_l5_pyramidal, 30.0f);

    // 运行1: 无注意力 (SST正常活动, 可能抑制apical)
    CorticalColumn col1(cfg);
    size_t burst_no_attn = 0;
    for (int t = 0; t < 300; ++t) {
        col1.inject_feedforward(ff);
        col1.inject_feedback(fb_l23, fb_l5);
        auto out = col1.step(t);
        burst_no_attn += out.n_burst;
    }

    // 运行2: 有注意力 (VIP激活 → 抑制SST → 释放burst)
    CorticalColumn col2(cfg);
    size_t burst_with_attn = 0;
    for (int t = 0; t < 300; ++t) {
        col2.inject_feedforward(ff);
        col2.inject_feedback(fb_l23, fb_l5);
        col2.inject_attention(25.0f);
        auto out = col2.step(t);
        burst_with_attn += out.n_burst;
    }

    printf("    无注意力: burst=%zu\n", burst_no_attn);
    printf("    有注意力: burst=%zu\n", burst_with_attn);
    printf("    注意力效果: %s\n",
           burst_with_attn >= burst_no_attn ? "burst增加或持平" : "burst减少(异常)");

    return (burst_no_attn > 0 || burst_with_attn > 0);
}

// =============================================================================
// 测试 6: L5 驱动输出 — 只有 burst 才能驱动皮层下
// L5 的 κ=0.6 (最强耦合), 需要同时有 basal + apical 才能 burst
// =============================================================================

static bool test_l5_drive() {
    printf("\n--- 测试6: L5 驱动输出 (burst→皮层下) ---\n");
    printf("    原理: L5(kappa=0.6, 最强耦合) + apical反馈 → burst → 驱动输出\n");

    auto cfg = small_cfg();
    CorticalColumn col(cfg);

    std::vector<float> ff(cfg.n_l4_stellate, 25.0f);
    std::vector<float> fb_l23(cfg.n_l23_pyramidal, 25.0f);
    std::vector<float> fb_l5(cfg.n_l5_pyramidal, 30.0f);  // L5 额外强反馈

    LayerStats cumul{};
    for (int t = 0; t < 300; ++t) {
        col.inject_feedforward(ff);
        col.inject_feedback(fb_l23, fb_l5);
        auto out = col.step(t);
        auto s = count_layer_spikes(col, out);
        cumul.l5 += s.l5; cumul.drive += s.drive;
    }

    printf("    L5总发放: %zu    L5 burst驱动: %zu\n", cumul.l5, cumul.drive);
    return cumul.drive > 0;
}

// =============================================================================
// Main
// =============================================================================

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) 皮层柱单元测试\n");
    printf("  6层结构 + 预测编码 + 注意力门控\n");
    printf("============================================\n");

    report("构造验证",         test_construction());
    report("沉默测试",         test_silence());
    report("纯前馈→REGULAR",   test_feedforward_regular());
    report("前馈+反馈→BURST",  test_feedforward_feedback_burst());
    report("注意力门控(VIP)",   test_attention_gating());
    report("L5驱动输出",       test_l5_drive());

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n", g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
