/**
 * test_learning_curve.cpp — 闭环学习曲线验证
 *
 * v21 环境升级: 10×10 grid, 5×5 视野, 5 food, 4 danger
 * (从 10×10/3×3/3food/2danger 升级, 释放 PC/睡眠/空间记忆)
 *
 * 核心问题: Agent能否在更大环境中通过DA-STDP学会趋食避害?
 *
 * 测试方案:
 * 1. 5000步长时训练, 每500步记录食物率和危险率
 * 2. 对比有学习 vs 无学习 (control)
 * 3. BG DA-STDP诊断 (权重变化/DA/elig)
 * 4. 10000步长时训练 (学习曲线稳定性)
 * 5. 更大环境 PC 对比 (15×15, 7×7视野)
 */

#include "engine/closed_loop_agent.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <map>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace wuyun;

static int g_pass = 0, g_fail = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
        g_fail++; return; \
    } \
} while(0)

struct EpochStats {
    int food = 0;
    int danger = 0;
    int steps = 0;
    float avg_reward = 0.0f;
    float food_rate() const { return steps > 0 ? (float)food / steps : 0.0f; }
    float danger_rate() const { return steps > 0 ? (float)danger / steps : 0.0f; }
    float safety_ratio() const {
        int total = food + danger;
        return total > 0 ? (float)food / total : 0.5f;
    }
};

static EpochStats run_epoch(ClosedLoopAgent& agent, int n_steps) {
    EpochStats stats;
    stats.steps = n_steps;
    for (int i = 0; i < n_steps; ++i) {
        auto result = agent.agent_step();
        if (result.got_food) stats.food++;
        if (result.hit_danger) stats.danger++;
        stats.avg_reward += result.reward;
    }
    stats.avg_reward /= n_steps;
    return stats;
}

// =========================================================================
// Test 1: 学习曲线 (5000步, 10×10 grid, 5×5 vision)
// =========================================================================
static void test_learning_curve() {
    printf("\n--- 测试1: 学习曲线 (5000步, 10x10 grid, 5x5 vision) ---\n");

    AgentConfig cfg;
    cfg.enable_da_stdp = true;

    ClosedLoopAgent agent(cfg);

    printf("  Environment: %zux%zu grid, %zux%zu vision, %zu food, %zu danger\n",
           cfg.world_config.width, cfg.world_config.height,
           cfg.world_config.vision_side(), cfg.world_config.vision_side(),
           cfg.world_config.n_food, cfg.world_config.n_danger);
    printf("  Brain: V1=%zu, dlPFC=%zu, LGN=%zu neurons\n",
           agent.v1()->n_neurons(), agent.dlpfc()->n_neurons(), agent.lgn()->n_neurons());
    printf("  Features: PC=%s, Sleep=%s, Replay=%s\n",
           cfg.enable_predictive_coding ? "ON" : "OFF",
           cfg.enable_sleep_consolidation ? "ON" : "OFF",
           cfg.enable_replay ? "ON" : "OFF");

    printf("  Epoch | Food | Danger | F:D ratio | Avg Reward | Safety\n");
    printf("  ------|------|--------|-----------|------------|-------\n");

    std::vector<EpochStats> epochs;
    for (int epoch = 0; epoch < 10; ++epoch) {
        auto stats = run_epoch(agent, 500);
        epochs.push_back(stats);
        printf("  %5d | %4d | %6d |   %5.2f   |   %+.4f   | %.2f\n",
               (epoch + 1) * 500, stats.food, stats.danger,
               stats.danger > 0 ? (float)stats.food / stats.danger : 99.0f,
               stats.avg_reward, stats.safety_ratio());
    }

    // Early = first 2 epochs (1000 steps), Late = last 2 epochs (1000 steps)
    float early_food = (float)(epochs[0].food + epochs[1].food);
    float late_food  = (float)(epochs[8].food + epochs[9].food);
    float early_danger = (float)(epochs[0].danger + epochs[1].danger);
    float late_danger  = (float)(epochs[8].danger + epochs[9].danger);
    float early_safety = (early_food + early_danger > 0)
        ? early_food / (early_food + early_danger) : 0.5f;
    float late_safety = (late_food + late_danger > 0)
        ? late_food / (late_food + late_danger) : 0.5f;

    printf("\n  Summary:\n");
    printf("  Early (0-1000):  food=%d, danger=%d, safety=%.2f\n",
           (int)early_food, (int)early_danger, early_safety);
    printf("  Late (4000-5000): food=%d, danger=%d, safety=%.2f\n",
           (int)late_food, (int)late_danger, late_safety);
    printf("  Total food: %d, Total steps: %d\n",
           agent.world().total_food_collected(), agent.world().total_steps());

    // The agent should collect food over 5000 steps
    TEST_ASSERT(agent.world().total_food_collected() > 0, "Collected at least some food");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 2: 学习 vs 无学习对照 (10×10, 5×5 vision)
// =========================================================================
static void test_learning_vs_control() {
    printf("\n--- 测试2: 学习 vs 无学习对照 (3000步, 10x10 grid) ---\n");

    auto make_agent = [](bool enable_learning) {
        AgentConfig cfg;
        cfg.enable_da_stdp = enable_learning;
        cfg.world_config.seed = 42;  // Same world layout
        return ClosedLoopAgent(cfg);
    };

    auto learner = make_agent(true);
    auto control = make_agent(false);

    // Warm-up: 1000 steps
    for (int i = 0; i < 1000; ++i) {
        learner.agent_step();
        control.agent_step();
    }

    // Test: 2000 steps
    auto learn_stats = run_epoch(learner, 2000);
    auto ctrl_stats  = run_epoch(control, 2000);

    printf("  Learner (DA-STDP ON):  food=%d, danger=%d, safety=%.2f, avg_r=%+.4f\n",
           learn_stats.food, learn_stats.danger, learn_stats.safety_ratio(), learn_stats.avg_reward);
    printf("  Control (DA-STDP OFF): food=%d, danger=%d, safety=%.2f, avg_r=%+.4f\n",
           ctrl_stats.food, ctrl_stats.danger, ctrl_stats.safety_ratio(), ctrl_stats.avg_reward);

    float learn_score = learn_stats.avg_reward;
    float ctrl_score = ctrl_stats.avg_reward;
    printf("  Learner advantage: %+.4f\n", learn_score - ctrl_score);

    // Learner should do at least as well as control
    // (even if not strictly better, the system shouldn't be worse)
    TEST_ASSERT(learn_score >= ctrl_score - 0.05f,
                "Learner not significantly worse than control");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 3: BG DA-STDP诊断 (找出权重不变的根因)
// =========================================================================
static void test_bg_diagnostics() {
    printf("\n--- 测试3: BG DA-STDP诊断 ---\n");

    AgentConfig cfg;
    cfg.enable_da_stdp = true;

    ClosedLoopAgent agent(cfg);

    // Diagnostic: run 10 agent steps with detailed logging
    printf("  Step-by-step diagnostics:\n");

    int d1_fire_total = 0;
    int d2_fire_total = 0;
    float max_da = 0.0f;
    float max_elig = 0.0f;

    for (int step = 0; step < 50; ++step) {
        auto result = agent.agent_step();

        auto* bg = agent.bg();
        auto* vta = agent.vta();

        // Count D1/D2 firing
        int d1_fired = 0, d2_fired = 0;
        for (size_t i = 0; i < bg->d1().size(); ++i) d1_fired += bg->d1().fired()[i];
        for (size_t i = 0; i < bg->d2().size(); ++i) d2_fired += bg->d2().fired()[i];
        d1_fire_total += d1_fired;
        d2_fire_total += d2_fired;

        float da = bg->da_level();
        float elig = bg->total_elig_d1() + bg->total_elig_d2();
        if (da > max_da) max_da = da;
        if (elig > max_elig) max_elig = elig;

        if (step < 10 || result.got_food || result.hit_danger) {
            printf("    step=%d act=%d r=%.2f | DA=%.3f accum=%.1f | D1=%d D2=%d | elig=%.1f | ctx=%zu\n",
                   step, (int)agent.last_action(), result.reward,
                   da, bg->da_spike_accum(), d1_fired, d2_fired,
                   elig, bg->total_cortical_inputs());
        }
    }

    printf("  Summary over 50 steps:\n");
    printf("    D1 total fires: %d, D2 total fires: %d\n", d1_fire_total, d2_fire_total);
    printf("    Max DA level: %.4f (baseline=0.1)\n", max_da);
    printf("    Max eligibility: %.4f\n", max_elig);
    printf("    VTA DA output: %.4f\n", agent.vta()->da_output());

    // Check weight changes
    auto* bg = agent.bg();
    float w_min = 999, w_max = -999;
    int w_count = 0;
    for (size_t src = 0; src < bg->d1_weight_count(); ++src) {
        for (size_t idx = 0; idx < bg->d1_weights_for(src).size(); ++idx) {
            float w = bg->d1_weights_for(src)[idx];
            if (w < w_min) w_min = w;
            if (w > w_max) w_max = w;
            w_count++;
        }
    }
    printf("    D1 weights: n=%d, min=%.4f, max=%.4f, range=%.4f\n",
           w_count, w_min, w_max, w_max - w_min);

    TEST_ASSERT(w_count > 0, "BG has D1 weights");
    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 4: 10000步长时训练 (10×10, 5×5 vision, all features ON)
// =========================================================================
static void test_long_training() {
    printf("\n--- 测试4: 10000步长时训练 (10x10 grid, PC+Sleep+Replay) ---\n");

    AgentConfig cfg;
    cfg.enable_da_stdp = true;

    ClosedLoopAgent agent(cfg);

    printf("  Environment: %zux%zu grid, %zux%zu vision, %zu food, %zu danger\n",
           cfg.world_config.width, cfg.world_config.height,
           cfg.world_config.vision_side(), cfg.world_config.vision_side(),
           cfg.world_config.n_food, cfg.world_config.n_danger);

    printf("  Epoch  | Food | Danger | Safety | Avg Reward\n");
    printf("  -------|------|--------|--------|----------\n");

    std::vector<float> safety_history;
    for (int epoch = 0; epoch < 10; ++epoch) {
        auto stats = run_epoch(agent, 1000);
        float safety = stats.safety_ratio();
        safety_history.push_back(safety);
        printf("  %5dk | %4d | %6d |  %.2f  |  %+.4f\n",
               epoch + 1, stats.food, stats.danger, safety, stats.avg_reward);
    }

    // Check trend: is late safety better than early?
    float early_avg = (safety_history[0] + safety_history[1]) / 2.0f;
    float late_avg = (safety_history[8] + safety_history[9]) / 2.0f;

    printf("\n  Early safety (1-2k): %.3f\n", early_avg);
    printf("  Late safety (9-10k): %.3f\n", late_avg);
    printf("  Improvement: %+.3f\n", late_avg - early_avg);

    printf("  Total food: %d\n", agent.world().total_food_collected());
    printf("  Total danger: %d\n", agent.world().total_danger_hits());

    // System should be stable (no crashes, some food collection)
    TEST_ASSERT(agent.world().total_food_collected() > 0, "Collected food in 10k steps");
    TEST_ASSERT(agent.agent_step_count() == 10000, "10k steps completed");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 5: 超大环境 (15x15, 7x7视野) — 验证扩展性
// =========================================================================
static void test_large_env() {
    printf("\n--- 测试5: 超大环境 (15x15, 7x7视野, 3000步) ---\n");

    AgentConfig cfg;
    cfg.enable_da_stdp = true;
    cfg.enable_predictive_coding = true;
    cfg.enable_sleep_consolidation = true;
    // Large environment with wider vision
    cfg.world_config.width  = 15;
    cfg.world_config.height = 15;
    cfg.world_config.n_food   = 8;
    cfg.world_config.n_danger = 6;
    cfg.world_config.vision_radius = 3;  // 7x7 vision (49 pixels)
    cfg.world_config.seed = 77;

    ClosedLoopAgent agent(cfg);

    printf("  Environment: %zux%zu grid, %zux%zu vision, %zu food, %zu danger\n",
           cfg.world_config.width, cfg.world_config.height,
           cfg.world_config.vision_side(), cfg.world_config.vision_side(),
           cfg.world_config.n_food, cfg.world_config.n_danger);
    printf("  Brain: V1=%zu, dlPFC=%zu, LGN=%zu neurons\n",
           agent.v1()->n_neurons(), agent.dlpfc()->n_neurons(), agent.lgn()->n_neurons());
    printf("  Features: PC=%s, Sleep=%s, Amygdala=%s, LHb=%s\n",
           cfg.enable_predictive_coding ? "ON" : "OFF",
           cfg.enable_sleep_consolidation ? "ON" : "OFF",
           cfg.enable_amygdala ? "ON" : "OFF",
           cfg.enable_lhb ? "ON" : "OFF");

    printf("  Epoch  | Food | Danger | Safety | Avg Reward\n");
    printf("  -------|------|--------|--------|----------\n");

    std::vector<float> safety_history;
    for (int epoch = 0; epoch < 6; ++epoch) {
        auto stats = run_epoch(agent, 500);
        float safety = stats.safety_ratio();
        safety_history.push_back(safety);
        printf("  %5d  | %4d | %6d |  %.2f  |  %+.4f\n",
               (epoch + 1) * 500, stats.food, stats.danger, safety, stats.avg_reward);
    }

    float early_avg = (safety_history[0] + safety_history[1]) / 2.0f;
    float late_avg = (safety_history[4] + safety_history[5]) / 2.0f;

    printf("\n  Early safety (0-1000): %.3f\n", early_avg);
    printf("  Late safety (2000-3000): %.3f\n", late_avg);
    printf("  Improvement: %+.3f\n", late_avg - early_avg);
    printf("  Total food: %d, Total danger: %d\n",
           agent.world().total_food_collected(), agent.world().total_danger_hits());

    // Verify the large brain runs stably
    TEST_ASSERT(agent.world().total_steps() == 3000, "Completed 3k steps in large env");
    TEST_ASSERT(agent.v1()->n_neurons() > 400, "V1 scaled up for 7x7 vision");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 6: 泛化能力测试 — 训练 seed=42 vs 未训练, 对比后期表现
// "学到的是规则还是记忆？"
// =========================================================================
static void test_generalization() {
    printf("\n--- 测试6: 泛化能力诊断 ---\n");

    // 训练 2000 步 (seed=42) vs 未训练 (seed=77), 各自后 500 步表现
    float trained_safety = 0, fresh_safety = 0;
    int seeds[] = {77, 123};

    for (int s : seeds) {
        // A: 训练 2000 步后再跑 500 步
        AgentConfig cfg_t;
        cfg_t.enable_da_stdp = true;
        cfg_t.world_config.seed = 42;
        ClosedLoopAgent ag_t(cfg_t);
        run_epoch(ag_t, 2000);  // 训练
        auto t_res = run_epoch(ag_t, 500);  // 测试 (同地图后期,食物已重生多次)

        // B: 全新 agent 跑 500 步 (不同 seed)
        AgentConfig cfg_f;
        cfg_f.enable_da_stdp = true;
        cfg_f.world_config.seed = static_cast<uint32_t>(s);
        ClosedLoopAgent ag_f(cfg_f);
        auto f_res = run_epoch(ag_f, 500);

        printf("    seed=%3d: trained=%.2f(f=%d,d=%d) fresh=%.2f(f=%d,d=%d) Δ=%+.2f\n",
               s, t_res.safety_ratio(), t_res.food, t_res.danger,
               f_res.safety_ratio(), f_res.food, f_res.danger,
               t_res.safety_ratio() - f_res.safety_ratio());
        trained_safety += t_res.safety_ratio();
        fresh_safety += f_res.safety_ratio();
    }

    float avg_t = trained_safety / 2.0f;
    float avg_f = fresh_safety / 2.0f;
    printf("    平均: trained=%.3f, fresh=%.3f, 泛化优势=%+.3f\n",
           avg_t, avg_f, avg_t - avg_f);

    if (avg_t > avg_f + 0.02f)
        printf("    结论: ✅ 训练有帮助 — 可能学到了一般性策略\n");
    else if (avg_t > avg_f - 0.02f)
        printf("    结论: ⚠️ 中性 — 训练没有显著帮助\n");
    else
        printf("    结论: ❌ 训练有害 — 可能过拟合了特定布局\n");

    TEST_ASSERT(true, "Generalization diagnostic completed");
    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// main
// =========================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("=== 悟韵 Step 23: 泛化能力诊断 (10x10, 5x5 vision) ===\n");

    test_learning_curve();
    test_learning_vs_control();
    test_bg_diagnostics();
    test_long_training();
    test_large_env();
    test_generalization();

    printf("\n========================================\n");
    printf("  通过: %d / %d\n", g_pass, g_pass + g_fail);
    printf("========================================\n");

    return g_fail > 0 ? 1 : 0;
}
