/**
 * test_learning_curve.cpp — 闭环学习曲线验证 (多线程并行)
 *
 * v21 环境升级: 10×10 grid, 5×5 视野, 5 food, 3 danger
 * v26: 6 个测试多线程并行, ~4-6× 加速
 *
 * 测试方案:
 * 1. 5000步学习曲线
 * 2. 学习 vs 无学习对照
 * 3. BG DA-STDP 诊断
 * 4. 10000步长时训练
 * 5. 超大环境 (15×15, 7×7)
 * 6. 泛化能力诊断
 */

#include "engine/closed_loop_agent.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace wuyun;

// Thread-safe test result
struct TestResult {
    bool passed = true;
    std::string output;
};

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
// Test 1: 学习曲线 (5000步)
// =========================================================================
static TestResult test_learning_curve() {
    std::ostringstream out;
    out << "\n--- 测试1: 学习曲线 (5000步, 10x10 grid, 5x5 vision) ---\n";

    AgentConfig cfg;
    cfg.enable_da_stdp = true;
    ClosedLoopAgent agent(cfg);

    out << "  Environment: " << cfg.world_config.width << "x" << cfg.world_config.height
        << " grid, " << cfg.world_config.vision_side() << "x" << cfg.world_config.vision_side()
        << " vision, " << cfg.world_config.n_food << " food, " << cfg.world_config.n_danger << " danger\n";
    out << "  Brain: V1=" << agent.v1()->n_neurons()
        << ", V2=" << (agent.v2() ? agent.v2()->n_neurons() : 0)
        << ", V4=" << (agent.v4() ? agent.v4()->n_neurons() : 0)
        << ", IT=" << (agent.it_ctx() ? agent.it_ctx()->n_neurons() : 0)
        << ", dlPFC=" << agent.dlpfc()->n_neurons()
        << ", LGN=" << agent.lgn()->n_neurons() << " neurons\n";
    out << "  Features: PC=" << (cfg.enable_predictive_coding ? "ON" : "OFF")
        << ", Sleep=" << (cfg.enable_sleep_consolidation ? "ON" : "OFF")
        << ", Replay=" << (cfg.enable_replay ? "ON" : "OFF") << "\n";

    char buf[256];
    snprintf(buf, sizeof(buf), "  Epoch | Food | Danger | F:D ratio | Avg Reward | Safety\n");
    out << buf;
    snprintf(buf, sizeof(buf), "  ------|------|--------|-----------|------------|-------\n");
    out << buf;

    std::vector<EpochStats> epochs;
    for (int epoch = 0; epoch < 10; ++epoch) {
        auto stats = run_epoch(agent, 500);
        epochs.push_back(stats);
        snprintf(buf, sizeof(buf), "  %5d | %4d | %6d |   %5.2f   |   %+.4f   | %.2f\n",
               (epoch + 1) * 500, stats.food, stats.danger,
               stats.danger > 0 ? (float)stats.food / stats.danger : 99.0f,
               stats.avg_reward, stats.safety_ratio());
        out << buf;
    }

    float early_food = (float)(epochs[0].food + epochs[1].food);
    float late_food  = (float)(epochs[8].food + epochs[9].food);
    float early_danger = (float)(epochs[0].danger + epochs[1].danger);
    float late_danger  = (float)(epochs[8].danger + epochs[9].danger);
    float early_safety = (early_food + early_danger > 0)
        ? early_food / (early_food + early_danger) : 0.5f;
    float late_safety = (late_food + late_danger > 0)
        ? late_food / (late_food + late_danger) : 0.5f;

    snprintf(buf, sizeof(buf), "\n  Summary:\n  Early (0-1000): food=%d, danger=%d, safety=%.2f\n",
           (int)early_food, (int)early_danger, early_safety);
    out << buf;
    snprintf(buf, sizeof(buf), "  Late (4000-5000): food=%d, danger=%d, safety=%.2f\n",
           (int)late_food, (int)late_danger, late_safety);
    out << buf;
    snprintf(buf, sizeof(buf), "  Total food: %d, Total steps: %d\n",
           agent.world().total_food_collected(), agent.world().total_steps());
    out << buf;

    TestResult r;
    r.passed = (agent.world().total_food_collected() > 0);
    out << (r.passed ? "  [PASS]\n" : "  [FAIL] No food collected\n");
    r.output = out.str();
    return r;
}

// =========================================================================
// Test 2: 学习 vs 无学习对照
// =========================================================================
static TestResult test_learning_vs_control() {
    std::ostringstream out;
    out << "\n--- 测试2: 学习 vs 无学习对照 (3000步, 10x10 grid) ---\n";

    // Run learner and control in parallel (2 threads)
    EpochStats learn_stats, ctrl_stats;
    std::thread t_learn([&]{
        AgentConfig cfg;
        cfg.enable_da_stdp = true;
        cfg.world_config.seed = 42;
        ClosedLoopAgent learner(cfg);
        for (int i = 0; i < 1000; ++i) learner.agent_step();
        learn_stats = run_epoch(learner, 2000);
    });
    std::thread t_ctrl([&]{
        AgentConfig cfg;
        cfg.enable_da_stdp = false;
        cfg.world_config.seed = 42;
        ClosedLoopAgent control(cfg);
        for (int i = 0; i < 1000; ++i) control.agent_step();
        ctrl_stats = run_epoch(control, 2000);
    });
    t_learn.join();
    t_ctrl.join();

    char buf[256];
    snprintf(buf, sizeof(buf), "  Learner (DA-STDP ON):  food=%d, danger=%d, safety=%.2f, avg_r=%+.4f\n",
           learn_stats.food, learn_stats.danger, learn_stats.safety_ratio(), learn_stats.avg_reward);
    out << buf;
    snprintf(buf, sizeof(buf), "  Control (DA-STDP OFF): food=%d, danger=%d, safety=%.2f, avg_r=%+.4f\n",
           ctrl_stats.food, ctrl_stats.danger, ctrl_stats.safety_ratio(), ctrl_stats.avg_reward);
    out << buf;

    float learn_score = learn_stats.avg_reward;
    float ctrl_score = ctrl_stats.avg_reward;
    snprintf(buf, sizeof(buf), "  Learner advantage: %+.4f\n", learn_score - ctrl_score);
    out << buf;

    TestResult r;
    r.passed = (learn_score >= ctrl_score - 0.05f);
    out << (r.passed ? "  [PASS]\n" : "  [FAIL] Learner significantly worse than control\n");
    r.output = out.str();
    return r;
}

// =========================================================================
// Test 3: BG DA-STDP 诊断
// =========================================================================
static TestResult test_bg_diagnostics() {
    std::ostringstream out;
    out << "\n--- 测试3: BG DA-STDP诊断 ---\n";

    AgentConfig cfg;
    cfg.enable_da_stdp = true;
    ClosedLoopAgent agent(cfg);

    out << "  Step-by-step diagnostics:\n";
    int d1_fire_total = 0, d2_fire_total = 0;
    float max_da = 0.0f, max_elig = 0.0f;
    char buf[256];

    for (int step = 0; step < 50; ++step) {
        auto result = agent.agent_step();
        auto* bg = agent.bg();

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
            snprintf(buf, sizeof(buf),
                "    step=%d act=%d r=%.2f | DA=%.3f | D1=%d D2=%d | elig=%.1f | ctx=%zu\n",
                step, (int)agent.last_action(), result.reward,
                da, d1_fired, d2_fired, elig, bg->total_cortical_inputs());
            out << buf;
        }
    }

    snprintf(buf, sizeof(buf), "  Summary: D1=%d, D2=%d fires. Max DA=%.4f, Max elig=%.1f\n",
           d1_fire_total, d2_fire_total, max_da, max_elig);
    out << buf;

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
    snprintf(buf, sizeof(buf), "  D1 weights: n=%d, min=%.4f, max=%.4f, range=%.4f\n",
           w_count, w_min, w_max, w_max - w_min);
    out << buf;

    TestResult r;
    r.passed = (w_count > 0);
    out << (r.passed ? "  [PASS]\n" : "  [FAIL] No D1 weights\n");
    r.output = out.str();
    return r;
}

// =========================================================================
// Test 4: 10000步长时训练
// =========================================================================
static TestResult test_long_training() {
    std::ostringstream out;
    out << "\n--- 测试4: 10000步长时训练 (10x10 grid, PC+Sleep+Replay) ---\n";

    AgentConfig cfg;
    cfg.enable_da_stdp = true;
    ClosedLoopAgent agent(cfg);

    out << "  Environment: " << cfg.world_config.width << "x" << cfg.world_config.height
        << " grid, " << cfg.world_config.vision_side() << "x" << cfg.world_config.vision_side()
        << " vision, " << cfg.world_config.n_food << " food, " << cfg.world_config.n_danger << " danger\n";

    char buf[256];
    snprintf(buf, sizeof(buf), "  Epoch  | Food | Danger | Safety | Avg Reward\n");
    out << buf;
    snprintf(buf, sizeof(buf), "  -------|------|--------|--------|----------\n");
    out << buf;

    std::vector<float> safety_history;
    for (int epoch = 0; epoch < 10; ++epoch) {
        auto stats = run_epoch(agent, 1000);
        float safety = stats.safety_ratio();
        safety_history.push_back(safety);
        snprintf(buf, sizeof(buf), "  %5dk | %4d | %6d |  %.2f  |  %+.4f\n",
               epoch + 1, stats.food, stats.danger, safety, stats.avg_reward);
        out << buf;
    }

    float early_avg = (safety_history[0] + safety_history[1]) / 2.0f;
    float late_avg = (safety_history[8] + safety_history[9]) / 2.0f;

    snprintf(buf, sizeof(buf), "\n  Early (1-2k): %.3f, Late (9-10k): %.3f, Improvement: %+.3f\n",
           early_avg, late_avg, late_avg - early_avg);
    out << buf;
    snprintf(buf, sizeof(buf), "  Total food: %d, Total danger: %d\n",
           agent.world().total_food_collected(), agent.world().total_danger_hits());
    out << buf;

    TestResult r;
    r.passed = (agent.world().total_food_collected() > 0 && agent.agent_step_count() == 10000);
    out << (r.passed ? "  [PASS]\n" : "  [FAIL]\n");
    r.output = out.str();
    return r;
}

// =========================================================================
// Test 5: 超大环境 (15x15, 7x7视野)
// =========================================================================
static TestResult test_large_env() {
    std::ostringstream out;
    out << "\n--- 测试5: 超大环境 (15x15, 7x7视野, 3000步) ---\n";

    AgentConfig cfg;
    cfg.enable_da_stdp = true;
    cfg.enable_predictive_coding = true;
    cfg.enable_sleep_consolidation = true;
    cfg.world_config.width = 15;
    cfg.world_config.height = 15;
    cfg.world_config.n_food = 8;
    cfg.world_config.n_danger = 6;
    cfg.world_config.vision_radius = 3;
    cfg.world_config.seed = 77;
    ClosedLoopAgent agent(cfg);

    out << "  Brain: V1=" << agent.v1()->n_neurons()
        << ", dlPFC=" << agent.dlpfc()->n_neurons()
        << ", LGN=" << agent.lgn()->n_neurons() << " neurons\n";

    char buf[256];
    std::vector<float> safety_history;
    for (int epoch = 0; epoch < 6; ++epoch) {
        auto stats = run_epoch(agent, 500);
        float safety = stats.safety_ratio();
        safety_history.push_back(safety);
        snprintf(buf, sizeof(buf), "  %5d | food=%2d danger=%2d safety=%.2f\n",
               (epoch + 1) * 500, stats.food, stats.danger, safety);
        out << buf;
    }

    float early_avg = (safety_history[0] + safety_history[1]) / 2.0f;
    float late_avg = (safety_history[4] + safety_history[5]) / 2.0f;
    snprintf(buf, sizeof(buf), "  Early: %.3f, Late: %.3f, Improvement: %+.3f\n",
           early_avg, late_avg, late_avg - early_avg);
    out << buf;

    TestResult r;
    r.passed = (agent.world().total_steps() == 3000 && agent.v1()->n_neurons() > 400);
    out << (r.passed ? "  [PASS]\n" : "  [FAIL]\n");
    r.output = out.str();
    return r;
}

// =========================================================================
// Test 6: 泛化能力诊断
// =========================================================================
static TestResult test_generalization() {
    std::ostringstream out;
    out << "\n--- 测试6: 泛化能力诊断 ---\n";

    // 4 agents in parallel: 2 trained (seed=42→test) + 2 fresh (seed=77,123)
    int seeds[] = {77, 123};
    EpochStats t_results[2], f_results[2];
    std::thread threads[4];

    for (int i = 0; i < 2; ++i) {
        threads[i*2] = std::thread([&, i]{
            AgentConfig cfg;
            cfg.enable_da_stdp = true;
            cfg.world_config.seed = 42;
            ClosedLoopAgent ag(cfg);
            run_epoch(ag, 2000);  // train
            t_results[i] = run_epoch(ag, 500);  // test
        });
        threads[i*2+1] = std::thread([&, i]{
            AgentConfig cfg;
            cfg.enable_da_stdp = true;
            cfg.world_config.seed = static_cast<uint32_t>(seeds[i]);
            ClosedLoopAgent ag(cfg);
            f_results[i] = run_epoch(ag, 500);
        });
    }
    for (auto& t : threads) t.join();

    float trained_safety = 0, fresh_safety = 0;
    char buf[256];
    for (int i = 0; i < 2; ++i) {
        snprintf(buf, sizeof(buf),
            "    seed=%3d: trained=%.2f(f=%d,d=%d) fresh=%.2f(f=%d,d=%d) D=%+.2f\n",
            seeds[i], t_results[i].safety_ratio(), t_results[i].food, t_results[i].danger,
            f_results[i].safety_ratio(), f_results[i].food, f_results[i].danger,
            t_results[i].safety_ratio() - f_results[i].safety_ratio());
        out << buf;
        trained_safety += t_results[i].safety_ratio();
        fresh_safety += f_results[i].safety_ratio();
    }

    float avg_t = trained_safety / 2.0f;
    float avg_f = fresh_safety / 2.0f;
    snprintf(buf, sizeof(buf), "    平均: trained=%.3f, fresh=%.3f, 泛化优势=%+.3f\n",
           avg_t, avg_f, avg_t - avg_f);
    out << buf;

    if (avg_t > avg_f + 0.02f)
        out << "    结论: 训练有帮助\n";
    else if (avg_t > avg_f - 0.02f)
        out << "    结论: 中性\n";
    else
        out << "    结论: 训练有害\n";

    TestResult r;
    r.passed = true;
    out << "  [PASS]\n";
    r.output = out.str();
    return r;
}

// =========================================================================
// main — 6 tests in parallel
// =========================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    unsigned hw = std::thread::hardware_concurrency();
    printf("=== 悟韵 闭环学习验证 (6 tests, multi-threaded, %u hw threads) ===\n", hw);

    auto t0 = std::chrono::steady_clock::now();

    // Launch all 6 tests in parallel
    TestResult results[6];
    std::thread threads[6];

    // Track completion for progress reporting
    std::atomic<int> done{0};
    const char* names[] = {"学习曲线5k", "学习vs对照", "BG诊断", "长训10k", "大环境15x15", "泛化"};

    auto run_test = [&](int idx, std::function<TestResult()> fn) {
        printf("  [开始] 测试%d: %s\n", idx+1, names[idx]);
        fflush(stdout);
        results[idx] = fn();
        int d = ++done;
        printf("  [完成] 测试%d: %s (%d/6)\n", idx+1, names[idx], d);
        fflush(stdout);
    };

    threads[0] = std::thread([&]{ run_test(0, test_learning_curve); });
    threads[1] = std::thread([&]{ run_test(1, test_learning_vs_control); });
    threads[2] = std::thread([&]{ run_test(2, test_bg_diagnostics); });
    threads[3] = std::thread([&]{ run_test(3, test_long_training); });
    threads[4] = std::thread([&]{ run_test(4, test_large_env); });
    threads[5] = std::thread([&]{ run_test(5, test_generalization); });

    // Wait for all
    for (auto& t : threads) t.join();

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // Print results sequentially (clean output)
    int pass = 0, fail = 0;
    for (int i = 0; i < 6; ++i) {
        printf("%s", results[i].output.c_str());
        if (results[i].passed) pass++; else fail++;
    }

    printf("\n========================================\n");
    printf("  通过: %d / %d  (%.1f秒, peak ~12 threads on %u hw)\n", pass, pass + fail, elapsed, hw);
    printf("========================================\n");

    return fail > 0 ? 1 : 0;
}
