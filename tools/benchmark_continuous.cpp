/**
 * benchmark_continuous — v55 连续移动 vs 离散移动 A/B 对比
 *
 * 用法: benchmark_continuous [steps] [seeds]
 * 默认: 2000 步, 5 个种子
 *
 * 对比:
 *   A: 离散模式 (act(Action) 直接调用 GridWorld)
 *   B: 连续模式 (ClosedLoopAgent 默认, act_continuous)
 *
 * 输出: food, danger, improvement, late_safety 对比
 */

#include "engine/closed_loop_agent.h"
#include "engine/grid_world_env.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using namespace wuyun;

struct BenchResult {
    float early_safety = 0;
    float late_safety = 0;
    float improvement = 0;
    int food = 0;
    int danger = 0;
    float elapsed_sec = 0;
};

BenchResult run_one(bool /*continuous — now always true*/, uint32_t seed, size_t steps) {
    AgentConfig cfg;
    cfg.continuous_step_size = 0.8f;
    GridWorldConfig wcfg;
    wcfg.width = 10; wcfg.height = 10;
    wcfg.n_food = 5; wcfg.n_danger = 3;
    wcfg.seed = seed;

    auto t0 = std::chrono::steady_clock::now();
    ClosedLoopAgent agent(std::make_unique<GridWorldEnv>(wcfg), cfg);

    size_t early_steps = steps / 5;
    size_t late_steps = steps - early_steps;

    int e_food = 0, e_danger = 0;
    for (size_t i = 0; i < early_steps; ++i) {
        auto r = agent.agent_step();
        if (r.positive_event) e_food++;
        if (r.negative_event) e_danger++;
    }

    int l_food = 0, l_danger = 0;
    for (size_t i = 0; i < late_steps; ++i) {
        auto r = agent.agent_step();
        if (r.positive_event) l_food++;
        if (r.negative_event) l_danger++;
    }

    auto t1 = std::chrono::steady_clock::now();

    BenchResult res;
    res.early_safety = static_cast<float>(e_food) /
                       std::max(1.0f, static_cast<float>(e_food + e_danger));
    res.late_safety = static_cast<float>(l_food) /
                      std::max(1.0f, static_cast<float>(l_food + l_danger));
    res.improvement = res.late_safety - res.early_safety;
    res.food = agent.env().positive_count();
    res.danger = agent.env().negative_count();
    res.elapsed_sec = std::chrono::duration<float>(t1 - t0).count();
    return res;
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    setvbuf(stdout, NULL, _IONBF, 0);

    int steps = (argc > 1) ? std::atoi(argv[1]) : 500;
    int n_seeds = (argc > 2) ? std::atoi(argv[2]) : 5;

    std::vector<uint32_t> seeds = {42, 77, 123, 256, 789};
    if (n_seeds < static_cast<int>(seeds.size()))
        seeds.resize(static_cast<size_t>(n_seeds));

    printf("=== v55 连续移动 A/B 对比 ===\n");
    printf("  Steps: %d, Seeds: %d\n\n", steps, static_cast<int>(seeds.size()));

    printf("%-6s %-5s %6s %6s %8s %8s %8s %6s\n",
           "Mode", "Seed", "Food", "Dngr", "Early%", "Late%", "Improv", "Time");
    printf("------ ----- ------ ------ -------- -------- -------- ------\n");

    float sum_discrete[4] = {0, 0, 0, 0};  // early, late, improve, time
    float sum_continuous[4] = {0, 0, 0, 0};
    int sum_d_food = 0, sum_d_danger = 0;
    int sum_c_food = 0, sum_c_danger = 0;

    for (auto seed : seeds) {
        // Discrete
        auto d = run_one(false, seed, static_cast<size_t>(steps));
        printf("%-6s %5u %6d %6d %7.1f%% %7.1f%% %+7.3f %5.1fs\n",
               "离散", seed, d.food, d.danger,
               d.early_safety * 100, d.late_safety * 100, d.improvement, d.elapsed_sec);
        sum_discrete[0] += d.early_safety;
        sum_discrete[1] += d.late_safety;
        sum_discrete[2] += d.improvement;
        sum_discrete[3] += d.elapsed_sec;
        sum_d_food += d.food;
        sum_d_danger += d.danger;

        // Continuous
        auto c = run_one(true, seed, static_cast<size_t>(steps));
        printf("%-6s %5u %6d %6d %7.1f%% %7.1f%% %+7.3f %5.1fs\n",
               "连续", seed, c.food, c.danger,
               c.early_safety * 100, c.late_safety * 100, c.improvement, c.elapsed_sec);
        sum_continuous[0] += c.early_safety;
        sum_continuous[1] += c.late_safety;
        sum_continuous[2] += c.improvement;
        sum_continuous[3] += c.elapsed_sec;
        sum_c_food += c.food;
        sum_c_danger += c.danger;

        printf("\n");
    }

    float n = static_cast<float>(seeds.size());
    printf("====== 平均 ======\n");
    printf("%-6s       %6.1f %6.1f %7.1f%% %7.1f%% %+7.3f %5.1fs\n",
           "离散",
           static_cast<float>(sum_d_food) / n,
           static_cast<float>(sum_d_danger) / n,
           sum_discrete[0] / n * 100,
           sum_discrete[1] / n * 100,
           sum_discrete[2] / n,
           sum_discrete[3] / n);
    printf("%-6s       %6.1f %6.1f %7.1f%% %7.1f%% %+7.3f %5.1fs\n",
           "连续",
           static_cast<float>(sum_c_food) / n,
           static_cast<float>(sum_c_danger) / n,
           sum_continuous[0] / n * 100,
           sum_continuous[1] / n * 100,
           sum_continuous[2] / n,
           sum_continuous[3] / n);

    printf("\n结论: ");
    float d_score = sum_discrete[2] / n;
    float c_score = sum_continuous[2] / n;
    if (c_score > d_score + 0.01f) {
        printf("连续模式更优 (+%.3f improvement)\n", c_score - d_score);
    } else if (d_score > c_score + 0.01f) {
        printf("离散模式更优 (+%.3f improvement), 需诊断连续模式\n", d_score - c_score);
    } else {
        printf("两模式相当 (差异 < 0.01)\n");
    }

    return 0;
}
