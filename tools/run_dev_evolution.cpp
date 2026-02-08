/**
 * run_dev_evolution — 间接编码发育基因组进化
 *
 * 用法: run_dev_evolution [generations] [population]
 * 默认: 30 代, 40 体
 *
 * 与 run_evolution (直接编码) 对比:
 *   run_evolution:     23 基因 → AgentConfig → build_brain()
 *   run_dev_evolution: 124 基因 → Developer::develop() → 大脑涌现
 */

#include "genome/dev_evolution.h"
#include "genome/dev_genome.h"
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    setvbuf(stdout, NULL, _IONBF, 0);

    int n_gen = (argc > 1) ? std::atoi(argv[1]) : 30;
    int n_pop = (argc > 2) ? std::atoi(argv[2]) : 40;

    printf("=== WuYun DevGenome Evolution (v53: 多任务天才评估) ===\n");
    printf("  Population: %d, Generations: %d\n", n_pop, n_gen);
    printf("  Tasks: Open Field + Sparse Reward + Reversal Learning\n");
    printf("  Fitness: open*1 + sparse*1 + reversal*1.5 (通用学习能力)\n\n");

    wuyun::EvolutionConfig config;
    config.n_generations = static_cast<size_t>(n_gen);
    config.population_size = static_cast<size_t>(n_pop);
    config.eval_steps = 400;   // v53: 每个任务 400 步
    config.ga_seed = 2026;

    wuyun::DevEvolutionEngine engine(config);
    auto best = engine.run();

    printf("\n=== Best DevGenome ===\n");
    auto genes = best.all_genes();
    for (auto* g : genes) {
        printf("  %-20s = %10.5f  [%.4f, %.4f]\n",
               g->name.c_str(), g->value, g->min_val, g->max_val);
    }

    // 详细多任务评估
    printf("\n=== Best DevGenome Multi-Task Evaluation ===\n");
    auto result = engine.evaluate(best);
    printf("  Fitness (total): %.4f\n", result.fitness);
    printf("  Open Field:      %.3f\n", result.open_field);
    printf("  Sparse Reward:   %.3f\n", result.sparse_reward);
    printf("  Reversal Learn:  %.3f\n", result.reversal);

    return 0;
}
