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

    printf("=== WuYun DevGenome Evolution (间接编码) ===\n");
    printf("  Population: %d, Generations: %d\n", n_pop, n_gen);
    printf("  Genome: 124 发育规则基因 (增殖/导向/分化/修剪)\n");
    printf("  Fitness: improvement*3 + late_safety*1 (Baldwin effect)\n\n");

    wuyun::EvolutionConfig config;
    config.n_generations = static_cast<size_t>(n_gen);
    config.population_size = static_cast<size_t>(n_pop);
    config.eval_steps = 5000;  // v49: 1000→5000 (Step 16 教训: 短评估优化短期表现)
    config.eval_seeds = {42, 77, 123, 256, 789};
    config.ga_seed = 2026;

    // 默认环境: 10x10 开放场地
    config.world_config.width = 10;
    config.world_config.height = 10;
    config.world_config.n_food = 5;
    config.world_config.n_danger = 3;
    config.world_config.vision_radius = 2;

    wuyun::DevEvolutionEngine engine(config);
    auto best = engine.run();

    printf("\n=== Best DevGenome ===\n");
    auto genes = best.all_genes();
    for (auto* g : genes) {
        printf("  %-20s = %10.5f  [%.4f, %.4f]\n",
               g->name.c_str(), g->value, g->min_val, g->max_val);
    }

    // 详细评估
    printf("\n=== Best DevGenome Detailed Evaluation ===\n");
    auto result = engine.evaluate(best);
    printf("  Fitness:     %.4f\n", result.fitness);
    printf("  Early safety: %.3f\n", result.early_safety);
    printf("  Late safety:  %.3f\n", result.late_safety);
    printf("  Improvement:  %+.3f\n", result.improvement);
    printf("  Total food:   %d\n", result.total_food);
    printf("  Total danger: %d\n", result.total_danger);

    return 0;
}
