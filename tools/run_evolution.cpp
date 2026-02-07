/**
 * run_evolution — 基因层 v1 进化运行器
 *
 * 用遗传算法搜索 ClosedLoopAgent 的最优参数组合。
 * 输出: 每代最佳基因组 + 最终 Hall of Fame JSON
 *
 * Usage: run_evolution [generations] [population]
 *   defaults: 30 generations, 60 population
 */

#include "genome/evolution.h"
#include <cstdio>
#include <fstream>
#include <string>

int main(int argc, char* argv[]) {
    using namespace wuyun;

    // Parse arguments
    size_t n_gen = 20;
    size_t n_pop = 30;
    if (argc >= 2) n_gen = static_cast<size_t>(std::atoi(argv[1]));
    if (argc >= 3) n_pop = static_cast<size_t>(std::atoi(argv[2]));

    printf("=== WuYun Genome Layer v1: Evolution (Step 22) ===\n");
    printf("  Population: %zu, Generations: %zu\n", n_pop, n_gen);
    printf("  Genes: 23 closed-loop parameters\n");
    printf("  Eval: 5000 steps x 3 seeds (10x10 grid, 5x5 vision)\n");
    printf("  Fitness: late_safety + improvement*2 - danger*0.002 + food*0.001\n\n");

    EvolutionConfig ecfg;
    ecfg.population_size = n_pop;
    ecfg.n_generations = n_gen;
    ecfg.tournament_size = 5;
    ecfg.mutation_rate = 0.15f;
    ecfg.mutation_sigma = 0.10f;
    ecfg.elite_fraction = 0.10f;
    ecfg.eval_steps = 5000;                    // v22: 3000→5000, capture full learning curve
    ecfg.eval_seeds = {42, 77, 123};           // v22: 2→3 seeds for generalization
    ecfg.ga_seed = 2024;

    // v22: use default 10x10 grid, 5x5 vision, 5 food, 3 danger
    // (GridWorldConfig defaults are already correct since Step 21)

    EvolutionEngine engine(ecfg);

    // Run evolution
    Genome best = engine.run();

    // Print best genome details
    printf("\n=== Best Genome ===\n");
    auto genes = best.all_genes();
    for (const auto* g : genes) {
        printf("  %-25s = %10.5f  [%.4f, %.4f]\n",
               g->name.c_str(), g->value, g->min_val, g->max_val);
    }

    // Evaluate best genome in detail
    printf("\n=== Best Genome Detailed Evaluation ===\n");
    FitnessResult res = engine.evaluate(best);
    printf("  Fitness:     %.4f\n", res.fitness);
    printf("  Early safety: %.3f\n", res.early_safety);
    printf("  Late safety:  %.3f\n", res.late_safety);
    printf("  Improvement:  %+.3f\n", res.improvement);
    printf("  Total food:   %d\n", res.total_food);
    printf("  Total danger: %d\n", res.total_danger);

    // Compare with manual baseline
    printf("\n=== Manual Baseline Comparison ===\n");
    Genome manual;  // Default genome = current hand-tuned parameters
    FitnessResult manual_res = engine.evaluate(manual);
    printf("  Manual:   fitness=%.4f, late_safety=%.3f, improvement=%+.3f\n",
           manual_res.fitness, manual_res.late_safety, manual_res.improvement);
    printf("  Evolved:  fitness=%.4f, late_safety=%.3f, improvement=%+.3f\n",
           res.fitness, res.late_safety, res.improvement);
    printf("  Delta:    fitness=%+.4f, late_safety=%+.3f, improvement=%+.3f\n",
           res.fitness - manual_res.fitness,
           res.late_safety - manual_res.late_safety,
           res.improvement - manual_res.improvement);

    // Save best genome to JSON
    std::string json = best.to_json();
    std::ofstream ofs("best_genome.json");
    if (ofs.is_open()) {
        ofs << json;
        ofs.close();
        printf("\n  Saved to best_genome.json\n");
    }

    // Save Hall of Fame
    const auto& hof = engine.hall_of_fame();
    std::ofstream hof_ofs("hall_of_fame.json");
    if (hof_ofs.is_open()) {
        hof_ofs << "[\n";
        for (size_t i = 0; i < hof.size(); ++i) {
            hof_ofs << hof[i].to_json();
            if (i + 1 < hof.size()) hof_ofs << ",";
            hof_ofs << "\n";
        }
        hof_ofs << "]\n";
        hof_ofs.close();
        printf("  Saved Hall of Fame (%zu entries) to hall_of_fame.json\n", hof.size());
    }

    return 0;
}
