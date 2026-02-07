#pragma once
/**
 * EvolutionEngine — 遗传算法引擎 v1
 *
 * 纯遗传算法: 锦标赛选择 + 均匀交叉 + 高斯变异
 * 适应度: late_safety + learning_improvement × 2.0 (Baldwin效应)
 * 多种子评估: 每个基因组在多个随机种子上平均 (防止过拟合)
 *
 * 生物学对应: 自然选择 + 有性繁殖 + 基因突变
 * 超越生物: 精英保留 (Hall of Fame), 多种子泛化
 */

#include "genome/genome.h"
#include "engine/closed_loop_agent.h"
#include <vector>
#include <functional>
#include <string>

namespace wuyun {

// =============================================================================
// Evolution configuration
// =============================================================================

struct EvolutionConfig {
    size_t population_size  = 60;    // 个体数/代
    size_t n_generations    = 30;    // 进化代数
    size_t tournament_size  = 5;     // 锦标赛选择大小
    float  mutation_rate    = 0.15f; // 基因突变概率
    float  mutation_sigma   = 0.1f;  // 突变幅度 (σ × range)
    float  elite_fraction   = 0.1f;  // 精英直接保留比例
    size_t eval_steps       = 5000;  // 每个个体的评估步数
    std::vector<uint32_t> eval_seeds = {42, 77, 123}; // 多种子评估
    uint32_t ga_seed        = 2024;  // GA随机种子

    // GridWorld environment config (shared by all individuals)
    GridWorldConfig world_config;
};

// =============================================================================
// Fitness result
// =============================================================================

struct FitnessResult {
    float fitness       = 0.0f;
    float early_safety  = 0.0f;
    float late_safety   = 0.0f;
    float improvement   = 0.0f;
    int   total_food    = 0;
    int   total_danger  = 0;
};

// =============================================================================
// EvolutionEngine
// =============================================================================

class EvolutionEngine {
public:
    explicit EvolutionEngine(const EvolutionConfig& config = {});

    /** Run the full evolutionary loop. Returns the best genome found. */
    Genome run();

    /** Evaluate a single genome (averaged over eval_seeds) */
    FitnessResult evaluate(const Genome& genome) const;

    /** Get the Hall of Fame (top genomes across all generations) */
    const std::vector<Genome>& hall_of_fame() const { return hall_of_fame_; }

    /** Set progress callback: (generation, best_fitness, best_genome_summary) */
    using ProgressCallback = std::function<void(int, float, const std::string&)>;
    void set_progress_callback(ProgressCallback cb) { progress_cb_ = std::move(cb); }

private:
    EvolutionConfig config_;
    std::mt19937 rng_;
    std::vector<Genome> population_;
    std::vector<Genome> hall_of_fame_;
    ProgressCallback progress_cb_;

    // GA operators
    void initialize_population();
    Genome tournament_select(const std::vector<Genome>& pop);
    std::vector<Genome> next_generation(std::vector<Genome>& current);

    // Fitness evaluation for a single seed
    FitnessResult evaluate_single(const Genome& genome, uint32_t seed) const;
};

} // namespace wuyun
