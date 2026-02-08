#pragma once
/**
 * DevEvolutionEngine — 间接编码进化引擎
 *
 * v53: 多任务"天才基因"评估
 *   不再只考一道题 (10×10 觅食), 而是用多种任务选通用学习能力:
 *     Task 1: 开放觅食 (基础趋近/回避)
 *     Task 2: 稀疏奖赏 (耐心 + 探索效率)
 *     Task 3: 反转学习 (灵活性, 旧策略失效时快速适应)
 *
 *   专才在某一项满分但其他项崩溃。天才在所有项都及格。
 *   → 进化选出的是通用学习器, 不是应试专家。
 */

#include "genome/dev_genome.h"
#include "genome/evolution.h"  // FitnessResult, EvolutionConfig
#include "engine/grid_world.h"
#include <vector>
#include <functional>
#include <random>

namespace wuyun {

/** v53: 多任务适应度 — 各任务分数 + 加权总分 */
struct MultitaskFitness {
    float fitness       = 0.0f;   // 加权总分
    float open_field    = 0.0f;   // Task 1: 开放觅食分
    float sparse_reward = 0.0f;   // Task 2: 稀疏奖赏分
    float reversal      = 0.0f;   // Task 3: 反转学习分
    int   total_food    = 0;
    int   total_danger  = 0;
};

class DevEvolutionEngine {
public:
    explicit DevEvolutionEngine(const EvolutionConfig& config = {});

    /** 运行完整进化循环, 返回最佳发育基因组 */
    DevGenome run();

    /** v53: 多任务评估 (开放觅食 + 稀疏奖赏 + 反转学习) */
    MultitaskFitness evaluate(const DevGenome& genome) const;

    /** Hall of Fame */
    const std::vector<DevGenome>& hall_of_fame() const { return hall_of_fame_; }

private:
    EvolutionConfig config_;
    std::mt19937 rng_;
    std::vector<DevGenome> population_;
    std::vector<DevGenome> hall_of_fame_;
    DevGenome best_ever_;
    size_t n_elite_ = 0;
    int stagnation_count_ = 0;

    void initialize_population();
    DevGenome tournament_select(const std::vector<DevGenome>& pop);
    std::vector<DevGenome> next_generation(std::vector<DevGenome>& current);

    // v53: 各任务评估器 — 每个构建独立 agent, 返回单任务分数
    float eval_open_field(const AgentConfig& base_cfg, uint32_t seed, size_t steps) const;
    float eval_sparse(const AgentConfig& base_cfg, uint32_t seed, size_t steps) const;
    float eval_reversal(const AgentConfig& base_cfg, uint32_t seed_a, uint32_t seed_b, size_t steps) const;

    // 通用: 跑 agent N 步, 返回 early×1 + improvement×2 + late×2
    static float run_and_score(ClosedLoopAgent& agent, size_t steps,
                               int& out_food, int& out_danger);
};

} // namespace wuyun
