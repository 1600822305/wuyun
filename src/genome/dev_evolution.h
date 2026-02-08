#pragma once
/**
 * DevEvolutionEngine — 间接编码进化引擎
 *
 * 用 DevGenome + Developer 替代 Genome + build_brain:
 *   DevGenome → Developer::develop() → SimulationEngine → 在 GridWorld 中评估
 *
 * 复用 EvolutionEngine 的 GA 机制 (锦标赛/交叉/变异/精英/多种子),
 * 但评估函数用发育大脑而非手工大脑。
 */

#include "genome/dev_genome.h"
#include "genome/evolution.h"  // FitnessResult, EvolutionConfig
#include "engine/grid_world.h"
#include <vector>
#include <functional>
#include <random>

namespace wuyun {

class DevEvolutionEngine {
public:
    explicit DevEvolutionEngine(const EvolutionConfig& config = {});

    /** 运行完整进化循环, 返回最佳发育基因组 */
    DevGenome run();

    /** 评估单个发育基因组 (多种子平均) */
    FitnessResult evaluate(const DevGenome& genome) const;

    /** Hall of Fame */
    const std::vector<DevGenome>& hall_of_fame() const { return hall_of_fame_; }

private:
    EvolutionConfig config_;
    std::mt19937 rng_;
    std::vector<DevGenome> population_;
    std::vector<DevGenome> hall_of_fame_;
    DevGenome best_ever_;          // 历史最佳, 永不丢失
    size_t n_elite_ = 0;          // 当代精英数 (评估时跳过)
    int stagnation_count_ = 0;    // 连续无进步代数 (自适应变异)

    void initialize_population();
    DevGenome tournament_select(const std::vector<DevGenome>& pop);
    std::vector<DevGenome> next_generation(std::vector<DevGenome>& current);
    FitnessResult evaluate_single(const DevGenome& genome, uint32_t seed) const;
};

} // namespace wuyun
