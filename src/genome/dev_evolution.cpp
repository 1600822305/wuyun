#include "genome/dev_evolution.h"
#include "development/developer.h"
#include "engine/closed_loop_agent.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstdio>

namespace wuyun {

// =============================================================================
// GA 操作 (与 EvolutionEngine 相同逻辑)
// =============================================================================

DevEvolutionEngine::DevEvolutionEngine(const EvolutionConfig& config)
    : config_(config), rng_(config.ga_seed) {}

void DevEvolutionEngine::initialize_population() {
    population_.resize(config_.population_size);
    for (auto& g : population_) {
        g.randomize(rng_);
        g.generation = 0;
    }
}

DevGenome DevEvolutionEngine::tournament_select(const std::vector<DevGenome>& pop) {
    std::uniform_int_distribution<size_t> pick(0, pop.size() - 1);
    size_t best_idx = pick(rng_);
    float best_fit = pop[best_idx].fitness;
    for (size_t i = 1; i < config_.tournament_size; ++i) {
        size_t idx = pick(rng_);
        if (pop[idx].fitness > best_fit) {
            best_idx = idx;
            best_fit = pop[idx].fitness;
        }
    }
    return pop[best_idx];
}

std::vector<DevGenome> DevEvolutionEngine::next_generation(std::vector<DevGenome>& current) {
    std::sort(current.begin(), current.end(),
              [](const DevGenome& a, const DevGenome& b) { return a.fitness > b.fitness; });

    std::vector<DevGenome> next;
    next.reserve(config_.population_size);

    // 精英保留: best_ever_ (历史最佳) + 当代 top 3, 不变异直接进入下一代
    // 这些精英在下一代评估时会被跳过 (保留原 fitness)
    next.push_back(best_ever_);  // 位置 0: 历史最佳, 永不丢失
    for (size_t i = 0; i < 3 && i < current.size(); ++i) {
        next.push_back(current[i]);  // 位置 1-3: 当代 top 3
    }
    n_elite_ = next.size();  // 记录精英数量, 评估时跳过

    // 剩余位置: 交叉 + 变异
    while (next.size() < config_.population_size) {
        auto child = DevGenome::crossover(tournament_select(current),
                                           tournament_select(current), rng_);
        child.mutate(rng_, config_.mutation_rate, config_.mutation_sigma);
        next.push_back(child);
    }
    return next;
}

// =============================================================================
// 评估: DevGenome → Developer → AgentConfig → ClosedLoopAgent → GridWorld
// =============================================================================

FitnessResult DevEvolutionEngine::evaluate_single(const DevGenome& genome, uint32_t seed) const {
    // --- 早停 1: 连通性检查 (0 步, 瞬间) ---
    // 如果没有皮层类型同时兼容 LGN 和 BG → 信号永远到不了运动输出
    int conn = Developer::check_connectivity(genome);
    if (conn == 0) {
        FitnessResult bad{};
        bad.fitness = -2.0f;  // 直接淘汰, 不浪费计算
        return bad;
    }

    // 1. 发育: DevGenome → AgentConfig (骨架固定 + 皮层涌现)
    AgentConfig cfg = Developer::to_agent_config(genome);
    cfg.fast_eval = true;
    cfg.world_config = config_.world_config;
    cfg.world_config.seed = seed;

    // 2. 构建完整 ClosedLoopAgent
    ClosedLoopAgent agent(cfg);

    // --- 早停 2: 100 步内是否有运动 ---
    int early_events = 0;
    for (size_t i = 0; i < 100; ++i) {
        auto result = agent.agent_step();
        if (result.got_food || result.hit_danger) early_events++;
    }
    if (early_events == 0) {
        // 100 步零事件 = agent 不动或被困
        FitnessResult bad{};
        bad.fitness = -1.5f;
        return bad;
    }

    // 3. 正式评估 (剩余步数)
    size_t remaining = config_.eval_steps - 100;
    size_t early_steps = remaining / 5;
    size_t late_steps = remaining - early_steps;

    int e_food = 0, e_danger = 0;
    for (size_t i = 0; i < early_steps; ++i) {
        auto result = agent.agent_step();
        if (result.got_food) e_food++;
        if (result.hit_danger) e_danger++;
    }

    int l_food = 0, l_danger = 0;
    for (size_t i = 0; i < late_steps; ++i) {
        auto result = agent.agent_step();
        if (result.got_food) l_food++;
        if (result.hit_danger) l_danger++;
    }

    FitnessResult res;
    res.early_safety = static_cast<float>(e_food) /
                       std::max(1.0f, static_cast<float>(e_food + e_danger));
    res.late_safety = static_cast<float>(l_food) /
                      std::max(1.0f, static_cast<float>(l_food + l_danger));
    res.improvement = res.late_safety - res.early_safety;
    res.total_food = agent.world().total_food_collected();
    res.total_danger = agent.world().total_danger_hits();

    // Baldwin 适应度 + 连通性奖励
    res.fitness = res.improvement * 3.0f
                + res.late_safety * 1.0f
                + static_cast<float>(res.total_food) * 0.001f
                - static_cast<float>(res.total_danger) * 0.001f
                + static_cast<float>(conn) * 0.05f;  // 连通性奖励
    return res;
}

FitnessResult DevEvolutionEngine::evaluate(const DevGenome& genome) const {
    FitnessResult avg{};
    for (uint32_t seed : config_.eval_seeds) {
        auto r = evaluate_single(genome, seed);
        avg.fitness      += r.fitness;
        avg.early_safety += r.early_safety;
        avg.late_safety  += r.late_safety;
        avg.improvement  += r.improvement;
        avg.total_food   += r.total_food;
        avg.total_danger += r.total_danger;
    }
    float n = static_cast<float>(config_.eval_seeds.size());
    avg.fitness /= n;  avg.early_safety /= n;
    avg.late_safety /= n;  avg.improvement /= n;
    return avg;
}

// =============================================================================
// 完整进化循环
// =============================================================================

DevGenome DevEvolutionEngine::run() {
    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();

    initialize_population();

    best_ever_.fitness = -999.0f;

    for (size_t gen = 0; gen < config_.n_generations; ++gen) {
        auto t_gen = Clock::now();

        size_t n_threads = std::min<size_t>(std::thread::hardware_concurrency(), population_.size());
        if (n_threads == 0) n_threads = 4;
        printf("  Evaluating %zu individuals (%zu threads): ", population_.size(), n_threads);
        fflush(stdout);

        std::atomic<size_t> done{0};
        std::vector<FitnessResult> results(population_.size());

        // 精英 (前 n_elite_ 个) 已有 fitness, 不重新评估
        size_t skip = (gen == 0) ? 0 : n_elite_;
        for (size_t i = 0; i < skip; ++i) {
            results[i].fitness = population_[i].fitness;  // 保留原 fitness
            done.fetch_add(1);
        }

        auto worker = [&](size_t s, size_t e) {
            for (size_t i = s; i < e; ++i) {
                if (i < skip) continue;  // 精英跳过
                results[i] = evaluate(population_[i]);
                done.fetch_add(1);
            }
        };

        std::vector<std::thread> threads;
        size_t chunk = (population_.size() + n_threads - 1) / n_threads;
        for (size_t t = 0; t < n_threads; ++t) {
            size_t s = t * chunk, e = std::min(s + chunk, population_.size());
            if (s < e) threads.emplace_back(worker, s, e);
        }
        while (done.load() < population_.size()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            printf(".");
            fflush(stdout);
        }
        for (auto& th : threads) th.join();
        printf(" done\n");

        for (size_t i = 0; i < population_.size(); ++i) {
            population_[i].fitness = results[i].fitness;
            population_[i].generation = static_cast<int>(gen);
        }

        auto best_it = std::max_element(population_.begin(), population_.end(),
            [](const DevGenome& a, const DevGenome& b) { return a.fitness < b.fitness; });

        bool improved = (best_it->fitness > best_ever_.fitness);
        if (improved) {
            best_ever_ = *best_it;
            stagnation_count_ = 0;
        } else {
            stagnation_count_++;
        }
        hall_of_fame_.push_back(*best_it);

        // 自适应变异: 停滞时加大探索 (逃离局部最优)
        // 没停滞: 正常变异 (15% × 0.10σ)
        // 停滞越久 → 变异越大 → 强制跳出当前山顶
        float scale = 1.0f + static_cast<float>(stagnation_count_) * 0.3f;
        float adapt_rate = std::min(config_.mutation_rate * scale, 0.50f);
        float adapt_sigma = std::min(config_.mutation_sigma * scale, 0.30f);

        float gen_sec = std::chrono::duration<float>(Clock::now() - t_gen).count();
        float avg_fit = 0;
        for (auto& g : population_) avg_fit += g.fitness;
        avg_fit /= static_cast<float>(population_.size());

        printf("  Gen %2zu/%zu | best=%.4f avg=%.4f | best_ever=%.4f | %.1fs",
               gen + 1, config_.n_generations, best_it->fitness, avg_fit, best_ever_.fitness, gen_sec);
        if (stagnation_count_ > 0) printf(" [stag=%d mr=%.0f%% σ=%.3f]", stagnation_count_, adapt_rate*100, adapt_sigma);
        printf("\n    %s\n", best_it->summary().c_str());

        // 用自适应参数生成下一代
        float old_mr = config_.mutation_rate;
        float old_ms = config_.mutation_sigma;
        config_.mutation_rate = adapt_rate;
        config_.mutation_sigma = adapt_sigma;
        population_ = next_generation(population_);
        config_.mutation_rate = old_mr;
        config_.mutation_sigma = old_ms;
    }

    float total_sec = std::chrono::duration<float>(Clock::now() - t_start).count();
    printf("\n  DevEvolution complete: %.1f sec, best fitness=%.4f\n", total_sec, best_ever_.fitness);
    printf("    %s\n", best_ever_.summary().c_str());

    return best_ever_;
}

} // namespace wuyun
