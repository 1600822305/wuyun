#include "genome/dev_evolution.h"
#include "development/developer.h"
#include "engine/closed_loop_agent.h"
#include "engine/grid_world_env.h"
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
// v53: 多任务"天才基因"评估
//
// 3 种任务, 7 次评估, 加权平均:
//   Task 1: 开放觅食 (3 seeds) — 基本趋近/回避
//   Task 2: 稀疏奖赏 (2 seeds) — 耐心 + 稀疏信号学习
//   Task 3: 反转学习 (2 seed pairs) — 灵活性 (×1.5 权重)
//
// 专才: 某一项满分但其他项崩溃
// 天才: 所有项都及格 → 通用学习能力
// =============================================================================

// 通用: 跑 agent N 步, 计算 early×1 + improvement×2 + late×2
float DevEvolutionEngine::run_and_score(ClosedLoopAgent& agent, size_t steps,
                                         int& out_food, int& out_danger) {
    // 早停: 50 步内是否有运动 (用位移判断, 不依赖 food/danger 事件)
    // v56 fix: 稀疏环境 (1 food, 0 danger) 中旧检查用 food/danger 事件判断运动,
    //   但 100 格只有 1 食物 → 60% 概率 50 步内没碰到 → 误判为"不动" → -1.0
    size_t warmup = std::min<size_t>(50, steps / 4);
    float start_x = agent.env().pos_x();
    float start_y = agent.env().pos_y();
    for (size_t i = 0; i < warmup; ++i) {
        agent.agent_step();
    }
    float dx = agent.env().pos_x() - start_x;
    float dy = agent.env().pos_y() - start_y;
    float displacement = dx * dx + dy * dy;
    if (displacement < 1.0f) {
        out_food = 0; out_danger = 0;
        return -1.0f;  // 真的不动 = 差评
    }

    size_t remaining = steps - warmup;
    size_t early_steps = remaining / 5;
    size_t late_steps = remaining - early_steps;

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

    float early_safety = static_cast<float>(e_food) /
                         std::max(1.0f, static_cast<float>(e_food + e_danger));
    float late_safety = static_cast<float>(l_food) /
                        std::max(1.0f, static_cast<float>(l_food + l_danger));
    float improvement = late_safety - early_safety;

    out_food = agent.env().positive_count();
    out_danger = agent.env().negative_count();

    return early_safety * 1.0f + improvement * 2.0f + late_safety * 2.0f;
}

// Task 1: 开放觅食 — 10×10, 5 food, 3 danger
float DevEvolutionEngine::eval_open_field(const AgentConfig& base_cfg,
                                           uint32_t seed, size_t steps) const {
    AgentConfig cfg = base_cfg;
    cfg.fast_eval = true;
    GridWorldConfig wcfg;
    wcfg.width = 10; wcfg.height = 10;
    wcfg.n_food = 5; wcfg.n_danger = 3;
    wcfg.maze_type = MazeType::OPEN_FIELD;
    wcfg.seed = seed;

    ClosedLoopAgent agent(std::make_unique<GridWorldEnv>(wcfg), cfg);
    int food = 0, danger = 0;
    return run_and_score(agent, steps, food, danger);
}

// Task 2: 稀疏奖赏 — 10×10, 1 food, 0 danger
// 测试耐心和探索效率: 食物少且无危险参考点
float DevEvolutionEngine::eval_sparse(const AgentConfig& base_cfg,
                                       uint32_t seed, size_t steps) const {
    AgentConfig cfg = base_cfg;
    cfg.fast_eval = true;
    GridWorldConfig wcfg;
    wcfg.width = 10; wcfg.height = 10;
    wcfg.n_food = 1; wcfg.n_danger = 0;
    wcfg.maze_type = MazeType::OPEN_FIELD;
    wcfg.seed = seed;

    ClosedLoopAgent agent(std::make_unique<GridWorldEnv>(wcfg), cfg);
    int food = 0, danger = 0;
    float score = run_and_score(agent, steps, food, danger);
    // 稀疏奖赏: 找到食物就给额外奖励 (因为只有 1 个, 很难找)
    score += static_cast<float>(food) * 0.1f;
    return score;
}

// Task 3: 反转学习 — 前半 seed_a, 后半 seed_b (大脑保留)
// 测试灵活性: 旧策略失效时能否快速适应
float DevEvolutionEngine::eval_reversal(const AgentConfig& base_cfg,
                                         uint32_t seed_a, uint32_t seed_b,
                                         size_t steps) const {
    AgentConfig cfg = base_cfg;
    cfg.fast_eval = true;
    GridWorldConfig wcfg;
    wcfg.width = 10; wcfg.height = 10;
    wcfg.n_food = 5; wcfg.n_danger = 3;
    wcfg.maze_type = MazeType::OPEN_FIELD;
    wcfg.seed = seed_a;

    ClosedLoopAgent agent(std::make_unique<GridWorldEnv>(wcfg), cfg);
    size_t half = steps / 2;

    // Phase 1: 正常学习 (seed_a)
    for (size_t i = 0; i < half; ++i) {
        agent.agent_step();
    }

    // Phase 2: 世界变了, 大脑保留 (seed_b)
    agent.reset_world_with_seed(seed_b);

    // 评估 Phase 2 表现 (反转后的适应能力)
    int food = 0, danger = 0;
    return run_and_score(agent, half, food, danger);
}

// 多任务评估: 3 种任务加权平均
MultitaskFitness DevEvolutionEngine::evaluate(const DevGenome& genome) const {
    // 连通性检查
    int conn = Developer::check_connectivity(genome);
    if (conn == 0) {
        MultitaskFitness bad{};
        bad.fitness = -2.0f;
        return bad;
    }

    AgentConfig base_cfg = Developer::to_agent_config(genome);
    size_t steps = config_.eval_steps;

    // Task 1: 开放觅食 (3 seeds, 权重 1.0)
    float open = 0;
    open += eval_open_field(base_cfg, 42,  steps);
    open += eval_open_field(base_cfg, 77,  steps);
    open += eval_open_field(base_cfg, 123, steps);
    open /= 3.0f;

    // Task 2: 稀疏奖赏 (2 seeds, 权重 1.0)
    float sparse = 0;
    sparse += eval_sparse(base_cfg, 256, steps);
    sparse += eval_sparse(base_cfg, 789, steps);
    sparse /= 2.0f;

    // Task 3: 反转学习 (2 seed pairs, 权重 1.5)
    float reversal = 0;
    reversal += eval_reversal(base_cfg, 42, 789, steps);
    reversal += eval_reversal(base_cfg, 77, 256, steps);
    reversal /= 2.0f;

    MultitaskFitness res;
    res.open_field = open;
    res.sparse_reward = sparse;
    res.reversal = reversal;
    // 加权: 开放×1 + 稀疏×1 + 反转×1.5
    // 反转高权重: 灵活性是区分专才和天才的关键
    res.fitness = open * 1.0f + sparse * 1.0f + reversal * 1.5f;
    res.fitness /= 3.5f;  // 归一化 (1+1+1.5=3.5)
    res.fitness += static_cast<float>(conn) * 0.05f;  // 连通性奖励
    return res;
}

// =============================================================================
// 完整进化循环
// =============================================================================

DevGenome DevEvolutionEngine::run() {
    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();

    initialize_population();

    best_ever_.fitness = -999.0f;

    // v53: 保留上一代精英的多任务分数 (修复精英显示 0.00 bug)
    std::vector<MultitaskFitness> prev_results;

    for (size_t gen = 0; gen < config_.n_generations; ++gen) {
        auto t_gen = Clock::now();

        size_t n_threads = std::min<size_t>(std::thread::hardware_concurrency(), population_.size());
        if (n_threads == 0) n_threads = 4;
        printf("  Evaluating %zu individuals (%zu threads): ", population_.size(), n_threads);
        fflush(stdout);

        std::atomic<size_t> done{0};
        std::vector<MultitaskFitness> results(population_.size());

        // 精英 (前 n_elite_ 个) 已有 fitness, 不重新评估
        // v53 fix: 保留完整 MultitaskFitness (不只是 fitness 标量)
        size_t skip = (gen == 0) ? 0 : n_elite_;
        for (size_t i = 0; i < skip; ++i) {
            if (i < prev_results.size()) {
                results[i] = prev_results[i];  // 保留完整多任务分数
            } else {
                results[i].fitness = population_[i].fitness;
            }
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

        // 找到最佳个体 + 保存其多任务分数
        size_t best_idx = 0;
        for (size_t i = 0; i < population_.size(); ++i) {
            population_[i].fitness = results[i].fitness;
            population_[i].generation = static_cast<int>(gen);
            if (results[i].fitness > results[best_idx].fitness) best_idx = i;
        }
        MultitaskFitness best_result = results[best_idx];

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
        printf("\n    %s | open=%.2f sparse=%.2f rev=%.2f\n",
               best_it->summary().c_str(),
               best_result.open_field, best_result.sparse_reward, best_result.reversal);

        // v53: 保存精英的多任务分数供下一代复用
        // next_generation() 把 best_ever 放位置 0, top-3 放位置 1-3
        {
            // 按 fitness 找当代 top-3 的 result 索引
            std::vector<size_t> sorted_idx(population_.size());
            for (size_t i = 0; i < sorted_idx.size(); ++i) sorted_idx[i] = i;
            std::sort(sorted_idx.begin(), sorted_idx.end(),
                [&](size_t a, size_t b) { return results[a].fitness > results[b].fitness; });
            prev_results.resize(std::min<size_t>(4, population_.size()) + 1);
            prev_results[0] = best_result;  // 位置 0: best_ever 用当前最佳结果
            if (improved) prev_results[0] = best_result;
            for (size_t i = 0; i < 3 && i < sorted_idx.size(); ++i) {
                prev_results[i + 1] = results[sorted_idx[i]];
            }
        }

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
