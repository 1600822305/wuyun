#include "genome/evolution.h"
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

namespace wuyun {

// =============================================================================
// Constructor
// =============================================================================

EvolutionEngine::EvolutionEngine(const EvolutionConfig& config)
    : config_(config)
    , rng_(config.ga_seed)
{
}

// =============================================================================
// Initialize population with random genomes
// =============================================================================

void EvolutionEngine::initialize_population() {
    population_.resize(config_.population_size);
    for (auto& genome : population_) {
        genome.randomize(rng_);
        genome.generation = 0;
    }
}

// =============================================================================
// Tournament selection: pick tournament_size random individuals, return best
// =============================================================================

Genome EvolutionEngine::tournament_select(const std::vector<Genome>& pop) {
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

// =============================================================================
// Generate next generation via selection + crossover + mutation
// =============================================================================

std::vector<Genome> EvolutionEngine::next_generation(std::vector<Genome>& current) {
    // Sort by fitness (descending)
    std::sort(current.begin(), current.end(),
              [](const Genome& a, const Genome& b) { return a.fitness > b.fitness; });

    std::vector<Genome> next;
    next.reserve(config_.population_size);

    // Elite: top fraction survives unchanged
    size_t n_elite = std::max<size_t>(1,
        static_cast<size_t>(config_.elite_fraction * config_.population_size));
    for (size_t i = 0; i < n_elite && i < current.size(); ++i) {
        next.push_back(current[i]);
    }

    // Fill rest with crossover + mutation
    while (next.size() < config_.population_size) {
        Genome parent_a = tournament_select(current);
        Genome parent_b = tournament_select(current);
        Genome child = Genome::crossover(parent_a, parent_b, rng_);
        child.mutate(rng_, config_.mutation_rate, config_.mutation_sigma);
        next.push_back(child);
    }

    return next;
}

// =============================================================================
// Evaluate a single genome on a single seed
// =============================================================================

FitnessResult EvolutionEngine::evaluate_single(const Genome& genome, uint32_t seed) const {
    AgentConfig cfg = genome.to_agent_config();

    // Apply genome-specific parameters not in AgentConfig
    // (bg_to_m1_gain, lgn_gain, etc. are hardcoded in build_brain/agent_step)
    // For v1, we pass them through AgentConfig extensions or use the direct fields.
    cfg.world_config = config_.world_config;
    cfg.world_config.seed = seed;

    ClosedLoopAgent agent(cfg);

    // Warm-up: first 20% of steps
    size_t warmup = config_.eval_steps / 5;
    size_t test_half = (config_.eval_steps - warmup) / 2;

    int warmup_food = 0;
    for (size_t i = 0; i < warmup; ++i) {
        auto result = agent.agent_step();
        if (result.got_food) warmup_food++;
    }

    // Early termination: if 0 food after warmup, this genome is broken
    if (warmup_food == 0 && warmup >= 500) {
        FitnessResult bad{};
        bad.fitness = -2.0f;
        return bad;
    }

    // Early phase
    int early_food = 0, early_danger = 0;
    for (size_t i = 0; i < test_half; ++i) {
        auto result = agent.agent_step();
        if (result.got_food) early_food++;
        if (result.hit_danger) early_danger++;
    }

    // Late phase
    int late_food = 0, late_danger = 0;
    for (size_t i = 0; i < test_half; ++i) {
        auto result = agent.agent_step();
        if (result.got_food) late_food++;
        if (result.hit_danger) late_danger++;
    }

    FitnessResult res;
    res.early_safety = static_cast<float>(early_food) /
                       std::max(1.0f, static_cast<float>(early_food + early_danger));
    res.late_safety = static_cast<float>(late_food) /
                      std::max(1.0f, static_cast<float>(late_food + late_danger));
    res.improvement = res.late_safety - res.early_safety;
    res.total_food = agent.world().total_food_collected();
    res.total_danger = agent.world().total_danger_hits();

    // Fitness: late performance + learning ability (Baldwin effect)
    res.fitness = res.late_safety * 1.0f
                + res.improvement * 2.0f
                - static_cast<float>(res.total_danger) * 0.002f
                + static_cast<float>(res.total_food) * 0.001f;

    return res;
}

// =============================================================================
// Evaluate genome averaged over multiple seeds
// =============================================================================

FitnessResult EvolutionEngine::evaluate(const Genome& genome) const {
    FitnessResult avg{};
    for (uint32_t seed : config_.eval_seeds) {
        FitnessResult r = evaluate_single(genome, seed);
        avg.fitness      += r.fitness;
        avg.early_safety += r.early_safety;
        avg.late_safety  += r.late_safety;
        avg.improvement  += r.improvement;
        avg.total_food   += r.total_food;
        avg.total_danger += r.total_danger;
    }
    float n = static_cast<float>(config_.eval_seeds.size());
    avg.fitness      /= n;
    avg.early_safety /= n;
    avg.late_safety  /= n;
    avg.improvement  /= n;
    return avg;
}

// =============================================================================
// Run the full evolutionary loop
// =============================================================================

Genome EvolutionEngine::run() {
    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();

    initialize_population();

    Genome best_ever;
    best_ever.fitness = -999.0f;

    for (size_t gen = 0; gen < config_.n_generations; ++gen) {
        auto t_gen_start = Clock::now();

        // Evaluate all individuals in parallel
        size_t n_threads = std::min<size_t>(
            std::thread::hardware_concurrency(),
            population_.size());
        if (n_threads == 0) n_threads = 4;
        printf("  Evaluating %zu individuals (%zu threads): ", population_.size(), n_threads);
        fflush(stdout);

        std::atomic<size_t> done_count{0};
        std::vector<FitnessResult> results(population_.size());

        auto worker = [&](size_t start, size_t end) {
            for (size_t idx = start; idx < end; ++idx) {
                results[idx] = evaluate(population_[idx]);
                done_count.fetch_add(1);
            }
        };

        // Distribute work across threads
        std::vector<std::thread> threads;
        size_t chunk = (population_.size() + n_threads - 1) / n_threads;
        for (size_t t = 0; t < n_threads; ++t) {
            size_t start = t * chunk;
            size_t end = std::min(start + chunk, population_.size());
            if (start < end) {
                threads.emplace_back(worker, start, end);
            }
        }

        // Wait with progress dots
        while (done_count.load() < population_.size()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            printf(".");
            fflush(stdout);
        }
        for (auto& th : threads) th.join();
        printf(" done\n");

        // Apply results
        for (size_t idx = 0; idx < population_.size(); ++idx) {
            population_[idx].fitness = results[idx].fitness;
            population_[idx].generation = static_cast<int>(gen);
        }

        // Find generation best
        auto best_it = std::max_element(population_.begin(), population_.end(),
            [](const Genome& a, const Genome& b) { return a.fitness < b.fitness; });

        // Update Hall of Fame
        if (best_it->fitness > best_ever.fitness) {
            best_ever = *best_it;
        }
        hall_of_fame_.push_back(*best_it);

        auto t_gen_end = Clock::now();
        float gen_sec = std::chrono::duration<float>(t_gen_end - t_gen_start).count();

        // Progress report
        float avg_fit = 0;
        for (const auto& g : population_) avg_fit += g.fitness;
        avg_fit /= static_cast<float>(population_.size());

        printf("  Gen %2zu/%zu | best=%.4f avg=%.4f | best_ever=%.4f | %.1fs\n",
               gen + 1, config_.n_generations,
               best_it->fitness, avg_fit, best_ever.fitness, gen_sec);
        printf("    %s\n", best_it->summary().c_str());

        if (progress_cb_) {
            progress_cb_(static_cast<int>(gen), best_ever.fitness, best_ever.summary());
        }

        // Generate next generation
        population_ = next_generation(population_);
    }

    auto t_end = Clock::now();
    float total_sec = std::chrono::duration<float>(t_end - t_start).count();
    printf("\n  Evolution complete: %.1f sec total, best fitness=%.4f\n", total_sec, best_ever.fitness);

    return best_ever;
}

} // namespace wuyun
