#include "genome/genome.h"
#include "engine/closed_loop_agent.h"
#include <sstream>
#include <iomanip>
#include <cmath>

namespace wuyun {

// =============================================================================
// Gene access (flat vector for generic GA operations)
// =============================================================================

std::vector<Gene*> Genome::all_genes() {
    return {
        &da_stdp_lr, &reward_scale, &cortical_a_plus, &cortical_a_minus, &cortical_w_max,
        &exploration_noise, &bg_to_m1_gain, &attractor_ratio, &background_ratio,
        &replay_passes, &replay_da_scale,
        &lgn_gain, &lgn_baseline, &lgn_noise,
        &homeostatic_target, &homeostatic_eta,
        &v1_size, &dlpfc_size, &bg_size,
        &brain_steps, &reward_steps,
        &ne_food_scale, &ne_floor
    };
}

std::vector<const Gene*> Genome::all_genes() const {
    return {
        &da_stdp_lr, &reward_scale, &cortical_a_plus, &cortical_a_minus, &cortical_w_max,
        &exploration_noise, &bg_to_m1_gain, &attractor_ratio, &background_ratio,
        &replay_passes, &replay_da_scale,
        &lgn_gain, &lgn_baseline, &lgn_noise,
        &homeostatic_target, &homeostatic_eta,
        &v1_size, &dlpfc_size, &bg_size,
        &brain_steps, &reward_steps,
        &ne_food_scale, &ne_floor
    };
}

size_t Genome::n_genes() const { return 23; }

// =============================================================================
// Randomize
// =============================================================================

void Genome::randomize(std::mt19937& rng) {
    for (Gene* g : all_genes()) {
        std::uniform_real_distribution<float> dist(g->min_val, g->max_val);
        g->value = dist(rng);
    }
}

// =============================================================================
// Mutate
// =============================================================================

void Genome::mutate(std::mt19937& rng, float mutation_rate, float sigma) {
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    for (Gene* g : all_genes()) {
        if (coin(rng) < mutation_rate) {
            g->mutate(rng, sigma);
        }
    }
}

// =============================================================================
// Crossover (uniform: each gene 50/50 from parent a or b)
// =============================================================================

Genome Genome::crossover(const Genome& a, const Genome& b, std::mt19937& rng) {
    Genome child;
    auto a_genes = a.all_genes();
    auto b_genes = b.all_genes();
    auto c_genes = child.all_genes();

    std::uniform_int_distribution<int> coin(0, 1);
    for (size_t i = 0; i < c_genes.size(); ++i) {
        c_genes[i]->value = coin(rng) ? a_genes[i]->value : b_genes[i]->value;
    }
    return child;
}

// =============================================================================
// Convert Genome → AgentConfig
// =============================================================================

AgentConfig Genome::to_agent_config() const {
    AgentConfig cfg;
    cfg.fast_eval = true;  // Skip hippocampus + cortical STDP for evolution speed

    // Learning
    cfg.da_stdp_lr = da_stdp_lr.value;
    cfg.reward_scale = reward_scale.value;
    cfg.cortical_stdp_a_plus = cortical_a_plus.value;
    cfg.cortical_stdp_a_minus = -cortical_a_minus.value;  // stored positive, applied negative
    cfg.cortical_stdp_w_max = cortical_w_max.value;

    // Exploration
    cfg.exploration_noise = exploration_noise.value;
    cfg.bg_to_m1_gain = bg_to_m1_gain.value;
    cfg.attractor_drive_ratio = attractor_ratio.value;
    cfg.background_drive_ratio = background_ratio.value;

    // NE modulation
    cfg.ne_food_scale = ne_food_scale.value;
    cfg.ne_floor = ne_floor.value;

    // Replay
    cfg.replay_passes = std::max(1, static_cast<int>(std::round(replay_passes.value)));
    cfg.replay_da_scale = replay_da_scale.value;

    // Visual encoding
    cfg.lgn_gain = lgn_gain.value;
    cfg.lgn_baseline = lgn_baseline.value;
    cfg.lgn_noise_amp = lgn_noise.value;

    // Homeostatic
    cfg.homeostatic_target_rate = homeostatic_target.value;
    cfg.homeostatic_eta = homeostatic_eta.value;

    // Brain size
    cfg.v1_size_factor = v1_size.value;
    cfg.dlpfc_size_factor = dlpfc_size.value;
    cfg.bg_size_factor = bg_size.value;

    // Timing — clamp brain_steps to 10 in fast_eval for speed
    // (full pipeline needs ~14, but 10 is enough to propagate LGN→V1→dlPFC→BG)
    size_t bs = static_cast<size_t>(std::round(brain_steps.value));
    cfg.brain_steps_per_action = std::clamp(bs, static_cast<size_t>(5), static_cast<size_t>(10));
    cfg.reward_processing_steps = std::max(
        static_cast<size_t>(1),
        std::min(static_cast<size_t>(3),
                 static_cast<size_t>(std::round(reward_steps.value))));

    return cfg;
}

// =============================================================================
// JSON serialization
// =============================================================================

std::string Genome::to_json() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    ss << "{\n";
    ss << "  \"generation\": " << generation << ",\n";
    ss << "  \"fitness\": " << fitness << ",\n";
    ss << "  \"genes\": {\n";

    auto genes = all_genes();
    for (size_t i = 0; i < genes.size(); ++i) {
        ss << "    \"" << genes[i]->name << "\": " << genes[i]->value;
        if (i + 1 < genes.size()) ss << ",";
        ss << "\n";
    }
    ss << "  }\n}";
    return ss.str();
}

Genome Genome::from_json(const std::string& /* json */) {
    // Minimal parser — to be expanded later
    // For now, return default genome
    return Genome{};
}

// =============================================================================
// Summary
// =============================================================================

std::string Genome::summary() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);
    ss << "Gen" << generation << " fit=" << fitness
       << " lr=" << da_stdp_lr.value
       << " noise=" << exploration_noise.value
       << " bg_gain=" << bg_to_m1_gain.value
       << " lgn=" << lgn_gain.value
       << " v1=" << v1_size.value
       << " replay=" << static_cast<int>(replay_passes.value);
    return ss.str();
}

} // namespace wuyun
