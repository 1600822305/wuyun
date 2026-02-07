#include "engine/closed_loop_agent.h"
#include "region/neuromod/lc_ne.h"
#include "region/neuromod/drn_5ht.h"
#include "region/neuromod/nbm_ach.h"
#include <algorithm>
#include <cstdio>
#include <numeric>

namespace wuyun {

// =============================================================================
// Construction
// =============================================================================

ClosedLoopAgent::ClosedLoopAgent(const AgentConfig& config)
    : config_(config)
    , world_(config.world_config)
    , engine_(10)
    , reward_history_(1000, 0.0f)
    , food_history_(1000, 0)
{
    build_brain();

    // Setup visual encoder for 3x3 patch → LGN
    VisualInputConfig vcfg;
    vcfg.input_width  = config_.vision_width;
    vcfg.input_height = config_.vision_height;
    vcfg.n_lgn_neurons = lgn_->n_neurons();
    vcfg.gain = 45.0f;      // Strong enough to drive LGN
    vcfg.baseline = 3.0f;
    vcfg.noise_amp = 1.5f;
    visual_encoder_ = VisualInput(vcfg);
}

// =============================================================================
// Build the brain circuit for closed-loop control
// =============================================================================

void ClosedLoopAgent::build_brain() {
    int s = config_.brain_scale;

    // --- Thalamic relay: LGN (visual input gate) ---
    ThalamicConfig lgn_cfg;
    lgn_cfg.name = "LGN";
    lgn_cfg.n_relay = 30 * s;
    lgn_cfg.n_trn   = 10 * s;
    engine_.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    // --- Cortical regions ---
    auto add_ctx = [&](const std::string& name, size_t l4, size_t l23,
                       size_t l5, size_t l6, size_t pv, size_t sst, size_t vip) {
        ColumnConfig c;
        c.n_l4_stellate = l4 * s;  c.n_l23_pyramidal = l23 * s;
        c.n_l5_pyramidal = l5 * s; c.n_l6_pyramidal = l6 * s;
        c.n_pv_basket = pv * s;    c.n_sst_martinotti = sst * s;
        c.n_vip = vip * s;
        engine_.add_region(std::make_unique<CorticalRegion>(name, c));
    };

    // V1: primary visual cortex
    add_ctx("V1", 30, 60, 30, 25, 10, 6, 3);
    // dlPFC: decision making + working memory
    add_ctx("dlPFC", 20, 50, 30, 20, 8, 5, 3);
    // M1: motor output (L5 → action decoding)
    add_ctx("M1", 20, 40, 40, 15, 8, 5, 2);

    // --- Basal Ganglia: action selection ---
    BasalGangliaConfig bg_cfg;
    bg_cfg.name = "BG";
    bg_cfg.n_d1_msn = 30 * s;
    bg_cfg.n_d2_msn = 30 * s;
    bg_cfg.n_gpi = 10 * s;
    bg_cfg.n_gpe = 10 * s;
    bg_cfg.n_stn = 8 * s;
    bg_cfg.da_stdp_enabled = config_.enable_da_stdp;
    bg_cfg.da_stdp_lr = config_.da_stdp_lr;
    engine_.add_region(std::make_unique<BasalGanglia>(bg_cfg));

    // --- Motor thalamus: BG → M1 relay ---
    ThalamicConfig mthal_cfg;
    mthal_cfg.name = "MotorThal";
    mthal_cfg.n_relay = 20 * s;
    mthal_cfg.n_trn   = 6 * s;
    engine_.add_region(std::make_unique<ThalamicRelay>(mthal_cfg));

    // --- VTA: dopamine / reward signal ---
    VTAConfig vta_cfg;
    vta_cfg.n_da_neurons = 30 * s;
    engine_.add_region(std::make_unique<VTA_DA>(vta_cfg));

    // --- Hippocampus: spatial memory ---
    HippocampusConfig hipp_cfg;
    hipp_cfg.n_ec  = 40 * s;
    hipp_cfg.n_dg  = 80 * s;
    hipp_cfg.n_ca3 = 30 * s;
    hipp_cfg.n_ca1 = 30 * s;
    hipp_cfg.n_sub = 15 * s;
    hipp_cfg.ca3_stdp_enabled = true;  // Memory encoding
    engine_.add_region(std::make_unique<Hippocampus>(hipp_cfg));

    // --- Projections (core closed-loop circuit) ---

    // Visual pathway: LGN → V1 → dlPFC
    engine_.add_projection("LGN", "V1", 2);
    engine_.add_projection("V1", "dlPFC", 2);

    // Decision → action: dlPFC → BG → MotorThal → M1
    engine_.add_projection("dlPFC", "BG", 2);
    engine_.add_projection("BG", "MotorThal", 2);
    engine_.add_projection("MotorThal", "M1", 2);

    // Feedback: M1 → dlPFC (efference copy)
    engine_.add_projection("M1", "dlPFC", 3);

    // Memory: dlPFC → Hippocampus (encode context)
    engine_.add_projection("dlPFC", "Hippocampus", 3);
    engine_.add_projection("V1", "Hippocampus", 3);

    // VTA DA → BG (reward signal)
    engine_.add_projection("VTA", "BG", 2);

    // --- Neuromodulator registration ---
    engine_.register_neuromod_source("VTA", SimulationEngine::NeuromodType::DA);

    // --- Wire DA source for BG ---
    auto* bg_ptr = dynamic_cast<BasalGanglia*>(engine_.find_region("BG"));
    auto* vta_ptr = engine_.find_region("VTA");
    if (bg_ptr && vta_ptr) bg_ptr->set_da_source_region(vta_ptr->region_id());

    // --- Cache region pointers ---
    lgn_   = engine_.find_region("LGN");
    v1_    = dynamic_cast<CorticalRegion*>(engine_.find_region("V1"));
    dlpfc_ = dynamic_cast<CorticalRegion*>(engine_.find_region("dlPFC"));
    m1_    = dynamic_cast<CorticalRegion*>(engine_.find_region("M1"));
    bg_    = dynamic_cast<BasalGanglia*>(engine_.find_region("BG"));
    vta_   = dynamic_cast<VTA_DA*>(engine_.find_region("VTA"));
    hipp_  = dynamic_cast<Hippocampus*>(engine_.find_region("Hippocampus"));

    // --- Enable homeostatic plasticity ---
    if (config_.enable_homeostatic) {
        HomeostaticParams hp;
        hp.target_rate = 5.0f;
        hp.eta = 0.001f;
        hp.scale_interval = 100;
        v1_->enable_homeostatic(hp);
        dlpfc_->enable_homeostatic(hp);
        m1_->enable_homeostatic(hp);
        hipp_->enable_homeostatic(hp);
    }

    // --- Enable working memory on dlPFC ---
    dlpfc_->enable_working_memory();
}

// =============================================================================
// Closed loop step
// =============================================================================

void ClosedLoopAgent::reset_world() {
    world_.reset();
    agent_step_count_ = 0;
    std::fill(reward_history_.begin(), reward_history_.end(), 0.0f);
    std::fill(food_history_.begin(), food_history_.end(), 0);
    history_idx_ = 0;
}

StepResult ClosedLoopAgent::agent_step() {
    // 1. Observe environment
    inject_observation();

    // 2. Run brain for N steps (accumulate M1 activity)

    // Motor exploration: pick ONE action group for this entire action period
    // and give it sustained drive (mimics BG thalamic disinhibition selecting one action)
    int boosted_group = -1;
    if (config_.exploration_noise > 0.0f) {
        std::uniform_int_distribution<int> group_pick(0, 3);
        boosted_group = group_pick(motor_rng_);
    }

    // Accumulate M1 L5 spike counts across all brain steps
    const auto& col = m1_->column();
    size_t l4_size  = col.l4().size();
    size_t l23_size = col.l23().size();
    size_t l5_size  = col.l5().size();
    size_t l5_offset = l4_size + l23_size;
    std::vector<int> l5_accum(l5_size, 0);

    for (size_t i = 0; i < config_.brain_steps_per_action; ++i) {
        // Re-inject observation every few brain steps to sustain input
        if (i > 0 && i % 3 == 0) {
            inject_observation();
        }

        // Sustained motor exploration noise into the selected M1 L5 group
        // (Biological basis: motor variability for exploration, Todorov 2004)
        if (boosted_group >= 0) {
            auto& l5 = m1_->column().l5();
            if (l5_size >= 4) {
                size_t group_size = l5_size / 4;
                float bias = config_.exploration_noise * 0.6f;
                float jitter_range = config_.exploration_noise * 0.4f;
                std::uniform_real_distribution<float> jitter(-jitter_range, jitter_range);
                size_t start = static_cast<size_t>(boosted_group) * group_size;
                size_t end = (boosted_group < 3)
                    ? static_cast<size_t>(boosted_group + 1) * group_size : l5_size;
                for (size_t j = start; j < end; ++j) {
                    l5.inject_basal(j, bias + jitter(motor_rng_));
                }
            }
        }

        engine_.step();

        // Accumulate M1 L5 fired state
        const auto& m1_fired = m1_->fired();
        for (size_t j = 0; j < l5_size && (l5_offset + j) < m1_fired.size(); ++j) {
            l5_accum[j] += m1_fired[l5_offset + j];
        }
    }

    // 3. Decode M1 output → action (from accumulated L5 spikes)
    Action action = decode_m1_action(l5_accum);

    // 4. Execute action in GridWorld
    StepResult result = world_.act(action);

    // 5. Inject reward to VTA
    inject_reward(result.reward);

    // 6. Update state
    last_action_ = action;
    last_reward_ = result.reward;
    agent_step_count_++;

    // Record history
    size_t hi = history_idx_ % reward_history_.size();
    reward_history_[hi] = result.reward;
    food_history_[hi] = result.got_food ? 1 : 0;
    history_idx_++;

    // Callback
    if (callback_) {
        callback_(agent_step_count_, action, result.reward, result.agent_x, result.agent_y);
    }

    return result;
}

void ClosedLoopAgent::run(int n_steps) {
    for (int i = 0; i < n_steps; ++i) {
        agent_step();
    }
}

// =============================================================================
// Perception: observe → encode → inject LGN
// =============================================================================

void ClosedLoopAgent::inject_observation() {
    auto obs = world_.observe();  // 3x3 = 9 pixels
    visual_encoder_.encode_and_inject(obs, lgn_);
}

// =============================================================================
// Action decoding: M1 L5 fired → winner-take-all over 4 directions
// =============================================================================

Action ClosedLoopAgent::decode_m1_action(const std::vector<int>& l5_accum) const {
    // M1 L5 pyramidal neurons are divided into 4 groups:
    //   Group 0: UP,   Group 1: DOWN,  Group 2: LEFT,  Group 3: RIGHT
    // Count accumulated spikes in each group, pick winner

    size_t l5_size = l5_accum.size();
    if (l5_size < 4) return Action::STAY;

    size_t group_size = l5_size / 4;
    int counts[4] = {0, 0, 0, 0};

    for (size_t g = 0; g < 4; ++g) {
        size_t start = g * group_size;
        size_t end = (g < 3) ? (g + 1) * group_size : l5_size;
        for (size_t i = start; i < end; ++i) {
            counts[g] += l5_accum[i];
        }
    }

    // Winner-take-all
    int max_count = *std::max_element(counts, counts + 4);
    if (max_count == 0) return Action::STAY;  // All silent

    // Find winner (with tie-breaking: prefer first)
    for (int g = 0; g < 4; ++g) {
        if (counts[g] == max_count) {
            return static_cast<Action>(g);
        }
    }
    return Action::STAY;
}

// =============================================================================
// Reward: inject to VTA
// =============================================================================

void ClosedLoopAgent::inject_reward(float reward) {
    if (vta_ && std::abs(reward) > 0.001f) {
        vta_->inject_reward(reward * config_.reward_scale);
    }
}

// =============================================================================
// Statistics
// =============================================================================

float ClosedLoopAgent::avg_reward(size_t window) const {
    size_t n = std::min(window, static_cast<size_t>(agent_step_count_));
    if (n == 0) return 0.0f;

    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        size_t idx = (history_idx_ - 1 - i) % reward_history_.size();
        sum += reward_history_[idx];
    }
    return sum / static_cast<float>(n);
}

float ClosedLoopAgent::food_rate(size_t window) const {
    size_t n = std::min(window, static_cast<size_t>(agent_step_count_));
    if (n == 0) return 0.0f;

    int sum = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t idx = (history_idx_ - 1 - i) % food_history_.size();
        sum += food_history_[idx];
    }
    return static_cast<float>(sum) / static_cast<float>(n);
}

} // namespace wuyun
