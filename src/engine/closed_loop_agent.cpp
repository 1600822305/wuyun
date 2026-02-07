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
    , replay_buffer_(config.replay_buffer_size, config.brain_steps_per_action)
{
    // Auto-compute vision size from world config
    config_.vision_width  = config_.world_config.vision_side();
    config_.vision_height = config_.world_config.vision_side();

    build_brain();

    // Setup visual encoder for NxN patch → LGN
    VisualInputConfig vcfg;
    vcfg.input_width  = config_.vision_width;
    vcfg.input_height = config_.vision_height;
    vcfg.n_lgn_neurons = lgn_->n_neurons();
    vcfg.gain = 200.0f;     // Strong: food pixel response ~0.3 → I=5+60=65, V_ss=0 (fires fast)
    vcfg.baseline = 5.0f;   // Empty pixels: V_ss=-60 (below threshold, no firing)
    vcfg.noise_amp = 2.0f;  // Slight noise for stochastic variation
    visual_encoder_ = VisualInput(vcfg);
}

// =============================================================================
// Build the brain circuit for closed-loop control
// =============================================================================

void ClosedLoopAgent::build_brain() {
    int s = config_.brain_scale;

    // --- Thalamic relay: LGN (visual input gate) ---
    // Scale LGN proportionally to vision size: ~3.3 LGN neurons per pixel
    size_t n_vis_pixels = config_.vision_width * config_.vision_height;
    size_t lgn_per_pixel = 3;  // ~3 LGN neurons per input pixel (ON/OFF + margin)
    size_t lgn_base = std::max<size_t>(30, n_vis_pixels * lgn_per_pixel);
    ThalamicConfig lgn_cfg;
    lgn_cfg.name = "LGN";
    lgn_cfg.n_relay = lgn_base * s;
    lgn_cfg.n_trn   = (lgn_base / 3) * s;
    engine_.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    // --- Cortical regions ---
    auto add_ctx = [&](const std::string& name, size_t l4, size_t l23,
                       size_t l5, size_t l6, size_t pv, size_t sst, size_t vip,
                       bool stdp = false) {
        ColumnConfig c;
        c.n_l4_stellate = l4 * s;  c.n_l23_pyramidal = l23 * s;
        c.n_l5_pyramidal = l5 * s; c.n_l6_pyramidal = l6 * s;
        c.n_pv_basket = pv * s;    c.n_sst_martinotti = sst * s;
        c.n_vip = vip * s;
        if (stdp) {
            c.stdp_enabled = true;
            c.stdp_a_plus  = config_.cortical_stdp_a_plus;
            c.stdp_a_minus = config_.cortical_stdp_a_minus;
            c.stdp_w_max   = config_.cortical_stdp_w_max;
        }
        engine_.add_region(std::make_unique<CorticalRegion>(name, c));
    };

    bool ctx_stdp = config_.enable_cortical_stdp;
    // V1: primary visual cortex — STDP learns visual feature selectivity
    // Scale with vision size: more pixels → more V1 neurons for feature extraction
    float vis_scale = static_cast<float>(n_vis_pixels) / 9.0f;  // 1.0 for 3×3, ~2.8 for 5×5
    size_t v1_l4  = static_cast<size_t>(30 * vis_scale);
    size_t v1_l23 = static_cast<size_t>(60 * vis_scale);
    size_t v1_l5  = static_cast<size_t>(30 * vis_scale);
    size_t v1_l6  = static_cast<size_t>(25 * vis_scale);
    size_t v1_pv  = static_cast<size_t>(10 * vis_scale);
    size_t v1_sst = static_cast<size_t>(6 * vis_scale);
    add_ctx("V1", v1_l4, v1_l23, v1_l5, v1_l6, v1_pv, v1_sst, 3, ctx_stdp);
    // dlPFC: decision making + working memory — NO STDP (representation stability
    //        needed for BG DA-STDP to learn stable action associations)
    // Slight scale for larger V1 input (sqrt to prevent over-scaling)
    float dlpfc_scale = std::sqrt(vis_scale);
    size_t dl_l4  = static_cast<size_t>(20 * dlpfc_scale);
    size_t dl_l23 = static_cast<size_t>(50 * dlpfc_scale);
    size_t dl_l5  = static_cast<size_t>(30 * dlpfc_scale);
    size_t dl_l6  = static_cast<size_t>(20 * dlpfc_scale);
    size_t dl_pv  = static_cast<size_t>(8 * dlpfc_scale);
    size_t dl_sst = static_cast<size_t>(5 * dlpfc_scale);
    add_ctx("dlPFC", dl_l4, dl_l23, dl_l5, dl_l6, dl_pv, dl_sst, 3, false);
    // M1: motor output (L5 → action decoding) — NO STDP (driven by noise + BG bias)
    add_ctx("M1", 20, 40, 40, 15, 8, 5, 2, false);

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

    // Predictive coding: dlPFC → V1 (top-down attentional feedback)
    // Gated by config: doesn't help in small 3x3 visual field (5 rounds of tuning confirmed).
    // Infrastructure ready for larger environments.
    if (config_.enable_predictive_coding) {
        engine_.add_projection("dlPFC", "V1", 3);
    }

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
    // DA传递用neuromodulatory broadcast (体积传递), 不走SpikeBus
    // VTA→BG投射保留用于其他信号, DA level直接同步
    // bg_ptr->set_da_source_region(vta_ptr->region_id()); // DISABLED: use direct DA broadcast

    // --- Cache region pointers ---
    lgn_   = engine_.find_region("LGN");
    v1_    = dynamic_cast<CorticalRegion*>(engine_.find_region("V1"));
    dlpfc_ = dynamic_cast<CorticalRegion*>(engine_.find_region("dlPFC"));
    m1_    = dynamic_cast<CorticalRegion*>(engine_.find_region("M1"));
    bg_    = dynamic_cast<BasalGanglia*>(engine_.find_region("BG"));
    vta_   = dynamic_cast<VTA_DA*>(engine_.find_region("VTA"));
    hipp_  = dynamic_cast<Hippocampus*>(engine_.find_region("Hippocampus"));

    // --- Register topographic V1→dlPFC mapping (retinotopic) ---
    // Preserves spatial structure: V1 left-field → dlPFC left zone
    // Without this, neuron_id % L4_size scrambles all spatial information
    if (v1_ && dlpfc_) {
        dlpfc_->add_topographic_input(v1_->region_id(), v1_->n_neurons());
    }

    // --- Register topographic dlPFC→BG mapping (corticostriatal somatotopy) ---
    // dlPFC spatial zone → D1/D2 action subgroup (proportional mapping)
    // Without this, random 20% connectivity scrambles direction information
    // at the "last mile" before action selection
    if (dlpfc_ && bg_) {
        bg_->set_topographic_cortical_source(dlpfc_->region_id(), dlpfc_->n_neurons());
    }

    // --- Enable predictive coding on V1 (if configured) ---
    // dlPFC sends top-down attentional feedback to V1 L2/3 apical.
    // Disabled by default: doesn't help in small 3x3 visual field.
    if (config_.enable_predictive_coding && v1_ && dlpfc_) {
        v1_->enable_predictive_coding();
        v1_->add_feedback_source(dlpfc_->region_id());
        v1_->add_topographic_input(dlpfc_->region_id(), dlpfc_->n_neurons());
    }

    // --- Enable homeostatic plasticity ---
    if (config_.enable_homeostatic) {
        HomeostaticParams hp;
        hp.target_rate = 5.0f;
        hp.eta = 0.001f;
        hp.scale_interval = 100;
        v1_->enable_homeostatic(hp);
        dlpfc_->enable_homeostatic(hp);
        // M1 intentionally excluded: motor cortex driven by exploration noise,
        // homeostatic plasticity would suppress noise-driven firing → agent freezes
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
    // =====================================================================
    // Temporal credit assignment: reward → DA → BG eligibility traces
    //
    // Timeline per agent_step:
    //   Phase A: Inject PREVIOUS reward → run reward_processing_steps
    //            VTA produces DA burst → BG DA-STDP modulates traces from prev action
    //   Phase B: Inject NEW observation → run brain_steps_per_action
    //            Cortex processes visual → BG builds new eligibility traces
    //            M1 L5 accumulates spikes → decode action
    //   Phase C: Act in world → store reward as pending for next step
    // =====================================================================

    // --- Phase A: Process pending reward (from previous action) ---
    if (has_pending_reward_) {
        inject_reward(pending_reward_);
        // Run a few steps so DA can modulate BG eligibility traces
        // DA broadcast: VTA computes DA level, BG reads it directly (volume transmission)
        for (size_t i = 0; i < config_.reward_processing_steps; ++i) {
            bg_->set_da_level(vta_->da_output());  // Neuromodulatory broadcast
            engine_.step();
        }
        has_pending_reward_ = false;
    }

    // --- Phase B: Observe + decide ---

    // Begin recording episode for awake SWR replay
    if (config_.enable_replay) {
        replay_buffer_.begin_episode();
    }

    // B1. Inject new visual observation
    inject_observation();

    // =====================================================================
    // Biologically correct motor architecture:
    //
    //   dlPFC → BG D1/D2 (corticostriatal: sensory context)
    //   D1 → GPi(inhibit) → MotorThal(disinhibit) → M1 L5 (Go)
    //   D2 → GPe → GPi(disinhibit) → MotorThal(inhibit) (NoGo)
    //   M1 L5 = sole motor output (action decoded here)
    //
    //   Exploration = diffuse cortical spontaneous activity (all M1 L5)
    //   BG influence = D1 subgroup firing → bias corresponding M1 L5 group
    //                  (simplified proxy for BG→MotorThal→M1 disinhibition)
    //   Learning naturally shifts M1 firing from noise-driven to BG-biased
    // =====================================================================

    // B2. Setup accumulators
    const auto& col = m1_->column();
    size_t l4_size  = col.l4().size();
    size_t l23_size = col.l23().size();
    size_t l5_size  = col.l5().size();
    size_t l5_offset = l4_size + l23_size;
    std::vector<int> l5_accum(l5_size, 0);

    // BG D1 subgroup parameters (for bias injection into M1)
    size_t d1_size = bg_->d1().size();
    size_t d1_group = (d1_size >= 4) ? d1_size / 4 : d1_size;
    float bg_to_m1_gain = 8.0f;  // BG Go signal → M1 drive strength

    // Motor exploration: cortical attractor dynamics + NE-modulated arousal
    // Biology: LC-NE system scales exploration based on learning progress
    //   Getting food regularly → low NE → exploit learned policy
    //   Not finding food → high NE → explore more
    //   Floor at 70% ensures M1 always fires (attractor_drive ≥ 0.7*55*0.6 = 23)
    float noise_scale = 1.0f;
    if (agent_step_count_ > 500 && reward_history_.size() > 0) {
        int food_count = 0;
        int total = static_cast<int>(std::min(history_idx_, reward_history_.size()));
        for (int k = 0; k < total; ++k) {
            if (food_history_[k]) food_count++;
        }
        float food_rate = static_cast<float>(food_count) / static_cast<float>(std::max(total, 1));
        // More food found → reduce exploration (exploit). Scale: 1.0→0.7 as food_rate 0→0.1
        noise_scale = std::max(0.7f, 1.0f - food_rate * 3.0f);
    }
    float effective_noise = config_.exploration_noise * noise_scale;

    int attractor_group = -1;
    if (effective_noise > 0.0f) {
        std::uniform_int_distribution<int> group_pick(0, 3);
        attractor_group = group_pick(motor_rng_);
    }
    float attractor_drive = effective_noise * 0.6f;   // Strong: selected direction
    float attractor_jitter = effective_noise * 0.4f;  // Variability
    float background_drive = effective_noise * 0.1f;  // Weak: other directions

    for (size_t i = 0; i < config_.brain_steps_per_action; ++i) {
        // Inject observation EVERY brain step to provide sustained drive to LGN.
        // Thalamic relay neurons (tau_m=20, threshold=-50, rest=-65) need ~7 steps
        // of sustained I=45 current to charge from rest to threshold.
        // Previous: inject every 3 steps → single-pulse ΔV=2.25mV, never fires.
        inject_observation();

        // DA neuromodulatory broadcast: VTA → BG (volume transmission, every step)
        bg_->set_da_level(vta_->da_output());

        // (1) M1 L5 exploration: attractor direction + background activity
        //     Attractor group: strong drive (cortical attractor settled on this direction)
        //     Other groups: weak background (cortical spontaneous activity)
        //     BG bias can override attractor as learning progresses
        auto& l5 = m1_->column().l5();
        if (l5_size >= 4) {
            size_t l5_group = l5_size / 4;
            std::uniform_real_distribution<float> jitter(-attractor_jitter, attractor_jitter);
            for (int g = 0; g < 4; ++g) {
                size_t m1_start = g * l5_group;
                size_t m1_end = (g < 3) ? (g + 1) * l5_group : l5_size;
                float drive = (g == attractor_group) ? attractor_drive : background_drive;
                for (size_t j = m1_start; j < m1_end; ++j) {
                    float current = drive + jitter(motor_rng_);
                    if (current > 0.0f) l5.inject_basal(j, current);
                }
            }
        }

        // (2) BG D1 → M1 L5 bias: simplified BG→MotorThal→M1 disinhibition
        //     D1 subgroup fires → corresponding M1 L5 group gets extra drive
        //     As DA-STDP changes D1 weights, specific M1 groups get stronger bias
        //     → learned actions emerge from BG modulation of M1
        if (d1_size >= 4 && l5_size >= 4) {
            const auto& d1_fired = bg_->d1().fired();
            size_t l5_group = l5_size / 4;
            for (int g = 0; g < 4; ++g) {
                size_t d1_start = g * d1_group;
                size_t d1_end = (g < 3) ? (g + 1) * d1_group : d1_size;
                int d1_fires = 0;
                for (size_t k = d1_start; k < d1_end; ++k) {
                    if (d1_fired[k]) d1_fires++;
                }
                if (d1_fires > 0) {
                    float bias = static_cast<float>(d1_fires) * bg_to_m1_gain;
                    size_t m1_start = g * l5_group;
                    size_t m1_end = (g < 3) ? (g + 1) * l5_group : l5_size;
                    for (size_t j = m1_start; j < m1_end; ++j) {
                        l5.inject_basal(j, bias);
                    }
                }
            }
        }

        engine_.step();

        // Capture dlPFC spike pattern for awake SWR replay buffer
        if (config_.enable_replay) {
            capture_dlpfc_spikes(attractor_group);
        }

        // Accumulate M1 L5 fired state (sole motor output)
        const auto& m1_fired = m1_->fired();
        for (size_t j = 0; j < l5_size && (l5_offset + j) < m1_fired.size(); ++j) {
            l5_accum[j] += m1_fired[l5_offset + j];
        }

        // Motor efference copy: mark current exploration direction in BG sensory slots.
        // Combined with topographic V1→dlPFC→BG visual context, enables DA-STDP to
        // learn joint "visual context + action → reward" associations.
        // Only in late brain steps (i>=10) after visual pipeline establishes context.
        if (i >= 10 && attractor_group >= 0) {
            bg_->mark_motor_efference(attractor_group);
        }
    }

    // B3. Decode action from M1 L5 only (biological: M1 is the motor output)
    Action action = decode_m1_action(l5_accum);

    // --- Phase C: Act in world + store reward ---
    StepResult result = world_.act(action);

    // Store reward as pending (will be processed at START of next agent_step)
    // Only trigger Phase A for significant rewards (food/danger), not step penalties
    pending_reward_ = result.reward * config_.reward_scale;
    has_pending_reward_ = (std::abs(result.reward) > 0.05f);

    // End episode recording and trigger awake SWR replay for significant rewards
    if (config_.enable_replay) {
        replay_buffer_.end_episode(result.reward, static_cast<int>(action));
        // Only replay POSITIVE rewards (food found).
        // Biology: awake SWR preferentially replays reward-associated trajectories.
        // Negative experiences are learned online via DA dip, not amplified by replay.
        if (result.reward > 0.05f && agent_step_count_ >= 10) {
            run_awake_replay(result.reward);
        }
    }

    // Update state
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
    auto obs = world_.observe();  // NxN patch (N = 2*vision_radius+1)
    visual_encoder_.encode_and_inject(obs, lgn_);
}
// =============================================================================
// Action decoding: M1 L5 fired → winner-take-all over 4 directions
// =============================================================================

Action ClosedLoopAgent::decode_m1_action(const std::vector<int>& l5_accum) const {
    // Biological: action decoded ONLY from M1 L5 (sole motor output)
    // BG influence reaches M1 through MotorThal pathway (bias injection above)
    // M1 L5 divided into 4 groups: UP / DOWN / LEFT / RIGHT

    size_t l5_size = l5_accum.size();
    if (l5_size < 4) return Action::STAY;

    float scores[4] = {0, 0, 0, 0};
    size_t group_size = l5_size / 4;
    for (size_t g = 0; g < 4; ++g) {
        size_t start = g * group_size;
        size_t end = (g < 3) ? (g + 1) * group_size : l5_size;
        for (size_t i = start; i < end; ++i) {
            scores[g] += static_cast<float>(l5_accum[i]);
        }
    }

    // Winner-take-all
    float max_score = *std::max_element(scores, scores + 4);
    if (max_score <= 0.0f) return Action::STAY;

    for (int g = 0; g < 4; ++g) {
        if (scores[g] >= max_score - 0.001f) {
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
        vta_->inject_reward(reward);
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

// =============================================================================
// Awake SWR Replay: capture cortical spikes + replay on reward
// =============================================================================

void ClosedLoopAgent::capture_dlpfc_spikes(int action_group) {
    if (!dlpfc_ || !bg_) return;

    // Capture dlPFC fired neurons as SpikeEvents (for BG replay)
    const auto& fired = dlpfc_->fired();
    const auto& stypes = dlpfc_->spike_type();
    uint32_t rid = dlpfc_->region_id();

    std::vector<SpikeEvent> cortical_events;
    for (size_t i = 0; i < fired.size(); ++i) {
        if (fired[i]) {
            SpikeEvent evt;
            evt.region_id  = rid;
            evt.neuron_id  = static_cast<uint32_t>(i);
            evt.spike_type = stypes[i];
            evt.timestamp  = 0;
            cortical_events.push_back(evt);
        }
    }

    // Also capture V1 fired neurons (for cortical consolidation)
    // Biology: SWR replay reactivates both sensory (V1) and association (dlPFC)
    // cortex representations, strengthening V1→dlPFC feature pathways
    std::vector<SpikeEvent> sensory_events;
    if (v1_) {
        const auto& v1_fired = v1_->fired();
        const auto& v1_stypes = v1_->spike_type();
        uint32_t v1_rid = v1_->region_id();
        for (size_t i = 0; i < v1_fired.size(); ++i) {
            if (v1_fired[i]) {
                SpikeEvent evt;
                evt.region_id  = v1_rid;
                evt.neuron_id  = static_cast<uint32_t>(i);
                evt.spike_type = v1_stypes[i];
                evt.timestamp  = 0;
                sensory_events.push_back(evt);
            }
        }
    }

    replay_buffer_.record_step(cortical_events, action_group, sensory_events);
}

void ClosedLoopAgent::run_awake_replay(float reward) {
    // Awake Sharp-Wave Ripple replay — memory consolidation:
    //
    //   When a new reward event occurs, replay OLDER successful episodes
    //   (NOT the current one — that's learned normally via Phase A).
    //   This combats weight decay on previously learned associations,
    //   preventing the agent from "forgetting" old strategies.
    //
    //   Biology: awake SWR preferentially replays remote reward-associated
    //   sequences for consolidation, not just the immediate experience.
    //   (Foster & Wilson 2006, Jadhav et al. 2012)

    if (!bg_ || !vta_ || config_.replay_passes <= 0) return;
    if (replay_buffer_.size() < 2) return;  // Need at least 1 old episode

    // Collect older episodes with positive reward (skip most recent = current)
    auto recent = replay_buffer_.recent(std::min(replay_buffer_.size(), (size_t)10));
    std::vector<const Episode*> replay_candidates;
    for (size_t i = 1; i < recent.size(); ++i) {  // Skip index 0 = current episode
        if (recent[i]->reward > 0.05f && !recent[i]->steps.empty()) {
            replay_candidates.push_back(recent[i]);
        }
    }
    if (replay_candidates.empty()) return;

    // Save current BG state
    float saved_da = bg_->da_level();

    // Replay DA: above baseline (positive consolidation signal)
    float da_baseline = 0.3f;
    float da_replay_level = std::clamp(
        da_baseline + reward * config_.replay_da_scale, 0.0f, 1.0f);

    // Enter replay mode (suppresses weight decay in DA-STDP)
    bg_->set_replay_mode(true);

    // Replay each candidate episode once
    size_t n_replay = std::min(replay_candidates.size(), (size_t)config_.replay_passes);
    for (size_t ep_idx = 0; ep_idx < n_replay; ++ep_idx) {
        const Episode& ep = *replay_candidates[ep_idx];

        bg_->set_da_level(da_replay_level);

        // Replay later brain steps (i>=8) where visual context is established
        size_t start_step = (ep.steps.size() > 8) ? 8 : 0;
        for (size_t i = start_step; i < ep.steps.size(); ++i) {
            const SpikeSnapshot& snap = ep.steps[i];

            // --- BG consolidation (existing): dlPFC spikes → BG DA-STDP ---
            if (!snap.cortical_events.empty()) {
                bg_->receive_spikes(snap.cortical_events);
            }
            if (snap.action_group >= 0) {
                bg_->mark_motor_efference(snap.action_group);
            }
            bg_->replay_learning_step(0, 1.0f);

            // --- Cortical consolidation: DEFERRED ---
            // V1 spikes are recorded in sensory_events (infrastructure ready)
            // but NOT injected into dlPFC during awake replay because:
            // 1. Full replay_cortical_step → LTD dominates (L4 fires, L23 doesn't)
            // 2. PSP priming → residual contaminates next real visual input
            // Proper cortical consolidation requires NREM sleep replay where
            // the entire brain state is controlled (slow-wave up/down states).
            // For awake SWR, BG-only replay is biologically correct:
            // awake SWR primarily consolidates striatal action values.
        }
    }

    // Exit replay mode and restore DA level
    bg_->set_replay_mode(false);
    bg_->set_da_level(saved_da);
}

} // namespace wuyun
