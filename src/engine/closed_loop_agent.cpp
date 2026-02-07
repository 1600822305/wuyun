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
    vcfg.gain = config_.lgn_gain;
    vcfg.baseline = config_.lgn_baseline;
    vcfg.noise_amp = config_.lgn_noise_amp;
    visual_encoder_ = VisualInput(vcfg);
}

// =============================================================================
// Build the brain circuit for closed-loop control
// =============================================================================

void ClosedLoopAgent::build_brain() {
    int s = config_.brain_scale;

    // ===================================================================
    // INFORMATION-DRIVEN NEURON ALLOCATION
    // Each neuron has a clear information-theoretic purpose.
    // n_pix pixels → 4 actions. No wasted neurons.
    // Architecture (6-layer column, STDP, DA-STDP) is unchanged.
    // ===================================================================

    size_t n_pix = config_.vision_width * config_.vision_height;  // 25 for 5×5
    size_t n_act = 4;  // UP/DOWN/LEFT/RIGHT

    // Helper: create CorticalRegion from total neuron count N
    // Distributes N across 7 populations maintaining biological ratios
    auto add_ctx = [&](const std::string& name, size_t N, bool stdp = false) {
        ColumnConfig c;
        c.n_l4_stellate    = std::max<size_t>(2, N * 25 / 100) * s;  // 25%
        c.n_l23_pyramidal  = std::max<size_t>(3, N * 35 / 100) * s;  // 35%
        c.n_l5_pyramidal   = std::max<size_t>(2, N * 20 / 100) * s;  // 20%
        c.n_l6_pyramidal   = std::max<size_t>(2, N * 12 / 100) * s;  // 12%
        c.n_pv_basket      = std::max<size_t>(2, N * 4 / 100) * s;   // 4%
        c.n_sst_martinotti = std::max<size_t>(1, N * 3 / 100) * s;   // 3%
        c.n_vip            = std::max<size_t>(1, N * 1 / 100) * s;   // 1%
        // Higher connection probability for small networks (ensures sufficient connections)
        if (N <= 30) {
            c.p_l4_to_l23 = 0.5f;   // was 0.2
            c.p_l23_to_l5 = 0.5f;
            c.p_l5_to_l6  = 0.5f;
            c.p_l6_to_l4  = 0.5f;
            c.p_l23_recurrent = 0.3f;
        }
        if (stdp) {
            c.stdp_enabled = true;
            c.stdp_a_plus  = config_.cortical_stdp_a_plus;
            c.stdp_a_minus = config_.cortical_stdp_a_minus;
            c.stdp_w_max   = config_.cortical_stdp_w_max;
        }
        engine_.add_region(std::make_unique<CorticalRegion>(name, c));
    };

    bool ctx_stdp = config_.enable_cortical_stdp && !config_.fast_eval;

    // --- LGN: 1 relay per pixel (ON/OFF encoded in gain, not neuron count) ---
    ThalamicConfig lgn_cfg;
    lgn_cfg.name = "LGN";
    lgn_cfg.n_relay = n_pix * s;                               // 25 for 5×5
    lgn_cfg.n_trn   = std::max<size_t>(3, n_pix / 3) * s;     // 8
    engine_.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    // --- Visual hierarchy: each level compresses information ---
    add_ctx("V1",  n_pix,                             ctx_stdp);  // 25: 1 per pixel
    add_ctx("V2",  std::max<size_t>(8, n_pix*6/10),   ctx_stdp);  // 15: texture combinations
    add_ctx("V4",  std::max<size_t>(6, n_pix*3/10),   ctx_stdp);  // 8: shape features
    add_ctx("IT",  std::max<size_t>(8, n_act * 2),     false);     // 8: object categories (stable)

    // --- Decision + motor ---
    add_ctx("dlPFC", n_act * 3,  false);   // 12: 4 directions × 3 (approach/avoid/neutral)
    add_ctx("M1",    n_act * 5,  false);   // 20: need ≥4 L5 neurons for winner-take-all

    // --- Basal Ganglia: 4 Go + 4 NoGo = minimal action selection ---
    BasalGangliaConfig bg_cfg;
    bg_cfg.name = "BG";
    bg_cfg.n_d1_msn = n_act * 2 * s;  // 8: 4 directions × 2 Go neurons
    bg_cfg.n_d2_msn = n_act * 2 * s;  // 8: 4 directions × 2 NoGo neurons
    bg_cfg.n_gpi = n_act * s;         // 4: 1 per action
    bg_cfg.n_gpe = n_act * s;         // 4
    bg_cfg.n_stn = std::max<size_t>(4, n_act) * s;  // 4
    bg_cfg.p_ctx_to_d1 = 0.5f;   // Higher connectivity for small network
    bg_cfg.p_ctx_to_d2 = 0.5f;
    bg_cfg.p_d1_to_gpi = 0.6f;
    bg_cfg.p_d2_to_gpe = 0.6f;
    bg_cfg.da_stdp_enabled = config_.enable_da_stdp;
    bg_cfg.da_stdp_lr = config_.da_stdp_lr;
    bg_cfg.synaptic_consolidation = config_.enable_synaptic_consolidation;
    engine_.add_region(std::make_unique<BasalGanglia>(bg_cfg));

    // --- Motor thalamus ---
    ThalamicConfig mthal_cfg;
    mthal_cfg.name = "MotorThal";
    mthal_cfg.n_relay = n_act * 2 * s;  // 8
    mthal_cfg.n_trn   = std::max<size_t>(2, n_act / 2) * s;  // 2
    engine_.add_region(std::make_unique<ThalamicRelay>(mthal_cfg));

    // --- VTA: dopamine ---
    VTAConfig vta_cfg;
    vta_cfg.n_da_neurons = std::max<size_t>(4, n_act) * s;  // 4
    engine_.add_region(std::make_unique<VTA_DA>(vta_cfg));

    // --- LHb: negative RPE center ---
    // Biology: LHb encodes negative prediction errors and aversive stimuli
    //   LHb → RMTg(GABA) → VTA: inhibits DA release → DA pause → D2 NoGo learning
    //   Essential for learning to AVOID danger (Matsumoto & Hikosaka 2007)
    // --- LHb: negative RPE (minimal: 4 neurons) ---
    if (config_.enable_lhb) {
        LHbConfig lhb_cfg;
        lhb_cfg.n_neurons = std::max<size_t>(4, n_act) * s;
        lhb_cfg.punishment_gain = config_.lhb_punishment_gain;
        lhb_cfg.frustration_gain = config_.lhb_frustration_gain;
        engine_.add_region(std::make_unique<LateralHabenula>(lhb_cfg));
    }

    // --- Amygdala: fear conditioning (minimal: 4+6+3+2 = 15 neurons) ---
    if (config_.enable_amygdala) {
        AmygdalaConfig amyg_cfg;
        amyg_cfg.n_la  = std::max<size_t>(4, n_act) * s;
        amyg_cfg.n_bla = std::max<size_t>(6, n_act + 2) * s;
        amyg_cfg.n_cea = std::max<size_t>(3, n_act - 1) * s;
        amyg_cfg.n_itc = 2 * s;
        amyg_cfg.fear_stdp_enabled = true;
        engine_.add_region(std::make_unique<Amygdala>(amyg_cfg));
    }

    // --- Hippocampus: spatial memory (minimal: compressed) ---
    if (!config_.fast_eval) {
        HippocampusConfig hipp_cfg;
        hipp_cfg.n_ec  = std::max<size_t>(6, n_act + 2) * s;
        hipp_cfg.n_dg  = std::max<size_t>(10, n_pix / 3) * s;
        hipp_cfg.n_ca3 = std::max<size_t>(6, n_act + 2) * s;
        hipp_cfg.n_ca1 = std::max<size_t>(6, n_act + 2) * s;
        hipp_cfg.n_sub = std::max<size_t>(3, n_act - 1) * s;
        hipp_cfg.ca3_stdp_enabled = true;
        engine_.add_region(std::make_unique<Hippocampus>(hipp_cfg));
    }

    // --- v30: Cerebellum forward model (Yoshida 2025: CB-BG synergistic RL) ---
    // Predicts sensory consequences of actions, CF error corrects predictions
    if (config_.enable_cerebellum) {
        CerebellumConfig cb_cfg;
        cb_cfg.n_granule  = std::max<size_t>(12, n_act * 3) * s;  // 12: input expansion
        cb_cfg.n_purkinje = n_act * s;                             // 4: one per action
        cb_cfg.n_dcn      = n_act * s;                             // 4: prediction output
        cb_cfg.n_mli      = 2 * s;                                 // 2: feedforward inhibition
        cb_cfg.n_golgi    = 2 * s;                                 // 2: feedback inhibition
        // Higher connectivity for small network
        cb_cfg.p_mf_to_grc = 0.4f;
        cb_cfg.p_pf_to_pc  = 0.6f;
        cb_cfg.p_pc_to_dcn = 0.6f;
        engine_.add_region(std::make_unique<Cerebellum>(cb_cfg));
    }

    // --- Projections (core closed-loop circuit) ---

    // Visual hierarchy (ventral "what" pathway): LGN → V1 → V2 → V4 → IT → dlPFC
    // Each level extracts increasingly abstract/invariant features
    // IT provides position-invariant "food"/"danger" representations to dlPFC
    engine_.add_projection("LGN", "V1", 2);
    engine_.add_projection("V1", "V2", 2);       // edges → textures
    engine_.add_projection("V2", "V4", 2);       // textures → shapes
    engine_.add_projection("V4", "IT", 2);       // shapes → objects (invariant)
    engine_.add_projection("IT", "dlPFC", 2);    // objects → decisions

    // Feedback projections (top-down prediction, slower)
    engine_.add_projection("V2", "V1", 3);
    engine_.add_projection("V4", "V2", 3);
    engine_.add_projection("IT", "V4", 3);

    // Decision → action: dlPFC → BG → MotorThal → M1
    engine_.add_projection("dlPFC", "BG", 2);
    engine_.add_projection("BG", "MotorThal", 2);
    engine_.add_projection("MotorThal", "M1", 2);

    // Feedback: M1 → dlPFC (efference copy)
    engine_.add_projection("M1", "dlPFC", 3);

    // Predictive coding: dlPFC → IT (top-down attentional feedback)
    // With visual hierarchy, dlPFC feeds back to IT (not V1 directly)
    // IT propagates predictions down through V4→V2→V1 via existing feedback projections
    if (config_.enable_predictive_coding) {
        engine_.add_projection("dlPFC", "IT", 3);
    }

    // Memory: dlPFC + IT → Hippocampus (encode context + object identity)
    if (!config_.fast_eval) {
        engine_.add_projection("dlPFC", "Hippocampus", 3);
        engine_.add_projection("IT", "Hippocampus", 3);   // v24: V1→IT, invariant object memory
        // Hippocampus → dlPFC (memory retrieval → decision bias)
        engine_.add_projection("Hippocampus", "dlPFC", 3);
    }

    // VTA DA → BG (reward signal)
    engine_.add_projection("VTA", "BG", 2);

    // LHb → VTA (inhibitory, via RMTg GABA interneurons)
    // Biology: LHb glutamatergic output excites RMTg GABAergic neurons
    //          which then inhibit VTA DA neurons. Simplified as direct projection.
    if (config_.enable_lhb) {
        engine_.add_projection("LHb", "VTA", 2);
    }

    // Amygdala fear circuit projections
    if (config_.enable_amygdala) {
        // Two fear pathways (LeDoux 1996):
        //   Fast: V1 → Amygdala La (crude, fast, subcortical-like)
        //   Slow: IT → Amygdala La (refined, invariant, cortical)
        engine_.add_projection("V1", "Amygdala", 2);   // Fast: raw visual → fear (crude but quick)
        engine_.add_projection("IT", "Amygdala", 3);   // Slow: invariant object → fear (precise)
        // v33: Amygdala→VTA SpikeBus投射已移除 (错误接线)
        // 原问题: SpikeBus把所有杏仁核脉冲(LA+BLA+CeA+ITC=15)当兴奋性送给VTA
        //         导致DA不降反升，与生物学CeA→RMTg(GABA)→VTA(抑制)完全相反
        // 修复: CeA→VTA抑制功能通过inject_lhb_inhibition(cea_drive)正确实现
        // engine_.add_projection("Amygdala", "VTA", 2);  // REMOVED: 兴奋性SpikeBus ≠ 抑制性通路
        // v33: Amygdala→LHb SpikeBus投射已移除 (同一类错误接线)
        // 原问题: 15个杏仁核脉冲全部兴奋LHb → LHb持续抑制VTA → DA慢性压制
        // 修复: CeA→VTA功能通过inject_lhb_inhibition(cea_drive)正确实现
        // if (config_.enable_lhb) {
        //     engine_.add_projection("Amygdala", "LHb", 2);
        // }
    }

    // --- v30: Cerebellum projections ---
    if (config_.enable_cerebellum) {
        // M1 → Cerebellum (mossy fiber: efference copy of motor commands)
        engine_.add_projection("M1", "Cerebellum", 1);
        // V1 → Cerebellum (mossy fiber: visual context for prediction)
        engine_.add_projection("V1", "Cerebellum", 1);
        // Cerebellum DCN → MotorThalamus (prediction-corrected motor signal)
        engine_.add_projection("Cerebellum", "MotorThal", 1);
        // Cerebellum DCN → BG (prediction confidence → modulate action selection)
        engine_.add_projection("Cerebellum", "BG", 1);
    }

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
    v2_    = dynamic_cast<CorticalRegion*>(engine_.find_region("V2"));
    v4_    = dynamic_cast<CorticalRegion*>(engine_.find_region("V4"));
    it_    = dynamic_cast<CorticalRegion*>(engine_.find_region("IT"));
    dlpfc_ = dynamic_cast<CorticalRegion*>(engine_.find_region("dlPFC"));
    m1_    = dynamic_cast<CorticalRegion*>(engine_.find_region("M1"));
    bg_    = dynamic_cast<BasalGanglia*>(engine_.find_region("BG"));
    vta_   = dynamic_cast<VTA_DA*>(engine_.find_region("VTA"));
    hipp_  = dynamic_cast<Hippocampus*>(engine_.find_region("Hippocampus"));
    lhb_   = dynamic_cast<LateralHabenula*>(engine_.find_region("LHb"));
    amyg_  = dynamic_cast<Amygdala*>(engine_.find_region("Amygdala"));
    cb_    = dynamic_cast<Cerebellum*>(engine_.find_region("Cerebellum"));

    // --- Topographic mappings through visual hierarchy ---
    // V1→V2: retinotopic (preserve spatial layout)
    if (v1_ && v2_) v2_->add_topographic_input(v1_->region_id(), v1_->n_neurons());
    // V2→V4: partial retinotopy
    if (v2_ && v4_) v4_->add_topographic_input(v2_->region_id(), v2_->n_neurons());
    // V4→IT: coarse spatial mapping (position invariance emerges through STDP)
    if (v4_ && it_) it_->add_topographic_input(v4_->region_id(), v4_->n_neurons());
    // IT→dlPFC: object identity → decision (replaces V1→dlPFC)
    if (it_ && dlpfc_) dlpfc_->add_topographic_input(it_->region_id(), it_->n_neurons());

    // --- Register topographic dlPFC→BG mapping (corticostriatal somatotopy) ---
    if (dlpfc_ && bg_) {
        bg_->set_topographic_cortical_source(dlpfc_->region_id(), dlpfc_->n_neurons());
    }

    // --- Enable predictive coding through visual hierarchy ---
    // With V2/V4/IT, predictions flow top-down: dlPFC→IT→V4→V2→V1
    if (config_.enable_predictive_coding) {
        // Enable PC on each visual area with feedback from the level above
        if (v1_ && v2_) {
            v1_->enable_predictive_coding();
            v1_->add_feedback_source(v2_->region_id());
        }
        if (v2_ && v4_) {
            v2_->enable_predictive_coding();
            v2_->add_feedback_source(v4_->region_id());
        }
        if (v4_ && it_) {
            v4_->enable_predictive_coding();
            v4_->add_feedback_source(it_->region_id());
        }
        if (it_ && dlpfc_) {
            it_->enable_predictive_coding();
            it_->add_feedback_source(dlpfc_->region_id());
        }
    }

    // --- Enable homeostatic plasticity ---
    if (config_.enable_homeostatic) {
        HomeostaticParams hp;
        hp.target_rate = config_.homeostatic_target_rate;
        hp.eta = config_.homeostatic_eta;
        hp.scale_interval = 100;
        v1_->enable_homeostatic(hp);
        if (v2_) v2_->enable_homeostatic(hp);
        if (v4_) v4_->enable_homeostatic(hp);
        if (it_) it_->enable_homeostatic(hp);
        dlpfc_->enable_homeostatic(hp);
        // M1 intentionally excluded: motor cortex driven by exploration noise
        if (hipp_) hipp_->enable_homeostatic(hp);
    }

    // --- v27: Enable predictive coding learning on visual hierarchy ---
    // L6 learns to predict L2/3, L4→L2/3 STDP becomes error-gated
    if (config_.enable_predictive_learning && config_.enable_cortical_stdp) {
        if (v1_) v1_->column().enable_predictive_learning();
        if (v2_) v2_->column().enable_predictive_learning();
        if (v4_) v4_->column().enable_predictive_learning();
        // IT intentionally excluded: NO STDP (representation stability)
    }

    // --- v26: Tonic drive for visual hierarchy (Pulvinar → V2/V4/IT) ---
    // Biology: Pulvinar thalamic nucleus provides sustained activation to extrastriate
    // visual areas, preventing signal extinction through the hierarchy.
    // Without this, V1=809 → V2=246 → V4=35 → IT=2 (signal dies)
    if (v2_) v2_->set_tonic_drive(3.0f);
    if (v4_) v4_->set_tonic_drive(2.5f);
    if (it_) it_->set_tonic_drive(2.0f);

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

    // --- Sleep consolidation: periodic offline replay ---
    // Biology: after sustained waking, NREM sleep replays recent experiences
    // via hippocampal SWR, consolidating both positive and negative memories
    // in BG (striatal action values). No environment interaction during sleep.
    // (Diekelmann & Born 2010: sleep for memory consolidation)
    if (config_.enable_sleep_consolidation && config_.wake_steps_before_sleep > 0) {
        ++wake_step_counter_;
        if (wake_step_counter_ >= config_.wake_steps_before_sleep) {
            run_sleep_consolidation();
            wake_step_counter_ = 0;
        }
    }

    // --- v27: Developmental period — no reward learning, just visual STDP ---
    // Biology: critical period for visual feature self-organization
    bool in_dev_period = (config_.dev_period_steps > 0 &&
                          static_cast<size_t>(agent_step_count_) < config_.dev_period_steps);
    if (in_dev_period) {
        has_pending_reward_ = false;  // Suppress reward processing during development
    }

    // --- Phase A: Process pending reward (from previous action) ---
    if (has_pending_reward_) {
        inject_reward(pending_reward_);

        // Hippocampal reward tagging: encode current location with reward value
        // Biology: VTA DA → hippocampus enhances LTP at active CA3 synapses
        // (Lisman & Grace 2005: DA gates hippocampal memory formation)
        if (hipp_ && std::abs(pending_reward_) > 0.01f) {
            hipp_->inject_reward_tag(std::abs(pending_reward_));
        }

        // Amygdala US injection: danger → BLA activation → La→BLA STDP
        // Biology: pain/danger (US) directly activates BLA. When paired with
        // visual CS (already flowing via V1→La SpikeBus), STDP strengthens
        // CS→BLA association. One trial = fear memory established.
        // (LeDoux 2000: one-shot fear conditioning)
        if (amyg_ && pending_reward_ < -0.01f) {
            float us_mag = -pending_reward_ * config_.amyg_us_gain;
            amyg_->inject_us(us_mag);
        }

        // v32: LHb NO LONGER receives direct punishment (was double-counting with VTA RPE)
        // Biology: LHb encodes frustrative non-reward (expected food not received),
        // NOT direct punishment. Direct punishment is handled by VTA negative RPE.
        // Previous bug: same pending_reward_ fed both VTA RPE AND LHb → 2× DA suppression
        // inject_frustration() below handles the correct LHb function.

        // Frustrative non-reward: expected reward didn't arrive
        // Biology: when food is expected (high food_rate) but not received,
        //          LHb activates to signal "worse than expected" (Bromberg-Martin 2010)
        if (lhb_ && pending_reward_ < 0.01f && expected_reward_level_ > 0.05f) {
            float frustration = expected_reward_level_ * 0.3f;  // Mild frustration signal
            lhb_->inject_frustration(frustration);
        }

        // v26: ACh-gated visual STDP (Froemke et al. 2007)
        // Biology: NBM ACh burst during salient events → visual cortex STDP enhanced
        // Effect: V2/V4 learn "what food/danger looks like" faster after reward events
        float ach_boost = 1.0f + std::abs(pending_reward_) * 0.5f;  // v26: gentler ACh (reward ±1.5 → gain 1.75)
        if (v1_)  v1_->column().set_ach_stdp_gain(ach_boost);
        if (v2_)  v2_->column().set_ach_stdp_gain(ach_boost);
        if (v4_)  v4_->column().set_ach_stdp_gain(ach_boost);
        // IT intentionally excluded (NO STDP, representation stability)

        // Run a few steps so DA can modulate BG eligibility traces
        // DA broadcast: VTA computes DA level, BG reads it directly (volume transmission)
        for (size_t i = 0; i < config_.reward_processing_steps; ++i) {
            // LHb → VTA inhibition: direct neuromodulatory broadcast
            // (supplements SpikeBus projection with immediate DA level effect)
            if (lhb_) {
                vta_->inject_lhb_inhibition(lhb_->vta_inhibition());
            }
            bg_->set_da_level(vta_->da_output());  // Neuromodulatory broadcast
            engine_.step();
        }
        has_pending_reward_ = false;

        // Reset ACh STDP boost after reward processing
        if (v1_)  v1_->column().set_ach_stdp_gain(1.0f);
        if (v2_)  v2_->column().set_ach_stdp_gain(1.0f);
        if (v4_)  v4_->column().set_ach_stdp_gain(1.0f);
    }

    // --- Phase B: Observe + decide ---

    // Begin recording episode for awake SWR replay
    if (config_.enable_replay) {
        replay_buffer_.begin_episode();
    }

    // B1. Inject new visual observation
    inject_observation();

    // B1b. Inject spatial position to hippocampus (grid cell activation)
    // Biology: EC grid cells encode agent position → DG → CA3 place cells
    // This creates a position-dependent activation pattern that CA3 stores via STDP
    if (hipp_) {
        hipp_->inject_spatial_context(
            world_.agent_x(), world_.agent_y(),
            static_cast<int>(world_.width()), static_cast<int>(world_.height()));
    }

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
    float bg_to_m1_gain = config_.bg_to_m1_gain;

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
        noise_scale = std::max(config_.ne_floor, 1.0f - food_rate * config_.ne_food_scale);
    }
    float effective_noise = config_.exploration_noise * noise_scale;

    int attractor_group = -1;
    if (effective_noise > 0.0f) {
        std::uniform_int_distribution<int> group_pick(0, 3);
        attractor_group = group_pick(motor_rng_);
    }
    float attractor_drive = effective_noise * config_.attractor_drive_ratio;
    float attractor_jitter = effective_noise * (1.0f - config_.attractor_drive_ratio);
    float background_drive = effective_noise * config_.background_drive_ratio;

    for (size_t i = 0; i < config_.brain_steps_per_action; ++i) {
        // Inject observation EVERY brain step to provide sustained drive to LGN.
        // Thalamic relay neurons (tau_m=20, threshold=-50, rest=-65) need ~7 steps
        // of sustained I=45 current to charge from rest to threshold.
        // Previous: inject every 3 steps → single-pulse ΔV=2.25mV, never fires.
        inject_observation();

        // LHb → VTA inhibition broadcast (every brain step during action processing)
        if (lhb_) {
            vta_->inject_lhb_inhibition(lhb_->vta_inhibition());
        }
        // Amygdala CeA → VTA/LHb: fear-driven DA pause
        // Biology: when Amygdala detects threatening visual pattern (learned CS),
        // CeA fires → drives VTA DA pause via RMTg, amplifying avoidance signal.
        // This is the "fast fear" pathway: bypasses slow DA-STDP learning.
        if (amyg_) {
            float cea_drive = amyg_->cea_vta_drive();
            if (cea_drive > 0.01f) {
                vta_->inject_lhb_inhibition(cea_drive);  // CeA → VTA DA轻微抑制
            }
            // v33: 主动消退 — 安全步骤时PFC驱动ITC抑制CeA
            // 生物学: mPFC在安全环境中持续激活ITC(闰细胞)，
            //   ITC(GABA)抑制CeA → 恐惧输出降低 → 恐惧消退
            //   (Milad & Quirk 2002, Phelps et al. 2004)
            // 只在没有pending reward(安全)时驱动消退
            if (!has_pending_reward_ || pending_reward_ > -0.01f) {
                std::vector<float> itc_drive(amyg_->itc().size(), 5.0f);
                amyg_->inject_pfc_to_itc(itc_drive);
            }
        }
        // v30: Cerebellum climbing fiber injection (every brain step)
        // Reward-as-error: unexpected food/danger = prediction failure → CF signal
        // CF drives PF→PC LTD → cerebellum learns to predict action outcomes
        if (cb_ && std::abs(last_reward_) > 0.05f) {
            float cf_error = std::min(1.0f, std::abs(last_reward_));
            cb_->inject_climbing_fiber(cf_error);
        }

        // DA neuromodulatory broadcast: VTA → BG (volume transmission, every step)
        bg_->set_da_level(vta_->da_output());

        // Hippocampal spatial memory → dlPFC: handled via SpikeBus projection
        // (Hippocampus → dlPFC added in build_brain)
        // When agent revisits a familiar location:
        //   EC grid cells fire position-specific pattern →
        //   CA3 pattern completion (if STDP encoded this place) →
        //   CA1 → Sub fires → SpikeBus → dlPFC receives memory signal →
        //   dlPFC→BG pathway naturally biases action selection
        // No direct BG injection needed — the cortical pathway handles it.

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
        // Combined with visual hierarchy IT→dlPFC→BG context, enables DA-STDP to
        // learn joint "visual context + action → reward" associations.
        // v29: i>=10: evolved brain_steps=17, pipeline ~10 steps
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

    // Update expected reward level (slow-moving average of food rate)
    // Biology: striatal tonically active neurons (TANs) track reward expectation
    // Used by LHb for frustrative non-reward detection
    if (agent_step_count_ > 100) {
        float recent_food = food_rate(200);
        expected_reward_level_ = expected_reward_level_ * 0.99f + recent_food * 0.01f;
    }

    // End episode recording and trigger awake SWR replay for significant rewards
    if (config_.enable_replay) {
        replay_buffer_.end_episode(result.reward, static_cast<int>(action));
        // Positive replay: food found → replay old successes (consolidate Go)
        if (result.reward > 0.05f && agent_step_count_ >= 10) {
            run_awake_replay(result.reward);
        }
        // Negative replay: danger hit → replay old failures (consolidate NoGo)
        // Previously disabled (D2 over-strengthening). Now safe with LHb-controlled DA pause.
        // Biology: aversive SWR replay strengthens avoidance memories
        // (Wu et al. 2017, de Lavilléon et al. 2015)
        if (config_.enable_negative_replay && config_.enable_lhb &&
            result.reward < -0.05f && agent_step_count_ >= 200) {
            run_negative_replay(result.reward);
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

void ClosedLoopAgent::run_negative_replay(float reward) {
    // Negative experience replay — LHb-controlled avoidance learning:
    //
    //   When a danger event occurs, replay OLDER danger episodes
    //   with DA level BELOW baseline (LHb-mediated DA pause).
    //   This strengthens D2 NoGo pathway for the action context
    //   that led to danger, teaching the agent to AVOID it.
    //
    //   Key difference from positive replay:
    //   - DA below baseline (not above) → D2 LTP, D1 LTD
    //   - Fewer passes (2 vs 5) to prevent D2 over-strengthening
    //   - Only enabled when LHb is active (provides graded control)
    //
    //   Previous issue without LHb: raw DA dip was uncontrolled,
    //   leading to D2 over-strengthening → behavioral oscillation.
    //   LHb provides biologically realistic graded DA pause.

    if (!bg_ || !vta_ || !lhb_ || config_.negative_replay_passes <= 0) return;
    if (replay_buffer_.size() < 3) return;  // Need sufficient history

    // Collect older episodes with negative reward (skip most recent = current)
    auto recent = replay_buffer_.recent(std::min(replay_buffer_.size(), (size_t)10));
    std::vector<const Episode*> replay_candidates;
    for (size_t i = 1; i < recent.size(); ++i) {  // Skip index 0 = current
        if (recent[i]->reward < -0.05f && !recent[i]->steps.empty()) {
            replay_candidates.push_back(recent[i]);
        }
    }
    if (replay_candidates.empty()) return;

    // Save current BG state
    float saved_da = bg_->da_level();

    // Replay DA: below baseline (LHb-mediated DA pause)
    // Biology: LHb activation during replay drives VTA DA below tonic level
    //   da_replay = baseline - |reward| × scale = 0.3 - 1.0×0.3 = 0.0
    //   Clamped to [0.05, 0.25] to prevent complete DA washout
    float da_baseline = 0.3f;
    float da_dip = std::abs(reward) * config_.negative_replay_da_scale;
    float da_replay_level = std::clamp(da_baseline - da_dip, 0.05f, 0.25f);

    // Enter replay mode (suppresses weight decay)
    bg_->set_replay_mode(true);

    // Replay each candidate episode once
    size_t n_replay = std::min(replay_candidates.size(), (size_t)config_.negative_replay_passes);
    for (size_t ep_idx = 0; ep_idx < n_replay; ++ep_idx) {
        const Episode& ep = *replay_candidates[ep_idx];

        bg_->set_da_level(da_replay_level);

        // Replay later brain steps (i>=8) where visual context is established
        size_t start_step = (ep.steps.size() > 8) ? 8 : 0;
        for (size_t i = start_step; i < ep.steps.size(); ++i) {
            const SpikeSnapshot& snap = ep.steps[i];

            // Inject cortical spikes → BG DA-STDP with low DA
            // D2: Δw = -lr × (da_replay - baseline) × elig
            //     = -lr × (-0.15) × elig = +0.0045 × elig (D2 strengthened)
            // D1: Δw = +lr × (-0.15) × elig = -0.0045 × elig (D1 weakened)
            if (!snap.cortical_events.empty()) {
                bg_->receive_spikes(snap.cortical_events);
            }
            if (snap.action_group >= 0) {
                bg_->mark_motor_efference(snap.action_group);
            }
            bg_->replay_learning_step(0, 1.0f);
        }
    }

    // Exit replay mode and restore DA level
    bg_->set_replay_mode(false);
    bg_->set_da_level(saved_da);
}

void ClosedLoopAgent::run_awake_replay(float reward) {
    // v33: Awake SWR replay with INTERLEAVED positive + negative episodes
    //
    //   When a new reward event occurs, replay OLDER episodes (both positive AND
    //   negative) to consolidate learned associations AND prevent catastrophic
    //   forgetting of avoidance behaviors.
    //
    //   Biology: awake SWR replays both reward and aversive sequences in an
    //   interleaved pattern, maintaining balanced Go/NoGo representations.
    //   (Foster & Wilson 2006, Wu et al. 2017)
    //
    //   Without interleaving: learning to approach food overwrites danger-avoidance
    //   weights, and vice versa → behavioral oscillation = catastrophic forgetting.

    if (!bg_ || !vta_ || config_.replay_passes <= 0) return;
    if (replay_buffer_.size() < 2) return;

    auto recent = replay_buffer_.recent(std::min(replay_buffer_.size(), (size_t)15));

    // Collect positive AND negative candidates (skip index 0 = current)
    std::vector<const Episode*> pos_candidates, neg_candidates;
    for (size_t i = 1; i < recent.size(); ++i) {
        if (recent[i]->steps.empty()) continue;
        if (recent[i]->reward > 0.05f)
            pos_candidates.push_back(recent[i]);
        else if (recent[i]->reward < -0.05f)
            neg_candidates.push_back(recent[i]);
    }
    if (pos_candidates.empty() && neg_candidates.empty()) return;

    float saved_da = bg_->da_level();
    float da_baseline = 0.3f;
    bg_->set_replay_mode(true);

    // Build interleaved replay schedule: alternate positive and negative
    // Positive episodes get more passes (they're the trigger context)
    std::vector<std::pair<const Episode*, float>> schedule;

    // Primary: positive episodes (with high DA)
    float da_pos = std::clamp(da_baseline + reward * config_.replay_da_scale, 0.0f, 1.0f);
    size_t n_pos = std::min(pos_candidates.size(), (size_t)config_.replay_passes);
    for (size_t i = 0; i < n_pos; ++i) {
        schedule.push_back({pos_candidates[i], da_pos});
    }

    // v33: Interleave negative episodes (with low DA) if enabled
    // This maintains avoidance learning while consolidating approach learning
    if (config_.enable_interleaved_replay && config_.enable_lhb && !neg_candidates.empty()) {
        float da_neg = std::clamp(da_baseline - std::abs(reward) * config_.negative_replay_da_scale,
                                   0.05f, 0.25f);
        // Insert 1-2 negative episodes between positive ones
        size_t n_neg = std::min(neg_candidates.size(), (size_t)2);
        for (size_t i = 0; i < n_neg; ++i) {
            // Insert after every 2 positive episodes (interleave)
            size_t insert_pos = std::min((i + 1) * 2, schedule.size());
            schedule.insert(schedule.begin() + insert_pos,
                           {neg_candidates[i], da_neg});
        }
    }

    // Execute interleaved replay schedule
    for (const auto& [ep, da_level] : schedule) {
        bg_->set_da_level(da_level);

        size_t start_step = (ep->steps.size() > 8) ? 8 : 0;
        for (size_t i = start_step; i < ep->steps.size(); ++i) {
            const SpikeSnapshot& snap = ep->steps[i];

            if (!snap.cortical_events.empty()) {
                bg_->receive_spikes(snap.cortical_events);
            }
            if (snap.action_group >= 0) {
                bg_->mark_motor_efference(snap.action_group);
            }
            bg_->replay_learning_step(0, 1.0f);
        }
    }

    bg_->set_replay_mode(false);
    bg_->set_da_level(saved_da);
}

// =============================================================================
// Sleep consolidation: NREM SWR replay for offline memory consolidation
// =============================================================================

void ClosedLoopAgent::run_sleep_consolidation() {
    // v31: Corrected NREM sleep consolidation
    //
    // Biology (2024-2025 Nature):
    //   1. NREM DA is LOW (at or below baseline) → no new BG learning
    //   2. Hippocampus CA3 spontaneously generates SWR → reactivates patterns
    //   3. SWR propagates via SpikeBus to cortex (Sub→dlPFC projection)
    //   4. Cortical STDP in Up state consolidates hippocampal→cortical transfer
    //   5. BG is NOT the target of sleep consolidation (awake replay does that)
    //
    // Previous bugs fixed:
    //   - DA was 0.35 (above baseline) → caused over-consolidation
    //   - Episode buffer was directly injected into BG → bypassed hippocampus
    //   - Hippocampus SWR output was disconnected from cortex

    if (!bg_ || !vta_) return;

    // --- Enter sleep ---
    sleep_mgr_.enter_sleep();
    if (hipp_) hipp_->enable_sleep_replay();

    // DA at baseline during NREM (no new BG learning)
    float saved_da = bg_->da_level();
    bg_->set_da_level(config_.sleep_positive_da);  // = 0.30 (baseline)

    // --- NREM consolidation: let hippocampus SWR drive cortex via SpikeBus ---
    // No episode buffer injection. Hippocampus generates SWR spontaneously.
    // SWR → CA1 → Sub → SpikeBus → dlPFC (existing projection).
    // Cortex receives SWR patterns → internal STDP consolidates (if enabled).
    size_t total_nrem = config_.sleep_nrem_steps;

    for (size_t i = 0; i < total_nrem; ++i) {
        // Step the ENTIRE brain (hippocampus SWR → SpikeBus → cortex)
        // No visual input (sleeping), no motor output, just internal replay
        engine_.step();
        sleep_mgr_.step();
    }

    // --- Wake up ---
    sleep_mgr_.wake_up();
    if (hipp_) hipp_->disable_sleep_replay();
    bg_->set_da_level(saved_da);
}

} // namespace wuyun
