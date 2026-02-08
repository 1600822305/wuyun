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

ClosedLoopAgent::ClosedLoopAgent(std::unique_ptr<Environment> env, const AgentConfig& config)
    : config_(config)
    , env_(std::move(env))
    , engine_(10)
    , reward_history_(1000, 0.0f)
    , food_history_(1000, 0)
    , replay_buffer_(config.replay_buffer_size, config.brain_steps_per_action)
{
    // Auto-compute vision size from environment
    config_.vision_width  = env_->vis_width();
    config_.vision_height = env_->vis_height();

    build_brain();

    // v36: Initialize spatial value map (cognitive map)
    spatial_map_w_ = static_cast<int>(env_->world_width());
    spatial_map_h_ = static_cast<int>(env_->world_height());
    size_t map_size = static_cast<size_t>(spatial_map_w_) * spatial_map_h_;
    spatial_value_map_.assign(map_size, 0.0f);

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
        // v39: L4/L5 minimum 2→3 (ensures reliable signal propagation through hierarchy)
        // With 2 neurons, one missed spike kills the chain. 3 provides redundancy.
        c.n_l4_stellate    = std::max<size_t>(3, N * 25 / 100) * s;  // 25%
        c.n_l23_pyramidal  = std::max<size_t>(3, N * 35 / 100) * s;  // 35%
        c.n_l5_pyramidal   = std::max<size_t>(3, N * 20 / 100) * s;  // 20%
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
    // v41: FPC 前额极皮层 (BA10) — 层级最高的前额叶 (Koechlin 2003)
    // Strong recurrent connections for sustained goal maintenance
    // FPC → dlPFC top-down: long-term goals modulate immediate decisions
    if (config_.enable_fpc) {
        add_ctx("FPC", n_act * 3, false);  // 12: same size as dlPFC
    }
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
        hipp_cfg.n_sub = std::max<size_t>(8, n_act * 2) * s;  // v36: ≥8 for retrieval_bias (needs ≥4 for 4 direction groups)
        // v36: Scale inhibitory populations proportional to excitatory
        // Defaults (20/10/15) are for n_dg=200/ca3=60/ca1=80.
        // With compressed sizes, E/I ratio was inverted → pathway smothered.
        hipp_cfg.n_dg_inh  = std::max<size_t>(2, hipp_cfg.n_dg / 5);   // ~5:1 E/I
        hipp_cfg.n_ca3_inh = std::max<size_t>(2, hipp_cfg.n_ca3 / 3);  // ~3:1 E/I
        hipp_cfg.n_ca1_inh = std::max<size_t>(2, hipp_cfg.n_ca1 / 3);  // ~3:1 E/I
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
    engine_.add_projection("V1", "LGN", 3);    // v56: L6→TC corticothalamic prediction
    engine_.add_projection("V2", "V1", 3);
    engine_.add_projection("V4", "V2", 3);
    engine_.add_projection("IT", "V4", 3);

    // v38: Thalamostriatal direct pathway (CM/Pf → striatum, Smith et al. 2004)
    // LGN spikes reach BG in 1 step, providing fast sensory salience drive.
    // Cortical pathway (LGN→V1→...→dlPFC→BG) takes ~14 steps but carries
    // action-specific learned information via DA-STDP.
    // Together: thalamic maintains MSN up-state, cortical determines WHICH direction.
    engine_.add_projection("LGN", "BG", 1);

    // Decision → action: dlPFC → BG → MotorThal → M1
    engine_.add_projection("dlPFC", "BG", 2);
    engine_.add_projection("BG", "MotorThal", 2);
    engine_.add_projection("MotorThal", "M1", 2);

    // Feedback: M1 → dlPFC (efference copy)
    engine_.add_projection("M1", "dlPFC", 3);

    // v41: FPC (BA10) — highest prefrontal hierarchy (Koechlin 2003)
    // FPC maintains long-term goals and modulates dlPFC decisions top-down
    if (config_.enable_fpc) {
        // IT → FPC (object identity → goal planning: "food exists → seek it")
        engine_.add_projection("IT", "FPC", 3);
        // v43 fix: dlPFC→FPC removed (feedback loop amplified noise)
        //   FPC→dlPFC is one-way top-down: goals modulate decisions, not vice versa
        engine_.add_projection("FPC", "dlPFC", 3);
        // Hippocampus → FPC (memory-guided planning: "I remember food at X")
        if (!config_.fast_eval) {
            engine_.add_projection("Hippocampus", "FPC", 3);
        }
    }

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

    // --- v35: ACC 前扣带回 (Botvinick 2001 冲突监测 + Alexander & Brown 2011 PRO) ---
    // 输入: BG D1发放率(冲突), VTA DA(惊讶), Amygdala CeA(威胁)
    // 输出: ACC→LC(唤醒/探索), ACC→dlPFC(注意力), 波动性→学习率调制
    // 替代硬编码 ne_floor, 用神经动力学驱动探索/利用平衡
    if (config_.enable_acc) {
        ACCConfig acc_cfg;
        acc_cfg.n_dacc = 12 * s;
        acc_cfg.n_vacc = 8 * s;
        acc_cfg.n_inh  = 6 * s;
        engine_.add_region(std::make_unique<AnteriorCingulate>(acc_cfg));
        // ACC SpikeBus projections:
        //   dlPFC → ACC (current context/state for prediction)
        //   ACC → dlPFC (cognitive control signal)
        engine_.add_projection("dlPFC", "ACC", 3);
        engine_.add_projection("ACC", "dlPFC", 3);
    }

    // --- v40: NAcc 伏隔核 (Mogenson 1980: limbic-motor interface) ---
    // Ventral striatum: motivation/reward integration, independent of dorsal BG motor selection
    // VTA→NAcc (mesolimbic DA): reward prediction → approach motivation
    // Amygdala→NAcc: emotional valence → avoidance motivation
    // NAcc→VP: motivation output → modulates BG motor vigor
    if (config_.enable_nacc) {
        NAccConfig nacc_cfg;
        nacc_cfg.n_core_d1 = n_act * s;   // 4: approach motivation
        nacc_cfg.n_core_d2 = n_act * s;   // 4: avoidance motivation
        nacc_cfg.n_shell   = n_act * s;   // 4: novelty detection
        nacc_cfg.n_vp      = n_act * s;   // 4: ventral pallidum output
        engine_.add_region(std::make_unique<NucleusAccumbens>(nacc_cfg));

        // VTA → NAcc (mesolimbic DA pathway, core reward signal)
        engine_.add_projection("VTA", "NAcc", 2);
        // Hippocampus → NAcc (contextual motivation: "this place had food")
        if (!config_.fast_eval) {
            engine_.add_projection("Hippocampus", "NAcc", 3);
        }
        // Amygdala → NAcc (emotional valence: fear → avoidance motivation)
        if (config_.enable_amygdala) {
            engine_.add_projection("Amygdala", "NAcc", 2);
        }
        // IT → NAcc (object identity: "I see food" → approach)
        engine_.add_projection("IT", "NAcc", 2);
        // NAcc → M1 (VP output → motor drive via SpikeBus)
        // Anti-cheat: replaces motivation_output() scalar bypass.
        // Biology: NAcc Core D1 → VP inhibition → VP releases downstream targets.
        //   VP→MD thalamus→PFC is the full pathway; we shortcut VP→M1 for motor vigor.
        //   High D1 (approach) → VP inhibited → less VP→M1 inhibition → more M1 activity.
        //   High D2 (avoidance) → VP excited → more VP→M1 inhibition → less M1 activity.
        engine_.add_projection("NAcc", "M1", 2);
    }

    // --- v41: PAG 导水管周围灰质 (LeDoux 1996: defense circuit) ---
    // CeA → PAG → brainstem: hardwired defense bypassing BG
    // dlPAG: active flight, vlPAG: passive freeze
    if (config_.enable_pag) {
        PAGConfig pag_cfg;
        pag_cfg.n_dlpag = n_act * s;   // 4: active defense
        pag_cfg.n_vlpag = n_act * s;   // 4: passive defense
        engine_.add_region(std::make_unique<PeriaqueductalGray>(pag_cfg));
        // Amygdala CeA → PAG (fear drive, fast: delay=1)
        if (config_.enable_amygdala) {
            engine_.add_projection("Amygdala", "PAG", 1);
        }
        // v43 fix: PAG→M1 removed (PAG has no directional info → blind motor bias
        //   caused agent to move randomly during fear → walked INTO danger)
        //   Correct role: PAG→LC (fear→NE arousal, heightened vigilance)
        //   Motor defense comes from BG learning (D2 NoGo for dangerous directions)
        // PAG → LC (fear → NE arousal, via SpikeBus)
        // Anti-cheat: replaces lc_->inject_arousal(pag_->arousal_drive()) scalar bypass.
        // Biology: PAG→LC increases NE release during threat (Aston-Jones 1991)
        if (config_.enable_lc_ne) {
            engine_.add_projection("PAG", "LC", 1);
        }
    }

    // --- v40: Superior Colliculus 上丘 (Krauzlis 2013: subcortical saliency) ---
    // Fast visual saliency pathway: LGN → SC → Pulvinar (enhances cortical processing)
    // Also SC → BG (fast saliency signal, supplements thalamostriatal pathway)
    if (config_.enable_sc) {
        SCConfig sc_cfg;
        sc_cfg.n_superficial = n_act * s;   // 4: visual map
        sc_cfg.n_deep        = n_act * s;   // 4: multimodal output
        engine_.add_region(std::make_unique<SuperiorColliculus>(sc_cfg));

        // LGN → SC (retinal input, fast: delay=1)
        engine_.add_projection("LGN", "SC", 1);
        // v43 fix: SC→BG changed to SC→dlPFC
        //   SC→BG was redundant with LGN→BG thalamostriatal → doubled noise in BG
        //   SC→dlPFC: saliency enhances cortical decision-making instead
        engine_.add_projection("SC", "dlPFC", 2);
        // V1 → SC (cortical feedback to SC deep layer, delay=2)
        engine_.add_projection("V1", "SC", 2);
    }

    // --- v42: OFC 眶额皮层 (Rolls 2000: stimulus-outcome value) ---
    // Encodes expected value of visual stimuli: "food-like pattern → positive value"
    // DA modulates value updates (volume transmission, not SpikeBus)
    if (config_.enable_ofc) {
        OFCConfig ofc_cfg;
        ofc_cfg.n_value_pos = n_act * s;    // 4: positive value
        ofc_cfg.n_value_neg = n_act * s;    // 4: negative value
        ofc_cfg.n_inh       = n_act * s;    // 4: E/I balance
        engine_.add_region(std::make_unique<OrbitofrontalCortex>(ofc_cfg));

        // IT → OFC (object identity → value association: "food" → pos value)
        engine_.add_projection("IT", "OFC", 2);
        // Amygdala → OFC (emotional valence → value modulation)
        if (config_.enable_amygdala) {
            engine_.add_projection("Amygdala", "OFC", 2);
        }
        // OFC → dlPFC (value signal → decision bias)
        engine_.add_projection("OFC", "dlPFC", 2);
        // OFC → NAcc (value → motivation)
        if (config_.enable_nacc) {
            engine_.add_projection("OFC", "NAcc", 2);
        }
    }

    // --- v42: vmPFC 腹内侧前额叶 (Milad & Quirk 2002: fear extinction) ---
    // Safety signal: context + value → "this place is safe now" → Amygdala ITC → CeA inhibition
    if (config_.enable_vmpfc) {
        add_ctx("vmPFC", n_act * 2, false);  // 8: smaller than dlPFC

        // OFC → vmPFC (value information → safety assessment)
        if (config_.enable_ofc) {
            engine_.add_projection("OFC", "vmPFC", 2);
        }
        // Hippocampus → vmPFC (context: "this location was safe before")
        if (!config_.fast_eval) {
            engine_.add_projection("Hippocampus", "vmPFC", 3);
        }
        // vmPFC → Amygdala (safety signal → ITC excitation → CeA suppression)
        if (config_.enable_amygdala) {
            engine_.add_projection("vmPFC", "Amygdala", 2);
        }
        // vmPFC → NAcc (value-based motivation)
        if (config_.enable_nacc) {
            engine_.add_projection("vmPFC", "NAcc", 2);
        }
    }

    // --- v40: SNc 黑质致密部 (Yin & Knowlton 2006: habit learning) ---
    // Nigrostriatal pathway: tonic DA → dorsal BG, maintains learned habits
    // Unlike VTA phasic RPE: SNc is stable, slowly tracks reward history
    if (config_.enable_snc) {
        SNcConfig snc_cfg;
        snc_cfg.n_da_neurons = std::max<size_t>(4, n_act) * s;
        engine_.add_region(std::make_unique<SNc_DA>(snc_cfg));
        // M1 → SNc (motor efference copy: active movement → SNc maintenance)
        engine_.add_projection("M1", "SNc", 2);
        // BG → SNc (striatonigral D1 feedback: active D1 MSN → SNc tonic maintenance)
        // Anti-cheat: replaces inject_d1_activity() scalar bypass.
        // Biology: D1 MSN project back to SNc (Haber 2003), maintaining tonic DA
        // for well-learned actions. This feedback arrives through SpikeBus.
        engine_.add_projection("BG", "SNc", 2);
    }

    // --- v34: 神经调质系统 (LC-NE, NBM-ACh, DRN-5HT) ---
    // Volume transmission: 不走SpikeBus，通过inject_arousal/surprise/wellbeing驱动
    // SimulationEngine每步自动收集输出并广播到所有区域
    if (config_.enable_lc_ne) {
        LCConfig lc_cfg;
        lc_cfg.n_ne_neurons = 4 * s;
        engine_.add_region(std::make_unique<LC_NE>(lc_cfg));
    }
    if (config_.enable_nbm_ach) {
        NBMConfig nbm_cfg;
        nbm_cfg.n_ach_neurons = 4 * s;
        engine_.add_region(std::make_unique<NBM_ACh>(nbm_cfg));
    }
    if (config_.enable_drn_5ht) {
        DRNConfig drn_cfg;
        drn_cfg.n_5ht_neurons = 4 * s;
        engine_.add_region(std::make_unique<DRN_5HT>(drn_cfg));
    }

    // --- Neuromodulator registration ---
    engine_.register_neuromod_source("VTA", SimulationEngine::NeuromodType::DA);
    if (config_.enable_lc_ne)
        engine_.register_neuromod_source("LC", SimulationEngine::NeuromodType::NE);
    if (config_.enable_nbm_ach)
        engine_.register_neuromod_source("NBM", SimulationEngine::NeuromodType::ACh);
    if (config_.enable_drn_5ht)
        engine_.register_neuromod_source("DRN", SimulationEngine::NeuromodType::SHT);

    // --- v46: Hypothalamus — hedonic sensory interface for reward signals ---
    // Biology: LH (lateral hypothalamus) → VTA drives DA release during feeding (Nieh 2015)
    //   PVN (paraventricular) → stress response during pain
    // Analogous to LGN for vision: the "body" converting environmental reward to neural signals.
    // Agent injects reward into Hypothalamus (inject_hedonic), NOT into VTA.
    {
        HypothalamusConfig hypo_cfg;
        hypo_cfg.name = "Hypothalamus";
        // Use small populations for closed-loop (only LH and PVN matter for reward)
        hypo_cfg.n_lh  = std::max<size_t>(4, n_act) * s;   // 4: food satisfaction
        hypo_cfg.n_pvn = std::max<size_t>(4, n_act) * s;   // 4: pain/stress
        // Keep other nuclei small (not critical for GridWorld reward path)
        hypo_cfg.n_scn = 4;
        hypo_cfg.n_vlpo = 4;
        hypo_cfg.n_orexin = 4;
        hypo_cfg.n_vmh = 4;
        engine_.add_region(std::make_unique<Hypothalamus>(hypo_cfg));

        // Hypothalamus → VTA (hedonic signal: LH excitation → DA burst)
        // Fast pathway: delay=1 (reward must reach VTA quickly)
        engine_.add_projection("Hypothalamus", "VTA", 1);
    }

    // OFC → VTA (value prediction: expected reward → suppress DA surprise)
    // Biology: OFC projects to VTA providing expected value signal (Takahashi 2011)
    if (config_.enable_ofc) {
        engine_.add_projection("OFC", "VTA", 2);
    }

    // --- Wire DA source for BG ---
    auto* bg_ptr = dynamic_cast<BasalGanglia*>(engine_.find_region("BG"));
    auto* vta_ptr = engine_.find_region("VTA");
    // DA传递用neuromodulatory broadcast (体积传递), 不走SpikeBus
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
    lc_    = dynamic_cast<LC_NE*>(engine_.find_region("LC"));
    nbm_   = dynamic_cast<NBM_ACh*>(engine_.find_region("NBM"));
    drn_   = dynamic_cast<DRN_5HT*>(engine_.find_region("DRN"));
    acc_   = dynamic_cast<AnteriorCingulate*>(engine_.find_region("ACC"));
    nacc_  = dynamic_cast<NucleusAccumbens*>(engine_.find_region("NAcc"));
    snc_   = dynamic_cast<SNc_DA*>(engine_.find_region("SNc"));
    sc_    = dynamic_cast<SuperiorColliculus*>(engine_.find_region("SC"));
    pag_   = dynamic_cast<PeriaqueductalGray*>(engine_.find_region("PAG"));
    fpc_   = dynamic_cast<CorticalRegion*>(engine_.find_region("FPC"));
    ofc_   = dynamic_cast<OrbitofrontalCortex*>(engine_.find_region("OFC"));
    vmpfc_ = dynamic_cast<CorticalRegion*>(engine_.find_region("vmPFC"));
    hypo_  = dynamic_cast<Hypothalamus*>(engine_.find_region("Hypothalamus"));

    // v46: Register VTA spike sources for internal RPE computation
    // VTA distinguishes Hypothalamus spikes (hedonic) from OFC spikes (prediction)
    if (vta_ && hypo_) {
        vta_->register_hedonic_source(hypo_->region_id());
    }
    if (vta_ && ofc_) {
        vta_->register_prediction_source(ofc_->region_id());
    }

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

    // --- v38: Register LGN as thalamic source for thalamostriatal pathway ---
    if (lgn_ && bg_) {
        bg_->set_thalamic_source(lgn_->region_id());
    }

    // --- v56: Register V1 as cortical feedback source for LGN ---
    // Biology: V1 L6 corticothalamic → LGN relay apical (prediction modulation)
    // Predicted visual input is suppressed; novel/unexpected input passes through.
    // This completes the thalamocortical prediction loop: LGN→V1 + V1→LGN.
    if (lgn_ && v1_) {
        auto* lgn_thal = dynamic_cast<ThalamicRelay*>(lgn_);
        if (lgn_thal) {
            lgn_thal->add_cortical_feedback_source(v1_->region_id());
        }
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

    // --- v45: Population vector encoding (Georgopoulos 1986) ---
    // Each M1 L5 neuron and BG D1 neuron gets a random preferred direction.
    // Direction selection emerges from population activity, not hardcoded groups.
    // Deterministic seed for reproducibility across runs.
    {
        std::mt19937 pv_rng(42);
        std::normal_distribution<float> jitter_dist(0.0f, 0.3f);  // ~17° jitter

        // v48: Evenly spaced preferred directions with small jitter
        // Biology: M1 has roughly equal representation of all movement directions
        // (Georgopoulos 1986: uniform coverage of directional space).
        // Pure random angles can create directional bias (e.g., more rightward neurons).
        // Fix: space angles evenly around the circle, add Gaussian jitter for diversity.

        // M1 L5 preferred directions
        size_t l5_sz = m1_->column().l5().size();
        m1_preferred_dir_.resize(l5_sz);
        for (size_t i = 0; i < l5_sz; ++i) {
            float base_angle = 6.2831853f * static_cast<float>(i) / static_cast<float>(l5_sz);
            m1_preferred_dir_[i] = base_angle + jitter_dist(pv_rng);
        }

        // BG D1 preferred directions
        size_t d1_sz = bg_->d1().size();
        d1_preferred_dir_.resize(d1_sz);
        for (size_t i = 0; i < d1_sz; ++i) {
            float base_angle = 6.2831853f * static_cast<float>(i) / static_cast<float>(d1_sz);
            d1_preferred_dir_[i] = base_angle + jitter_dist(pv_rng);
        }
    }
}

// =============================================================================
// Closed loop step
// =============================================================================

void ClosedLoopAgent::reset_world() {
    env_->reset();
    agent_step_count_ = 0;
    std::fill(reward_history_.begin(), reward_history_.end(), 0.0f);
    std::fill(food_history_.begin(), food_history_.end(), 0);
    history_idx_ = 0;
}

void ClosedLoopAgent::reset_world_with_seed(uint32_t seed) {
    // v53: 反转学习 — 大脑保留, 世界换布局
    // 只重置世界和历史, 不重置 agent_step_count_ (大脑继续成长)
    // 不重置 novelty (已在旧世界 habituate, 新世界也不是第一次了)
    env_->reset_with_seed(seed);
    // 清空空间价值图 (旧世界的空间记忆不适用新布局)
    if (!spatial_value_map_.empty()) {
        std::fill(spatial_value_map_.begin(), spatial_value_map_.end(), 0.0f);
    }
    // 清空回放缓冲 (旧世界的经验不适用新布局)
    if (config_.enable_replay) {
        replay_buffer_.clear();
    }
    // 清空奖励历史 (重新统计)
    std::fill(reward_history_.begin(), reward_history_.end(), 0.0f);
    std::fill(food_history_.begin(), food_history_.end(), 0);
    history_idx_ = 0;
    expected_reward_level_ = 0.0f;
    has_pending_reward_ = false;
    pending_reward_ = 0.0f;
}

Environment::Result ClosedLoopAgent::agent_step() {
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

        // v36: Update spatial value map (cognitive map)
        update_spatial_value_map(pending_reward_);

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

        // v34: ACh-gated visual STDP — 由 NBM-ACh 神经元驱动
        // Biology: NBM ACh burst during salient events → visual cortex STDP enhanced
        // 输入: DA偏离baseline = "意外" → ACh phasic burst
        // 效果: 意外事件(食物/危险)时皮层学得快，平常时学得慢
        if (nbm_) {
            float da_error = std::abs(vta_->da_output() - 0.3f);
            nbm_->inject_surprise(da_error);  // DA偏离→意外→ACh↑
        }
        float ach_boost = nbm_ ? (nbm_->ach_output() / 0.2f)
                                : (1.0f + std::abs(pending_reward_) * 0.5f);  // fallback
        if (v1_)  v1_->column().set_ach_stdp_gain(ach_boost);
        if (v2_)  v2_->column().set_ach_stdp_gain(ach_boost);
        if (v4_)  v4_->column().set_ach_stdp_gain(ach_boost);
        // IT intentionally excluded (NO STDP, representation stability)

        // v38: ACh → BG consolidation gating (volume transmission)
        // High ACh during reward events → opens plasticity window for reversal learning
        // Biology: NBM ACh signals uncertainty/novelty → striatal consolidation reduced
        //   (Hasselmo 1999: ACh switches cortex from recall to encoding mode)
        if (bg_) {
            float ach = nbm_ ? nbm_->ach_output() : 0.2f;
            bg_->set_ach_level(ach);
        }

        // v40: Feed VTA DA to NAcc (mesolimbic pathway)
        // NAcc processes reward signal for motivation, independent of BG motor selection
        if (nacc_) {
            nacc_->set_da_level(vta_->da_output());
        }
        // v42: Feed VTA DA to OFC (value update signal, volume transmission)
        // DA+ → strengthen positive value associations, DA- → strengthen negative
        if (ofc_) {
            ofc_->set_da_level(vta_->da_output());
        }

        // v40: SNc habit maintenance — blend tonic DA with VTA phasic DA
        // Biology: BG DA = VTA phasic (new learning) + SNc tonic (habit maintenance)
        //   As habits form, SNc contribution stabilizes BG weights against VTA fluctuations
        // Anti-cheat: SNc tonic adapts from BG→SNc SpikeBus spikes (D1 feedback),
        //   NOT from agent-computed avg_reward or D1 firing rate scalars.
        //   See snc_da.cpp step() for spike-driven tonic adaptation logic.

        // Run a few steps so DA can modulate BG eligibility traces
        // DA broadcast: VTA computes DA level, BG reads it directly (volume transmission)
        for (size_t i = 0; i < config_.reward_processing_steps; ++i) {
            // LHb → VTA inhibition: direct neuromodulatory broadcast
            // (supplements SpikeBus projection with immediate DA level effect)
            if (lhb_) {
                vta_->inject_lhb_inhibition(lhb_->vta_inhibition());
            }
            engine_.step();
            // v37: Read DA AFTER engine step (VTA processes reward during step)
            // Previous bug: bg read da BEFORE engine step → missed the DA change
            // on the first reward processing step entirely.
            // v40: Blend VTA phasic + SNc tonic DA for BG
            // Biology: dorsal striatum receives DA from both VTA and SNc
            //   VTA: fast phasic RPE (new learning)
            //   SNc: stable tonic (habit maintenance, resists fluctuations)
            //   Blend: 70% VTA + 30% SNc → habits stabilize as SNc tonic rises
            float da = vta_->da_output();
            if (snc_) {
                da = da * 0.7f + snc_->da_output() * 0.3f;
            }
            bg_->set_da_level(da);
        }
        has_pending_reward_ = false;

        // Reset ACh STDP boost after reward processing
        if (v1_)  v1_->column().set_ach_stdp_gain(1.0f);
        if (v2_)  v2_->column().set_ach_stdp_gain(1.0f);
        if (v4_)  v4_->column().set_ach_stdp_gain(1.0f);
        // v38: Reset BG ACh to baseline (consolidation fully protected during routine Phase B)
        if (bg_) bg_->set_ach_level(0.2f);
    }

    // --- Phase B: Observe + decide ---

    // Begin recording episode for awake SWR replay
    if (config_.enable_replay) {
        replay_buffer_.begin_episode();
    }

    // B1. Inject new visual observation
    inject_observation();

    // B1b: spatial context + retrieval bias are now injected EVERY brain step
    // (moved into brain steps loop below, same fix as inject_observation)

    // =====================================================================
    // Biologically correct motor architecture:
    //
    //   dlPFC → BG D1/D2 (corticostriatal: sensory context)
    //   D1 → GPi(inhibit) → MotorThal(disinhibit) → M1 L5 (Go)
    //   D2 → GPe → GPi(disinhibit) → MotorThal(inhibit) (NoGo)
    //   M1 L5 = sole motor output (action decoded here)
    //
    //   Exploration = cosine-weighted noise along random attractor_angle (all M1 L5)
    //   BG influence = D1 population vector → cosine bias on M1 L5 (Georgopoulos 1986)
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

    // v45: BG D1 population vector parameters (for bias injection into M1)
    size_t d1_size = bg_->d1().size();
    float bg_to_m1_gain = config_.bg_to_m1_gain;

    // v35: ACC 冲突/惊讶/波動性计算 → 驱动 LC-NE 探索
    // Biology: dACC检测BG D1方向冲突 → ACC→LC phasic burst → NE↑ → 探索
    //          PRO模型惊讶信号 → 额外唤醒
    //          替代硬编码 ne_floor 和手工 food_rate arousal 计算
    if (acc_ && bg_) {
        // v45: D1 按 preferred direction 最近基数方向分组 (替代旧的索引切分)
        // Biology: ACC 检测 BG 中不同方向 D1 MSN 的竞争强度
        //   如果多个方向的 D1 同等活跃 → 高冲突 → 需要更多探索
        // 旧方式(v35-v44): 按索引位置 d1_size/4 切分 → 群体向量后无意义
        // 新方式(v45+): 按 d1_preferred_dir_ 最近基数方向分组
        //   Direction angles: UP=π/2, DOWN=-π/2, LEFT=π, RIGHT=0
        constexpr float ACC_DIR_ANGLES[4] = {1.5708f, -1.5708f, 3.14159f, 0.0f};
        std::array<float, 4> d1_rates = {0.0f, 0.0f, 0.0f, 0.0f};
        std::array<int, 4> d1_counts = {0, 0, 0, 0};  // neurons per direction group
        const auto& d1_f = bg_->d1().fired();
        size_t d1_sz = bg_->d1().size();
        for (size_t k = 0; k < d1_sz && k < d1_preferred_dir_.size(); ++k) {
            // Assign to nearest cardinal direction by preferred angle
            int best_dir = 0;
            float best_cos = -2.0f;
            for (int d = 0; d < 4; ++d) {
                float c = std::cos(d1_preferred_dir_[k] - ACC_DIR_ANGLES[d]);
                if (c > best_cos) { best_cos = c; best_dir = d; }
            }
            d1_counts[best_dir]++;
            if (d1_f[k]) d1_rates[best_dir] += 1.0f;
        }
        // Normalize by group size
        for (int d = 0; d < 4; ++d) {
            if (d1_counts[d] > 0) {
                d1_rates[d] /= static_cast<float>(d1_counts[d]);
            }
        }
        acc_->inject_d1_rates(d1_rates);

        // 注入上一步的奖励结果 (PRO模型: 预测vs实际)
        acc_->inject_outcome(last_reward_);

        // 注入威胁信号 (Amygdala CeA → vACC)
        if (amyg_) {
            acc_->inject_threat(amyg_->cea_vta_drive());
        }
    }

    // v35: ACC→LC arousal 驱动 (替代手工 food_rate 计算)
    // Biology: dACC冲突+惊讶→LC phasic NE burst→探索噪声↑
    //          ACC arousal_drive 是冲突+惊讶+觅食+威胁的加权组合
    if (lc_) {
        float arousal = 0.0f;
        if (acc_) {
            // ACC驱动: 冲突高→探索多, 惊讶高→更警觉, 觅食信号→切换策略
            arousal = acc_->arousal_drive() * 0.15f;
        } else {
            // fallback: 旧的温和arousal
            float fr = food_rate(200);
            arousal = std::max(0.0f, 0.05f - fr * 0.1f);
        }
        lc_->inject_arousal(arousal);
    }

    // v35b: ACC→dlPFC 注意力增益 (冲突/惊讶 → dlPFC更专注)
    // Biology: ACC→dlPFC投射增强PFC上下控制 (Shenhav 2013 EVC)
    // 高冲突/惊讶 → attention_gain↑ → dlPFC神经元对输入更敏感 → 决策更精确
    if (acc_ && dlpfc_) {
        float att_gain = 1.0f + acc_->attention_signal() * 0.5f;  // [1.0, 1.5]
        dlpfc_->set_attention_gain(att_gain);
    }
    // v34: DRN-5HT wellbeing 注入 (持续获得食物→5-HT↑→更耐心)
    if (drn_) {
        drn_->inject_wellbeing(food_rate(200));
    }

    float noise_scale = 1.0f;
    if (lc_) {
        // NE驱动探索: ne_tonic≈0.22时scale=1.0, 高NE=探索多, 低NE=利用多
        float ne = lc_->ne_output();
        float ne_tonic = 0.22f;  // LC自然tonic发放对应的NE水平
        noise_scale = std::clamp(ne / ne_tonic, 0.5f, 1.5f);
    } else if (agent_step_count_ > 500 && reward_history_.size() > 0) {
        // fallback: 旧的手工逻辑
        int food_count = 0;
        int total = static_cast<int>(std::min(history_idx_, reward_history_.size()));
        for (int k = 0; k < total; ++k) {
            if (food_history_[k]) food_count++;
        }
        float fr = static_cast<float>(food_count) / static_cast<float>(std::max(total, 1));
        noise_scale = std::max(config_.ne_floor, 1.0f - fr * config_.ne_food_scale);
    }

    // v35b: ACC foraging_signal → 探索噪声增强 (Kolling 2012)
    // Biology: dACC觅食信号: 当前策略不如全局平均 → 应该切换策略 → 探索↑
    // foraging_signal高 → noise_scale↑ → 更多随机动作 → 突破当前局部最优
    if (acc_) {
        noise_scale *= (1.0f + acc_->foraging_signal() * 0.3f);
        noise_scale = std::clamp(noise_scale, 0.3f, 2.0f);
    }

    // v40 anti-cheat: NAcc motivation no longer bypasses SpikeBus.
    // NAcc VP spikes flow through NAcc→M1 projection (SpikeBus), modulating
    // M1 activity directly via neural pathways. motivation_output() remains
    // available as a diagnostic but is NOT used for noise scaling.

    // v59: 探索饥饿重置 — 长时间无奖赏 → noise 翻倍
    // Biology: LC NE burst mode when no reward for extended period (Aston-Jones 2005)
    // Prevents agent from pacing in the same area indefinitely
    if (config_.starvation_threshold > 0 &&
        steps_since_reward_ > config_.starvation_threshold) {
        noise_scale *= config_.starvation_noise_boost;
        noise_scale = std::clamp(noise_scale, 0.3f, 4.0f);
    }

    float effective_noise = config_.exploration_noise * noise_scale;

    // v45: Population vector exploration — pick random direction angle
    //   instead of random group. Derive group for backward compat (efference copy, replay).
    //   Direction angles: UP=π/2, DOWN=-π/2, LEFT=π, RIGHT=0
    constexpr float DIR_ANGLES[4] = {1.5708f, -1.5708f, 3.14159f, 0.0f};
    float attractor_angle = 0.0f;
    int attractor_group = -1;
    if (effective_noise > 0.0f) {
        std::uniform_real_distribution<float> angle_pick(0.0f, 6.2831853f);
        attractor_angle = angle_pick(motor_rng_);
        // Derive closest cardinal direction for efference copy compatibility
        float best_cos = -2.0f;
        for (int d = 0; d < 4; ++d) {
            float c = std::cos(attractor_angle - DIR_ANGLES[d]);
            if (c > best_cos) { best_cos = c; attractor_group = d; }
        }
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

        // v36: Inject spatial context EVERY brain step (sustained-drive)
        // EC grid cells need ~5+ steps of sustained current to charge and fire.
        if (hipp_) {
            hipp_->inject_spatial_context(
                static_cast<int>(env_->pos_x()), static_cast<int>(env_->pos_y()),
                spatial_map_w_, spatial_map_h_);
        }

        // v36: CLS — spatial value gradient → BG sensory context
        // Biology: Hippocampal place cells + OFC value = cognitive map (Tolman 1948)
        //   Agent remembers reward/danger outcomes at each position.
        //   Adjacent positions' values form a gradient → bias BG toward food.
        //   Equivalent to EC successor representation (Stachenfeld 2017).
        // This replaces get_retrieval_bias (which had wrong direction mapping).
        if (bg_ && !spatial_value_map_.empty()) {
            int ax = static_cast<int>(env_->pos_x()), ay = static_cast<int>(env_->pos_y());
            int w = spatial_map_w_, h = spatial_map_h_;
            // Look up value of adjacent cells: UP(y-1), DOWN(y+1), LEFT(x-1), RIGHT(x+1)
            float adj[4] = {0, 0, 0, 0};
            if (ay > 0)     adj[0] = spatial_value_map_[(ay - 1) * w + ax];  // UP
            if (ay < h - 1) adj[1] = spatial_value_map_[(ay + 1) * w + ax];  // DOWN
            if (ax > 0)     adj[2] = spatial_value_map_[ay * w + (ax - 1)];  // LEFT
            if (ax < w - 1) adj[3] = spatial_value_map_[ay * w + (ax + 1)];  // RIGHT
            bg_->inject_sensory_context(adj);
        }

        // LHb → VTA inhibition broadcast (every brain step during action processing)
        if (lhb_) {
            vta_->inject_lhb_inhibition(lhb_->vta_inhibition());
        }
        // v41 anti-cheat: PAG defense is fully spike-driven.
        // CeA→PAG and PAG→M1/LC all via SpikeBus projections.
        // No inject_fear() or arousal_drive() scalar bypasses.
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
        // v34: DRN-5HT 调制 DA 信号增益
        // 5-HT↑(顺利) → DA error 被衰减 → 学习更稳定(不轻易改变已学到的行为)
        // 5-HT↓(不顺) → DA error 被放大 → 学习更快(需要快速调整策略)
        {
            float da = vta_->da_output();
            if (drn_) {
                float sht = drn_->sht_output();
                // sht=0.3(baseline)→gain=1.0, sht=0.5(顺利)→gain=0.7, sht=0.1(不顺)→gain=1.3
                float da_gain = std::clamp(1.6f - 2.0f * sht, 0.5f, 1.5f);
                float da_baseline = 0.3f;
                da = da_baseline + (da - da_baseline) * da_gain;
                da = std::clamp(da, 0.0f, 1.0f);
            }
            // v35b: ACC波动性 → DA误差缩放 (Behrens 2007)
            // Biology: 高波动性 → 环境在变 → DA RPE误差放大 → 学得快
            //          低波动性 → 环境稳定 → DA RPE误差压缩 → 保持稳定
            // learning_rate_modulation ∈ [0.5, 2.0], baseline=1.0
            if (acc_) {
                float lr_mod = acc_->learning_rate_modulation();
                float da_baseline = 0.3f;
                da = da_baseline + (da - da_baseline) * lr_mod;
                da = std::clamp(da, 0.0f, 1.0f);
            }
            bg_->set_da_level(da);
        }

        // Hippocampal spatial memory → dlPFC: handled via SpikeBus projection
        // (Hippocampus → dlPFC added in build_brain)
        // When agent revisits a familiar location:
        //   EC grid cells fire position-specific pattern →
        //   CA3 pattern completion (if STDP encoded this place) →
        //   CA1 → Sub fires → SpikeBus → dlPFC receives memory signal →
        //   dlPFC→BG pathway naturally biases action selection
        // No direct BG injection needed — the cortical pathway handles it.

        // (0) v52: SC 趋近反射 — 皮层下快通路
        //     SC 从视觉 patch 计算显著性方向 → 深层方向性神经元激活
        //     深层群体向量 → M1 L5 cos 驱动 = "看到东西就走过去"
        //     2-3 步出结果, 比皮层慢通路 (14 步) 快 5-7 倍
        //     生物学: 视网膜→SC 浅层→SC 深层→脑干运动核 (Ingle 1973)
        auto& l5 = m1_->column().l5();
        if (sc_ && config_.sc_approach_gain > 0.01f) {
            // 注入视觉 patch → SC 计算显著性方向
            auto obs = env_->observe();
            sc_->inject_visual_patch(obs, static_cast<int>(config_.vision_width),
                                     static_cast<int>(config_.vision_height),
                                     config_.sc_approach_gain);

            // SC 深层群体向量 → M1 cos 驱动
            const auto& sc_deep_fired = sc_->deep().fired();
            const auto& sc_dirs = sc_->deep_preferred_dir();
            float sc_vx = 0.0f, sc_vy = 0.0f;
            for (size_t k = 0; k < sc_deep_fired.size(); ++k) {
                if (sc_deep_fired[k]) {
                    sc_vx += std::cos(sc_dirs[k]);
                    sc_vy += std::sin(sc_dirs[k]);
                }
            }
            float sc_mag = std::sqrt(sc_vx * sc_vx + sc_vy * sc_vy);
            if (sc_mag > 0.1f) {
                float sc_angle = std::atan2(sc_vy, sc_vx);
                for (size_t j = 0; j < l5_size; ++j) {
                    float cos_sim = std::cos(m1_preferred_dir_[j] - sc_angle);
                    if (cos_sim > 0.0f) {
                        l5.inject_basal(j, cos_sim * sc_mag * config_.sc_approach_gain);
                    }
                }
            }
        }

        // (0b) v52: PAG 冻结反射 — 恐惧→运动抑制
        //     PAG dlPAG 激活 (CeA→PAG 脉冲驱动) → 抑制全部 M1 L5
        //     = "害怕就不动" (freeze response)
        //     v43 教训: PAG→M1 激活是错的 (无方向信息→盲目运动→走进危险)
        //     v52: PAG→M1 抑制 — 冻结不需要方向, 只需要停
        //     生物学: PAG→脑干抑制性网状核→运动神经元池抑制 (Bandler 1994)
        if (pag_ && config_.pag_freeze_gain > 0.01f) {
            const auto& pag_dl_fired = pag_->dlpag().fired();
            int dl_fires = 0;
            for (size_t k = 0; k < pag_dl_fired.size(); ++k)
                if (pag_dl_fired[k]) dl_fires++;
            if (dl_fires > 0) {
                float inhibition = static_cast<float>(dl_fires) * config_.pag_freeze_gain;
                for (size_t j = 0; j < l5_size; ++j) {
                    l5.inject_basal(j, -inhibition);
                }
            }
        }

        // (0c) v59: 墙壁回避反射 — 视觉 patch 检测前方墙壁 → M1 偏离
        //     视觉 patch 中 vis_wall (0.1) 的像素 → 计算墙壁质心方向
        //     M1 注入反方向 cos 驱动 = "看到墙就转向"
        //     生物学: 触须/视动反射 — 不需要学习, 硬连线回避 (Goodale 2011)
        if (config_.wall_avoid_gain > 0.01f && i == 0) {
            auto obs = env_->observe();
            int vw = static_cast<int>(config_.vision_width);
            int vh = static_cast<int>(config_.vision_height);
            int cx = vw / 2, cy = vh / 2;
            float wall_vx = 0.0f, wall_vy = 0.0f;
            int wall_count = 0;
            for (int vy = 0; vy < vh; ++vy) {
                for (int vx = 0; vx < vw; ++vx) {
                    float val = obs[vy * vw + vx];
                    // Wall = 0.1, check within tolerance
                    if (std::abs(val - 0.1f) < 0.05f && !(vx == cx && vy == cy)) {
                        wall_vx += static_cast<float>(vx - cx);
                        wall_vy += static_cast<float>(vy - cy);
                        wall_count++;
                    }
                }
            }
            if (wall_count >= 2) {
                // Normalize and reverse: move AWAY from wall centroid
                float wm = std::sqrt(wall_vx * wall_vx + wall_vy * wall_vy);
                if (wm > 0.1f) {
                    float avoid_angle = std::atan2(-wall_vy, -wall_vx);
                    float strength = std::min(static_cast<float>(wall_count), 8.0f)
                                   / 8.0f * config_.wall_avoid_gain;
                    for (size_t j = 0; j < l5_size; ++j) {
                        float cos_sim = std::cos(m1_preferred_dir_[j] - avoid_angle);
                        if (cos_sim > 0.0f) {
                            l5.inject_basal(j, cos_sim * strength);
                        }
                    }
                }
            }
        }

        // (1) v45: M1 L5 population vector exploration (Georgopoulos 1986)
        //     Each L5 neuron has a preferred direction θ_i.
        //     Neurons aligned with attractor_angle get strong drive (cosine weighting).
        //     BG population vector can override as learning progresses.
        {
            std::uniform_real_distribution<float> jitter(-attractor_jitter, attractor_jitter);
            for (size_t j = 0; j < l5_size; ++j) {
                float cos_sim = std::cos(m1_preferred_dir_[j] - attractor_angle);
                // Aligned neurons (cos>0): scale from background to attractor drive
                // Opposite neurons (cos<0): background only
                float drive = background_drive
                    + std::max(0.0f, cos_sim) * (attractor_drive - background_drive);
                drive += jitter(motor_rng_);
                if (drive > 0.0f) l5.inject_basal(j, drive);
            }
        }

        // (2) v45: BG D1 population vector → M1 directional bias
        //     D1 fired neurons contribute their preferred direction to a population vector.
        //     The BG vector biases M1 neurons via cosine similarity.
        //     DA-STDP changes WHICH D1 neurons fire → shapes the population vector direction.
        {
            const auto& d1_fired = bg_->d1().fired();
            float bg_vx = 0.0f, bg_vy = 0.0f;
            for (size_t k = 0; k < d1_size; ++k) {
                if (d1_fired[k]) {
                    bg_vx += std::cos(d1_preferred_dir_[k]);
                    bg_vy += std::sin(d1_preferred_dir_[k]);
                }
            }
            float bg_mag = std::sqrt(bg_vx * bg_vx + bg_vy * bg_vy);
            if (bg_mag > 0.1f) {
                float bg_angle = std::atan2(bg_vy, bg_vx);
                for (size_t j = 0; j < l5_size; ++j) {
                    float cos_sim = std::cos(m1_preferred_dir_[j] - bg_angle);
                    if (cos_sim > 0.0f) {
                        l5.inject_basal(j, cos_sim * bg_mag * bg_to_m1_gain);
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

    // B3. Decode action from M1 L5 (biological: M1 is the motor output)
    // v55: continuous movement is the ONLY mode — no discrete 4-direction path
    auto [dx, dy] = decode_m1_continuous(l5_accum);
    Action action = decode_m1_action(l5_accum);  // nearest cardinal for efference copy/replay

    // --- Phase C: Act in environment ---
    Environment::Result result = env_->step(dx, dy);

    // Store reward as pending (will be processed at START of next agent_step)
    // Only trigger Phase A for significant rewards (food/danger), not step penalties
    pending_reward_ = result.reward * config_.reward_scale;
    has_pending_reward_ = (std::abs(result.reward) > 0.05f);

    // v52b: 新奇性 habituation 更新 (不放大奖励信号!)
    // 生物学: 第一口食物不是 10 倍好吃, 而是被记了 10 倍久
    //   奖励放大会炸掉 DA-STDP 权重 (reward_scale=30 × novelty=10 = 300×)
    //   新奇性只影响回放次数, 不影响奖励强度
    if (has_pending_reward_) {
        if (result.reward > 0.05f) {
            food_novelty_ *= 0.5f;   // habituation: 每次食物减半
        } else if (result.reward < -0.05f) {
            danger_novelty_ *= 0.5f;  // habituation: 每次危险减半
        }
    }

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
        // v52b: 新奇性放大回放 — 第一次食物回放更多次
        //   生物学: 新奇经验触发更强 SWR replay (Atherton 2015)
        //   第一次食物: novelty=1.0 → 回放 1 + boost 轮 (最多 1+boost 轮)
        //   第 N 次: novelty≈0 → 回放 1 轮 (正常)
        //   这是"一次学习"的核心: 不是 DA 更强, 而是练得更多
        if (result.reward > 0.05f && agent_step_count_ >= 10) {
            int replay_rounds = 1 + static_cast<int>(
                food_novelty_ * (config_.novelty_da_boost - 1.0f));
            for (int p = 0; p < replay_rounds; ++p) {
                run_awake_replay(result.reward);
            }
        }
        // Negative replay: danger hit → replay old failures (consolidate NoGo)
        // v52b: 新奇性放大 — 第一次危险回放更多次
        if (config_.enable_negative_replay && config_.enable_lhb &&
            result.reward < -0.05f && agent_step_count_ >= 200) {
            int replay_rounds = 1 + static_cast<int>(
                danger_novelty_ * (config_.novelty_da_boost - 1.0f));
            for (int p = 0; p < replay_rounds; ++p) {
                run_negative_replay(result.reward);
            }
        }
    }

    // Update state
    last_action_ = action;
    last_reward_ = result.reward;
    agent_step_count_++;

    // Record history
    size_t hi = history_idx_ % reward_history_.size();
    reward_history_[hi] = result.reward;
    food_history_[hi] = result.positive_event ? 1 : 0;
    history_idx_++;

    // v59: 探索饥饿计数器
    if (result.positive_event || result.negative_event) {
        steps_since_reward_ = 0;
    } else {
        steps_since_reward_++;
    }

    // Callback
    if (callback_) {
        callback_(agent_step_count_, action, result.reward,
                  static_cast<int>(result.pos_x), static_cast<int>(result.pos_y));
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
    auto obs = env_->observe();  // NxN patch from environment
    visual_encoder_.encode_and_inject(obs, lgn_);
}
// =============================================================================
// Action decoding: M1 L5 population vector → direction (Georgopoulos 1986)
// =============================================================================

Action ClosedLoopAgent::decode_m1_action(const std::vector<int>& l5_accum) const {
    // v45: Population vector decoding — each L5 neuron has a preferred direction.
    // The population vector = Σ (fire_count_i × preferred_direction_i).
    // The angle of this vector determines movement direction.
    // Biology: motor cortex neurons have broad directional tuning curves.
    //   Population vector accurately predicts arm movement direction
    //   even though individual neurons are noisy (Georgopoulos et al. 1986).

    if (l5_accum.size() < 2 || m1_preferred_dir_.empty()) return Action::STAY;

    float vx = 0.0f, vy = 0.0f;
    for (size_t i = 0; i < l5_accum.size() && i < m1_preferred_dir_.size(); ++i) {
        if (l5_accum[i] > 0) {
            vx += static_cast<float>(l5_accum[i]) * std::cos(m1_preferred_dir_[i]);
            vy += static_cast<float>(l5_accum[i]) * std::sin(m1_preferred_dir_[i]);
        }
    }

    float mag = std::sqrt(vx * vx + vy * vy);
    if (mag < 0.01f) return Action::STAY;  // no coherent direction

    // Map population vector angle to closest cardinal direction
    // UP=π/2, DOWN=-π/2, LEFT=π, RIGHT=0
    float angle = std::atan2(vy, vx);
    constexpr float DIR_ANGLES[4] = {1.5708f, -1.5708f, 3.14159f, 0.0f};
    float best_cos = -2.0f;
    int best_dir = 4;  // STAY
    for (int d = 0; d < 4; ++d) {
        float c = std::cos(angle - DIR_ANGLES[d]);
        if (c > best_cos) {
            best_cos = c;
            best_dir = d;
        }
    }
    return static_cast<Action>(best_dir);
}

// =============================================================================
// v55: Continuous action decoding — population vector → (dx, dy) displacement
// =============================================================================

std::pair<float, float> ClosedLoopAgent::decode_m1_continuous(
        const std::vector<int>& l5_accum) const {
    // Same population vector computation as decode_m1_action, but output is
    // continuous (dx, dy) instead of discrete Action.
    // Magnitude controls speed: weak coherence → slow, strong → fast.
    // Biology: motor cortex population vector predicts both direction AND speed
    //   of arm movements (Georgopoulos 1988, Moran & Schwartz 1999).

    if (l5_accum.size() < 2 || m1_preferred_dir_.empty())
        return {0.0f, 0.0f};

    float vx = 0.0f, vy = 0.0f;
    float total_fires = 0.0f;
    for (size_t i = 0; i < l5_accum.size() && i < m1_preferred_dir_.size(); ++i) {
        if (l5_accum[i] > 0) {
            float f = static_cast<float>(l5_accum[i]);
            vx += f * std::cos(m1_preferred_dir_[i]);
            vy += f * std::sin(m1_preferred_dir_[i]);
            total_fires += f;
        }
    }

    if (total_fires < 1.0f) return {0.0f, 0.0f};

    // Normalize by total fires → coherence ∈ [0, 1]
    // All neurons firing same direction → coherence=1 → full speed
    // Neurons firing uniformly → coherence≈0 → near-zero speed (STAY equivalent)
    float mag = std::sqrt(vx * vx + vy * vy);
    float coherence = mag / total_fires;

    if (coherence < 0.05f) return {0.0f, 0.0f};  // too incoherent → stay

    // Scale to step size: coherence × max_step
    float speed = coherence * config_.continuous_step_size;
    float angle = std::atan2(vy, vx);

    // GridWorld convention: +x = RIGHT, +y = DOWN
    float dx = speed * std::cos(angle);
    float dy = -speed * std::sin(angle);  // negate: math +y is UP, grid +y is DOWN

    return {dx, dy};
}

// =============================================================================
// Reward: inject to Hypothalamus (hedonic sensory interface)
// v46: Reward no longer injected directly into VTA. Instead, it enters through
// Hypothalamus LH/PVN (sensory pathway), propagates to VTA via SpikeBus.
// Analogous to visual input entering through LGN, not directly into V1.
// =============================================================================

void ClosedLoopAgent::inject_reward(float reward) {
    if (hypo_ && std::abs(reward) > 0.001f) {
        hypo_->inject_hedonic(reward);
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

    // --- NREM consolidation: hippocampus SWR + systems consolidation ---
    // v36: CLS systems consolidation (McClelland 1995, Kumaran 2016)
    //   1. Hippocampus generates SWR spontaneously (CA3 noise → pattern completion)
    //   2. SWR propagates via SpikeBus to cortex (Sub→dlPFC)
    //   3. NEW: During SWR, hippocampal retrieval_bias actively teaches BG
    //      → DA is briefly elevated above baseline (reward prediction from memory)
    //      → BG DA-STDP updates weights based on hippocampal spatial knowledge
    //   4. This transfers knowledge from hippocampal CA3 STDP (stable, no decay)
    //      to BG DA-STDP (volatile, subject to weight decay)
    //   → BG weights are periodically "refreshed" from hippocampal memory
    //   → Catastrophic forgetting is prevented because hippocampus retains memories
    size_t total_nrem = config_.sleep_nrem_steps;

    for (size_t i = 0; i < total_nrem; ++i) {
        // Step the ENTIRE brain (hippocampus SWR → SpikeBus → cortex)
        engine_.step();
        sleep_mgr_.step();

        // v36: Systems consolidation — spatial value map → BG during SWR
        // Biology: During SWR, hippocampal replay reactivates place cells.
        //   Spatial value map provides the correct directional gradient.
        //   VTA DA burst during SWR enables BG DA-STDP weight update.
        //   Effect: BG "re-learns" navigation from spatial memory during sleep.
        if (hipp_ && hipp_->is_swr() && !spatial_value_map_.empty()) {
            // Pick a random position with nonzero value to "replay"
            // Biology: SWR replays trajectories through valued locations
            int w = spatial_map_w_;
            int h = spatial_map_h_;
            float best_val = 0.0f;
            int best_x = -1, best_y = -1;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    float v = std::abs(spatial_value_map_[y * w + x]);
                    if (v > best_val) { best_val = v; best_x = x; best_y = y; }
                }
            }
            if (best_val > 0.05f) {
                // Compute value gradient at the replayed position
                float adj[4] = {0, 0, 0, 0};
                if (best_y > 0)     adj[0] = spatial_value_map_[(best_y-1)*w + best_x];
                if (best_y < h - 1) adj[1] = spatial_value_map_[(best_y+1)*w + best_x];
                if (best_x > 0)     adj[2] = spatial_value_map_[best_y*w + (best_x-1)];
                if (best_x < w - 1) adj[3] = spatial_value_map_[best_y*w + (best_x+1)];
                bg_->inject_sensory_context(adj);

                // SWR-triggered DA burst (Gomperts 2015)
                float swr_da = config_.sleep_positive_da + 0.15f;
                bg_->set_da_level(std::min(swr_da, 0.6f));
            }
        } else {
            bg_->set_da_level(config_.sleep_positive_da);
        }
    }

    // --- Wake up ---
    sleep_mgr_.wake_up();
    if (hipp_) hipp_->disable_sleep_replay();
    bg_->set_da_level(saved_da);
}

void ClosedLoopAgent::update_spatial_value_map(float reward) {
    // v36: Update cognitive map with reward outcome at current position.
    // Biology: Hippocampal place cells + OFC value coding.
    //   Food location → positive value, Danger location → negative value.
    //   Values diffuse to neighbors (spatial generalization).
    //   Slow decay ensures long-term spatial memory.

    if (spatial_value_map_.empty()) return;
    int w = spatial_map_w_;
    int h = spatial_map_h_;
    int ax = static_cast<int>(env_->pos_x()), ay = static_cast<int>(env_->pos_y());

    // 1. Asymmetric decay: food memories persist, danger memories extinguish
    // Biology: fear extinction (Milad & Quirk 2002) is faster than reward memory
    for (auto& v : spatial_value_map_) {
        v *= (v >= 0.0f) ? SPATIAL_VALUE_DECAY_POS : SPATIAL_VALUE_DECAY_NEG;
    }

    // 2. Update current position with reward signal (food/danger only, not step cost)
    if (std::abs(reward) > 0.1f) {
        int idx = ay * w + ax;
        spatial_value_map_[idx] += SPATIAL_VALUE_LR * (reward - spatial_value_map_[idx]);
        // Clamp to prevent runaway accumulation
        spatial_value_map_[idx] = std::clamp(spatial_value_map_[idx],
                                              SPATIAL_VALUE_MIN, SPATIAL_VALUE_MAX);

        // 3. Diffuse to adjacent cells (spatial generalization)
        // Only diffuse positive values (food attracts neighbors)
        // Negative values don't diffuse — danger is localized, not area-wide
        if (reward > 0.0f) {
            float diffuse = SPATIAL_VALUE_LR * reward * 0.3f;
            if (ay > 0)     spatial_value_map_[(ay-1)*w + ax] += diffuse;
            if (ay < h - 1) spatial_value_map_[(ay+1)*w + ax] += diffuse;
            if (ax > 0)     spatial_value_map_[ay*w + (ax-1)] += diffuse;
            if (ax < w - 1) spatial_value_map_[ay*w + (ax+1)] += diffuse;
        }
    }
}

} // namespace wuyun
