#include "circuit/cortical_column.h"
#include <random>
#include <algorithm>
#include <cmath>

namespace wuyun {

// =============================================================================
// Helper: generate random sparse connections (COO format)
// =============================================================================

struct COO {
    std::vector<int32_t> pre;
    std::vector<int32_t> post;
    std::vector<float>   weights;
    std::vector<int32_t> delays;
};

static COO make_random_connections(
    size_t n_pre, size_t n_post,
    float prob, float weight, int32_t delay,
    uint32_t seed
) {
    COO coo;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < n_pre; ++i) {
        for (size_t j = 0; j < n_post; ++j) {
            if (dist(rng) < prob) {
                coo.pre.push_back(static_cast<int32_t>(i));
                coo.post.push_back(static_cast<int32_t>(j));
                coo.weights.push_back(weight);
                coo.delays.push_back(delay);
            }
        }
    }
    return coo;
}

// =============================================================================
// Helper: make a dummy SynapseGroup (0 synapses) as placeholder
// =============================================================================

static SynapseGroup make_empty_synapse(size_t n_pre, size_t n_post,
                                        const SynapseParams& params,
                                        CompartmentType target) {
    return SynapseGroup(n_pre, n_post, {}, {}, {}, {}, params, target);
}

// =============================================================================
// Constructor
// =============================================================================

// Helper: neuron params factories
static NeuronParams make_l4_stellate_params() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 20.0f;
    p.somatic.r_s = 1.0f; p.somatic.a = 0.01f;
    p.somatic.b = 3.0f; p.somatic.tau_w = 200.0f;
    p.somatic.refractory_period = 3;
    p.kappa = 0.1f; p.kappa_backward = 0.05f;
    p.burst_spike_count = 2; p.burst_isi = 3;
    return p;
}
static NeuronParams make_l6_params() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 25.0f;
    p.somatic.r_s = 0.9f; p.somatic.a = 0.01f;
    p.somatic.b = 4.0f; p.somatic.tau_w = 250.0f;
    p.somatic.refractory_period = 3;
    p.kappa = 0.2f; p.kappa_backward = 0.1f;
    p.burst_spike_count = 2; p.burst_isi = 3;
    return p;
}
static NeuronParams make_sst_params() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 20.0f;
    p.somatic.r_s = 0.9f; p.somatic.a = 0.05f;
    p.somatic.b = 2.0f; p.somatic.tau_w = 100.0f;
    p.somatic.refractory_period = 2;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}
static NeuronParams make_vip_params() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 15.0f;
    p.somatic.r_s = 1.0f; p.somatic.a = 0.03f;
    p.somatic.b = 1.0f; p.somatic.tau_w = 80.0f;
    p.somatic.refractory_period = 2;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

#define EMPTY_SYN(p, t) make_empty_synapse(1, 1, p, t)

CorticalColumn::CorticalColumn(const ColumnConfig& cfg)
    : config_(cfg)
    // --- Excitatory populations ---
    , l4_stellate_(cfg.n_l4_stellate, make_l4_stellate_params())
    , l23_pyramidal_(cfg.n_l23_pyramidal, L23_PYRAMIDAL_PARAMS())
    , l5_pyramidal_(cfg.n_l5_pyramidal, L5_PYRAMIDAL_PARAMS())
    , l6_pyramidal_(cfg.n_l6_pyramidal, make_l6_params())
    // --- Inhibitory populations ---
    , pv_basket_(cfg.n_pv_basket, PV_BASKET_PARAMS())
    , sst_martinotti_(cfg.n_sst_martinotti, make_sst_params())
    , vip_(cfg.n_vip, make_vip_params())
    // --- AMPA synapses (placeholders) ---
    , syn_l4_to_l23_(EMPTY_SYN(AMPA_PARAMS, CompartmentType::BASAL))
    , syn_l23_to_l5_(EMPTY_SYN(AMPA_PARAMS, CompartmentType::BASAL))
    , syn_l5_to_l6_(EMPTY_SYN(AMPA_PARAMS, CompartmentType::BASAL))
    , syn_l6_to_l4_(EMPTY_SYN(AMPA_PARAMS, CompartmentType::BASAL))
    , syn_l23_recurrent_(EMPTY_SYN(AMPA_PARAMS, CompartmentType::BASAL))
    // --- NMDA synapses (placeholders) ---
    , syn_l4_to_l23_nmda_(EMPTY_SYN(NMDA_PARAMS, CompartmentType::BASAL))
    , syn_l23_to_l5_nmda_(EMPTY_SYN(NMDA_PARAMS, CompartmentType::BASAL))
    , syn_l23_rec_nmda_(EMPTY_SYN(NMDA_PARAMS, CompartmentType::BASAL))
    // --- Exc -> Inh ---
    , syn_exc_to_pv_(EMPTY_SYN(AMPA_PARAMS, CompartmentType::SOMA))
    , syn_exc_to_sst_(EMPTY_SYN(AMPA_PARAMS, CompartmentType::SOMA))
    , syn_exc_to_vip_(EMPTY_SYN(AMPA_PARAMS, CompartmentType::SOMA))
    // --- PV -> all exc soma ---
    , syn_pv_to_l23_(EMPTY_SYN(GABA_A_PARAMS, CompartmentType::SOMA))
    , syn_pv_to_l4_(EMPTY_SYN(GABA_A_PARAMS, CompartmentType::SOMA))
    , syn_pv_to_l5_(EMPTY_SYN(GABA_A_PARAMS, CompartmentType::SOMA))
    , syn_pv_to_l6_(EMPTY_SYN(GABA_A_PARAMS, CompartmentType::SOMA))
    // --- SST -> apical ---
    , syn_sst_to_l23_api_(EMPTY_SYN(GABA_B_PARAMS, CompartmentType::APICAL))
    , syn_sst_to_l5_api_(EMPTY_SYN(GABA_B_PARAMS, CompartmentType::APICAL))
    // --- VIP -> SST ---
    , syn_vip_to_sst_(EMPTY_SYN(GABA_A_PARAMS, CompartmentType::SOMA))
    // --- L6 -> L2/3 prediction (v27) ---
    , syn_l6_to_l23_predict_(EMPTY_SYN(AMPA_PARAMS, CompartmentType::APICAL))
{
    build_synapses();
}

#undef EMPTY_SYN

// =============================================================================
// Build synapses with random sparse connectivity
// =============================================================================

void CorticalColumn::build_synapses() {
    const auto& c = config_;
    uint32_t seed = 42;

    // Helper lambda for building a SynapseGroup
    auto build = [&](size_t npre, size_t npost, float prob, float w,
                     const SynapseParams& sp, CompartmentType tgt) -> SynapseGroup {
        auto coo = make_random_connections(npre, npost, prob, w, 1, seed++);
        return SynapseGroup(npre, npost, coo.pre, coo.post, coo.weights, coo.delays, sp, tgt);
    };

    // ===================== Excitatory AMPA =====================
    syn_l4_to_l23_     = build(c.n_l4_stellate,   c.n_l23_pyramidal, c.p_l4_to_l23,    c.w_exc,       AMPA_PARAMS, CompartmentType::BASAL);
    syn_l23_to_l5_     = build(c.n_l23_pyramidal,  c.n_l5_pyramidal,  c.p_l23_to_l5,    c.w_exc,       AMPA_PARAMS, CompartmentType::BASAL);
    syn_l5_to_l6_      = build(c.n_l5_pyramidal,   c.n_l6_pyramidal,  c.p_l5_to_l6,     c.w_exc,       AMPA_PARAMS, CompartmentType::BASAL);
    syn_l6_to_l4_      = build(c.n_l6_pyramidal,   c.n_l4_stellate,   c.p_l6_to_l4,     c.w_l6_to_l4,  AMPA_PARAMS, CompartmentType::BASAL);
    syn_l23_recurrent_  = build(c.n_l23_pyramidal,  c.n_l23_pyramidal, c.p_l23_recurrent, c.w_recurrent, AMPA_PARAMS, CompartmentType::BASAL);

    // ===================== Excitatory NMDA (parallel slow) =====================
    syn_l4_to_l23_nmda_ = build(c.n_l4_stellate,   c.n_l23_pyramidal, c.p_l4_to_l23,    c.w_nmda, NMDA_PARAMS, CompartmentType::BASAL);
    syn_l23_to_l5_nmda_ = build(c.n_l23_pyramidal,  c.n_l5_pyramidal,  c.p_l23_to_l5,    c.w_nmda, NMDA_PARAMS, CompartmentType::BASAL);
    syn_l23_rec_nmda_   = build(c.n_l23_pyramidal,  c.n_l23_pyramidal, c.p_l23_recurrent, c.w_nmda * 0.5f, NMDA_PARAMS, CompartmentType::BASAL);

    // ===================== Auto-enable STDP if configured =====================
    if (c.stdp_enabled) {
        enable_stdp();
    }

    // ===================== Exc -> Inhibitory (AMPA) =====================
    syn_exc_to_pv_  = build(c.n_l23_pyramidal, c.n_pv_basket,      c.p_exc_to_pv,  c.w_exc, AMPA_PARAMS, CompartmentType::SOMA);
    syn_exc_to_sst_ = build(c.n_l23_pyramidal, c.n_sst_martinotti, c.p_exc_to_sst, c.w_exc, AMPA_PARAMS, CompartmentType::SOMA);
    syn_exc_to_vip_ = build(c.n_l23_pyramidal, c.n_vip,            c.p_exc_to_vip, c.w_exc, AMPA_PARAMS, CompartmentType::SOMA);

    // ===================== PV -> ALL excitatory soma (GABA_A) =====================
    syn_pv_to_l23_ = build(c.n_pv_basket, c.n_l23_pyramidal, c.p_pv_to_l23, c.w_inh, GABA_A_PARAMS, CompartmentType::SOMA);
    syn_pv_to_l4_  = build(c.n_pv_basket, c.n_l4_stellate,   c.p_pv_to_l4,  c.w_inh, GABA_A_PARAMS, CompartmentType::SOMA);
    syn_pv_to_l5_  = build(c.n_pv_basket, c.n_l5_pyramidal,  c.p_pv_to_l5,  c.w_inh, GABA_A_PARAMS, CompartmentType::SOMA);
    syn_pv_to_l6_  = build(c.n_pv_basket, c.n_l6_pyramidal,  c.p_pv_to_l6,  c.w_inh, GABA_A_PARAMS, CompartmentType::SOMA);

    // ===================== SST -> L2/3 AND L5 apical (GABA_B) =====================
    syn_sst_to_l23_api_ = build(c.n_sst_martinotti, c.n_l23_pyramidal, c.p_sst_to_l23_api, c.w_inh, GABA_B_PARAMS, CompartmentType::APICAL);
    syn_sst_to_l5_api_  = build(c.n_sst_martinotti, c.n_l5_pyramidal,  c.p_sst_to_l5_api,  c.w_inh, GABA_B_PARAMS, CompartmentType::APICAL);

    // ===================== VIP -> SST (GABA_A disinhibition) =====================
    syn_vip_to_sst_ = build(c.n_vip, c.n_sst_martinotti, c.p_vip_to_sst, c.w_inh, GABA_A_PARAMS, CompartmentType::SOMA);
}

// =============================================================================
// Spike delivery helper
// =============================================================================

void CorticalColumn::deliver_and_inject(
    const NeuronPopulation& pre,
    SynapseGroup& syn,
    NeuronPopulation& post,
    float dt
) {
    if (syn.n_synapses() == 0) return;

    syn.deliver_spikes(pre.fired(), pre.spike_type());
    auto currents = syn.step_and_compute(post.v_soma(), dt);

    CompartmentType target = syn.target();
    for (size_t i = 0; i < post.size(); ++i) {
        if (std::abs(currents[i]) < 1e-12f) continue;
        switch (target) {
            case CompartmentType::BASAL:
                post.inject_basal(i, currents[i]);
                break;
            case CompartmentType::APICAL:
                post.inject_apical(i, currents[i]);
                break;
            case CompartmentType::SOMA:
                post.inject_soma(i, currents[i]);
                break;
        }
    }
}

// =============================================================================
// External input injection
// =============================================================================

void CorticalColumn::inject_feedforward(const std::vector<float>& currents) {
    size_t n = std::min(currents.size(), l4_stellate_.size());
    for (size_t i = 0; i < n; ++i) {
        l4_stellate_.inject_basal(i, currents[i]);
    }
}

void CorticalColumn::inject_feedback(
    const std::vector<float>& currents_l23,
    const std::vector<float>& currents_l5
) {
    // Feedback -> L2/3 apical (via L1)
    size_t n23 = std::min(currents_l23.size(), l23_pyramidal_.size());
    for (size_t i = 0; i < n23; ++i) {
        l23_pyramidal_.inject_apical(i, currents_l23[i]);
    }
    // Feedback -> L5 apical (via L1)
    size_t n5 = std::min(currents_l5.size(), l5_pyramidal_.size());
    for (size_t i = 0; i < n5; ++i) {
        l5_pyramidal_.inject_apical(i, currents_l5[i]);
    }
}

void CorticalColumn::inject_attention(float vip_drive) {
    for (size_t i = 0; i < vip_.size(); ++i) {
        vip_.inject_soma(i, vip_drive);
    }
}

// =============================================================================
// Main step
// =============================================================================

ColumnOutput CorticalColumn::step(int t, float dt) {
    // ================================================================
    // STEP 1: Deliver intra-column spikes from previous step
    // ================================================================

    // --- Excitatory AMPA pathway: L4 → L2/3 → L5 → L6 → L4 ---
    deliver_and_inject(l4_stellate_,   syn_l4_to_l23_,    l23_pyramidal_, dt);
    deliver_and_inject(l23_pyramidal_, syn_l23_to_l5_,    l5_pyramidal_,  dt);
    deliver_and_inject(l5_pyramidal_,  syn_l5_to_l6_,     l6_pyramidal_,  dt);
    deliver_and_inject(l6_pyramidal_,  syn_l6_to_l4_,     l4_stellate_,   dt);
    deliver_and_inject(l23_pyramidal_, syn_l23_recurrent_, l23_pyramidal_, dt);

    // --- v27: L6→L2/3 prediction (apical) — top-down prediction within column ---
    if (predictive_learning_) {
        deliver_and_inject(l6_pyramidal_, syn_l6_to_l23_predict_, l23_pyramidal_, dt);
    }

    // --- Excitatory NMDA pathway (parallel slow channel) ---
    deliver_and_inject(l4_stellate_,   syn_l4_to_l23_nmda_, l23_pyramidal_, dt);
    deliver_and_inject(l23_pyramidal_, syn_l23_to_l5_nmda_, l5_pyramidal_,  dt);
    deliver_and_inject(l23_pyramidal_, syn_l23_rec_nmda_,   l23_pyramidal_, dt);

    // --- Excitatory → Inhibitory ---
    deliver_and_inject(l23_pyramidal_, syn_exc_to_pv_,  pv_basket_,      dt);
    deliver_and_inject(l23_pyramidal_, syn_exc_to_sst_, sst_martinotti_, dt);
    deliver_and_inject(l23_pyramidal_, syn_exc_to_vip_, vip_,            dt);

    // --- PV → ALL excitatory soma (GABA_A fast) ---
    deliver_and_inject(pv_basket_, syn_pv_to_l23_, l23_pyramidal_, dt);
    deliver_and_inject(pv_basket_, syn_pv_to_l4_,  l4_stellate_,   dt);
    deliver_and_inject(pv_basket_, syn_pv_to_l5_,  l5_pyramidal_,  dt);
    deliver_and_inject(pv_basket_, syn_pv_to_l6_,  l6_pyramidal_,  dt);

    // --- SST → L2/3 AND L5 apical (GABA_B slow, blocks burst!) ---
    deliver_and_inject(sst_martinotti_, syn_sst_to_l23_api_, l23_pyramidal_, dt);
    deliver_and_inject(sst_martinotti_, syn_sst_to_l5_api_,  l5_pyramidal_,  dt);

    // --- VIP → SST (GABA_A disinhibition) ---
    deliver_and_inject(vip_, syn_vip_to_sst_, sst_martinotti_, dt);

    // ================================================================
    // STEP 2: Update all populations
    // ================================================================
    l4_stellate_.step(t, dt);
    l23_pyramidal_.step(t, dt);
    l5_pyramidal_.step(t, dt);
    l6_pyramidal_.step(t, dt);
    pv_basket_.step(t, dt);
    sst_martinotti_.step(t, dt);
    vip_.step(t, dt);

    // ================================================================
    // STEP 2.5: Online plasticity (STDP)
    // ================================================================
    if (stdp_active_) {
        // v26: ACh modulation — temporarily scale STDP rates during salient events
        // Biology: NBM ACh enhances LTP window (Froemke 2007), making STDP
        // learn reward-relevant features faster during attention/arousal
        if (ach_stdp_gain_ > 1.01f || ach_stdp_gain_ < 0.99f) {
            auto scale_stdp = [&](SynapseGroup& sg) {
                sg.stdp_params().a_plus  *= ach_stdp_gain_;
                sg.stdp_params().a_minus *= ach_stdp_gain_;
            };
            scale_stdp(syn_l4_to_l23_);
            scale_stdp(syn_l23_recurrent_);
            scale_stdp(syn_l23_to_l5_);
            if (predictive_learning_) scale_stdp(syn_l6_to_l23_predict_);
        }

        if (predictive_learning_) {
            // v27: ERROR-GATED STDP (Whittington & Bogacz 2017)
            // L4→L2/3: only regular spikes (prediction errors) trigger LTP
            // burst spikes (prediction match) do NOT update feedforward weights
            // → "learn new features, don't overwrite already-learned ones"
            syn_l4_to_l23_.apply_stdp_error_gated(
                l4_stellate_.fired(), l23_pyramidal_.fired(),
                l23_pyramidal_.spike_type(),
                static_cast<int8_t>(SpikeType::REGULAR), t);

            // L6→L2/3 prediction STDP: L6 learns to predict L2/3 activity
            // L6 fires + L2/3 fires → LTP (good prediction)
            // L6 fires + L2/3 silent → LTD (false prediction)
            syn_l6_to_l23_predict_.apply_stdp(
                l6_pyramidal_.fired(), l23_pyramidal_.fired(), t);
        } else {
            // Original Hebbian STDP (no error gating)
            syn_l4_to_l23_.apply_stdp(l4_stellate_.fired(), l23_pyramidal_.fired(), t);
        }

        // L2/3 recurrent + L2/3→L5: always standard STDP (not error-gated)
        syn_l23_recurrent_.apply_stdp(l23_pyramidal_.fired(), l23_pyramidal_.fired(), t);
        syn_l23_to_l5_.apply_stdp(l23_pyramidal_.fired(), l5_pyramidal_.fired(), t);

        // Restore original STDP params (ACh modulation)
        if (ach_stdp_gain_ > 1.01f || ach_stdp_gain_ < 0.99f) {
            auto unscale = [&](SynapseGroup& sg) {
                sg.stdp_params().a_plus  /= ach_stdp_gain_;
                sg.stdp_params().a_minus /= ach_stdp_gain_;
            };
            unscale(syn_l4_to_l23_);
            unscale(syn_l23_recurrent_);
            unscale(syn_l23_to_l5_);
            if (predictive_learning_) unscale(syn_l6_to_l23_predict_);
        }
    }

    // ================================================================
    // STEP 2.6: Homeostatic plasticity (synaptic scaling)
    // ================================================================
    if (homeo_active_) {
        // Update rate estimates every step
        homeo_l4_->update_rates(l4_stellate_.fired().data(), dt);
        homeo_l23_->update_rates(l23_pyramidal_.fired().data(), dt);
        homeo_l5_->update_rates(l5_pyramidal_.fired().data(), dt);
        homeo_l6_->update_rates(l6_pyramidal_.fired().data(), dt);

        // Apply scaling periodically
        ++homeo_step_count_;
        if (homeo_step_count_ >= homeo_interval_) {
            homeo_step_count_ = 0;
            apply_homeostatic_scaling();
        }
    }

    // ================================================================
    // STEP 3: Classify output
    // ================================================================
    ColumnOutput out;
    classify_output(out);
    return out;
}

// =============================================================================
// Classify output
// =============================================================================

void CorticalColumn::classify_output(ColumnOutput& out) {
    size_t n23 = l23_pyramidal_.size();
    size_t n5  = l5_pyramidal_.size();
    size_t n6  = l6_pyramidal_.size();

    out.l23_regular.resize(n23, 0);
    out.l23_burst.resize(n23, 0);
    out.l5_burst.resize(n5, 0);
    out.l6_fired.resize(n6, 0);
    out.n_regular = 0;
    out.n_burst = 0;
    out.n_drive = 0;

    // L2/3: regular = prediction error, burst = match
    for (size_t i = 0; i < n23; ++i) {
        auto st = static_cast<SpikeType>(l23_pyramidal_.spike_type()[i]);
        if (st == SpikeType::REGULAR) {
            out.l23_regular[i] = 1;
            out.n_regular++;
        } else if (is_burst(st)) {
            out.l23_burst[i] = 1;
            out.n_burst++;
        }
    }

    // L5: only burst counts as drive output
    for (size_t i = 0; i < n5; ++i) {
        auto st = static_cast<SpikeType>(l5_pyramidal_.spike_type()[i]);
        if (is_burst(st)) {
            out.l5_burst[i] = 1;
            out.n_drive++;
        }
    }

    // L6: any firing = prediction output
    for (size_t i = 0; i < n6; ++i) {
        out.l6_fired[i] = l6_pyramidal_.fired()[i];
    }
}

// =============================================================================
// Info
// =============================================================================

size_t CorticalColumn::total_neurons() const {
    return l4_stellate_.size() + l23_pyramidal_.size() +
           l5_pyramidal_.size() + l6_pyramidal_.size() +
           pv_basket_.size() + sst_martinotti_.size() + vip_.size();
}

size_t CorticalColumn::total_synapses() const {
    return
        // AMPA excitatory
        syn_l4_to_l23_.n_synapses() + syn_l23_to_l5_.n_synapses() +
        syn_l5_to_l6_.n_synapses() + syn_l6_to_l4_.n_synapses() +
        syn_l23_recurrent_.n_synapses() +
        // NMDA excitatory
        syn_l4_to_l23_nmda_.n_synapses() + syn_l23_to_l5_nmda_.n_synapses() +
        syn_l23_rec_nmda_.n_synapses() +
        // Exc -> Inh
        syn_exc_to_pv_.n_synapses() + syn_exc_to_sst_.n_synapses() +
        syn_exc_to_vip_.n_synapses() +
        // PV -> all exc
        syn_pv_to_l23_.n_synapses() + syn_pv_to_l4_.n_synapses() +
        syn_pv_to_l5_.n_synapses() + syn_pv_to_l6_.n_synapses() +
        // SST -> apical
        syn_sst_to_l23_api_.n_synapses() + syn_sst_to_l5_api_.n_synapses() +
        // VIP -> SST
        syn_vip_to_sst_.n_synapses();
}

// =============================================================================
// Enable STDP on excitatory synapses
// =============================================================================

void CorticalColumn::enable_stdp() {
    STDPParams params;
    params.a_plus   = config_.stdp_a_plus;
    params.a_minus  = config_.stdp_a_minus;
    params.tau_plus  = config_.stdp_tau;
    params.tau_minus = config_.stdp_tau;
    params.w_min     = 0.0f;
    params.w_max     = config_.stdp_w_max;

    // L4→L2/3: feedforward feature learning (most important for self-organization)
    syn_l4_to_l23_.enable_stdp(params);
    // L2/3 recurrent: lateral attractor dynamics
    syn_l23_recurrent_.enable_stdp(params);
    // L2/3→L5: output pathway learning
    syn_l23_to_l5_.enable_stdp(params);

    stdp_active_ = true;
}

void CorticalColumn::enable_predictive_learning() {
    // v27: Predictive coding learning (Whittington & Bogacz 2017)
    // L6 learns to predict L2/3 activity. L4→L2/3 STDP becomes error-gated.
    // Requires STDP to be enabled first.
    if (!stdp_active_) enable_stdp();

    // Build L6→L2/3 prediction synapse (L6 projects to L2/3 APICAL dendrites)
    // Biology: L6 corticothalamic neurons send collaterals to L1/L2/3 apical,
    // providing top-down predictions within the same column
    {
        auto coo = make_random_connections(
            l6_pyramidal_.size(), l23_pyramidal_.size(),
            0.15f, 0.2f, 1, 777);
        syn_l6_to_l23_predict_ = SynapseGroup(
            l6_pyramidal_.size(), l23_pyramidal_.size(),
            coo.pre, coo.post, coo.weights, coo.delays,
            AMPA_PARAMS, CompartmentType::APICAL);
    }

    // Enable STDP on the prediction synapse
    // L6 pre + L2/3 post → LTP (prediction matches input = good, strengthen)
    // L6 pre + L2/3 silent → LTD (false prediction = bad, weaken)
    STDPParams pred_params;
    pred_params.a_plus   = config_.stdp_a_plus * 0.5f;  // Gentler than feedforward
    pred_params.a_minus  = config_.stdp_a_minus * 0.5f;
    pred_params.tau_plus  = config_.stdp_tau;
    pred_params.tau_minus = config_.stdp_tau;
    pred_params.w_min     = 0.0f;
    pred_params.w_max     = config_.stdp_w_max;
    syn_l6_to_l23_predict_.enable_stdp(pred_params);

    predictive_learning_ = true;
}

// =============================================================================
// Enable homeostatic plasticity
// =============================================================================

void CorticalColumn::enable_homeostatic(const HomeostaticParams& params) {
    homeo_l4_  = std::make_unique<SynapticScaler>(config_.n_l4_stellate, params);
    homeo_l23_ = std::make_unique<SynapticScaler>(config_.n_l23_pyramidal, params);
    homeo_l5_  = std::make_unique<SynapticScaler>(config_.n_l5_pyramidal, params);
    homeo_l6_  = std::make_unique<SynapticScaler>(config_.n_l6_pyramidal, params);
    homeo_interval_ = params.scale_interval;
    homeo_step_count_ = 0;
    homeo_active_ = true;
}

void CorticalColumn::apply_homeostatic_scaling() {
    // Scale feedforward excitatory AMPA synapses only.
    // Do NOT scale recurrent synapses (they store learned patterns).
    // Do NOT scale inhibitory synapses (separate regulation).

    auto scale_syn = [](SynapticScaler& scaler, SynapseGroup& syn) {
        if (syn.n_synapses() == 0) return;
        scaler.apply_scaling(syn.weights().data(), syn.n_synapses(),
                             syn.col_idx().data());
    };

    // L4 inputs: L6→L4 prediction loop
    scale_syn(*homeo_l4_, syn_l6_to_l4_);

    // L2/3 inputs: L4→L2/3 feedforward (main pathway)
    scale_syn(*homeo_l23_, syn_l4_to_l23_);

    // L5 inputs: L2/3→L5 feedforward
    scale_syn(*homeo_l5_, syn_l23_to_l5_);

    // L6 inputs: L5→L6 feedforward
    scale_syn(*homeo_l6_, syn_l5_to_l6_);
}

} // namespace wuyun
