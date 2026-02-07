#include "region/limbic/amygdala.h"
#include <random>
#include <algorithm>

namespace wuyun {

// =============================================================================
// Helper
// =============================================================================
static SynapseGroup make_empty(size_t n_pre, size_t n_post,
                                const SynapseParams& params,
                                CompartmentType target) {
    return SynapseGroup(n_pre, n_post, {}, {}, {}, {}, params, target);
}

static SynapseGroup build_synapse_group(
    size_t n_pre, size_t n_post, float prob, float weight,
    const SynapseParams& params, CompartmentType target, unsigned seed)
{
    std::vector<int32_t> pre, post, d;
    std::vector<float> w;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n_pre; ++i) {
        for (size_t j = 0; j < n_post; ++j) {
            if (dist(rng) < prob) {
                pre.push_back(static_cast<int32_t>(i));
                post.push_back(static_cast<int32_t>(j));
                w.push_back(weight);
                d.push_back(1);
            }
        }
    }
    if (pre.empty()) return make_empty(n_pre, n_post, params, target);
    return SynapseGroup(n_pre, n_post, pre, post, w, d, params, target);
}

// =============================================================================
// Constructor
// =============================================================================

Amygdala::Amygdala(const AmygdalaConfig& config)
    : BrainRegion(config.name,
                  config.n_la + config.n_bla + config.n_cea + config.n_itc +
                  config.n_mea + config.n_coa + config.n_ab)
    , config_(config)
    , la_(config.n_la, NeuronParams{})       // Standard excitatory
    , bla_(config.n_bla, NeuronParams{})     // Standard excitatory (learning via DA-STDP)
    , cea_(config.n_cea, NeuronParams{})     // Standard excitatory (output)
    , itc_(config.n_itc, PV_BASKET_PARAMS()) // Inhibitory gate
    , mea_(config.n_mea, NeuronParams{})      // MeA (optional)
    , coa_(config.n_coa, NeuronParams{})      // CoA (optional)
    , ab_(config.n_ab, NeuronParams{})        // AB (optional)
    // Synapses (initialized empty, filled in build_synapses)
    , syn_la_to_bla_(make_empty(config.n_la, config.n_bla, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_bla_to_cea_(make_empty(config.n_bla, config.n_cea, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_la_to_cea_(make_empty(config.n_la, config.n_cea, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_bla_to_itc_(make_empty(config.n_bla, config.n_itc, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_itc_to_cea_(make_empty(config.n_itc, config.n_cea, GABA_A_PARAMS, CompartmentType::BASAL))
    , syn_bla_rec_(make_empty(config.n_bla, config.n_bla, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_la_to_mea_(make_empty(std::max<size_t>(1,config.n_la), std::max<size_t>(1,config.n_mea), AMPA_PARAMS, CompartmentType::BASAL))
    , syn_la_to_coa_(make_empty(std::max<size_t>(1,config.n_la), std::max<size_t>(1,config.n_coa), AMPA_PARAMS, CompartmentType::BASAL))
    , syn_bla_to_ab_(make_empty(std::max<size_t>(1,config.n_bla), std::max<size_t>(1,config.n_ab), AMPA_PARAMS, CompartmentType::BASAL))
    , syn_ab_to_cea_(make_empty(std::max<size_t>(1,config.n_ab), std::max<size_t>(1,config.n_cea), AMPA_PARAMS, CompartmentType::BASAL))
    , syn_mea_to_cea_(make_empty(std::max<size_t>(1,config.n_mea), std::max<size_t>(1,config.n_cea), AMPA_PARAMS, CompartmentType::BASAL))
    // PSP buffer
    , psp_la_(config.n_la, 0.0f)
    , psp_itc_(config.n_itc, 0.0f)
    // Aggregate state
    , fired_all_(n_neurons_, 0)
    , spike_type_all_(n_neurons_, 0)
{
    build_synapses();
}

void Amygdala::build_synapses() {
    unsigned seed = 2000;

    // La → BLA (sensory input to learning center)
    syn_la_to_bla_ = build_synapse_group(
        config_.n_la, config_.n_bla, config_.p_la_to_bla, config_.w_la_bla,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // BLA → CeA (fear expression pathway)
    syn_bla_to_cea_ = build_synapse_group(
        config_.n_bla, config_.n_cea, config_.p_bla_to_cea, config_.w_bla_cea,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // La → CeA (direct fast fear pathway)
    syn_la_to_cea_ = build_synapse_group(
        config_.n_la, config_.n_cea, config_.p_la_to_cea, config_.w_la_cea,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // BLA → ITC (drives extinction gate)
    syn_bla_to_itc_ = build_synapse_group(
        config_.n_bla, config_.n_itc, config_.p_bla_to_itc, config_.w_bla_itc,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // ITC → CeA (inhibitory gate: extinction suppresses fear)
    syn_itc_to_cea_ = build_synapse_group(
        config_.n_itc, config_.n_cea, config_.p_itc_to_cea, config_.w_itc_cea,
        GABA_A_PARAMS, CompartmentType::BASAL, seed++);

    // BLA recurrent (maintains valence representations)
    syn_bla_rec_ = build_synapse_group(
        config_.n_bla, config_.n_bla, config_.p_bla_to_bla, config_.w_bla_rec,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // --- Enable fear conditioning STDP on La→BLA ---
    // Biology: BLA LTP is fast (one-shot), NMDA-dependent, gated by US.
    // Very asymmetric: strong LTP, weak LTD (fear is hard to extinguish).
    // (LeDoux 2000, Maren 2001, Rogan et al. 1997)
    if (config_.fear_stdp_enabled) {
        STDPParams fear_stdp;
        fear_stdp.a_plus   = config_.fear_stdp_a_plus;   // 0.10 (10x cortical)
        fear_stdp.a_minus  = config_.fear_stdp_a_minus;  // -0.03 (weak LTD)
        fear_stdp.tau_plus  = config_.fear_stdp_tau;
        fear_stdp.tau_minus = config_.fear_stdp_tau;
        fear_stdp.w_min     = 0.0f;
        fear_stdp.w_max     = config_.fear_stdp_w_max;   // 3.0 (high ceiling)
        syn_la_to_bla_.enable_stdp(fear_stdp);
    }

    // --- Optional nuclei ---
    if (config_.n_mea > 0) {
        syn_la_to_mea_ = build_synapse_group(
            config_.n_la, config_.n_mea, config_.p_la_to_mea, config_.w_mea,
            AMPA_PARAMS, CompartmentType::BASAL, seed++);
        syn_mea_to_cea_ = build_synapse_group(
            config_.n_mea, config_.n_cea, config_.p_mea_to_cea, config_.w_mea,
            AMPA_PARAMS, CompartmentType::BASAL, seed++);
    }
    if (config_.n_coa > 0) {
        syn_la_to_coa_ = build_synapse_group(
            config_.n_la, config_.n_coa, config_.p_la_to_coa, config_.w_coa,
            AMPA_PARAMS, CompartmentType::BASAL, seed++);
    }
    if (config_.n_ab > 0) {
        syn_bla_to_ab_ = build_synapse_group(
            config_.n_bla, config_.n_ab, config_.p_bla_to_ab, config_.w_ab,
            AMPA_PARAMS, CompartmentType::BASAL, seed++);
        syn_ab_to_cea_ = build_synapse_group(
            config_.n_ab, config_.n_cea, config_.p_ab_to_cea, config_.w_ab,
            AMPA_PARAMS, CompartmentType::BASAL, seed++);
    }
}

// =============================================================================
// Step
// =============================================================================

void Amygdala::step(int32_t t, float dt) {
    oscillation_.step(dt);
    neuromod_.step(dt);

    // Inject PSP buffer into La (sensory input)
    for (size_t i = 0; i < psp_la_.size(); ++i) {
        if (psp_la_[i] > 0.5f) la_.inject_basal(i, psp_la_[i]);
        psp_la_[i] *= PSP_DECAY;
    }
    // Inject PSP buffer into ITC (PFC top-down for extinction)
    for (size_t i = 0; i < psp_itc_.size(); ++i) {
        if (psp_itc_[i] > 0.5f) itc_.inject_basal(i, psp_itc_[i]);
        psp_itc_[i] *= PSP_DECAY;
    }

    // 1. La → BLA
    syn_la_to_bla_.deliver_spikes(la_.fired(), la_.spike_type());
    auto i_bla = syn_la_to_bla_.step_and_compute(bla_.v_soma(), dt);
    for (size_t i = 0; i < bla_.size(); ++i) bla_.inject_basal(i, i_bla[i]);

    // 2. BLA recurrent
    syn_bla_rec_.deliver_spikes(bla_.fired(), bla_.spike_type());
    auto i_bla_rec = syn_bla_rec_.step_and_compute(bla_.v_soma(), dt);
    for (size_t i = 0; i < bla_.size(); ++i) bla_.inject_basal(i, i_bla_rec[i]);

    // 3. BLA → CeA (fear expression)
    syn_bla_to_cea_.deliver_spikes(bla_.fired(), bla_.spike_type());
    auto i_cea_bla = syn_bla_to_cea_.step_and_compute(cea_.v_soma(), dt);
    for (size_t i = 0; i < cea_.size(); ++i) cea_.inject_basal(i, i_cea_bla[i]);

    // 4. La → CeA (direct fast path)
    syn_la_to_cea_.deliver_spikes(la_.fired(), la_.spike_type());
    auto i_cea_la = syn_la_to_cea_.step_and_compute(cea_.v_soma(), dt);
    for (size_t i = 0; i < cea_.size(); ++i) cea_.inject_basal(i, i_cea_la[i]);

    // 5. BLA → ITC (drives gate)
    syn_bla_to_itc_.deliver_spikes(bla_.fired(), bla_.spike_type());
    auto i_itc = syn_bla_to_itc_.step_and_compute(itc_.v_soma(), dt);
    for (size_t i = 0; i < itc_.size(); ++i) itc_.inject_basal(i, i_itc[i]);

    // 6. ITC → CeA (inhibitory gate: extinction)
    syn_itc_to_cea_.deliver_spikes(itc_.fired(), itc_.spike_type());
    auto i_cea_itc = syn_itc_to_cea_.step_and_compute(cea_.v_soma(), dt);
    for (size_t i = 0; i < cea_.size(); ++i) cea_.inject_basal(i, i_cea_itc[i]);

    // 7. Optional: La → MeA, MeA → CeA
    if (config_.n_mea > 0) {
        syn_la_to_mea_.deliver_spikes(la_.fired(), la_.spike_type());
        auto i_mea = syn_la_to_mea_.step_and_compute(mea_.v_soma(), dt);
        for (size_t i = 0; i < mea_.size(); ++i) mea_.inject_basal(i, i_mea[i]);

        syn_mea_to_cea_.deliver_spikes(mea_.fired(), mea_.spike_type());
        auto i_cea_mea = syn_mea_to_cea_.step_and_compute(cea_.v_soma(), dt);
        for (size_t i = 0; i < cea_.size(); ++i) cea_.inject_basal(i, i_cea_mea[i]);
    }

    // 8. Optional: La → CoA
    if (config_.n_coa > 0) {
        syn_la_to_coa_.deliver_spikes(la_.fired(), la_.spike_type());
        auto i_coa = syn_la_to_coa_.step_and_compute(coa_.v_soma(), dt);
        for (size_t i = 0; i < coa_.size(); ++i) coa_.inject_basal(i, i_coa[i]);
    }

    // 9. Optional: BLA → AB → CeA
    if (config_.n_ab > 0) {
        syn_bla_to_ab_.deliver_spikes(bla_.fired(), bla_.spike_type());
        auto i_ab = syn_bla_to_ab_.step_and_compute(ab_.v_soma(), dt);
        for (size_t i = 0; i < ab_.size(); ++i) ab_.inject_basal(i, i_ab[i]);

        syn_ab_to_cea_.deliver_spikes(ab_.fired(), ab_.spike_type());
        auto i_cea_ab = syn_ab_to_cea_.step_and_compute(cea_.v_soma(), dt);
        for (size_t i = 0; i < cea_.size(); ++i) cea_.inject_basal(i, i_cea_ab[i]);
    }

    // Inject US (unconditioned stimulus) drive to BLA
    // US decays over steps, modeling the transient pain/danger signal
    if (us_strength_ > 0.5f) {
        for (size_t i = 0; i < bla_.size(); ++i) {
            bla_.inject_basal(i, us_strength_);
        }
        us_strength_ *= US_DECAY;
    }

    // Step all populations
    la_.step(t, dt);
    bla_.step(t, dt);
    itc_.step(t, dt);
    cea_.step(t, dt);
    if (config_.n_mea > 0) mea_.step(t, dt);
    if (config_.n_coa > 0) coa_.step(t, dt);
    if (config_.n_ab > 0)  ab_.step(t, dt);

    // Fear conditioning STDP: La (CS) → BLA (US response)
    // When La fires (sensory CS) and BLA fires (US-driven), STDP strengthens
    // the La→BLA connection. Next time the CS appears, La alone can drive BLA.
    if (config_.fear_stdp_enabled) {
        syn_la_to_bla_.apply_stdp(la_.fired(), bla_.fired(), t);
    }

    aggregate_state();
}

// =============================================================================
// SpikeBus interface
// =============================================================================

void Amygdala::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 30.0f : 20.0f;

        // PFC spikes → ITC (top-down extinction control)
        if (evt.region_id == pfc_source_region_) {
            size_t base = evt.neuron_id % psp_itc_.size();
            for (size_t k = 0; k < 3 && (base + k) < psp_itc_.size(); ++k) {
                psp_itc_[base + k] += current;
            }
            continue;
        }

        // All other spikes → La (sensory input)
        size_t base = evt.neuron_id % psp_la_.size();
        for (size_t k = 0; k < 3 && (base + k) < psp_la_.size(); ++k) {
            psp_la_[base + k] += current;
        }
    }
}

void Amygdala::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_all_, spike_type_all_, t);
}

void Amygdala::inject_external(const std::vector<float>& currents) {
    inject_sensory(currents);
}

void Amygdala::inject_sensory(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), la_.size()); ++i) {
        la_.inject_basal(i, currents[i]);
    }
}

void Amygdala::inject_pfc_to_itc(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), itc_.size()); ++i) {
        itc_.inject_basal(i, currents[i]);
    }
}

// =============================================================================
// Fear conditioning closed-loop
// =============================================================================

void Amygdala::inject_us(float magnitude) {
    // Inject unconditioned stimulus (pain/danger) directly to BLA.
    // This causes BLA neurons to fire strongly, which when paired with
    // La sensory input (CS), drives La→BLA STDP to strengthen the
    // CS→fear association.
    //
    // Biology: US (foot shock, pain) activates BLA via direct thalamic
    // and cortical pathways. The strong BLA activation is the "teaching
    // signal" that establishes fear memory in one trial.
    // (LeDoux 2000, Fanselow & Poulos 2005)

    us_strength_ = magnitude * 40.0f;  // Strong drive: ensure BLA fires
}

float Amygdala::fear_output() const {
    // CeA firing rate as fear signal [0, 1].
    // Biology: CeA is the primary output of the amygdala fear circuit.
    // High CeA activity → freezing, startle, autonomic arousal, DA pause.
    size_t n_fired = 0;
    for (size_t i = 0; i < cea_.size(); ++i) {
        if (cea_.fired()[i]) ++n_fired;
    }
    return static_cast<float>(n_fired) / static_cast<float>(std::max(cea_.size(), (size_t)1));
}

float Amygdala::cea_vta_drive() const {
    // CeA → VTA/LHb inhibition signal.
    // Scaled fear_output for driving DA pause via VTA/LHb.
    // Biology: CeA → RMTg (GABA) → VTA DA neurons (inhibition)
    //          CeA → LHb (excitation) → RMTg → VTA DA (additional inhibition)
    // Combined effect: strong DA pause for aversive learning.
    float fear = fear_output();
    // Only produce drive when fear is significant (> 10% CeA firing)
    if (fear < 0.1f) return 0.0f;
    return fear * 1.5f;  // Amplify: amygdala fear → strong VTA inhibition
}

// =============================================================================
// Aggregate
// =============================================================================

void Amygdala::aggregate_state() {
    size_t offset = 0;
    auto copy_pop = [&](const NeuronPopulation& pop) {
        for (size_t i = 0; i < pop.size(); ++i) {
            fired_all_[offset + i]      = pop.fired()[i];
            spike_type_all_[offset + i] = pop.spike_type()[i];
        }
        offset += pop.size();
    };

    copy_pop(la_);
    copy_pop(bla_);
    copy_pop(cea_);
    copy_pop(itc_);
    if (config_.n_mea > 0) copy_pop(mea_);
    if (config_.n_coa > 0) copy_pop(coa_);
    if (config_.n_ab > 0)  copy_pop(ab_);
}

} // namespace wuyun
