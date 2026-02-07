#include "region/amygdala.h"
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
                  config.n_la + config.n_bla + config.n_cea + config.n_itc)
    , config_(config)
    , la_(config.n_la, NeuronParams{})       // Standard excitatory
    , bla_(config.n_bla, NeuronParams{})     // Standard excitatory (learning via DA-STDP)
    , cea_(config.n_cea, NeuronParams{})     // Standard excitatory (output)
    , itc_(config.n_itc, PV_BASKET_PARAMS()) // Inhibitory gate
    // Synapses (initialized empty, filled in build_synapses)
    , syn_la_to_bla_(make_empty(config.n_la, config.n_bla, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_bla_to_cea_(make_empty(config.n_bla, config.n_cea, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_la_to_cea_(make_empty(config.n_la, config.n_cea, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_bla_to_itc_(make_empty(config.n_bla, config.n_itc, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_itc_to_cea_(make_empty(config.n_itc, config.n_cea, GABA_A_PARAMS, CompartmentType::BASAL))
    , syn_bla_rec_(make_empty(config.n_bla, config.n_bla, AMPA_PARAMS, CompartmentType::BASAL))
    // PSP buffer
    , psp_la_(config.n_la, 0.0f)
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
}

// =============================================================================
// Step
// =============================================================================

void Amygdala::step(int32_t t, float dt) {
    oscillation_.step(dt);
    neuromod_.step(dt);

    // Inject PSP buffer into La
    for (size_t i = 0; i < psp_la_.size(); ++i) {
        if (psp_la_[i] > 0.5f) la_.inject_basal(i, psp_la_[i]);
        psp_la_[i] *= PSP_DECAY;
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

    // Step all populations
    la_.step(t, dt);
    bla_.step(t, dt);
    itc_.step(t, dt);
    cea_.step(t, dt);

    aggregate_state();
}

// =============================================================================
// SpikeBus interface
// =============================================================================

void Amygdala::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 30.0f : 20.0f;
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
}

} // namespace wuyun
