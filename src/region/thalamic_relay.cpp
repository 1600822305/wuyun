#include "region/thalamic_relay.h"
#include <random>
#include <algorithm>

namespace wuyun {

// Helper: build random sparse connections (same as cortical_column.cpp)
static void build_sparse_connections(
    size_t n_pre, size_t n_post, float prob, float weight,
    std::vector<int32_t>& pre_ids,
    std::vector<int32_t>& post_ids,
    std::vector<float>& weights,
    std::vector<int32_t>& delays,
    unsigned seed = 42
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n_pre; ++i) {
        for (size_t j = 0; j < n_post; ++j) {
            if (dist(rng) < prob) {
                pre_ids.push_back(static_cast<int32_t>(i));
                post_ids.push_back(static_cast<int32_t>(j));
                weights.push_back(weight);
                delays.push_back(1);
            }
        }
    }
}

static SynapseGroup make_empty_synapse(size_t n_pre, size_t n_post,
                                        const SynapseParams& params,
                                        CompartmentType target) {
    return SynapseGroup(n_pre, n_post, {}, {}, {}, {}, params, target);
}

ThalamicRelay::ThalamicRelay(const ThalamicConfig& config)
    : BrainRegion(config.name, config.n_relay + config.n_trn)
    , config_(config)
    , relay_(config.n_relay,
             config.burst_mode ? THALAMIC_RELAY_BURST_PARAMS()
                               : THALAMIC_RELAY_TONIC_PARAMS())
    , trn_(config.n_trn, TRN_PARAMS())
    , syn_relay_to_trn_(make_empty_synapse(config.n_relay, config.n_trn,
                                            AMPA_PARAMS, CompartmentType::BASAL))
    , syn_trn_to_relay_(make_empty_synapse(config.n_trn, config.n_relay,
                                            GABA_A_PARAMS, CompartmentType::BASAL))
    , fired_all_(n_neurons_, 0)
    , spike_type_all_(n_neurons_, 0)
{
    build_synapses();
}

void ThalamicRelay::build_synapses() {
    // Relay → TRN (excitatory AMPA)
    {
        std::vector<int32_t> pre, post;
        std::vector<float> w;
        std::vector<int32_t> d;
        build_sparse_connections(config_.n_relay, config_.n_trn,
                                  config_.p_relay_to_trn, config_.w_relay_trn,
                                  pre, post, w, d, 100);
        syn_relay_to_trn_ = SynapseGroup(config_.n_relay, config_.n_trn,
                                          pre, post, w, d,
                                          AMPA_PARAMS, CompartmentType::BASAL);
    }
    // TRN → Relay (inhibitory GABA_A)
    {
        std::vector<int32_t> pre, post;
        std::vector<float> w;
        std::vector<int32_t> d;
        build_sparse_connections(config_.n_trn, config_.n_relay,
                                  config_.p_trn_to_relay, config_.w_trn_inh,
                                  pre, post, w, d, 200);
        syn_trn_to_relay_ = SynapseGroup(config_.n_trn, config_.n_relay,
                                          pre, post, w, d,
                                          GABA_A_PARAMS, CompartmentType::BASAL);
    }
}

void ThalamicRelay::step(int32_t t, float dt) {
    oscillation_.step(dt);
    neuromod_.step(dt);

    // 1. Relay → TRN (excitatory drive)
    syn_relay_to_trn_.deliver_spikes(relay_.fired(), relay_.spike_type());
    auto i_trn = syn_relay_to_trn_.step_and_compute(trn_.v_soma(), dt);
    for (size_t i = 0; i < trn_.size(); ++i) {
        trn_.inject_basal(i, i_trn[i]);
    }

    // 2. TRN → Relay (inhibitory)
    syn_trn_to_relay_.deliver_spikes(trn_.fired(), trn_.spike_type());
    auto i_relay_inh = syn_trn_to_relay_.step_and_compute(relay_.v_soma(), dt);
    for (size_t i = 0; i < relay_.size(); ++i) {
        relay_.inject_basal(i, i_relay_inh[i]);
    }

    // 3. Step both populations
    relay_.step(t, dt);
    trn_.step(t, dt);

    aggregate_state();
}

void ThalamicRelay::receive_spikes(const std::vector<SpikeEvent>& events) {
    // Arriving spikes go to relay neurons (feedforward sensory input)
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 30.0f : 20.0f;
        size_t base = evt.neuron_id % relay_.size();
        for (size_t k = 0; k < 3 && (base + k) < relay_.size(); ++k) {
            relay_.inject_basal(base + k, current);
        }
    }
}

void ThalamicRelay::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_all_, spike_type_all_, t);
}

void ThalamicRelay::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), relay_.size()); ++i) {
        relay_.inject_basal(i, currents[i]);
    }
}

void ThalamicRelay::inject_cortical_feedback(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), relay_.size()); ++i) {
        relay_.inject_apical(i, currents[i]);
    }
}

void ThalamicRelay::inject_trn_modulation(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), trn_.size()); ++i) {
        trn_.inject_basal(i, currents[i]);
    }
}

void ThalamicRelay::set_mode(bool burst_mode) {
    config_.burst_mode = burst_mode;
    auto params = burst_mode ? THALAMIC_RELAY_BURST_PARAMS()
                              : THALAMIC_RELAY_TONIC_PARAMS();
    relay_ = NeuronPopulation(config_.n_relay, params);
    build_synapses();
}

void ThalamicRelay::aggregate_state() {
    // Relay neurons first, then TRN
    for (size_t i = 0; i < relay_.size(); ++i) {
        fired_all_[i]      = relay_.fired()[i];
        spike_type_all_[i] = relay_.spike_type()[i];
    }
    size_t off = relay_.size();
    for (size_t i = 0; i < trn_.size(); ++i) {
        fired_all_[off + i]      = trn_.fired()[i];
        spike_type_all_[off + i] = trn_.spike_type()[i];
    }
}

} // namespace wuyun
