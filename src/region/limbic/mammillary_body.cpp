#include "region/limbic/mammillary_body.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace wuyun {

static SynapseGroup make_empty_mb(size_t n_pre, size_t n_post,
                                  const SynapseParams& params,
                                  CompartmentType target) {
    return SynapseGroup(n_pre, n_post, {}, {}, {}, {}, params, target);
}

static SynapseGroup build_syn_mb(
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
    if (pre.empty()) return make_empty_mb(n_pre, n_post, params, target);
    return SynapseGroup(n_pre, n_post, pre, post, w, d, params, target);
}

MammillaryBody::MammillaryBody(const MammillaryConfig& config)
    : BrainRegion(config.name, config.n_medial + config.n_lateral)
    , config_(config)
    , medial_(config.n_medial, NeuronParams{})
    , lateral_(config.n_lateral, NeuronParams{})
    , syn_med_to_lat_(make_empty_mb(config.n_medial, config.n_lateral, AMPA_PARAMS, CompartmentType::BASAL))
    , psp_medial_(config.n_medial, 0.0f)
    , fired_all_(n_neurons_, 0)
    , spike_type_all_(n_neurons_, 0)
{
    unsigned seed = 6000;
    syn_med_to_lat_ = build_syn_mb(
        config_.n_medial, config_.n_lateral, config_.p_medial_to_lateral, config_.w_medial_lateral,
        AMPA_PARAMS, CompartmentType::BASAL, seed);
}

void MammillaryBody::step(int32_t t, float dt) {
    oscillation_.step(dt);
    neuromod_.step(dt);

    // Inject PSP to medial neurons (from Hippocampus Sub)
    for (size_t i = 0; i < psp_medial_.size(); ++i) {
        if (psp_medial_[i] > 0.5f) {
            medial_.inject_basal(i, psp_medial_[i]);
        }
        psp_medial_[i] *= PSP_DECAY;
    }

    medial_.step(t, dt);

    // Medial → Lateral
    syn_med_to_lat_.deliver_spikes(medial_.fired(), medial_.spike_type());
    auto lat_currents = syn_med_to_lat_.step_and_compute(lateral_.v_soma(), dt);
    for (size_t i = 0; i < lateral_.size(); ++i) {
        if (std::abs(lat_currents[i]) > 0.01f) {
            lateral_.inject_basal(i, lat_currents[i]);
        }
    }

    lateral_.step(t, dt);

    aggregate_state();
}

void MammillaryBody::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        size_t idx = evt.neuron_id % psp_medial_.size();
        size_t fan = std::max<size_t>(2, psp_medial_.size() / 8);
        for (size_t k = 0; k < fan; ++k) {
            size_t i = (idx + k) % psp_medial_.size();
            psp_medial_[i] += 25.0f;
        }
    }
}

void MammillaryBody::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_all_, spike_type_all_, t);
}

void MammillaryBody::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), medial_.size()); ++i) {
        medial_.inject_basal(i, currents[i]);
    }
}

void MammillaryBody::aggregate_state() {
    size_t offset = 0;
    auto copy_pop = [&](const NeuronPopulation& pop) {
        const auto& f = pop.fired();
        const auto& s = pop.spike_type();
        for (size_t i = 0; i < pop.size(); ++i) {
            fired_all_[offset + i]      = f[i];
            spike_type_all_[offset + i] = s[i];
        }
        offset += pop.size();
    };
    copy_pop(medial_);
    copy_pop(lateral_);
}

} // namespace wuyun
