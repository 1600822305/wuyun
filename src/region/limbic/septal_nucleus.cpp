#include "region/limbic/septal_nucleus.h"
#include <cmath>
#include <algorithm>
#include <random>

namespace wuyun {

static SynapseGroup make_empty_sn(size_t n_pre, size_t n_post,
                                  const SynapseParams& params,
                                  CompartmentType target) {
    return SynapseGroup(n_pre, n_post, {}, {}, {}, {}, params, target);
}

static SynapseGroup build_syn_sn(
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
    if (pre.empty()) return make_empty_sn(n_pre, n_post, params, target);
    return SynapseGroup(n_pre, n_post, pre, post, w, d, params, target);
}

SeptalNucleus::SeptalNucleus(const SeptalConfig& config)
    : BrainRegion(config.name, config.n_ach + config.n_gaba)
    , config_(config)
    , ach_(config.n_ach, NeuronParams{})
    , gaba_(config.n_gaba, NeuronParams{})
    , syn_gaba_to_ach_(make_empty_sn(config.n_gaba, config.n_ach, GABA_A_PARAMS, CompartmentType::BASAL))
    , psp_ach_(config.n_ach, 0.0f)
    , fired_all_(n_neurons_, 0)
    , spike_type_all_(n_neurons_, 0)
{
    ach_output_ = config_.ach_output;
    unsigned seed = 5000;
    syn_gaba_to_ach_ = build_syn_sn(
        config_.n_gaba, config_.n_ach, config_.p_gaba_to_ach, config_.w_gaba_ach,
        GABA_A_PARAMS, CompartmentType::BASAL, seed);
}

void SeptalNucleus::step(int32_t t, float dt) {
    oscillation_.step(dt);
    neuromod_.step(dt);

    // === Theta pacemaker: rhythmic drive to GABA neurons ===
    theta_phase_ += dt / config_.theta_period;
    if (theta_phase_ >= 1.0f) theta_phase_ -= 1.0f;

    // Theta burst window: GABA neurons fire in a narrow phase window
    // Phase 0.0~0.2 = burst, 0.2~1.0 = silent
    bool in_burst = theta_phase_ < 0.2f;
    if (in_burst) {
        for (size_t i = 0; i < gaba_.size(); ++i) {
            gaba_.inject_basal(i, config_.theta_drive);
        }
    }

    // Inject PSP to ACh neurons (from external input)
    for (size_t i = 0; i < psp_ach_.size(); ++i) {
        if (psp_ach_[i] > 0.5f) {
            ach_.inject_basal(i, psp_ach_[i]);
        }
        psp_ach_[i] *= PSP_DECAY;
    }

    // GABA → ACh synapse
    syn_gaba_to_ach_.deliver_spikes(gaba_.fired(), gaba_.spike_type());
    auto gaba_currents = syn_gaba_to_ach_.step_and_compute(ach_.v_soma(), dt);
    for (size_t i = 0; i < ach_.size(); ++i) {
        if (std::abs(gaba_currents[i]) > 0.01f) {
            ach_.inject_basal(i, gaba_currents[i]);
        }
    }

    // Step populations
    gaba_.step(t, dt);
    ach_.step(t, dt);

    // ACh output: proportional to ACh neuron firing
    size_t ach_spikes = 0;
    for (auto f : ach_.fired()) if (f) ach_spikes++;
    float spike_frac = static_cast<float>(ach_spikes) / static_cast<float>(ach_.size() + 1);
    ach_output_ = config_.ach_output + spike_frac * 0.3f;  // Tonic + phasic

    aggregate_state();
}

void SeptalNucleus::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        size_t idx = evt.neuron_id % psp_ach_.size();
        psp_ach_[idx] += 20.0f;
    }
}

void SeptalNucleus::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_all_, spike_type_all_, t);
}

void SeptalNucleus::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), ach_.size()); ++i) {
        ach_.inject_basal(i, currents[i]);
    }
}

void SeptalNucleus::aggregate_state() {
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
    copy_pop(ach_);
    copy_pop(gaba_);
}

} // namespace wuyun
