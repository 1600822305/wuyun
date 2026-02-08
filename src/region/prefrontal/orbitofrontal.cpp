#include "region/prefrontal/orbitofrontal.h"
#include "core/types.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

// OFC value neurons: moderate threshold, some adaptation
// Biology: OFC neurons have sustained firing for expected reward
static NeuronParams OFC_VALUE_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -62.0f;
    p.somatic.v_threshold = -48.0f;
    p.somatic.v_reset = -55.0f;
    p.somatic.tau_m = 15.0f;       // Moderate speed
    p.somatic.r_s = 1.0f;
    p.somatic.a = 0.01f;
    p.somatic.b = 1.0f;            // Some adaptation (value updates)
    p.somatic.tau_w = 200.0f;
    p.kappa = 0.0f;
    return p;
}

// OFC inhibitory: fast-spiking PV-like
static NeuronParams OFC_INH_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -60.0f;
    p.somatic.v_threshold = -45.0f;
    p.somatic.v_reset = -55.0f;
    p.somatic.tau_m = 8.0f;
    p.somatic.r_s = 1.2f;
    p.somatic.a = 0.0f;
    p.somatic.b = 0.0f;
    p.somatic.tau_w = 50.0f;
    p.kappa = 0.0f;
    return p;
}

OrbitofrontalCortex::OrbitofrontalCortex(const OFCConfig& config)
    : BrainRegion(config.name, config.n_value_pos + config.n_value_neg + config.n_inh)
    , config_(config)
    , value_pos_(config.n_value_pos, OFC_VALUE_PARAMS())
    , value_neg_(config.n_value_neg, OFC_VALUE_PARAMS())
    , inh_(config.n_inh, OFC_INH_PARAMS())
    , psp_pos_(config.n_value_pos, 0.0f)
    , psp_neg_(config.n_value_neg, 0.0f)
    , psp_inh_(config.n_inh, 0.0f)
    , fired_(config.n_value_pos + config.n_value_neg + config.n_inh, 0)
    , spike_type_(config.n_value_pos + config.n_value_neg + config.n_inh, 0)
{
}

void OrbitofrontalCortex::step(int32_t t, float dt) {
    // --- DA modulation of value neurons ---
    // High DA (reward present) → boost positive value neurons
    // Low DA (punishment/omission) → boost negative value neurons
    // Biology: OFC encodes expected value, DA signals actual outcome
    //   When DA > baseline: "better than expected" → strengthen pos associations
    //   When DA < baseline: "worse than expected" → strengthen neg associations
    float da_diff = da_level_ - 0.3f;  // Deviation from baseline

    // --- Inhibitory → value competition ---
    // PV interneurons enforce winner-take-all between pos and neg value
    float inh_drive = 0.0f;
    for (size_t i = 0; i < inh_.size(); ++i) {
        if (inh_.fired()[i]) inh_drive += 5.0f;
    }

    // --- Positive value neurons ---
    for (size_t i = 0; i < value_pos_.size(); ++i) {
        float da_boost = std::max(0.0f, da_diff) * 50.0f;  // DA+ boosts pos
        value_pos_.inject_basal(i, psp_pos_[i] + da_boost - inh_drive);
        psp_pos_[i] *= PSP_DECAY;
    }

    // --- Negative value neurons ---
    for (size_t i = 0; i < value_neg_.size(); ++i) {
        float da_boost = std::max(0.0f, -da_diff) * 50.0f; // DA- boosts neg
        value_neg_.inject_basal(i, psp_neg_[i] + da_boost - inh_drive);
        psp_neg_[i] *= PSP_DECAY;
    }

    // --- Inhibitory neurons ---
    // Driven by both pos and neg value neurons (E→I)
    float exc_drive = 0.0f;
    for (size_t i = 0; i < value_pos_.size(); ++i)
        if (value_pos_.fired()[i]) exc_drive += 6.0f;
    for (size_t i = 0; i < value_neg_.size(); ++i)
        if (value_neg_.fired()[i]) exc_drive += 6.0f;
    for (size_t i = 0; i < inh_.size(); ++i) {
        inh_.inject_basal(i, psp_inh_[i] + exc_drive);
        psp_inh_[i] *= PSP_DECAY;
    }

    value_pos_.step(t, dt);
    value_neg_.step(t, dt);
    inh_.step(t, dt);

    // --- Value signal computation (diagnostic, not used for decisions) ---
    size_t pos_fires = 0, neg_fires = 0;
    for (size_t i = 0; i < value_pos_.size(); ++i)
        if (value_pos_.fired()[i]) pos_fires++;
    for (size_t i = 0; i < value_neg_.size(); ++i)
        if (value_neg_.fired()[i]) neg_fires++;

    float pos_rate = static_cast<float>(pos_fires) / std::max<size_t>(value_pos_.size(), 1);
    float neg_rate = static_cast<float>(neg_fires) / std::max<size_t>(value_neg_.size(), 1);
    value_signal_ = value_signal_ * 0.9f + (pos_rate - neg_rate) * 0.1f;

    aggregate_state();
}

void OrbitofrontalCortex::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 35.0f : 22.0f;
        // Route to both pos and neg value neurons (let DA modulation decide winner)
        size_t pos_idx = evt.neuron_id % value_pos_.size();
        size_t neg_idx = evt.neuron_id % value_neg_.size();
        psp_pos_[pos_idx] += current;
        psp_neg_[neg_idx] += current;
        // Some drive to inhibitory
        size_t inh_idx = evt.neuron_id % inh_.size();
        psp_inh_[inh_idx] += current * 0.3f;
    }
}

void OrbitofrontalCortex::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void OrbitofrontalCortex::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), value_pos_.size()); ++i) {
        psp_pos_[i] += currents[i];
    }
}

void OrbitofrontalCortex::aggregate_state() {
    size_t offset = 0;
    auto copy_pop = [&](const NeuronPopulation& pop) {
        for (size_t i = 0; i < pop.size(); ++i) {
            fired_[offset + i] = pop.fired()[i];
            spike_type_[offset + i] = pop.spike_type()[i];
        }
        offset += pop.size();
    };
    copy_pop(value_pos_);
    copy_pop(value_neg_);
    copy_pop(inh_);
}

} // namespace wuyun
