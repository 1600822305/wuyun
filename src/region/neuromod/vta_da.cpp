#include "region/neuromod/vta_da.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

VTA_DA::VTA_DA(const VTAConfig& config)
    : BrainRegion(config.name, config.n_da_neurons)
    , config_(config)
    , da_neurons_(config.n_da_neurons, DOPAMINE_NEURON_PARAMS())
    , da_level_(config.tonic_rate)
    , psp_da_(config.n_da_neurons, 0.0f)
    , fired_(config.n_da_neurons, 0)
    , spike_type_(config.n_da_neurons, 0)
{}

void VTA_DA::step(int32_t t, float dt) {
    oscillation_.step(dt);

    // Compute RPE = actual reward - expected reward
    last_rpe_ = reward_input_ - expected_reward_;

    // RPE → DA neuron excitation
    // Positive RPE → phasic burst (strong excitation)
    // Negative RPE → pause (inhibition, below tonic)
    float rpe_current = last_rpe_ * config_.phasic_gain * 50.0f;

    // Inject PSP buffer (cross-region input, sustained)
    for (size_t i = 0; i < psp_da_.size(); ++i) {
        float psp_input = psp_da_[i] > 0.5f ? psp_da_[i] : 0.0f;
        // Tonic baseline drive + RPE + cross-region PSP
        da_neurons_.inject_basal(i, 5.0f + rpe_current + psp_input);
        psp_da_[i] *= PSP_DECAY;
    }

    da_neurons_.step(t, dt);

    // Compute DA output level from firing rate
    size_t n_fired = 0;
    for (size_t i = 0; i < da_neurons_.size(); ++i) {
        fired_[i]      = da_neurons_.fired()[i];
        spike_type_[i] = da_neurons_.spike_type()[i];
        if (fired_[i]) n_fired++;
    }

    // DA level = tonic + phasic (from firing rate)
    float firing_rate = static_cast<float>(n_fired) / static_cast<float>(da_neurons_.size());
    float phasic = firing_rate * config_.phasic_gain;
    da_level_ = std::clamp(config_.tonic_rate + phasic, 0.0f, 1.0f);

    // Reset reward input (consumed)
    reward_input_ = 0.0f;
}

void VTA_DA::receive_spikes(const std::vector<SpikeEvent>& events) {
    // Arriving spikes → PSP buffer (sustained drive via exponential decay)
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 20.0f : 12.0f;
        size_t base = evt.neuron_id % psp_da_.size();
        for (size_t k = 0; k < 3 && (base + k) < psp_da_.size(); ++k) {
            psp_da_[base + k] += current;
        }
    }
}

void VTA_DA::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void VTA_DA::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), da_neurons_.size()); ++i) {
        da_neurons_.inject_basal(i, currents[i]);
    }
}

void VTA_DA::inject_reward(float reward) {
    reward_input_ = reward;
}

void VTA_DA::set_expected_reward(float expected) {
    expected_reward_ = expected;
}

} // namespace wuyun
