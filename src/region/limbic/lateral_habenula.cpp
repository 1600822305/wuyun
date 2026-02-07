#include "region/limbic/lateral_habenula.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

LateralHabenula::LateralHabenula(const LHbConfig& config)
    : BrainRegion(config.name, config.n_neurons)
    , config_(config)
    , neurons_(config.n_neurons, NeuronParams{})  // Standard excitatory neurons
    , psp_(config.n_neurons, 0.0f)
    , fired_(config.n_neurons, 0)
    , spike_type_(config.n_neurons, 0)
{}

void LateralHabenula::step(int32_t t, float dt) {
    oscillation_.step(dt);

    // Accumulate aversive signals into sustained PSP buffer
    // Biology: LHb neurons have sustained responses to aversive events
    //          lasting 200-500ms (Matsumoto & Hikosaka 2007)
    float aversive_drive = 0.0f;

    if (punishment_input_ > 0.01f) {
        // Direct punishment (danger collision) → strong LHb activation
        aversive_psp_ += punishment_input_ * config_.punishment_gain * 150.0f;
    }
    if (frustration_input_ > 0.01f) {
        // Frustrative non-reward (expected food didn't appear) → moderate LHb activation
        aversive_psp_ += frustration_input_ * config_.frustration_gain * 100.0f;
    }

    aversive_drive = aversive_psp_;
    aversive_psp_ *= AVERSIVE_PSP_DECAY;  // Sustained decay

    // Inject drive into all LHb neurons
    for (size_t i = 0; i < neurons_.size(); ++i) {
        float psp_input = psp_[i] > 0.5f ? psp_[i] : 0.0f;
        // Tonic baseline + aversive drive + cross-region PSP
        neurons_.inject_basal(i, config_.tonic_drive + aversive_drive + psp_input);
        psp_[i] *= PSP_DECAY;
    }

    neurons_.step(t, dt);

    // Compute output level from firing rate
    size_t n_fired = 0;
    for (size_t i = 0; i < neurons_.size(); ++i) {
        fired_[i]      = neurons_.fired()[i];
        spike_type_[i] = neurons_.spike_type()[i];
        if (fired_[i]) n_fired++;
    }

    // Output level = normalized firing rate
    output_level_ = static_cast<float>(n_fired) / static_cast<float>(neurons_.size());

    // VTA inhibition: LHb firing → RMTg GABA → VTA DA suppression
    // Biology: LHb burst during negative RPE causes ~200ms DA pause
    // Implementation: firing_rate × gain = inhibition amount (subtracted from VTA DA level)
    vta_inhibition_ = std::clamp(output_level_ * config_.vta_inhibition_gain, 0.0f, 1.0f);

    // Reset inputs (consumed)
    punishment_input_  = 0.0f;
    frustration_input_ = 0.0f;
}

void LateralHabenula::receive_spikes(const std::vector<SpikeEvent>& events) {
    // Arriving spikes → PSP buffer (from GPb, PFC, etc.)
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 20.0f : 12.0f;
        size_t base = evt.neuron_id % psp_.size();
        for (size_t k = 0; k < 3 && (base + k) < psp_.size(); ++k) {
            psp_[base + k] += current;
        }
    }
}

void LateralHabenula::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void LateralHabenula::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), neurons_.size()); ++i) {
        neurons_.inject_basal(i, currents[i]);
    }
}

void LateralHabenula::inject_punishment(float punishment) {
    punishment_input_ = std::max(0.0f, punishment);
}

void LateralHabenula::inject_frustration(float frustration) {
    frustration_input_ = std::max(0.0f, frustration);
}

} // namespace wuyun
