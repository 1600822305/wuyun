#include "region/nbm_ach.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

// ACh neurons: moderate tonic firing, burst on surprise
static NeuronParams make_ach_neuron_params() {
    NeuronParams p;
    p.somatic.v_rest = -58.0f;
    p.somatic.v_threshold = -45.0f;
    p.somatic.v_reset = -53.0f;
    p.somatic.tau_m = 18.0f;
    p.somatic.r_s = 0.9f;
    p.somatic.a = 0.02f;
    p.somatic.b = 2.5f;
    p.somatic.tau_w = 250.0f;
    p.somatic.refractory_period = 3;
    p.kappa = 0.0f;
    p.kappa_backward = 0.0f;
    p.burst_spike_count = 2;
    p.burst_isi = 2;
    return p;
}

NBM_ACh::NBM_ACh(const NBMConfig& config)
    : BrainRegion(config.name, config.n_ach_neurons)
    , config_(config)
    , ach_neurons_(config.n_ach_neurons, make_ach_neuron_params())
    , ach_level_(config.tonic_rate)
    , psp_ach_(config.n_ach_neurons, 0.0f)
    , fired_(config.n_ach_neurons, 0)
    , spike_type_(config.n_ach_neurons, 0)
{}

void NBM_ACh::step(int32_t t, float dt) {
    oscillation_.step(dt);

    float surprise_current = surprise_input_ * 35.0f;

    for (size_t i = 0; i < psp_ach_.size(); ++i) {
        float psp_input = psp_ach_[i] > 0.5f ? psp_ach_[i] : 0.0f;
        // Tonic drive (moderate baseline)
        ach_neurons_.inject_basal(i, 7.0f + surprise_current + psp_input);
        psp_ach_[i] *= PSP_DECAY;
    }

    ach_neurons_.step(t, dt);

    size_t n_fired = 0;
    for (size_t i = 0; i < ach_neurons_.size(); ++i) {
        fired_[i]      = ach_neurons_.fired()[i];
        spike_type_[i] = ach_neurons_.spike_type()[i];
        if (fired_[i]) n_fired++;
    }

    float firing_rate = static_cast<float>(n_fired) / static_cast<float>(ach_neurons_.size());
    float phasic = firing_rate * config_.phasic_gain;
    float target = std::clamp(config_.tonic_rate + phasic, 0.0f, 1.0f);
    ach_level_ += (target - ach_level_) * 0.1f;

    surprise_input_ = 0.0f;
}

void NBM_ACh::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 16.0f : 9.0f;
        size_t base = evt.neuron_id % psp_ach_.size();
        for (size_t k = 0; k < 3 && (base + k) < psp_ach_.size(); ++k) {
            psp_ach_[base + k] += current;
        }
    }
}

void NBM_ACh::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void NBM_ACh::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), ach_neurons_.size()); ++i) {
        ach_neurons_.inject_basal(i, currents[i]);
    }
}

void NBM_ACh::inject_surprise(float surprise) {
    surprise_input_ = surprise;
}

} // namespace wuyun
