#include "region/neuromod/lc_ne.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

// NE neurons: similar to DA neurons but faster tonic firing
static NeuronParams make_ne_neuron_params() {
    NeuronParams p;
    p.somatic.v_rest = -55.0f;       // Higher resting (more excitable)
    p.somatic.v_threshold = -45.0f;
    p.somatic.v_reset = -52.0f;
    p.somatic.tau_m = 15.0f;          // Faster membrane
    p.somatic.r_s = 0.8f;
    p.somatic.a = 0.02f;
    p.somatic.b = 2.0f;
    p.somatic.tau_w = 300.0f;         // Slow adaptation
    p.somatic.refractory_period = 3;
    p.kappa = 0.0f;
    p.kappa_backward = 0.0f;
    p.burst_spike_count = 2;
    p.burst_isi = 2;
    return p;
}

LC_NE::LC_NE(const LCConfig& config)
    : BrainRegion(config.name, config.n_ne_neurons)
    , config_(config)
    , ne_neurons_(config.n_ne_neurons, make_ne_neuron_params())
    , ne_level_(config.tonic_rate)
    , psp_ne_(config.n_ne_neurons, 0.0f)
    , fired_(config.n_ne_neurons, 0)
    , spike_type_(config.n_ne_neurons, 0)
{}

void LC_NE::step(int32_t t, float dt) {
    oscillation_.step(dt);

    // Arousal → NE neuron excitation
    float arousal_current = arousal_input_ * 40.0f;

    // Inject PSP buffer + arousal + tonic baseline drive
    for (size_t i = 0; i < psp_ne_.size(); ++i) {
        float psp_input = psp_ne_[i] > 0.5f ? psp_ne_[i] : 0.0f;
        // Tonic drive (LC has spontaneous firing ~1-3 Hz)
        ne_neurons_.inject_basal(i, 8.0f + arousal_current + psp_input);
        psp_ne_[i] *= PSP_DECAY;
    }

    ne_neurons_.step(t, dt);

    // Compute NE output from firing rate
    size_t n_fired = 0;
    for (size_t i = 0; i < ne_neurons_.size(); ++i) {
        fired_[i]      = ne_neurons_.fired()[i];
        spike_type_[i] = ne_neurons_.spike_type()[i];
        if (fired_[i]) n_fired++;
    }

    float firing_rate = static_cast<float>(n_fired) / static_cast<float>(ne_neurons_.size());
    float phasic = firing_rate * config_.phasic_gain;
    float target = std::clamp(config_.tonic_rate + phasic, 0.0f, 1.0f);
    // Exponential smoothing (volume transmission has slow kinetics)
    ne_level_ += (target - ne_level_) * 0.1f;

    // Reset arousal input
    arousal_input_ = 0.0f;
}

void LC_NE::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 18.0f : 10.0f;
        size_t base = evt.neuron_id % psp_ne_.size();
        for (size_t k = 0; k < 3 && (base + k) < psp_ne_.size(); ++k) {
            psp_ne_[base + k] += current;
        }
    }
}

void LC_NE::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void LC_NE::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), ne_neurons_.size()); ++i) {
        ne_neurons_.inject_basal(i, currents[i]);
    }
}

void LC_NE::inject_arousal(float arousal) {
    arousal_input_ = arousal;
}

} // namespace wuyun
