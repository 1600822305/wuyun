#include "region/drn_5ht.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

// 5-HT neurons: slow, regular tonic firing
static NeuronParams make_5ht_neuron_params() {
    NeuronParams p;
    p.somatic.v_rest = -60.0f;
    p.somatic.v_threshold = -45.0f;
    p.somatic.v_reset = -55.0f;
    p.somatic.tau_m = 25.0f;          // Slow membrane (regular firing)
    p.somatic.r_s = 0.7f;
    p.somatic.a = 0.01f;
    p.somatic.b = 3.0f;
    p.somatic.tau_w = 400.0f;         // Very slow adaptation
    p.somatic.refractory_period = 4;  // Long refractory (slow firing ~1-2 Hz)
    p.kappa = 0.0f;
    p.kappa_backward = 0.0f;
    p.burst_spike_count = 1;
    p.burst_isi = 1;
    return p;
}

DRN_5HT::DRN_5HT(const DRNConfig& config)
    : BrainRegion(config.name, config.n_5ht_neurons)
    , config_(config)
    , sht_neurons_(config.n_5ht_neurons, make_5ht_neuron_params())
    , sht_level_(config.tonic_rate)
    , psp_5ht_(config.n_5ht_neurons, 0.0f)
    , fired_(config.n_5ht_neurons, 0)
    , spike_type_(config.n_5ht_neurons, 0)
{}

void DRN_5HT::step(int32_t t, float dt) {
    oscillation_.step(dt);

    float wellbeing_current = wellbeing_input_ * 30.0f;

    for (size_t i = 0; i < psp_5ht_.size(); ++i) {
        float psp_input = psp_5ht_[i] > 0.5f ? psp_5ht_[i] : 0.0f;
        // Tonic drive (DRN has slow regular firing)
        sht_neurons_.inject_basal(i, 6.0f + wellbeing_current + psp_input);
        psp_5ht_[i] *= PSP_DECAY;
    }

    sht_neurons_.step(t, dt);

    size_t n_fired = 0;
    for (size_t i = 0; i < sht_neurons_.size(); ++i) {
        fired_[i]      = sht_neurons_.fired()[i];
        spike_type_[i] = sht_neurons_.spike_type()[i];
        if (fired_[i]) n_fired++;
    }

    float firing_rate = static_cast<float>(n_fired) / static_cast<float>(sht_neurons_.size());
    float phasic = firing_rate * config_.phasic_gain;
    float target = std::clamp(config_.tonic_rate + phasic, 0.0f, 1.0f);
    sht_level_ += (target - sht_level_) * 0.1f;

    wellbeing_input_ = 0.0f;
}

void DRN_5HT::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 15.0f : 8.0f;
        size_t base = evt.neuron_id % psp_5ht_.size();
        for (size_t k = 0; k < 3 && (base + k) < psp_5ht_.size(); ++k) {
            psp_5ht_[base + k] += current;
        }
    }
}

void DRN_5HT::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void DRN_5HT::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), sht_neurons_.size()); ++i) {
        sht_neurons_.inject_basal(i, currents[i]);
    }
}

void DRN_5HT::inject_wellbeing(float wellbeing) {
    wellbeing_input_ = wellbeing;
}

} // namespace wuyun
