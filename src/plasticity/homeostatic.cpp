#include "plasticity/homeostatic.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace wuyun {

SynapticScaler::SynapticScaler(size_t n_neurons, const HomeostaticParams& params)
    : n_(n_neurons)
    , params_(params)
    , rates_(n_neurons, params.target_rate)  // Initialize at target
{}

void SynapticScaler::update_rates(const bool* fired, float dt) {
    // Exponential moving average of firing rate
    // rate += (spike/dt_s - rate) * dt / tau_rate
    float dt_s = dt * 0.001f;  // ms → s
    float alpha = dt / params_.tau_rate;

    for (size_t i = 0; i < n_; ++i) {
        float instant_rate = fired[i] ? (1.0f / dt_s) : 0.0f;
        rates_[i] += alpha * (instant_rate - rates_[i]);
    }
}

void SynapticScaler::apply_scaling(float* weights, size_t n_synapses,
                                    const int32_t* post_ids) {
    for (size_t s = 0; s < n_synapses; ++s) {
        size_t post = static_cast<size_t>(post_ids[s]);
        float error = (params_.target_rate - rates_[post]) / params_.target_rate;
        float dw = params_.eta * error * weights[s];
        weights[s] += dw;
        weights[s] = std::clamp(weights[s], params_.w_min, params_.w_max);
    }
}

} // namespace wuyun
