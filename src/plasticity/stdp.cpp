#include "plasticity/stdp.h"
#include <cmath>
#include <algorithm>

namespace wuyun {

float stdp_delta_w(float t_pre, float t_post, const STDPParams& params) {
    float dt = t_post - t_pre;
    if (dt > 0.0f) {
        // Pre before post → LTP
        return params.a_plus * std::exp(-dt / params.tau_plus);
    } else if (dt < 0.0f) {
        // Post before pre → LTD
        return params.a_minus * std::exp(dt / params.tau_minus);
    }
    return 0.0f;
}

void stdp_update_batch(
    float* weights,
    size_t n_synapses,
    const float* pre_times,
    const float* post_times,
    const int* pre_ids,
    const int* post_ids,
    const STDPParams& params
) {
    for (size_t s = 0; s < n_synapses; ++s) {
        float t_pre  = pre_times[pre_ids[s]];
        float t_post = post_times[post_ids[s]];

        // Skip if either neuron hasn't fired yet
        if (t_pre < 0.0f || t_post < 0.0f) continue;

        float dw = stdp_delta_w(t_pre, t_post, params);
        weights[s] += dw;
        weights[s] = std::clamp(weights[s], params.w_min, params.w_max);
    }
}

} // namespace wuyun
