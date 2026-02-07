#include "plasticity/istdp.h"
#include <cmath>
#include <algorithm>

namespace wuyun {

float istdp_delta_w(float t_pre, float t_post, const ISTDPParams& params) {
    float dt = std::abs(t_post - t_pre);

    if (dt < params.tau_window) {
        // Correlated: strengthen inhibition
        return params.a_corr * std::exp(-dt / params.tau);
    } else {
        // Uncorrelated: weaken inhibition
        return -params.b_uncorr;
    }
}

void istdp_update_batch(
    float* weights,
    size_t n_synapses,
    const float* pre_times,
    const float* post_times,
    const int32_t* pre_ids,
    const int32_t* post_ids,
    const ISTDPParams& params
) {
    for (size_t s = 0; s < n_synapses; ++s) {
        float t_pre  = pre_times[pre_ids[s]];
        float t_post = post_times[post_ids[s]];

        if (t_pre < 0.0f || t_post < 0.0f) continue;

        float dw = istdp_delta_w(t_pre, t_post, params);
        weights[s] += dw;
        weights[s] = std::clamp(weights[s], params.w_min, params.w_max);
    }
}

} // namespace wuyun
