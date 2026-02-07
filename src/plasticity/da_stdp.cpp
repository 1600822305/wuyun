#include "plasticity/da_stdp.h"
#include <cmath>
#include <algorithm>

namespace wuyun {

DASTDPProcessor::DASTDPProcessor(size_t n_synapses, const DASTDPParams& params)
    : n_(n_synapses)
    , params_(params)
    , eligibility_(n_synapses, 0.0f)
{}

void DASTDPProcessor::update_traces(
    const float* pre_times,
    const float* post_times,
    const int32_t* pre_ids,
    const int32_t* post_ids,
    float dt
) {
    float decay = dt / params_.tau_eligibility;

    for (size_t s = 0; s < n_; ++s) {
        // Decay existing trace
        eligibility_[s] -= eligibility_[s] * decay;

        // If both pre and post fired recently, add STDP contribution
        float t_pre  = pre_times[pre_ids[s]];
        float t_post = post_times[post_ids[s]];

        if (t_pre >= 0.0f && t_post >= 0.0f) {
            float dw = stdp_delta_w(t_pre, t_post, params_.stdp);
            eligibility_[s] += dw;
        }
    }
}

void DASTDPProcessor::apply_da_modulation(float* weights, float da_signal) {
    // DA relative to baseline: positive = reward, negative = punishment
    float da_relative = da_signal - params_.da_baseline;

    for (size_t s = 0; s < n_; ++s) {
        float dw = da_relative * eligibility_[s];
        weights[s] += dw;
        weights[s] = std::clamp(weights[s], params_.w_min, params_.w_max);
    }
}

} // namespace wuyun
