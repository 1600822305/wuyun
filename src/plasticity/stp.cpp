#include "plasticity/stp.h"
#include <algorithm>

namespace wuyun {

float stp_step(STPState& state, const STPParams& params, bool spiked, float dt) {
    // Recovery: dx/dt = (1 - x) / tau_D
    state.x += (1.0f - state.x) / params.tau_D * dt;

    // Facilitation decay: du/dt = (U - u) / tau_F
    state.u += (params.U - state.u) / params.tau_F * dt;

    float gain = state.u * state.x;

    if (spiked) {
        // On spike: u jumps up, x depletes
        state.u += params.U * (1.0f - state.u);
        state.x -= state.u * state.x;

        // Clamp
        state.x = std::clamp(state.x, 0.0f, 1.0f);
        state.u = std::clamp(state.u, 0.0f, 1.0f);
    }

    return gain;
}

void stp_step_batch(
    STPState* states,
    size_t n,
    const STPParams& params,
    const bool* fired,
    float* gains,
    float dt
) {
    for (size_t i = 0; i < n; ++i) {
        gains[i] = stp_step(states[i], params, fired[i], dt);
    }
}

} // namespace wuyun
