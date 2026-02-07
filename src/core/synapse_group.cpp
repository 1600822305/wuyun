#include "core/synapse_group.h"
#include "plasticity/stp.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace wuyun {

SynapseGroup::SynapseGroup(
    size_t n_pre,
    size_t n_post,
    const std::vector<int32_t>& pre_ids,
    const std::vector<int32_t>& post_ids,
    const std::vector<float>& weights,
    const std::vector<int32_t>& delays,
    const SynapseParams& syn_params,
    CompartmentType target
)
    : n_pre_(n_pre)
    , n_post_(n_post)
    , target_(target)
    , weights_(weights)
    , delays_(delays)
    , tau_decay_(syn_params.tau_decay)
    , e_rev_(syn_params.e_rev)
    , g_max_(syn_params.g_max)
    , mg_conc_(syn_params.mg_conc)
    , g_(weights.size(), 0.0f)
    , i_post_(n_post, 0.0f)
{
    // Build CSR from COO (pre_ids, post_ids)
    size_t n_syn = pre_ids.size();
    row_ptr_.resize(n_pre + 1, 0);

    // Count synapses per pre neuron
    for (size_t s = 0; s < n_syn; ++s) {
        row_ptr_[static_cast<size_t>(pre_ids[s]) + 1] += 1;
    }
    // Prefix sum
    for (size_t i = 1; i <= n_pre; ++i) {
        row_ptr_[i] += row_ptr_[i - 1];
    }

    // Sort by pre_id to fill CSR col_idx
    // Use temporary offset array
    col_idx_.resize(n_syn);
    std::vector<float> sorted_weights(n_syn);
    std::vector<int32_t> sorted_delays(n_syn);
    std::vector<int32_t> offset(n_pre, 0);

    for (size_t s = 0; s < n_syn; ++s) {
        size_t pre = static_cast<size_t>(pre_ids[s]);
        size_t pos = static_cast<size_t>(row_ptr_[pre]) + static_cast<size_t>(offset[pre]);
        col_idx_[pos]       = post_ids[s];
        sorted_weights[pos] = weights[s];
        sorted_delays[pos]  = delays[s];
        offset[pre] += 1;
    }
    weights_ = std::move(sorted_weights);
    delays_  = std::move(sorted_delays);
}

void SynapseGroup::enable_stp(const STPParams& params) {
    stp_enabled_ = true;
    stp_params_ = params;
    stp_states_.resize(n_pre_);
    for (auto& s : stp_states_) {
        s.x = 1.0f;
        s.u = params.U;
    }
}

void SynapseGroup::deliver_spikes(
    const std::vector<uint8_t>& pre_fired,
    const std::vector<int8_t>& pre_spike_type
) {
    for (size_t pre = 0; pre < n_pre_; ++pre) {
        bool fired = pre_fired[pre] != 0;

        // STP: update state every step, get gain
        float stp_gain = 1.0f;
        if (stp_enabled_) {
            stp_gain = stp_step(stp_states_[pre], stp_params_, fired);
        }

        if (!fired) continue;

        // Burst spikes carry stronger signal (x2) than regular (x1)
        auto st = static_cast<SpikeType>(pre_spike_type[pre]);
        float burst_gain = is_burst(st) ? 2.0f : 1.0f;

        float total_gain = burst_gain * stp_gain;

        int32_t start = row_ptr_[pre];
        int32_t end   = row_ptr_[pre + 1];
        for (int32_t s = start; s < end; ++s) {
            g_[static_cast<size_t>(s)] += total_gain;
        }
    }
}

std::vector<float> SynapseGroup::step_and_compute(
    const std::vector<float>& v_post,
    float dt
) {
    // Clear output buffer
    std::fill(i_post_.begin(), i_post_.end(), 0.0f);

    float decay = dt / tau_decay_;
    size_t n_syn = col_idx_.size();

    for (size_t s = 0; s < n_syn; ++s) {
        // Decay gating variable: ds/dt = -s / tau_decay
        g_[s] -= g_[s] * decay;

        // I_syn = g_max * w * s * B(V) * (E_rev - V_post)
        // B(V) = 1/(1 + [Mg²⁺]/3.57 · exp(-0.062·V))  (NMDA only)
        size_t post = static_cast<size_t>(col_idx_[s]);
        float v = v_post[post];
        float b_v = 1.0f;
        if (mg_conc_ > 0.0f) {
            b_v = 1.0f / (1.0f + (mg_conc_ / 3.57f) * std::exp(-0.062f * v));
        }
        float i_syn = g_max_ * weights_[s] * g_[s] * b_v * (e_rev_ - v);
        i_post_[post] += i_syn;
    }

    return i_post_;
}

void SynapseGroup::enable_stdp(const STDPParams& params) {
    stdp_enabled_ = true;
    stdp_params_ = params;
    last_spike_pre_.assign(n_pre_, -1000.0f);
    last_spike_post_.assign(n_post_, -1000.0f);
}

void SynapseGroup::apply_stdp(
    const std::vector<uint8_t>& pre_fired,
    const std::vector<uint8_t>& post_fired,
    int32_t t
) {
    if (!stdp_enabled_) return;

    float tf = static_cast<float>(t);

    // Update last spike times
    for (size_t i = 0; i < n_pre_; ++i) {
        if (pre_fired[i]) last_spike_pre_[i] = tf;
    }
    for (size_t i = 0; i < n_post_; ++i) {
        if (post_fired[i]) last_spike_post_[i] = tf;
    }

    // For each synapse: if pre or post fired this step, apply STDP
    for (size_t pre = 0; pre < n_pre_; ++pre) {
        int32_t start = row_ptr_[pre];
        int32_t end   = row_ptr_[pre + 1];

        for (int32_t s = start; s < end; ++s) {
            size_t post = static_cast<size_t>(col_idx_[s]);

            float dw = 0.0f;

            // Pre fired this step: check last post spike time (LTD if post was recent)
            if (pre_fired[pre]) {
                dw += stdp_delta_w(tf, last_spike_post_[post], stdp_params_);
            }

            // Post fired this step: check last pre spike time (LTP if pre was recent)
            if (post_fired[post]) {
                dw += stdp_delta_w(last_spike_pre_[pre], tf, stdp_params_);
            }

            if (dw != 0.0f) {
                weights_[s] += dw;
                weights_[s] = std::clamp(weights_[s], stdp_params_.w_min, stdp_params_.w_max);
            }
        }
    }
}

void SynapseGroup::apply_stdp_error_gated(
    const std::vector<uint8_t>& pre_fired,
    const std::vector<uint8_t>& post_fired,
    const std::vector<int8_t>& post_spike_type,
    int8_t required_type,
    int32_t t
) {
    if (!stdp_enabled_) return;

    float tf = static_cast<float>(t);

    // Update last spike times (all spikes, not just error)
    for (size_t i = 0; i < n_pre_; ++i) {
        if (pre_fired[i]) last_spike_pre_[i] = tf;
    }
    for (size_t i = 0; i < n_post_; ++i) {
        if (post_fired[i]) last_spike_post_[i] = tf;
    }

    // Error-gated: only update weights when post fires with required_type
    for (size_t pre = 0; pre < n_pre_; ++pre) {
        int32_t start = row_ptr_[pre];
        int32_t end_idx = row_ptr_[pre + 1];

        for (int32_t s = start; s < end_idx; ++s) {
            size_t post = static_cast<size_t>(col_idx_[s]);

            float dw = 0.0f;

            // Pre fired: LTD as normal (prediction without input = weaken)
            if (pre_fired[pre]) {
                dw += stdp_delta_w(tf, last_spike_post_[post], stdp_params_);
            }

            // Post fired: LTP ONLY if post spike type matches required_type
            // regular spike (error) → LTP; burst (match) → skip LTP
            if (post_fired[post] && post_spike_type[post] == required_type) {
                dw += stdp_delta_w(last_spike_pre_[pre], tf, stdp_params_);
            }

            if (dw != 0.0f) {
                weights_[s] += dw;
                weights_[s] = std::clamp(weights_[s], stdp_params_.w_min, stdp_params_.w_max);
            }
        }
    }
}

} // namespace wuyun
