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

} // namespace wuyun
