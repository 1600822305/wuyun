#include "region/cortical_region.h"
#include <algorithm>

namespace wuyun {

CorticalRegion::CorticalRegion(const std::string& name, const ColumnConfig& config)
    : BrainRegion(name, config.n_l4_stellate + config.n_l23_pyramidal +
                        config.n_l5_pyramidal + config.n_l6_pyramidal +
                        config.n_pv_basket + config.n_sst_martinotti +
                        config.n_vip)
    , column_(config)
    , fired_(n_neurons_, 0)
    , spike_type_(n_neurons_, 0)
    , psp_buffer_(config.n_l4_stellate, 0.0f)
    , psp_current_regular_(config.input_psp_regular)
    , psp_current_burst_(config.input_psp_burst)
    , psp_fan_out_(std::max<size_t>(3, static_cast<size_t>(config.n_l4_stellate * config.input_fan_out_frac)))
    , pc_prediction_buf_(config.n_l23_pyramidal, 0.0f)
{}

void CorticalRegion::step(int32_t t, float dt) {
    // Update oscillation and neuromodulation
    oscillation_.step(dt);
    neuromod_.step(dt);

    // NE gain modulation: neuromod system's gain affects all incoming PSP
    float ne_gain = neuromod_.compute_effect().gain;  // 0.5 ~ 2.0

    // === Predictive coding: update precision from neuromodulators ===
    if (pc_enabled_) {
        // NE -> sensory precision (bottom-up salience)
        // gain = 0.5 + 1.5*NE, so NE=0.5 -> gain=1.25
        pc_precision_sensory_ = ne_gain;

        // ACh -> inverse prior precision (high ACh = distrust predictions)
        // ACh typically 0.0~1.0, compute_effect doesn't have ACh directly
        // Use neuromod ACh level: high ACh -> low prior precision
        float ach = neuromod_.current().ach;
        pc_precision_prior_ = std::max(0.2f, 1.0f - 0.8f * ach);
    }

    // === Working memory: inject recurrent buffer into L2/3 basal ===
    if (wm_enabled_) {
        float da = neuromod_.current().da;
        wm_da_gain_ = 1.0f + WM_DA_SENSITIVITY * da;

        auto& l23 = column_.l23();
        for (size_t i = 0; i < wm_recurrent_buf_.size(); ++i) {
            if (wm_recurrent_buf_[i] > 0.5f) {
                l23.inject_basal(i, wm_recurrent_buf_[i] * wm_da_gain_);
            }
            wm_recurrent_buf_[i] *= WM_DECAY;
        }
    }

    // === Top-down attention: PSP gain + VIP disinhibition ===
    float att_gain = attention_gain_;
    if (att_gain > 1.01f) {
        // VIP activation → SST inhibition → L2/3 apical disinhibition
        // Letzkus/Pi (2013) disinhibitory attention circuit
        float vip_drive = (att_gain - 1.0f) * VIP_ATT_DRIVE;
        column_.inject_attention(vip_drive);
    }

    // Inject decaying PSP buffer into L4 basal (feedforward sensory input)
    auto& l4 = column_.l4();
    for (size_t i = 0; i < psp_buffer_.size(); ++i) {
        if (psp_buffer_[i] > 0.5f) {
            float current = psp_buffer_[i] * att_gain;  // Attention gain on feedforward
            if (pc_enabled_) current *= pc_precision_sensory_;
            else current *= ne_gain;
            l4.inject_basal(i, current);
        }
        psp_buffer_[i] *= PSP_DECAY;
    }

    // === Predictive coding: inject prediction into L2/3 apical ===
    if (pc_enabled_) {
        auto& l23 = column_.l23();
        float error_sum = 0.0f;
        for (size_t i = 0; i < pc_prediction_buf_.size(); ++i) {
            if (pc_prediction_buf_[i] > 0.5f) {
                // Prediction arrives as INHIBITORY input to L2/3 apical
                // (predictions suppress prediction error units)
                float pred = pc_prediction_buf_[i] * pc_precision_prior_;
                l23.inject_apical(i, -pred);  // Negative = suppressive
                error_sum += pc_prediction_buf_[i];
            }
            pc_prediction_buf_[i] *= PC_PRED_DECAY;
        }
        // Smooth prediction error tracking
        float instant_error = error_sum / static_cast<float>(pc_prediction_buf_.size() + 1);
        pc_error_smooth_ += PC_ERROR_SMOOTH * (instant_error - pc_error_smooth_);
    }

    // Step the cortical column
    last_output_ = column_.step(t, dt);

    // === Working memory: L2/3 firing feeds back into recurrent buffer ===
    if (wm_enabled_) {
        auto& l23 = column_.l23();
        const auto& l23_fired = l23.fired();
        size_t fan = static_cast<size_t>(WM_FAN_OUT);
        for (size_t i = 0; i < l23.size(); ++i) {
            if (l23_fired[i]) {
                for (size_t k = 0; k <= fan; ++k) {
                    size_t idx = (i + k) % l23.size();
                    wm_recurrent_buf_[idx] += WM_RECURRENT_STR;
                }
            }
        }
    }

    // Aggregate firing state from all populations
    aggregate_firing_state();
}

void CorticalRegion::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type))
                        ? psp_current_burst_ : psp_current_regular_;

        // Predictive coding: route feedback sources to prediction buffer
        if (pc_enabled_ && pc_feedback_sources_.count(evt.region_id)) {
            // Feedback → prediction buffer (L2/3 sized)
            size_t base = evt.neuron_id % pc_prediction_buf_.size();
            size_t fan = std::max<size_t>(3, pc_prediction_buf_.size() / 10);
            for (size_t k = 0; k < fan; ++k) {
                size_t idx = (base + k) % pc_prediction_buf_.size();
                pc_prediction_buf_[idx] += current * 0.5f;  // Prediction weaker than sensory
            }
        } else {
            // Feedforward → L4 PSP buffer (same as before)
            size_t base = evt.neuron_id % psp_buffer_.size();
            for (size_t k = 0; k < psp_fan_out_; ++k) {
                size_t idx = (base + k) % psp_buffer_.size();
                psp_buffer_[idx] += current;
            }
        }
    }
}

void CorticalRegion::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void CorticalRegion::inject_external(const std::vector<float>& currents) {
    inject_feedforward(currents);
}

void CorticalRegion::inject_feedforward(const std::vector<float>& currents) {
    column_.inject_feedforward(currents);
}

void CorticalRegion::inject_feedback(const std::vector<float>& currents) {
    // Feedback goes to L2/3 and L5 apical dendrites
    // Split: first n_l23 values -> L2/3, rest -> L5
    auto& l23 = column_.l23();
    auto& l5  = column_.l5();

    std::vector<float> l23_cur(l23.size(), 0.0f);
    std::vector<float> l5_cur(l5.size(), 0.0f);

    for (size_t i = 0; i < std::min(currents.size(), l23.size()); ++i) {
        l23_cur[i] = currents[i];
    }
    for (size_t i = 0; i < l5.size() && (i + l23.size()) < currents.size(); ++i) {
        l5_cur[i] = currents[i + l23.size()];
    }
    column_.inject_feedback(l23_cur, l5_cur);
}

void CorticalRegion::inject_attention(float vip_current) {
    column_.inject_attention(vip_current);
}

void CorticalRegion::aggregate_firing_state() {
    // Merge all population firing states into a single flat vector
    // Order: L4, L23, L5, L6, PV, SST, VIP
    size_t offset = 0;
    auto copy_pop = [&](const NeuronPopulation& pop) {
        const auto& f = pop.fired();
        const auto& s = pop.spike_type();
        for (size_t i = 0; i < pop.size(); ++i) {
            fired_[offset + i]      = f[i];
            spike_type_[offset + i] = s[i];
        }
        offset += pop.size();
    };

    copy_pop(column_.l4());
    copy_pop(column_.l23());
    copy_pop(column_.l5());
    copy_pop(column_.l6());

    // Access inhibitory populations through column internals
    // For now, the remaining slots stay 0 (inhibitory firing not exported)
    // This is fine — SpikeBus only needs excitatory output for cross-region routing
}

void CorticalRegion::enable_predictive_coding() {
    pc_enabled_ = true;
}

void CorticalRegion::add_feedback_source(uint32_t region_id) {
    pc_feedback_sources_.insert(region_id);
}

void CorticalRegion::enable_working_memory() {
    wm_enabled_ = true;
    wm_recurrent_buf_.assign(column_.l23().size(), 0.0f);
}

float CorticalRegion::wm_persistence() const {
    if (!wm_enabled_) return 0.0f;
    size_t active = 0;
    for (float v : wm_recurrent_buf_) {
        if (v > 1.0f) active++;
    }
    return static_cast<float>(active) / static_cast<float>(wm_recurrent_buf_.size());
}

} // namespace wuyun
