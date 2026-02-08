#include "region/neuromod/vta_da.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

VTA_DA::VTA_DA(const VTAConfig& config)
    : BrainRegion(config.name, config.n_da_neurons)
    , config_(config)
    , da_neurons_(config.n_da_neurons, DOPAMINE_NEURON_PARAMS())
    , da_level_(config.tonic_rate)
    , psp_da_(config.n_da_neurons, 0.0f)
    , fired_(config.n_da_neurons, 0)
    , spike_type_(config.n_da_neurons, 0)
{}

void VTA_DA::step(int32_t t, float dt) {
    oscillation_.step(dt);

    // =========================================================
    // v46: Spike-driven RPE (replaces inject_reward scalar)
    // Biology (Schultz 1997, Nieh 2015, Takahashi 2011):
    //   VTA DA neurons receive convergent inputs:
    //   - Hedonic (Hypothalamus LH): excitatory → "actual reward arrived"
    //   - Prediction (OFC): inhibitory → "expected reward" → suppresses DA
    //   - LHb: inhibitory → "negative RPE / frustration"
    //   RPE emerges from the net drive: hedonic excitation - prediction inhibition
    //   No scalar reward is injected. VTA only sees spike patterns.
    // =========================================================

    // Accumulate LHb inhibition into sustained PSP buffer
    if (lhb_inhibition_ > 0.01f) {
        lhb_inh_psp_ += lhb_inhibition_ * 180.0f;
    }

    // Inject neural drive into DA neurons:
    //   Baseline tonic (20.0) + hedonic excitation - prediction inhibition
    //   + general cortical/striatal PSP - LHb inhibition
    for (size_t i = 0; i < psp_da_.size(); ++i) {
        float psp_input = psp_da_[i] > 0.5f ? psp_da_[i] : 0.0f;
        // Hedonic spikes (LH) excite DA neurons → burst when reward arrives
        // Prediction spikes (OFC) suppress DA neurons → no burst when expected
        // Net effect: unexpected reward → hedonic high, prediction low → DA burst
        //             expected reward → hedonic high, prediction high → cancel → tonic
        //             expected omission → hedonic low, prediction high → DA pause
        float net_drive = 20.0f + hedonic_psp_ - prediction_psp_ * 0.7f
                        + psp_input - lhb_inh_psp_;
        da_neurons_.inject_basal(i, std::max(0.0f, net_drive));
        psp_da_[i] *= PSP_DECAY;
    }
    hedonic_psp_ *= HEDONIC_PSP_DECAY;
    prediction_psp_ *= PREDICTION_PSP_DECAY;
    lhb_inh_psp_ *= LHB_INH_PSP_DECAY;

    da_neurons_.step(t, dt);

    // Compute DA output level from firing rate
    size_t n_fired = 0;
    for (size_t i = 0; i < da_neurons_.size(); ++i) {
        fired_[i]      = da_neurons_.fired()[i];
        spike_type_[i] = da_neurons_.spike_type()[i];
        if (fired_[i]) n_fired++;
    }

    // v37: DA level based on firing rate deviation from tonic baseline
    float firing_rate = static_cast<float>(n_fired) / static_cast<float>(da_neurons_.size());

    float phasic = 0.0f;
    if (step_count_ >= WARMUP_STEPS) {
        phasic = (firing_rate - tonic_firing_smooth_) * config_.phasic_gain * 3.0f;
    }

    // LHb adds additional DA suppression
    float lhb_suppression = lhb_inhibition_ * config_.phasic_gain;

    da_level_ = std::clamp(config_.tonic_rate + phasic - lhb_suppression, 0.0f, 1.0f);

    // v46: RPE from spike rates (diagnostic, replaces old reward_input_ - expected_reward_)
    float hedonic_rate = hedonic_psp_ / std::max<float>(1.0f, static_cast<float>(da_neurons_.size()));
    float prediction_rate = prediction_psp_ / std::max<float>(1.0f, static_cast<float>(da_neurons_.size()));
    last_rpe_ = hedonic_rate - prediction_rate;

    // Update tonic firing rate estimate
    if (hedonic_psp_ < 5.0f && lhb_inh_psp_ < 5.0f) {
        float alpha = (step_count_ < WARMUP_STEPS) ? 0.1f : 0.01f;
        tonic_firing_smooth_ = tonic_firing_smooth_ * (1.0f - alpha) + firing_rate * alpha;
    }
    ++step_count_;

    // Reset inputs (consumed)
    lhb_inhibition_ = 0.0f;
}

void VTA_DA::receive_spikes(const std::vector<SpikeEvent>& events) {
    // v46: Route incoming spikes by source region (SpikeEvent.region_id)
    // Hedonic source (Hypothalamus) → hedonic_psp_ (excitatory, actual reward)
    // Prediction source (OFC) → prediction_psp_ (inhibitory, expected value)
    // All other sources → psp_da_ (general cortical/striatal modulation)
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 20.0f : 12.0f;

        if (has_hedonic_source_ && evt.region_id == hedonic_source_id_) {
            // Hypothalamus LH spikes → "actual reward arrived"
            hedonic_psp_ += current * 1.5f;  // Strong excitatory drive
        } else if (has_prediction_source_ && evt.region_id == prediction_source_id_) {
            // OFC spikes → "expected value" → will suppress DA (prediction inhibition)
            prediction_psp_ += current;
        } else {
            // General cortical/striatal modulation (existing behavior)
            size_t base = evt.neuron_id % psp_da_.size();
            for (size_t k = 0; k < 3 && (base + k) < psp_da_.size(); ++k) {
                psp_da_[base + k] += current;
            }
        }
    }
}

void VTA_DA::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void VTA_DA::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), da_neurons_.size()); ++i) {
        da_neurons_.inject_basal(i, currents[i]);
    }
}

// v46: inject_reward() and set_expected_reward() REMOVED (anti-cheat).
// VTA now computes RPE from spike rates: hedonic (Hypothalamus) - prediction (OFC).
// Reward enters through Hypothalamus→VTA SpikeBus, not agent scalar injection.

void VTA_DA::inject_lhb_inhibition(float inhibition) {
    lhb_inhibition_ = std::clamp(inhibition, 0.0f, 1.0f);
}

} // namespace wuyun
