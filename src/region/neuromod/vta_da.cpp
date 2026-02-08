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

    // Compute RPE = actual reward - expected reward
    last_rpe_ = reward_input_ - expected_reward_;

    // Accumulate reward into PSP buffer (sustained drive across multiple steps)
    if (std::abs(reward_input_) > 0.001f) {
        reward_psp_ += last_rpe_ * config_.phasic_gain * 200.0f;
    }

    // Accumulate LHb inhibition into sustained PSP buffer
    // Biology: LHb → RMTg (GABA) → VTA: inhibitory postsynaptic current
    // that hyperpolarizes DA neurons, causing a firing pause (200-500ms)
    if (lhb_inhibition_ > 0.01f) {
        lhb_inh_psp_ += lhb_inhibition_ * 180.0f;  // Strong inhibitory drive
    }

    // Inject PSP buffer (cross-region input, sustained)
    for (size_t i = 0; i < psp_da_.size(); ++i) {
        float psp_input = psp_da_[i] > 0.5f ? psp_da_[i] : 0.0f;
        // Tonic baseline drive + sustained reward PSP + cross-region PSP - LHb inhibition
        // 20.0 = enough for ~4Hz spontaneous firing (normal DA tonic activity)
        // LHb inhibition subtracts from drive → DA neurons hyperpolarize → firing pause
        float net_drive = 20.0f + reward_psp_ + psp_input - lhb_inh_psp_;
        da_neurons_.inject_basal(i, std::max(0.0f, net_drive));
        psp_da_[i] *= PSP_DECAY;
    }
    reward_psp_ *= REWARD_PSP_DECAY;  // Slow decay of reward signal
    lhb_inh_psp_ *= LHB_INH_PSP_DECAY;  // Slow decay of LHb inhibition

    da_neurons_.step(t, dt);

    // Compute DA output level from firing rate
    size_t n_fired = 0;
    for (size_t i = 0; i < da_neurons_.size(); ++i) {
        fired_[i]      = da_neurons_.fired()[i];
        spike_type_[i] = da_neurons_.spike_type()[i];
        if (fired_[i]) n_fired++;
    }

    // v37: DA level based on firing rate deviation from tonic baseline
    // Biology (Grace 1991, Schultz 1997):
    //   DA concentration depends on firing rate relative to tonic baseline:
    //   - Burst firing (reward) → DA release >> reuptake → DA rises above tonic
    //   - Firing pause (punishment/LHb) → reuptake > release → DA drops BELOW tonic
    //
    // Previous bug: negative phasic used instantaneous RPE (reward_input_, reset to 0
    //   after 1 step). Even though reward_psp_ kept DA neurons suppressed for ~15 steps,
    //   da_level_ returned to 0.3 after step 1 because phasic_negative = 0 when RPE = 0.
    //   Result: DA dip lasted only 1 engine step → BG never saw punishment signal.
    //
    // Fix: compute DA purely from firing rate. When neurons fire more than tonic → DA up.
    //   When neurons fire less (suppressed by negative reward_psp_) → DA down.
    //   This naturally produces sustained DA dip for the entire reward_psp_ decay period.
    float firing_rate = static_cast<float>(n_fired) / static_cast<float>(da_neurons_.size());

    // Phasic component: deviation of firing rate from tracked tonic baseline
    // phasic_gain * 3.0 amplifies the signal to produce meaningful DA swings
    float phasic = 0.0f;
    if (step_count_ >= WARMUP_STEPS) {
        // After warmup: firing-rate-based DA (both burst AND pause detected)
        phasic = (firing_rate - tonic_firing_smooth_) * config_.phasic_gain * 3.0f;
    }
    // During warmup: phasic=0 → DA stays at tonic_rate (no spurious D2 strengthening)

    // LHb adds additional DA suppression (separate from firing pause)
    float lhb_suppression = lhb_inhibition_ * config_.phasic_gain;

    da_level_ = std::clamp(config_.tonic_rate + phasic - lhb_suppression, 0.0f, 1.0f);

    // Update tonic firing rate estimate (always track, even during warmup)
    // Fast convergence during warmup (α=0.1), slow tracking after (α=0.01)
    if (std::abs(reward_psp_) < 5.0f && lhb_inh_psp_ < 5.0f) {
        float alpha = (step_count_ < WARMUP_STEPS) ? 0.1f : 0.01f;
        tonic_firing_smooth_ = tonic_firing_smooth_ * (1.0f - alpha) + firing_rate * alpha;
    }
    ++step_count_;

    // Reset inputs (consumed)
    reward_input_ = 0.0f;
    lhb_inhibition_ = 0.0f;
}

void VTA_DA::receive_spikes(const std::vector<SpikeEvent>& events) {
    // Arriving spikes → PSP buffer (sustained drive via exponential decay)
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 20.0f : 12.0f;
        size_t base = evt.neuron_id % psp_da_.size();
        for (size_t k = 0; k < 3 && (base + k) < psp_da_.size(); ++k) {
            psp_da_[base + k] += current;
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

void VTA_DA::inject_reward(float reward) {
    reward_input_ = reward;
}

void VTA_DA::set_expected_reward(float expected) {
    expected_reward_ = expected;
}

void VTA_DA::inject_lhb_inhibition(float inhibition) {
    lhb_inhibition_ = std::clamp(inhibition, 0.0f, 1.0f);
}

} // namespace wuyun
