#include "region/neuromod/snc_da.h"
#include "core/types.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

// SNc DA neurons: tonic firing, less bursty than VTA
static NeuronParams SNC_DA_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -60.0f;
    p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -55.0f;
    p.somatic.tau_m = 15.0f;
    p.somatic.r_s = 1.0f;
    p.somatic.a = 0.02f;   // mild subthreshold adaptation
    p.somatic.b = 1.0f;    // mild spike-triggered adaptation (less than MSN)
    p.somatic.tau_w = 200.0f;
    p.kappa = 0.0f;        // no apical coupling (DA neurons are not pyramidal)
    return p;
}

SNc_DA::SNc_DA(const SNcConfig& config)
    : BrainRegion(config.name, config.n_da_neurons)
    , config_(config)
    , da_pop_(config.n_da_neurons, SNC_DA_PARAMS())
    , da_level_(config.tonic_rate)
    , tonic_baseline_(config.tonic_rate)
    , fired_(config.n_da_neurons, 0)
    , spike_type_(config.n_da_neurons, 0)
    , psp_buf_(config.n_da_neurons, 0.0f)
{
}

void SNc_DA::step(int32_t t, float dt) {
    // Inject tonic drive + cortical PSP into DA neurons
    for (size_t i = 0; i < da_pop_.size(); ++i) {
        // Tonic drive: keeps SNc neurons firing at baseline rate
        float tonic_drive = 25.0f;  // Sustained depolarization
        da_pop_.inject_basal(i, psp_buf_[i] + tonic_drive);
        psp_buf_[i] *= PSP_DECAY;
    }

    da_pop_.step(t, dt);

    // Copy firing state
    for (size_t i = 0; i < da_pop_.size(); ++i) {
        fired_[i] = da_pop_.fired()[i];
        spike_type_[i] = da_pop_.spike_type()[i];
    }

    // --- Spike-driven tonic adaptation (anti-cheat compliant) ---
    // Biology: striatonigral D1 MSN → SNc positive feedback (Haber 2003)
    //   Well-learned actions → D1 fires consistently → BG→SNc spikes arrive →
    //   SNc tonic maintained/rises → habits consolidate.
    //   Poorly learned / no activity → fewer spikes → tonic drifts down.
    //
    // Anti-cheat: Previously used inject_reward_history(avg_reward) which gave
    //   SNc "omniscient" access to the agent's computed average reward.
    //   Now SNc only knows what arrives through SpikeBus (BG→SNc projection).
    float spike_rate = static_cast<float>(received_spike_count_)
                     / std::max<float>(1.0f, static_cast<float>(da_pop_.size()));
    received_spike_count_ = 0;  // Reset for next step

    // Tonic baseline slowly adapts toward received spike rate
    // High spike rate (D1 active) → tonic rises → habit maintenance
    // Low spike rate → tonic drifts back to baseline → habits can be overwritten
    float spike_drive = spike_rate * 0.1f;  // Scaled D1 feedback signal
    tonic_baseline_ += config_.habit_lr * (spike_drive - (tonic_baseline_ - config_.tonic_rate));
    tonic_baseline_ = std::clamp(tonic_baseline_, 0.15f, 0.45f);

    // DA output: purely tonic (no direct agent-computed scalars)
    da_level_ = tonic_baseline_;
    da_level_ = std::clamp(da_level_, 0.1f, 0.5f);
}

void SNc_DA::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 15.0f : 10.0f;
        size_t idx = evt.neuron_id % da_pop_.size();
        psp_buf_[idx] += current;
    }
    // Track incoming spike count for tonic adaptation
    // (anti-cheat: replaces inject_d1_activity scalar bypass)
    received_spike_count_ += events.size();
}

void SNc_DA::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void SNc_DA::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), da_pop_.size()); ++i) {
        psp_buf_[i] += currents[i];
    }
}

} // namespace wuyun
