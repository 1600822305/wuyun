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

    // --- Compute DA level ---
    // SNc DA is tonic-dominant: slowly tracks reward history, not single RPE events
    // Biology: SNc DA supports motor execution and habit maintenance
    //   - Repeated rewards → tonic_baseline slowly increases → habits consolidate
    //   - Single failures don't crash tonic level (unlike VTA phasic)

    // Slow habit consolidation: tonic baseline drifts toward reward history
    // reward_history > 0: agent is doing well → tonic rises → habits strengthen
    // reward_history < 0: agent is struggling → tonic drops → habits weaken (slowly)
    tonic_baseline_ += config_.habit_lr * (reward_history_ - (tonic_baseline_ - config_.tonic_rate));
    tonic_baseline_ = std::clamp(tonic_baseline_, 0.15f, 0.45f);

    // D1 feedback: active D1 MSN → SNc maintenance signal
    // Biology: striatonigral D1 MSN project back to SNc (positive feedback loop)
    //   well-learned action → D1 fires consistently → SNc tonic maintained
    float d1_boost = d1_activity_ * 0.02f;  // Very small contribution

    // DA output: mostly tonic, small modulation from D1 feedback
    da_level_ = tonic_baseline_ + d1_boost;
    da_level_ = std::clamp(da_level_, 0.1f, 0.5f);
}

void SNc_DA::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 15.0f : 10.0f;
        size_t idx = evt.neuron_id % da_pop_.size();
        psp_buf_[idx] += current;
    }
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
