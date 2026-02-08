#include "region/subcortical/periaqueductal_gray.h"
#include "core/types.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

// dlPAG: active defense neurons (flight/fight)
// Low threshold, fast, minimal adaptation → rapid response
static NeuronParams DLPAG_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -60.0f;
    p.somatic.v_threshold = -45.0f;  // Low threshold → fast activation
    p.somatic.v_reset = -55.0f;
    p.somatic.tau_m = 8.0f;          // Very fast (emergency circuit)
    p.somatic.r_s = 1.2f;
    p.somatic.a = 0.0f;
    p.somatic.b = 0.5f;             // Minimal adaptation (sustained defense)
    p.somatic.tau_w = 100.0f;
    p.kappa = 0.0f;
    return p;
}

// vlPAG: passive defense neurons (freeze)
// Higher threshold, needs sustained fear input
static NeuronParams VLPAG_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -62.0f;
    p.somatic.v_threshold = -48.0f;
    p.somatic.v_reset = -57.0f;
    p.somatic.tau_m = 12.0f;        // Slightly slower than dlPAG
    p.somatic.r_s = 1.0f;
    p.somatic.a = 0.01f;
    p.somatic.b = 1.0f;
    p.somatic.tau_w = 150.0f;
    p.kappa = 0.0f;
    return p;
}

PeriaqueductalGray::PeriaqueductalGray(const PAGConfig& config)
    : BrainRegion(config.name, config.n_dlpag + config.n_vlpag)
    , config_(config)
    , dlpag_(config.n_dlpag, DLPAG_PARAMS())
    , vlpag_(config.n_vlpag, VLPAG_PARAMS())
    , psp_dl_(config.n_dlpag, 0.0f)
    , psp_vl_(config.n_vlpag, 0.0f)
    , fired_(config.n_dlpag + config.n_vlpag, 0)
    , spike_type_(config.n_dlpag + config.n_vlpag, 0)
{
}

void PeriaqueductalGray::step(int32_t t, float dt) {
    // --- Fear gating: only activate if fear exceeds threshold ---
    // Biology: PAG requires sufficient CeA excitation to trigger defense
    // Prevents random noise from causing defensive responses
    float gated_fear = std::max(0.0f, fear_input_ - FEAR_THRESHOLD);

    // --- dlPAG: active defense (flight) ---
    // Activated by moderate-to-high fear → "run away"
    // Biology: dlPAG stimulation produces active escape behaviors
    for (size_t i = 0; i < dlpag_.size(); ++i) {
        float fear_drive = gated_fear * 150.0f;  // Strong drive from fear
        dlpag_.inject_basal(i, psp_dl_[i] + fear_drive);
        psp_dl_[i] *= PSP_DECAY;
    }

    // --- vlPAG: passive defense (freeze) ---
    // Activated by sustained low-moderate fear → "freeze in place"
    // Biology: vlPAG inhibits dlPAG when threat is nearby (freeze > flight)
    // In GridWorld: freeze = suppress motor output (less effective than flight)
    for (size_t i = 0; i < vlpag_.size(); ++i) {
        float fear_drive = gated_fear * 100.0f;  // Weaker than dlPAG
        // vlPAG gets inhibited by dlPAG (mutual antagonism)
        float dl_inhibition = 0.0f;
        for (size_t j = 0; j < dlpag_.size(); ++j) {
            if (dlpag_.fired()[j]) dl_inhibition += 5.0f;
        }
        vlpag_.inject_basal(i, psp_vl_[i] + fear_drive - dl_inhibition);
        psp_vl_[i] *= PSP_DECAY;
    }

    dlpag_.step(t, dt);
    vlpag_.step(t, dt);

    // --- Compute defense outputs ---
    size_t dl_fires = 0, vl_fires = 0;
    for (size_t i = 0; i < dlpag_.size(); ++i)
        if (dlpag_.fired()[i]) dl_fires++;
    for (size_t i = 0; i < vlpag_.size(); ++i)
        if (vlpag_.fired()[i]) vl_fires++;

    float dl_rate = static_cast<float>(dl_fires) / static_cast<float>(std::max<size_t>(dlpag_.size(), 1));
    float vl_rate = static_cast<float>(vl_fires) / static_cast<float>(std::max<size_t>(vlpag_.size(), 1));

    // Defense: active flight response (smoothed)
    defense_level_ = defense_level_ * 0.8f + dl_rate * 0.2f;
    // Freeze: passive defense (smoothed)
    freeze_level_ = freeze_level_ * 0.8f + vl_rate * 0.2f;
    // Arousal: both PAG columns drive LC arousal
    arousal_ = arousal_ * 0.9f + (dl_rate + vl_rate) * 0.5f * 0.1f;

    // Decay fear input (must be re-injected each step)
    fear_input_ *= 0.5f;

    aggregate_state();
}

void PeriaqueductalGray::inject_fear(float cea_drive) {
    fear_input_ = cea_drive;
}

void PeriaqueductalGray::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 30.0f : 20.0f;
        // Route to dlPAG (active defense)
        size_t dl_idx = evt.neuron_id % dlpag_.size();
        psp_dl_[dl_idx] += current;
        // Some to vlPAG
        size_t vl_idx = evt.neuron_id % vlpag_.size();
        psp_vl_[vl_idx] += current * 0.5f;
    }
}

void PeriaqueductalGray::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void PeriaqueductalGray::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), dlpag_.size()); ++i) {
        psp_dl_[i] += currents[i];
    }
}

void PeriaqueductalGray::aggregate_state() {
    size_t offset = 0;
    auto copy_pop = [&](const NeuronPopulation& pop) {
        for (size_t i = 0; i < pop.size(); ++i) {
            fired_[offset + i] = pop.fired()[i];
            spike_type_[offset + i] = pop.spike_type()[i];
        }
        offset += pop.size();
    };
    copy_pop(dlpag_);
    copy_pop(vlpag_);
}

} // namespace wuyun
