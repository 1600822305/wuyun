#include "region/subcortical/superior_colliculus.h"
#include "core/types.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

// SC superficial neurons: fast visual processing, low threshold
static NeuronParams SC_VISUAL_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -60.0f;
    p.somatic.v_threshold = -48.0f;  // Low threshold → fast response
    p.somatic.v_reset = -55.0f;
    p.somatic.tau_m = 8.0f;          // Very fast membrane (faster than cortex)
    p.somatic.r_s = 1.2f;
    p.somatic.a = 0.0f;
    p.somatic.b = 0.5f;             // Minimal adaptation
    p.somatic.tau_w = 100.0f;
    p.kappa = 0.0f;                  // No apical (not pyramidal)
    return p;
}

// SC deep neurons: multimodal integration, motor-like output
static NeuronParams SC_MOTOR_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -58.0f;
    p.somatic.v_threshold = -45.0f;  // Slightly higher threshold (needs convergent input)
    p.somatic.v_reset = -53.0f;
    p.somatic.tau_m = 10.0f;
    p.somatic.r_s = 1.0f;
    p.somatic.a = 0.01f;
    p.somatic.b = 1.0f;
    p.somatic.tau_w = 150.0f;
    p.kappa = 0.0f;
    return p;
}

SuperiorColliculus::SuperiorColliculus(const SCConfig& config)
    : BrainRegion(config.name, config.n_superficial + config.n_deep)
    , config_(config)
    , superficial_(config.n_superficial, SC_VISUAL_PARAMS())
    , deep_(config.n_deep, SC_MOTOR_PARAMS())
    , psp_sup_(config.n_superficial, 0.0f)
    , psp_deep_(config.n_deep, 0.0f)
    , fired_(config.n_superficial + config.n_deep, 0)
    , spike_type_(config.n_superficial + config.n_deep, 0)
{
}

void SuperiorColliculus::step(int32_t t, float dt) {
    // --- Superficial layer: retinotopic visual map ---
    // Receives direct retinal/LGN input, detects visual events
    float total_input = 0.0f;
    for (size_t i = 0; i < superficial_.size(); ++i) {
        superficial_.inject_basal(i, psp_sup_[i]);
        total_input += psp_sup_[i];
        psp_sup_[i] *= PSP_DECAY;
    }

    // --- Deep layer: receives from superficial + cortical feedback ---
    // Superficial → Deep feedforward
    for (size_t i = 0; i < deep_.size(); ++i) {
        // Deep gets input from superficial (broad convergence)
        float sup_drive = 0.0f;
        for (size_t j = 0; j < superficial_.size(); ++j) {
            if (superficial_.fired()[j]) sup_drive += 8.0f;
        }
        deep_.inject_basal(i, psp_deep_[i] + sup_drive);
        psp_deep_[i] *= PSP_DECAY;
    }

    superficial_.step(t, dt);
    deep_.step(t, dt);

    // --- Saliency computation ---
    // Saliency = change detection (onset/offset of visual stimuli)
    // Biology: SC responds strongly to stimulus ONSET, habituates to static scenes
    float current_input = total_input / std::max<float>(1.0f, static_cast<float>(superficial_.size()));
    float input_change = std::abs(current_input - prev_input_level_);
    prev_input_level_ = prev_input_level_ * 0.95f + current_input * 0.05f;  // Slow adaptation

    // Count deep layer firing as saliency measure
    size_t deep_fires = 0;
    for (size_t i = 0; i < deep_.size(); ++i)
        if (deep_.fired()[i]) deep_fires++;

    float firing_saliency = static_cast<float>(deep_fires) / static_cast<float>(std::max<size_t>(deep_.size(), 1));
    saliency_ = saliency_ * 0.9f + (input_change * 0.5f + firing_saliency * 0.5f) * 0.1f;

    aggregate_state();
}

void SuperiorColliculus::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 40.0f : 25.0f;
        // Route to superficial (visual input) and deep (cortical feedback)
        size_t sup_idx = evt.neuron_id % superficial_.size();
        psp_sup_[sup_idx] += current;

        // Some input also reaches deep layer (broad routing)
        size_t deep_idx = evt.neuron_id % deep_.size();
        psp_deep_[deep_idx] += current * 0.3f;  // Weaker to deep
    }
}

void SuperiorColliculus::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void SuperiorColliculus::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), superficial_.size()); ++i) {
        psp_sup_[i] += currents[i];
    }
}

void SuperiorColliculus::aggregate_state() {
    size_t offset = 0;
    auto copy_pop = [&](const NeuronPopulation& pop) {
        for (size_t i = 0; i < pop.size(); ++i) {
            fired_[offset + i] = pop.fired()[i];
            spike_type_[offset + i] = pop.spike_type()[i];
        }
        offset += pop.size();
    };
    copy_pop(superficial_);
    copy_pop(deep_);
}

} // namespace wuyun
