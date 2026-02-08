#include "region/subcortical/nucleus_accumbens.h"
#include "core/types.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

// VP (Ventral Pallidum) params: tonic firing, similar to GPi/GPe
// Biology: VP neurons are tonically active GABAergic neurons, inhibited by NAcc MSN
static NeuronParams VP_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -55.0f;   // depolarized → tonic firing
    p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -60.0f;
    p.somatic.tau_m = 10.0f;     // fast dynamics
    p.somatic.r_s = 1.0f;
    p.somatic.a = 0.0f;   // no subthreshold adaptation
    p.somatic.b = 0.0f;   // no spike-triggered adaptation
    return p;
}

NucleusAccumbens::NucleusAccumbens(const NAccConfig& config)
    : BrainRegion(config.name,
                  config.n_core_d1 + config.n_core_d2 + config.n_shell + config.n_vp)
    , config_(config)
    , core_d1_(config.n_core_d1, MSN_D1_PARAMS())
    , core_d2_(config.n_core_d2, MSN_D2_PARAMS())
    , shell_(config.n_shell, MSN_D1_PARAMS())   // Shell MSN (D1-like)
    , vp_(config.n_vp, VP_PARAMS())             // VP tonic inhibitory
    , psp_d1_(config.n_core_d1, 0.0f)
    , psp_d2_(config.n_core_d2, 0.0f)
    , psp_shell_(config.n_shell, 0.0f)
    , psp_vp_(config.n_vp, 0.0f)
    , fired_(config.n_core_d1 + config.n_core_d2 + config.n_shell + config.n_vp, 0)
    , spike_type_(config.n_core_d1 + config.n_core_d2 + config.n_shell + config.n_vp, 0)
{
}

void NucleusAccumbens::step(int32_t t, float dt) {
    // --- DA modulation of Core MSN ---
    // D1 (approach): DA above baseline → enhance firing → "want to approach"
    // D2 (avoidance): DA below baseline → enhance firing → "want to avoid"
    float da_error = da_level_ - DA_BASELINE;

    // Inject PSP + DA-modulated up-state drive into Core MSN
    for (size_t i = 0; i < core_d1_.size(); ++i) {
        float da_boost = std::max(0.0f, da_error) * D1_DA_GAIN;  // D1: positive DA → approach
        core_d1_.inject_basal(i, psp_d1_[i] + 20.0f + da_boost);  // 20.0 = up-state drive
        psp_d1_[i] *= PSP_DECAY;
    }
    for (size_t i = 0; i < core_d2_.size(); ++i) {
        float da_boost = std::max(0.0f, -da_error) * D2_DA_GAIN;  // D2: negative DA → avoidance
        core_d2_.inject_basal(i, psp_d2_[i] + 20.0f + da_boost);
        psp_d2_[i] *= PSP_DECAY;
    }

    // Shell: novelty detection (receives hippocampal/contextual input)
    for (size_t i = 0; i < shell_.size(); ++i) {
        shell_.inject_basal(i, psp_shell_[i] + 15.0f);  // Lower tonic drive
        psp_shell_[i] *= PSP_DECAY;
    }

    // VP: tonically active, inhibited by Core D1 MSN
    // Biology: VP neurons fire tonically, NAcc Core D1→VP inhibition
    //   releases VP targets (MD thalamus → PFC motivation drive)
    float d1_inhibition = 0.0f;
    for (size_t i = 0; i < core_d1_.size(); ++i) {
        if (core_d1_.fired()[i]) d1_inhibition += 5.0f;
    }
    float d2_excitation = 0.0f;
    for (size_t i = 0; i < core_d2_.size(); ++i) {
        if (core_d2_.fired()[i]) d2_excitation += 3.0f;
    }
    for (size_t i = 0; i < vp_.size(); ++i) {
        // VP tonic drive - D1 inhibition + D2 excitation (indirect)
        vp_.inject_basal(i, psp_vp_[i] + 30.0f - d1_inhibition + d2_excitation);
        psp_vp_[i] *= PSP_DECAY;
    }

    // Step all populations
    core_d1_.step(t, dt);
    core_d2_.step(t, dt);
    shell_.step(t, dt);
    vp_.step(t, dt);

    // --- Compute motivation output ---
    // Count D1 (approach) vs D2 (avoidance) firing
    size_t d1_fires = 0, d2_fires = 0;
    for (size_t i = 0; i < core_d1_.size(); ++i)
        if (core_d1_.fired()[i]) d1_fires++;
    for (size_t i = 0; i < core_d2_.size(); ++i)
        if (core_d2_.fired()[i]) d2_fires++;

    // motivation > 0: approach dominant (increase motor vigor)
    // motivation < 0: avoidance dominant (suppress motor output)
    float raw_motivation = static_cast<float>(d1_fires) - static_cast<float>(d2_fires);
    motivation_ = motivation_ * 0.9f + raw_motivation * 0.1f;  // Smooth

    // --- Novelty detection from shell ---
    size_t shell_fires = 0;
    for (size_t i = 0; i < shell_.size(); ++i)
        if (shell_.fired()[i]) shell_fires++;

    float shell_rate = static_cast<float>(shell_fires) / static_cast<float>(shell_.size());
    // Novelty = deviation from expected shell activity
    float novelty_raw = std::abs(shell_rate - shell_activity_smooth_);
    novelty_ = novelty_ * 0.9f + novelty_raw * 0.1f;
    shell_activity_smooth_ = shell_activity_smooth_ * NOVELTY_TAU + shell_rate * (1.0f - NOVELTY_TAU);

    // Aggregate firing state
    aggregate_state();
}

void NucleusAccumbens::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 50.0f : 30.0f;
        size_t src = evt.neuron_id;

        // Route to Core D1 and D2 (broad input)
        for (size_t j = 0; j < core_d1_.size(); ++j) {
            size_t idx = (src + j) % core_d1_.size();
            psp_d1_[idx] += current * 0.5f;  // Weaker than dorsal BG (motivation, not selection)
        }
        for (size_t j = 0; j < core_d2_.size(); ++j) {
            size_t idx = (src + j) % core_d2_.size();
            psp_d2_[idx] += current * 0.5f;
        }
        // Route to Shell (contextual input)
        size_t shell_idx = src % shell_.size();
        psp_shell_[shell_idx] += current * 0.3f;
    }
}

void NucleusAccumbens::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void NucleusAccumbens::inject_external(const std::vector<float>& currents) {
    // External input → Core D1 (approach bias from spatial value etc.)
    for (size_t i = 0; i < std::min(currents.size(), core_d1_.size()); ++i) {
        psp_d1_[i] += currents[i];
    }
}

void NucleusAccumbens::aggregate_state() {
    size_t offset = 0;
    auto copy_pop = [&](const NeuronPopulation& pop) {
        for (size_t i = 0; i < pop.size(); ++i) {
            fired_[offset + i] = pop.fired()[i];
            spike_type_[offset + i] = pop.spike_type()[i];
        }
        offset += pop.size();
    };
    copy_pop(core_d1_);
    copy_pop(core_d2_);
    copy_pop(shell_);
    copy_pop(vp_);
}

} // namespace wuyun
