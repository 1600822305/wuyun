#include "region/limbic/hypothalamus.h"
#include <cmath>
#include <algorithm>
#include <random>

namespace wuyun {

// === Synapse builders (same pattern as septal_nucleus) ===

static SynapseGroup make_empty_hy(size_t n_pre, size_t n_post,
                                  const SynapseParams& params,
                                  CompartmentType target) {
    return SynapseGroup(n_pre, n_post, {}, {}, {}, {}, params, target);
}

static SynapseGroup build_syn_hy(
    size_t n_pre, size_t n_post, float prob, float weight,
    const SynapseParams& params, CompartmentType target, unsigned seed)
{
    std::vector<int32_t> pre, post, d;
    std::vector<float> w;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n_pre; ++i) {
        for (size_t j = 0; j < n_post; ++j) {
            if (dist(rng) < prob) {
                pre.push_back(static_cast<int32_t>(i));
                post.push_back(static_cast<int32_t>(j));
                w.push_back(weight);
                d.push_back(1);
            }
        }
    }
    if (pre.empty()) return make_empty_hy(n_pre, n_post, params, target);
    return SynapseGroup(n_pre, n_post, pre, post, w, d, params, target);
}

// === Constructor ===

Hypothalamus::Hypothalamus(const HypothalamusConfig& config)
    : BrainRegion(config.name,
        config.n_scn + config.n_vlpo + config.n_orexin +
        config.n_pvn + config.n_lh + config.n_vmh)
    , config_(config)
    , scn_(config.n_scn, NeuronParams{})
    , vlpo_(config.n_vlpo, NeuronParams{})
    , orexin_(config.n_orexin, NeuronParams{})
    , pvn_(config.n_pvn, NeuronParams{})
    , lh_(config.n_lh, NeuronParams{})
    , vmh_(config.n_vmh, NeuronParams{})
    , syn_vlpo_to_orexin_(make_empty_hy(config.n_vlpo, config.n_orexin, GABA_A_PARAMS, CompartmentType::BASAL))
    , syn_orexin_to_vlpo_(make_empty_hy(config.n_orexin, config.n_vlpo, GABA_A_PARAMS, CompartmentType::BASAL))
    , syn_scn_to_vlpo_(make_empty_hy(config.n_scn, config.n_vlpo, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_lh_to_vmh_(make_empty_hy(config.n_lh, config.n_vmh, GABA_A_PARAMS, CompartmentType::BASAL))
    , syn_vmh_to_lh_(make_empty_hy(config.n_vmh, config.n_lh, GABA_A_PARAMS, CompartmentType::BASAL))
    , psp_vlpo_(config.n_vlpo, 0.0f)
    , psp_orexin_(config.n_orexin, 0.0f)
    , psp_pvn_(config.n_pvn, 0.0f)
    , fired_all_(n_neurons_, 0)
    , spike_type_all_(n_neurons_, 0)
{
    unsigned seed = 7000;
    syn_vlpo_to_orexin_ = build_syn_hy(
        config_.n_vlpo, config_.n_orexin, config_.p_vlpo_to_orexin, config_.w_vlpo_orexin,
        GABA_A_PARAMS, CompartmentType::BASAL, seed++);
    syn_orexin_to_vlpo_ = build_syn_hy(
        config_.n_orexin, config_.n_vlpo, config_.p_orexin_to_vlpo, config_.w_orexin_vlpo,
        GABA_A_PARAMS, CompartmentType::BASAL, seed++);
    syn_scn_to_vlpo_ = build_syn_hy(
        config_.n_scn, config_.n_vlpo, config_.p_scn_to_vlpo, config_.w_scn_vlpo,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);
    syn_lh_to_vmh_ = build_syn_hy(
        config_.n_lh, config_.n_vmh, config_.p_lh_to_vmh, config_.w_lh_vmh,
        GABA_A_PARAMS, CompartmentType::BASAL, seed++);
    syn_vmh_to_lh_ = build_syn_hy(
        config_.n_vmh, config_.n_lh, config_.p_vmh_to_lh, config_.w_vmh_lh,
        GABA_A_PARAMS, CompartmentType::BASAL, seed++);
}

// === Step ===

void Hypothalamus::step(int32_t t, float dt) {
    oscillation_.step(dt);
    neuromod_.step(dt);

    // =========================================================
    // 1. SCN circadian pacemaker
    // =========================================================
    circadian_phase_ += dt / config_.circadian_period;
    if (circadian_phase_ >= 1.0f) circadian_phase_ -= 1.0f;

    // SCN output: cosine oscillation (peak at phase=0.5 = "noon")
    // Drive is high during "day" (phase ~0.3-0.7), low at "night"
    float scn_drive = config_.scn_drive_amp *
        (0.6f + 0.4f * std::cos(2.0f * 3.14159f * (circadian_phase_ - 0.5f)));
    for (size_t i = 0; i < scn_.size(); ++i) {
        scn_.inject_basal(i, scn_drive);
    }

    // =========================================================
    // 2. VLPO sleep drive
    // =========================================================
    // VLPO receives: SCN circadian gate + homeostatic pressure
    float sleep_drive = config_.homeostatic_sleep_pressure * 30.0f;
    for (size_t i = 0; i < vlpo_.size(); ++i) {
        vlpo_.inject_basal(i, sleep_drive);
        if (psp_vlpo_[i] > 0.5f) {
            vlpo_.inject_basal(i, psp_vlpo_[i]);
        }
        psp_vlpo_[i] *= PSP_DECAY;
    }

    // SCN → VLPO synapse (circadian gate: SCN inhibits VLPO during day)
    // Actually SCN's effect is complex - simplified: SCN excites VLPO at night
    syn_scn_to_vlpo_.deliver_spikes(scn_.fired(), scn_.spike_type());
    const auto& scn_cur = syn_scn_to_vlpo_.step_and_compute(vlpo_.v_soma(), dt);
    for (size_t i = 0; i < vlpo_.size(); ++i) {
        if (std::abs(scn_cur[i]) > 0.01f)
            vlpo_.inject_basal(i, scn_cur[i]);
    }

    // =========================================================
    // 3. Orexin wake drive
    // =========================================================
    // Orexin fires during wakefulness, suppressed by VLPO during sleep
    float wake_drive = (1.0f - config_.homeostatic_sleep_pressure) * 30.0f;
    for (size_t i = 0; i < orexin_.size(); ++i) {
        orexin_.inject_basal(i, wake_drive);
        if (psp_orexin_[i] > 0.5f) {
            orexin_.inject_basal(i, psp_orexin_[i]);
        }
        psp_orexin_[i] *= PSP_DECAY;
    }

    // =========================================================
    // 4. Flip-flop mutual inhibition
    // =========================================================
    // VLPO → Orexin (sleep inhibits wake)
    syn_vlpo_to_orexin_.deliver_spikes(vlpo_.fired(), vlpo_.spike_type());
    const auto& vlpo_cur = syn_vlpo_to_orexin_.step_and_compute(orexin_.v_soma(), dt);
    for (size_t i = 0; i < orexin_.size(); ++i) {
        if (std::abs(vlpo_cur[i]) > 0.01f)
            orexin_.inject_basal(i, vlpo_cur[i]);
    }

    // Orexin → VLPO (wake inhibits sleep)
    syn_orexin_to_vlpo_.deliver_spikes(orexin_.fired(), orexin_.spike_type());
    const auto& orx_cur = syn_orexin_to_vlpo_.step_and_compute(vlpo_.v_soma(), dt);
    for (size_t i = 0; i < vlpo_.size(); ++i) {
        if (std::abs(orx_cur[i]) > 0.01f)
            vlpo_.inject_basal(i, orx_cur[i]);
    }

    // =========================================================
    // 5. PVN stress response
    // =========================================================
    float stress_drive = config_.stress_level * 25.0f;
    for (size_t i = 0; i < pvn_.size(); ++i) {
        pvn_.inject_basal(i, stress_drive);
        if (psp_pvn_[i] > 0.5f) {
            pvn_.inject_basal(i, psp_pvn_[i]);
        }
        psp_pvn_[i] *= PSP_DECAY;
    }

    // =========================================================
    // 6. LH ⟷ VMH feeding balance
    // =========================================================
    float hunger_drive  = config_.hunger_level * 30.0f;
    float satiety_drive = config_.satiety_level * 30.0f;
    for (size_t i = 0; i < lh_.size(); ++i)
        lh_.inject_basal(i, hunger_drive);
    for (size_t i = 0; i < vmh_.size(); ++i)
        vmh_.inject_basal(i, satiety_drive);

    // LH → VMH (hunger inhibits satiety)
    syn_lh_to_vmh_.deliver_spikes(lh_.fired(), lh_.spike_type());
    const auto& lh_cur = syn_lh_to_vmh_.step_and_compute(vmh_.v_soma(), dt);
    for (size_t i = 0; i < vmh_.size(); ++i) {
        if (std::abs(lh_cur[i]) > 0.01f)
            vmh_.inject_basal(i, lh_cur[i]);
    }

    // VMH → LH (satiety inhibits hunger)
    syn_vmh_to_lh_.deliver_spikes(vmh_.fired(), vmh_.spike_type());
    const auto& vmh_cur = syn_vmh_to_lh_.step_and_compute(lh_.v_soma(), dt);
    for (size_t i = 0; i < lh_.size(); ++i) {
        if (std::abs(vmh_cur[i]) > 0.01f)
            lh_.inject_basal(i, vmh_cur[i]);
    }

    // =========================================================
    // Step all populations
    // =========================================================
    scn_.step(t, dt);
    vlpo_.step(t, dt);
    orexin_.step(t, dt);
    pvn_.step(t, dt);
    lh_.step(t, dt);
    vmh_.step(t, dt);

    // =========================================================
    // Compute output state variables
    // =========================================================
    auto fire_frac = [](const NeuronPopulation& pop) -> float {
        size_t n = 0;
        for (auto f : pop.fired()) if (f) n++;
        return static_cast<float>(n) / static_cast<float>(pop.size() + 1);
    };

    float vlpo_activity   = fire_frac(vlpo_);
    float orexin_activity = fire_frac(orexin_);

    // Wake level: orexin vs VLPO balance (exponential smoothing)
    // Orexin activity pushes wake up, VLPO pushes it down
    float delta = (orexin_activity - vlpo_activity) * 0.5f;
    wake_level_ += 0.1f * (delta + 0.0f);  // No bias; pure competition
    // Also driven by base drive levels as a floor/ceiling
    float drive_target = (1.0f - config_.homeostatic_sleep_pressure);
    wake_level_ += 0.02f * (drive_target - wake_level_);
    wake_level_ = std::clamp(wake_level_, 0.0f, 1.0f);

    // Stress output
    stress_output_ = config_.stress_level + fire_frac(pvn_) * 0.5f;
    stress_output_ = std::clamp(stress_output_, 0.0f, 1.0f);

    // Hunger/satiety output
    hunger_output_  = config_.hunger_level + fire_frac(lh_) * 0.3f;
    satiety_output_ = config_.satiety_level + fire_frac(vmh_) * 0.3f;
    hunger_output_  = std::clamp(hunger_output_, 0.0f, 1.0f);
    satiety_output_ = std::clamp(satiety_output_, 0.0f, 1.0f);

    aggregate_state();
}

// === Spike I/O ===

void Hypothalamus::receive_spikes(const std::vector<SpikeEvent>& events) {
    // Route incoming spikes to VLPO/Orexin/PVN based on modular target
    for (const auto& evt : events) {
        float current = 20.0f;
        // Simple routing: distribute across sub-populations
        size_t total = psp_vlpo_.size() + psp_orexin_.size() + psp_pvn_.size();
        size_t idx = evt.neuron_id % total;
        if (idx < psp_vlpo_.size()) {
            psp_vlpo_[idx] += current;
        } else if (idx < psp_vlpo_.size() + psp_orexin_.size()) {
            psp_orexin_[idx - psp_vlpo_.size()] += current;
        } else {
            psp_pvn_[idx - psp_vlpo_.size() - psp_orexin_.size()] += current;
        }
    }
}

void Hypothalamus::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_all_, spike_type_all_, t);
}

void Hypothalamus::inject_external(const std::vector<float>& currents) {
    // External input → Orexin (arousal pathway)
    for (size_t i = 0; i < std::min(currents.size(), orexin_.size()); ++i) {
        orexin_.inject_basal(i, currents[i]);
    }
}

// === Aggregate ===

void Hypothalamus::aggregate_state() {
    size_t offset = 0;
    auto copy_pop = [&](const NeuronPopulation& pop) {
        const auto& f = pop.fired();
        const auto& s = pop.spike_type();
        for (size_t i = 0; i < pop.size(); ++i) {
            fired_all_[offset + i]      = f[i];
            spike_type_all_[offset + i] = s[i];
        }
        offset += pop.size();
    };
    copy_pop(scn_);
    copy_pop(vlpo_);
    copy_pop(orexin_);
    copy_pop(pvn_);
    copy_pop(lh_);
    copy_pop(vmh_);
}

} // namespace wuyun
