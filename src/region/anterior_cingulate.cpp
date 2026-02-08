#include "region/anterior_cingulate.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace wuyun {

// =============================================================================
// Helper: build sparse random connections (same pattern as hippocampus.cpp)
// =============================================================================

static void acc_build_sparse(
    size_t n_pre, size_t n_post, float prob, float weight,
    std::vector<int32_t>& pre, std::vector<int32_t>& post,
    std::vector<float>& w, std::vector<int32_t>& d, unsigned seed = 42)
{
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
}

static SynapseGroup acc_make_empty(size_t n_pre, size_t n_post,
                                    const SynapseParams& params,
                                    CompartmentType target) {
    return SynapseGroup(n_pre, n_post, {}, {}, {}, {}, params, target);
}

static SynapseGroup acc_build_synapse_group(
    size_t n_pre, size_t n_post, float prob, float weight,
    const SynapseParams& params, CompartmentType target, unsigned seed)
{
    std::vector<int32_t> pre, post, d;
    std::vector<float> w;
    acc_build_sparse(n_pre, n_post, prob, weight, pre, post, w, d, seed);
    if (pre.empty()) return acc_make_empty(n_pre, n_post, params, target);
    return SynapseGroup(n_pre, n_post, pre, post, w, d, params, target);
}

// =============================================================================
// Construction
// =============================================================================

AnteriorCingulate::AnteriorCingulate(const ACCConfig& config)
    : BrainRegion(config.name, config.n_dacc + config.n_vacc + config.n_inh)
    , config_(config)
    , dacc_(config.n_dacc, L23_PYRAMIDAL_PARAMS())
    , vacc_(config.n_vacc, L23_PYRAMIDAL_PARAMS())
    , inh_(config.n_inh, PV_BASKET_PARAMS())
    // Synapses (initialized empty, filled in build_synapses)
    , syn_dacc_to_vacc_(acc_make_empty(config.n_dacc, config.n_vacc, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_vacc_to_dacc_(acc_make_empty(config.n_vacc, config.n_dacc, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_dacc_to_inh_(acc_make_empty(config.n_dacc, config.n_inh, AMPA_PARAMS, CompartmentType::SOMA))
    , syn_vacc_to_inh_(acc_make_empty(config.n_vacc, config.n_inh, AMPA_PARAMS, CompartmentType::SOMA))
    , syn_inh_to_dacc_(acc_make_empty(config.n_inh, config.n_dacc, GABA_A_PARAMS, CompartmentType::SOMA))
    , syn_inh_to_vacc_(acc_make_empty(config.n_inh, config.n_vacc, GABA_A_PARAMS, CompartmentType::SOMA))
    , psp_dacc_(config.n_dacc, 0.0f)
    , psp_vacc_(config.n_vacc, 0.0f)
    , fired_all_(config.n_dacc + config.n_vacc + config.n_inh, 0)
    , spike_type_all_(config.n_dacc + config.n_vacc + config.n_inh, 0)
{
    build_synapses();
}

void AnteriorCingulate::build_synapses() {
    unsigned seed = 5000;

    // dACC ↔ vACC (excitatory cross-talk)
    syn_dacc_to_vacc_ = acc_build_synapse_group(
        config_.n_dacc, config_.n_vacc,
        config_.p_dacc_to_vacc, config_.w_exc,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    syn_vacc_to_dacc_ = acc_build_synapse_group(
        config_.n_vacc, config_.n_dacc,
        config_.p_vacc_to_dacc, config_.w_exc,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // Exc → Inh (feedback inhibition)
    syn_dacc_to_inh_ = acc_build_synapse_group(
        config_.n_dacc, config_.n_inh,
        config_.p_exc_to_inh, config_.w_exc,
        AMPA_PARAMS, CompartmentType::SOMA, seed++);

    syn_vacc_to_inh_ = acc_build_synapse_group(
        config_.n_vacc, config_.n_inh,
        config_.p_exc_to_inh, config_.w_exc,
        AMPA_PARAMS, CompartmentType::SOMA, seed++);

    // Inh → Exc (E/I balance)
    syn_inh_to_dacc_ = acc_build_synapse_group(
        config_.n_inh, config_.n_dacc,
        config_.p_inh_to_exc, config_.w_inh,
        GABA_A_PARAMS, CompartmentType::SOMA, seed++);

    syn_inh_to_vacc_ = acc_build_synapse_group(
        config_.n_inh, config_.n_vacc,
        config_.p_inh_to_exc, config_.w_inh,
        GABA_A_PARAMS, CompartmentType::SOMA, seed++);
}

// =============================================================================
// Main step
// =============================================================================

void AnteriorCingulate::step(int32_t t, float dt) {
    // 1. Apply cross-region PSP input
    for (size_t i = 0; i < config_.n_dacc; ++i) {
        if (psp_dacc_[i] > 0.1f) dacc_.inject_basal(i, psp_dacc_[i]);
        psp_dacc_[i] *= PSP_DECAY;
    }
    for (size_t i = 0; i < config_.n_vacc; ++i) {
        if (psp_vacc_[i] > 0.1f) vacc_.inject_basal(i, psp_vacc_[i]);
        psp_vacc_[i] *= PSP_DECAY;
    }

    // 2. Inject conflict signal into dACC neurons
    //    Biology: dACC neurons receive convergent input from multiple
    //    response channels; when conflict is high, they fire more
    if (conflict_level_ > 0.05f) {
        float conflict_drive = conflict_level_ * config_.conflict_gain * 15.0f;
        // Distribute across dACC — proportional to conflict level
        for (size_t i = 0; i < config_.n_dacc; ++i) {
            dacc_.inject_basal(i, conflict_drive);
        }
    }

    // 3. Inject surprise signal into vACC neurons
    //    Biology: unexpected outcomes (PRO model) → vACC activation
    //    vACC is more sensitive to emotional/motivational significance
    if (surprise_level_ > 0.05f) {
        float surprise_drive = surprise_level_ * config_.surprise_gain * 12.0f;
        for (size_t i = 0; i < config_.n_vacc; ++i) {
            vacc_.inject_basal(i, surprise_drive);
        }
    }

    // 4. Inject threat signal into vACC
    //    Biology: CeA → vACC, urgent emotional arousal
    if (threat_input_ > 0.01f) {
        float threat_drive = threat_input_ * 15.0f;
        for (size_t i = 0; i < config_.n_vacc; ++i) {
            vacc_.inject_basal(i, threat_drive);
        }
        threat_input_ *= THREAT_DECAY;
    }

    // 5. Tonic drive (ACC has spontaneous activity ~3-5 Hz)
    for (size_t i = 0; i < config_.n_dacc; ++i) {
        dacc_.inject_basal(i, 6.0f);
    }
    for (size_t i = 0; i < config_.n_vacc; ++i) {
        vacc_.inject_basal(i, 5.0f);
    }

    // 6. Propagate internal synapses (deliver_spikes + step_and_compute pattern)

    // dACC → vACC
    syn_dacc_to_vacc_.deliver_spikes(dacc_.fired(), dacc_.spike_type());
    const auto& i_vacc_from_dacc = syn_dacc_to_vacc_.step_and_compute(vacc_.v_soma(), dt);
    for (size_t i = 0; i < vacc_.size(); ++i) vacc_.inject_basal(i, i_vacc_from_dacc[i]);

    // vACC → dACC
    syn_vacc_to_dacc_.deliver_spikes(vacc_.fired(), vacc_.spike_type());
    const auto& i_dacc_from_vacc = syn_vacc_to_dacc_.step_and_compute(dacc_.v_soma(), dt);
    for (size_t i = 0; i < dacc_.size(); ++i) dacc_.inject_basal(i, i_dacc_from_vacc[i]);

    // dACC → Inh
    syn_dacc_to_inh_.deliver_spikes(dacc_.fired(), dacc_.spike_type());
    const auto& i_inh_from_dacc = syn_dacc_to_inh_.step_and_compute(inh_.v_soma(), dt);
    for (size_t i = 0; i < inh_.size(); ++i) inh_.inject_soma(i, i_inh_from_dacc[i]);

    // vACC → Inh
    syn_vacc_to_inh_.deliver_spikes(vacc_.fired(), vacc_.spike_type());
    const auto& i_inh_from_vacc = syn_vacc_to_inh_.step_and_compute(inh_.v_soma(), dt);
    for (size_t i = 0; i < inh_.size(); ++i) inh_.inject_soma(i, i_inh_from_vacc[i]);

    // Inh → dACC
    syn_inh_to_dacc_.deliver_spikes(inh_.fired(), inh_.spike_type());
    const auto& i_dacc_from_inh = syn_inh_to_dacc_.step_and_compute(dacc_.v_soma(), dt);
    for (size_t i = 0; i < dacc_.size(); ++i) dacc_.inject_soma(i, i_dacc_from_inh[i]);

    // Inh → vACC
    syn_inh_to_vacc_.deliver_spikes(inh_.fired(), inh_.spike_type());
    const auto& i_vacc_from_inh = syn_inh_to_vacc_.step_and_compute(vacc_.v_soma(), dt);
    for (size_t i = 0; i < vacc_.size(); ++i) vacc_.inject_soma(i, i_vacc_from_inh[i]);

    // 7. Step all populations
    dacc_.step(t, dt);
    vacc_.step(t, dt);
    inh_.step(t, dt);

    // 8. Update computational signals
    update_conflict(t);
    update_surprise(t);
    update_volatility(t);
    update_foraging(t);
    compute_outputs(t);

    // 9. Aggregate firing state
    aggregate_state();
}

// =============================================================================
// Conflict monitoring (Botvinick et al. 2001)
// =============================================================================

void AnteriorCingulate::update_conflict(int32_t /*t*/) {
    // Hopfield energy-based conflict: conflict = Σ_{i≠j} rate_i × rate_j
    // Normalized by total activity to get [0,1] range
    // When all 4 groups equal: conflict = 6 × (0.25)² / (1.0)² = 0.375
    // When 1 group dominates: conflict ≈ 0
    float cross_product = 0.0f;
    float total_rate = 0.0f;
    for (int i = 0; i < 4; ++i) {
        total_rate += d1_rates_[i];
        for (int j = i + 1; j < 4; ++j) {
            cross_product += d1_rates_[i] * d1_rates_[j];
        }
    }
    float total_sq = total_rate * total_rate;
    float raw = (total_sq > 0.001f) ? cross_product / total_sq : 0.0f;

    // Exponential smoothing
    conflict_raw_ = conflict_raw_ * config_.conflict_decay + raw * (1.0f - config_.conflict_decay);
    conflict_level_ = std::clamp(conflict_raw_ * config_.conflict_gain, 0.0f, 1.0f);
}

// =============================================================================
// PRO model: prediction & surprise (Alexander & Brown 2011)
// =============================================================================

void AnteriorCingulate::update_surprise(int32_t /*t*/) {
    // Surprise = |actual - predicted| regardless of valence
    // "ACC doesn't care about good or bad, only if it was expected"
    float prediction_error = std::abs(last_outcome_ - predicted_reward_);

    // Update prediction (slow exponential moving average)
    predicted_reward_ = predicted_reward_ * config_.prediction_tau
                      + last_outcome_ * (1.0f - config_.prediction_tau);

    // Smooth surprise signal
    surprise_raw_ = surprise_raw_ * config_.surprise_decay
                  + prediction_error * (1.0f - config_.surprise_decay);
    surprise_level_ = std::clamp(surprise_raw_ * config_.surprise_gain, 0.0f, 1.0f);
}

// =============================================================================
// Volatility detection (Behrens et al. 2007)
// =============================================================================

void AnteriorCingulate::update_volatility(int32_t /*t*/) {
    // Fast and slow reward rate trackers
    // Volatility = |fast - slow| = how much the reward rate is changing
    float outcome_abs = std::abs(last_outcome_);
    reward_rate_fast_ = reward_rate_fast_ * config_.volatility_fast_tau
                      + outcome_abs * (1.0f - config_.volatility_fast_tau);
    reward_rate_slow_ = reward_rate_slow_ * config_.volatility_slow_tau
                      + outcome_abs * (1.0f - config_.volatility_slow_tau);

    volatility_raw_ = std::abs(reward_rate_fast_ - reward_rate_slow_);
    volatility_level_ = std::clamp(volatility_raw_ * config_.volatility_gain, 0.0f, 1.0f);
}

// =============================================================================
// Foraging decision (Kolling et al. 2012, Hayden et al. 2011)
// =============================================================================

void AnteriorCingulate::update_foraging(int32_t /*t*/) {
    // Local reward rate: recent performance (fast tracker)
    // Global reward rate: long-term average (slow tracker)
    // Foraging signal = max(0, global - local)
    // When local < global → current strategy is suboptimal → should switch
    local_reward_rate_ = reward_rate_fast_;
    global_reward_rate_ = global_reward_rate_ * config_.foraging_tau
                        + std::abs(last_outcome_) * (1.0f - config_.foraging_tau);

    float switch_pressure = std::max(0.0f, global_reward_rate_ - local_reward_rate_);
    foraging_signal_ = std::clamp(switch_pressure * 5.0f, 0.0f, 1.0f);
}

// =============================================================================
// Compute output signals
// =============================================================================

void AnteriorCingulate::compute_outputs(int32_t /*t*/) {
    // --- ACC→LC arousal drive ---
    // Weighted combination: conflict + surprise + foraging + threat
    // This replaces hardcoded ne_floor!
    // Biology: dACC→LC projection drives phasic NE bursts for exploration
    arousal_drive_ = std::clamp(
        conflict_level_ * 0.4f +     // Action competition → explore more
        surprise_level_ * 0.3f +     // Unexpected outcome → be more alert
        foraging_signal_ * 0.2f +    // Strategy failing → try something new
        threat_input_ * 0.1f,        // Danger → heightened vigilance
        0.0f, 1.0f
    ) * config_.acc_to_lc_gain;

    // --- ACC→dlPFC attention signal ---
    // High conflict/surprise → increase dlPFC attention gain
    // Biology: ACC→dlPFC projection enhances PFC top-down control
    attention_signal_ = std::clamp(
        conflict_level_ * 0.5f +
        surprise_level_ * 0.3f +
        volatility_level_ * 0.2f,
        0.0f, 1.0f
    ) * config_.acc_to_dlpfc_gain;

    // --- Learning rate modulation (Behrens 2007) ---
    // High volatility → lr×2.0 (fast adaptation)
    // Low volatility → lr×0.5 (stability)
    // Baseline (zero volatility) → lr×1.0
    lr_modulation_ = std::clamp(1.0f + volatility_level_ * 1.0f, 0.5f, 2.0f);
}

// =============================================================================
// Input injection
// =============================================================================

void AnteriorCingulate::inject_d1_rates(const std::array<float, 4>& d1_group_rates) {
    d1_rates_ = d1_group_rates;
}

void AnteriorCingulate::inject_outcome(float reward) {
    last_outcome_ = reward;
}

void AnteriorCingulate::inject_threat(float threat_level) {
    threat_input_ = std::max(threat_input_, threat_level);
}

// =============================================================================
// SpikeBus interface
// =============================================================================

void AnteriorCingulate::receive_spikes(const std::vector<SpikeEvent>& events) {
    // Route incoming spikes to dACC and vACC PSP buffers
    for (const auto& ev : events) {
        float psp = (ev.spike_type == static_cast<int8_t>(SpikeType::REGULAR)) ? 25.0f : 40.0f;
        // Distribute to dACC and vACC based on neuron index modulo
        size_t total_exc = config_.n_dacc + config_.n_vacc;
        size_t target = ev.neuron_id % total_exc;
        if (target < config_.n_dacc) {
            psp_dacc_[target] += psp;
        } else {
            psp_vacc_[target - config_.n_dacc] += psp;
        }
    }
}

void AnteriorCingulate::submit_spikes(SpikeBus& bus, int32_t t) {
    // Use aggregate fired/spike_type arrays (same pattern as Hippocampus)
    bus.submit_spikes(region_id_, fired_all_, spike_type_all_, t);
}

void AnteriorCingulate::inject_external(const std::vector<float>& currents) {
    size_t idx = 0;
    for (size_t i = 0; i < config_.n_dacc && idx < currents.size(); ++i, ++idx) {
        dacc_.inject_basal(i, currents[idx]);
    }
    for (size_t i = 0; i < config_.n_vacc && idx < currents.size(); ++i, ++idx) {
        vacc_.inject_basal(i, currents[idx]);
    }
}

// =============================================================================
// Aggregate firing state
// =============================================================================

void AnteriorCingulate::aggregate_state() {
    size_t idx = 0;
    for (size_t i = 0; i < config_.n_dacc; ++i, ++idx) {
        fired_all_[idx] = dacc_.fired()[i];
        spike_type_all_[idx] = dacc_.spike_type()[i];
    }
    for (size_t i = 0; i < config_.n_vacc; ++i, ++idx) {
        fired_all_[idx] = vacc_.fired()[i];
        spike_type_all_[idx] = vacc_.spike_type()[i];
    }
    for (size_t i = 0; i < config_.n_inh; ++i, ++idx) {
        fired_all_[idx] = inh_.fired()[i];
        spike_type_all_[idx] = inh_.spike_type()[i];
    }
}

} // namespace wuyun
