#include "region/hippocampus.h"
#include <random>
#include <algorithm>
#include <cmath>

namespace wuyun {

// =============================================================================
// Helper: build sparse random connections
// =============================================================================
static void build_sparse(
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

static SynapseGroup make_empty(size_t n_pre, size_t n_post,
                                const SynapseParams& params,
                                CompartmentType target) {
    return SynapseGroup(n_pre, n_post, {}, {}, {}, {}, params, target);
}

static SynapseGroup build_synapse_group(
    size_t n_pre, size_t n_post, float prob, float weight,
    const SynapseParams& params, CompartmentType target, unsigned seed)
{
    std::vector<int32_t> pre, post, d;
    std::vector<float> w;
    build_sparse(n_pre, n_post, prob, weight, pre, post, w, d, seed);
    if (pre.empty()) return make_empty(n_pre, n_post, params, target);
    return SynapseGroup(n_pre, n_post, pre, post, w, d, params, target);
}

// =============================================================================
// Constructor
// =============================================================================

Hippocampus::Hippocampus(const HippocampusConfig& config)
    : BrainRegion(config.name,
                  config.n_ec + config.n_dg + config.n_ca3 +
                  config.n_ca1 + config.n_sub +
                  config.n_dg_inh + config.n_ca3_inh + config.n_ca1_inh)
    , config_(config)
    // Excitatory populations
    , ec_(config.n_ec, GRID_CELL_PARAMS())
    , dg_(config.n_dg, GRANULE_CELL_PARAMS())
    , ca3_(config.n_ca3, PLACE_CELL_PARAMS())
    , ca1_(config.n_ca1, PLACE_CELL_PARAMS())
    , sub_(config.n_sub, NeuronParams{})  // Standard pyramidal defaults
    // Inhibitory populations (PV basket)
    , dg_inh_(config.n_dg_inh, PV_BASKET_PARAMS())
    , ca3_inh_(config.n_ca3_inh, PV_BASKET_PARAMS())
    , ca1_inh_(config.n_ca1_inh, PV_BASKET_PARAMS())
    // Trisynaptic path (initialized as empty, filled in build_synapses)
    , syn_ec_to_dg_(make_empty(config.n_ec, config.n_dg, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_dg_to_ca3_(make_empty(config.n_dg, config.n_ca3, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_ca3_to_ca3_(make_empty(config.n_ca3, config.n_ca3, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_ca3_to_ca1_(make_empty(config.n_ca3, config.n_ca1, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_ca1_to_sub_(make_empty(config.n_ca1, config.n_sub, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_sub_to_ec_(make_empty(config.n_sub, config.n_ec, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_ec_to_ca1_(make_empty(config.n_ec, config.n_ca1, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_ca3_to_dg_fb_(make_empty(config.n_ca3, config.n_dg, AMPA_PARAMS, CompartmentType::BASAL))
    // Inhibitory synapses
    , syn_ec_to_dg_inh_(make_empty(config.n_ec, config.n_dg_inh, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_dg_to_dg_inh_(make_empty(config.n_dg, config.n_dg_inh, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_dg_inh_to_dg_(make_empty(config.n_dg_inh, config.n_dg, GABA_A_PARAMS, CompartmentType::BASAL))
    , syn_ca3_to_ca3_inh_(make_empty(config.n_ca3, config.n_ca3_inh, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_ca3_inh_to_ca3_(make_empty(config.n_ca3_inh, config.n_ca3, GABA_A_PARAMS, CompartmentType::BASAL))
    , syn_ca1_to_ca1_inh_(make_empty(config.n_ca1, config.n_ca1_inh, AMPA_PARAMS, CompartmentType::BASAL))
    , syn_ca1_inh_to_ca1_(make_empty(config.n_ca1_inh, config.n_ca1, GABA_A_PARAMS, CompartmentType::BASAL))
    // PSP buffer
    , psp_ec_(config.n_ec, 0.0f)
    // Aggregate state
    , fired_all_(n_neurons_, 0)
    , spike_type_all_(n_neurons_, 0)
{
    build_synapses();
}

// =============================================================================
// Build all synapses
// =============================================================================

void Hippocampus::build_synapses() {
    unsigned seed = 1000;

    // --- Trisynaptic path ---
    // EC → DG (perforant path)
    syn_ec_to_dg_ = build_synapse_group(
        config_.n_ec, config_.n_dg, config_.p_ec_to_dg, config_.w_ec_dg,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // DG → CA3 (mossy fiber: sparse but VERY strong)
    syn_dg_to_ca3_ = build_synapse_group(
        config_.n_dg, config_.n_ca3, config_.p_dg_to_ca3, config_.w_dg_ca3,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // CA3 → CA3 (recurrent autoassociative, ~1-2%)
    // This is THE key memory substrate: pattern completion happens here
    syn_ca3_to_ca3_ = build_synapse_group(
        config_.n_ca3, config_.n_ca3, config_.p_ca3_to_ca3, config_.w_ca3_ca3,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // CA3 → CA1 (Schaffer collateral)
    syn_ca3_to_ca1_ = build_synapse_group(
        config_.n_ca3, config_.n_ca1, config_.p_ca3_to_ca1, config_.w_ca3_ca1,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // CA1 → Subiculum
    syn_ca1_to_sub_ = build_synapse_group(
        config_.n_ca1, config_.n_sub, config_.p_ca1_to_sub, config_.w_ca1_sub,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // Subiculum → EC (output loop back)
    syn_sub_to_ec_ = build_synapse_group(
        config_.n_sub, config_.n_ec, config_.p_sub_to_ec, config_.w_sub_ec,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // --- Direct path (bypasses DG/CA3) ---
    // EC L3 → CA1
    syn_ec_to_ca1_ = build_synapse_group(
        config_.n_ec, config_.n_ca1, config_.p_ec_to_ca1, config_.w_ec_ca1,
        AMPA_PARAMS, CompartmentType::APICAL, seed++);  // → apical (feedback-like)

    // --- Feedback ---
    // CA3 → DG backprojection
    syn_ca3_to_dg_fb_ = build_synapse_group(
        config_.n_ca3, config_.n_dg, config_.p_ca3_to_dg, config_.w_ca3_dg_fb,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);

    // --- Inhibitory circuits (E/I balance per subregion) ---
    // DG: feedforward + feedback inhibition (critical for ~2% sparsity)
    syn_ec_to_dg_inh_ = build_synapse_group(
        config_.n_ec, config_.n_dg_inh, config_.p_ec_to_dg_inh, config_.w_exc_to_inh,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);
    syn_dg_to_dg_inh_ = build_synapse_group(
        config_.n_dg, config_.n_dg_inh, config_.p_dg_to_dg_inh, config_.w_exc_to_inh,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);
    syn_dg_inh_to_dg_ = build_synapse_group(
        config_.n_dg_inh, config_.n_dg, config_.p_dg_inh_to_dg, config_.w_inh * 3.0f,
        GABA_A_PARAMS, CompartmentType::BASAL, seed++);

    // CA3
    syn_ca3_to_ca3_inh_ = build_synapse_group(
        config_.n_ca3, config_.n_ca3_inh, config_.p_ca3_to_ca3_inh, config_.w_exc_to_inh,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);
    syn_ca3_inh_to_ca3_ = build_synapse_group(
        config_.n_ca3_inh, config_.n_ca3, config_.p_ca3_inh_to_ca3, config_.w_inh,
        GABA_A_PARAMS, CompartmentType::BASAL, seed++);

    // CA1
    syn_ca1_to_ca1_inh_ = build_synapse_group(
        config_.n_ca1, config_.n_ca1_inh, config_.p_ca1_to_ca1_inh, config_.w_exc_to_inh,
        AMPA_PARAMS, CompartmentType::BASAL, seed++);
    syn_ca1_inh_to_ca1_ = build_synapse_group(
        config_.n_ca1_inh, config_.n_ca1, config_.p_ca1_inh_to_ca1, config_.w_inh,
        GABA_A_PARAMS, CompartmentType::BASAL, seed++);

    // --- Enable CA3 fast STDP (one-shot memory encoding) ---
    if (config_.ca3_stdp_enabled) {
        STDPParams ca3_stdp;
        ca3_stdp.a_plus   = config_.ca3_stdp_a_plus;
        ca3_stdp.a_minus  = config_.ca3_stdp_a_minus;
        ca3_stdp.tau_plus  = config_.ca3_stdp_tau;
        ca3_stdp.tau_minus = config_.ca3_stdp_tau;
        ca3_stdp.w_min     = 0.0f;
        ca3_stdp.w_max     = config_.ca3_stdp_w_max;
        syn_ca3_to_ca3_.enable_stdp(ca3_stdp);
    }
}

// =============================================================================
// Step
// =============================================================================

void Hippocampus::step(int32_t t, float dt) {
    oscillation_.step(dt);
    neuromod_.step(dt);

    // Inject PSP buffer into EC (cross-region input with temporal decay)
    for (size_t i = 0; i < psp_ec_.size(); ++i) {
        if (psp_ec_[i] > 0.5f) ec_.inject_basal(i, psp_ec_[i]);
        psp_ec_[i] *= PSP_DECAY;
    }

    // ========================================
    // Forward pass: EC → DG → CA3 → CA1 → Sub
    // ========================================

    // 1. EC → DG (perforant path)
    syn_ec_to_dg_.deliver_spikes(ec_.fired(), ec_.spike_type());
    auto i_dg_ec = syn_ec_to_dg_.step_and_compute(dg_.v_soma(), dt);
    for (size_t i = 0; i < dg_.size(); ++i) dg_.inject_basal(i, i_dg_ec[i]);

    // 2a. EC → DG_inh (feedforward inhibition, same timing as EC→DG)
    syn_ec_to_dg_inh_.deliver_spikes(ec_.fired(), ec_.spike_type());
    auto i_dg_inh_ff = syn_ec_to_dg_inh_.step_and_compute(dg_inh_.v_soma(), dt);
    for (size_t i = 0; i < dg_inh_.size(); ++i) dg_inh_.inject_basal(i, i_dg_inh_ff[i]);

    // 2b. DG → DG_inh (feedback inhibition)
    syn_dg_to_dg_inh_.deliver_spikes(dg_.fired(), dg_.spike_type());
    auto i_dg_inh = syn_dg_to_dg_inh_.step_and_compute(dg_inh_.v_soma(), dt);
    for (size_t i = 0; i < dg_inh_.size(); ++i) dg_inh_.inject_basal(i, i_dg_inh[i]);

    syn_dg_inh_to_dg_.deliver_spikes(dg_inh_.fired(), dg_inh_.spike_type());
    auto i_dg_fb = syn_dg_inh_to_dg_.step_and_compute(dg_.v_soma(), dt);
    for (size_t i = 0; i < dg_.size(); ++i) dg_.inject_basal(i, i_dg_fb[i]);

    // 3. DG → CA3 (mossy fiber, sparse but strong)
    syn_dg_to_ca3_.deliver_spikes(dg_.fired(), dg_.spike_type());
    auto i_ca3_dg = syn_dg_to_ca3_.step_and_compute(ca3_.v_soma(), dt);
    for (size_t i = 0; i < ca3_.size(); ++i) ca3_.inject_basal(i, i_ca3_dg[i]);

    // 4. CA3 → CA3 recurrent (autoassociative memory recall)
    syn_ca3_to_ca3_.deliver_spikes(ca3_.fired(), ca3_.spike_type());
    auto i_ca3_rec = syn_ca3_to_ca3_.step_and_compute(ca3_.v_soma(), dt);
    for (size_t i = 0; i < ca3_.size(); ++i) ca3_.inject_basal(i, i_ca3_rec[i]);

    // 5. CA3 feedback inhibition
    syn_ca3_to_ca3_inh_.deliver_spikes(ca3_.fired(), ca3_.spike_type());
    auto i_ca3_inh = syn_ca3_to_ca3_inh_.step_and_compute(ca3_inh_.v_soma(), dt);
    for (size_t i = 0; i < ca3_inh_.size(); ++i) ca3_inh_.inject_basal(i, i_ca3_inh[i]);

    syn_ca3_inh_to_ca3_.deliver_spikes(ca3_inh_.fired(), ca3_inh_.spike_type());
    auto i_ca3_inh_fb = syn_ca3_inh_to_ca3_.step_and_compute(ca3_.v_soma(), dt);
    for (size_t i = 0; i < ca3_.size(); ++i) ca3_.inject_basal(i, i_ca3_inh_fb[i]);

    // 6. CA3 → CA1 (Schaffer collateral)
    syn_ca3_to_ca1_.deliver_spikes(ca3_.fired(), ca3_.spike_type());
    auto i_ca1_ca3 = syn_ca3_to_ca1_.step_and_compute(ca1_.v_soma(), dt);
    for (size_t i = 0; i < ca1_.size(); ++i) ca1_.inject_basal(i, i_ca1_ca3[i]);

    // 7. EC → CA1 direct path (to apical dendrite)
    syn_ec_to_ca1_.deliver_spikes(ec_.fired(), ec_.spike_type());
    auto i_ca1_ec = syn_ec_to_ca1_.step_and_compute(ca1_.v_soma(), dt);
    for (size_t i = 0; i < ca1_.size(); ++i) {
        if (ca1_.has_apical()) {
            ca1_.inject_apical(i, i_ca1_ec[i]);
        } else {
            ca1_.inject_basal(i, i_ca1_ec[i]);
        }
    }

    // 8. CA1 feedback inhibition
    syn_ca1_to_ca1_inh_.deliver_spikes(ca1_.fired(), ca1_.spike_type());
    auto i_ca1_inh = syn_ca1_to_ca1_inh_.step_and_compute(ca1_inh_.v_soma(), dt);
    for (size_t i = 0; i < ca1_inh_.size(); ++i) ca1_inh_.inject_basal(i, i_ca1_inh[i]);

    syn_ca1_inh_to_ca1_.deliver_spikes(ca1_inh_.fired(), ca1_inh_.spike_type());
    auto i_ca1_inh_fb = syn_ca1_inh_to_ca1_.step_and_compute(ca1_.v_soma(), dt);
    for (size_t i = 0; i < ca1_.size(); ++i) ca1_.inject_basal(i, i_ca1_inh_fb[i]);

    // 9. CA1 → Subiculum
    syn_ca1_to_sub_.deliver_spikes(ca1_.fired(), ca1_.spike_type());
    auto i_sub_ca1 = syn_ca1_to_sub_.step_and_compute(sub_.v_soma(), dt);
    for (size_t i = 0; i < sub_.size(); ++i) sub_.inject_basal(i, i_sub_ca1[i]);

    // 10. Subiculum → EC (output loop)
    syn_sub_to_ec_.deliver_spikes(sub_.fired(), sub_.spike_type());
    auto i_ec_sub = syn_sub_to_ec_.step_and_compute(ec_.v_soma(), dt);
    for (size_t i = 0; i < ec_.size(); ++i) ec_.inject_basal(i, i_ec_sub[i]);

    // 11. CA3 → DG feedback
    syn_ca3_to_dg_fb_.deliver_spikes(ca3_.fired(), ca3_.spike_type());
    auto i_dg_ca3 = syn_ca3_to_dg_fb_.step_and_compute(dg_.v_soma(), dt);
    for (size_t i = 0; i < dg_.size(); ++i) dg_.inject_basal(i, i_dg_ca3[i]);

    // ========================================
    // Step all populations
    // ========================================
    ec_.step(t, dt);
    dg_.step(t, dt);
    dg_inh_.step(t, dt);
    ca3_.step(t, dt);
    ca3_inh_.step(t, dt);
    ca1_.step(t, dt);
    ca1_inh_.step(t, dt);
    sub_.step(t, dt);

    // ========================================
    // Online plasticity (after all neurons stepped)
    // ========================================
    // CA3 recurrent STDP: co-active CA3 neurons strengthen mutual connections
    if (config_.ca3_stdp_enabled) {
        syn_ca3_to_ca3_.apply_stdp(ca3_.fired(), ca3_.fired(), t);
    }

    aggregate_state();
}

// =============================================================================
// SpikeBus interface
// =============================================================================

void Hippocampus::receive_spikes(const std::vector<SpikeEvent>& events) {
    // Arriving spikes → EC (input gate of hippocampus)
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 30.0f : 20.0f;
        size_t base = evt.neuron_id % psp_ec_.size();
        for (size_t k = 0; k < 3 && (base + k) < psp_ec_.size(); ++k) {
            psp_ec_[base + k] += current;
        }
    }
}

void Hippocampus::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_all_, spike_type_all_, t);
}

void Hippocampus::inject_external(const std::vector<float>& currents) {
    inject_cortical_input(currents);
}

void Hippocampus::inject_cortical_input(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), ec_.size()); ++i) {
        ec_.inject_basal(i, currents[i]);
    }
}

float Hippocampus::dg_sparsity() const {
    size_t active = 0;
    for (size_t i = 0; i < dg_.size(); ++i) {
        if (dg_.fired()[i]) active++;
    }
    return static_cast<float>(active) / static_cast<float>(dg_.size());
}

// =============================================================================
// Aggregate firing state
// =============================================================================

void Hippocampus::aggregate_state() {
    size_t offset = 0;
    auto copy_pop = [&](const NeuronPopulation& pop) {
        for (size_t i = 0; i < pop.size(); ++i) {
            fired_all_[offset + i]      = pop.fired()[i];
            spike_type_all_[offset + i] = pop.spike_type()[i];
        }
        offset += pop.size();
    };

    copy_pop(ec_);
    copy_pop(dg_);
    copy_pop(ca3_);
    copy_pop(ca1_);
    copy_pop(sub_);
    copy_pop(dg_inh_);
    copy_pop(ca3_inh_);
    copy_pop(ca1_inh_);
}

} // namespace wuyun
