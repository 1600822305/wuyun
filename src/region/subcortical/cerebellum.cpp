#include "region/subcortical/cerebellum.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace wuyun {

// =============================================================================
// Helper: build sparse random connections (same pattern as hippocampus)
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
// Neuron parameter sets
// =============================================================================

static NeuronParams make_dcn_params() {
    NeuronParams p;
    p.somatic.v_rest = -60.0f;
    p.somatic.v_threshold = -48.0f;
    p.somatic.v_reset = -55.0f;
    p.somatic.tau_m = 15.0f;
    p.somatic.r_s = 1.0f;
    p.somatic.a = 0.01f;
    p.somatic.b = 2.0f;
    p.somatic.tau_w = 200.0f;
    p.somatic.refractory_period = 2;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

static NeuronParams make_mli_params() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f;
    p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -60.0f;
    p.somatic.tau_m = 8.0f;
    p.somatic.r_s = 1.2f;
    p.somatic.a = 0.0f;
    p.somatic.b = 0.0f;
    p.somatic.tau_w = 100.0f;
    p.somatic.refractory_period = 1;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

static NeuronParams make_golgi_params() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f;
    p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -58.0f;
    p.somatic.tau_m = 20.0f;
    p.somatic.r_s = 0.8f;
    p.somatic.a = 0.02f;
    p.somatic.b = 3.0f;
    p.somatic.tau_w = 300.0f;
    p.somatic.refractory_period = 3;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

static NeuronParams make_cerebellar_granule_params() {
    NeuronParams p;
    p.somatic.v_rest = -70.0f;
    p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -65.0f;
    p.somatic.tau_m = 12.0f;
    p.somatic.r_s = 1.0f;
    p.somatic.a = 0.0f;
    p.somatic.b = 0.0f;
    p.somatic.tau_w = 100.0f;
    p.somatic.refractory_period = 2;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

// =============================================================================
// Constructor
// =============================================================================

Cerebellum::Cerebellum(const CerebellumConfig& config)
    : BrainRegion(config.name,
                  config.n_granule + config.n_purkinje + config.n_dcn +
                  config.n_mli + config.n_golgi)
    , config_(config)
    , grc_(config.n_granule, make_cerebellar_granule_params())
    , pc_(config.n_purkinje, PURKINJE_PARAMS())
    , dcn_(config.n_dcn, make_dcn_params())
    , mli_(config.n_mli, make_mli_params())
    , golgi_(config.n_golgi, make_golgi_params())
    // Excitatory synapses
    , syn_mf_to_grc_(build_synapse_group(config.n_granule, config.n_granule,
        config.p_mf_to_grc, config.w_mf_grc, AMPA_PARAMS, CompartmentType::BASAL, 100))
    , syn_pf_to_pc_(build_synapse_group(config.n_granule, config.n_purkinje,
        config.p_pf_to_pc, config.w_pf_pc, AMPA_PARAMS, CompartmentType::BASAL, 101))
    , syn_pf_to_mli_(build_synapse_group(config.n_granule, config.n_mli,
        config.p_pf_to_mli, config.w_pf_mli, AMPA_PARAMS, CompartmentType::BASAL, 102))
    , syn_grc_to_golgi_(build_synapse_group(config.n_granule, config.n_golgi,
        config.p_grc_to_golgi, config.w_grc_golgi, AMPA_PARAMS, CompartmentType::BASAL, 103))
    // Inhibitory synapses
    , syn_mli_to_pc_(build_synapse_group(config.n_mli, config.n_purkinje,
        config.p_mli_to_pc, config.w_mli_pc, GABA_A_PARAMS, CompartmentType::BASAL, 104))
    , syn_pc_to_dcn_(build_synapse_group(config.n_purkinje, config.n_dcn,
        config.p_pc_to_dcn, config.w_pc_dcn, GABA_A_PARAMS, CompartmentType::BASAL, 105))
    , syn_golgi_to_grc_(build_synapse_group(config.n_golgi, config.n_granule,
        config.p_golgi_to_grc, config.w_golgi_grc, GABA_A_PARAMS, CompartmentType::BASAL, 106))
    // PSP buffer + aggregate state
    , psp_grc_(config.n_granule, 0.0f)
    , fired_(config.n_granule + config.n_purkinje + config.n_dcn +
             config.n_mli + config.n_golgi, 0)
    , spike_type_(config.n_granule + config.n_purkinje + config.n_dcn +
                  config.n_mli + config.n_golgi, 0)
{}

// =============================================================================
// Step
// =============================================================================

void Cerebellum::step(int32_t t, float dt) {
    oscillation_.step(dt);

    // 1. Inject PSP buffer into granule cells (from SpikeBus mossy fibers)
    for (size_t i = 0; i < psp_grc_.size(); ++i) {
        if (psp_grc_[i] > 0.5f) {
            grc_.inject_basal(i, psp_grc_[i]);
        }
        psp_grc_[i] *= PSP_DECAY;
    }

    // 2. DCN gets strong tonic excitatory drive
    //    (biology: DCN fires tonically at ~40-50Hz, PC only sculpts timing)
    for (size_t i = 0; i < dcn_.size(); ++i) {
        dcn_.inject_basal(i, 35.0f);
    }

    // 3. Step granule cells
    grc_.step(t, dt);

    // 4. GrC → PC (parallel fibers), GrC → MLI, GrC → Golgi
    syn_pf_to_pc_.deliver_spikes(grc_.fired(), grc_.spike_type());
    const auto& i_pc_pf = syn_pf_to_pc_.step_and_compute(pc_.v_soma(), dt);
    for (size_t i = 0; i < pc_.size(); ++i) pc_.inject_basal(i, i_pc_pf[i]);

    syn_pf_to_mli_.deliver_spikes(grc_.fired(), grc_.spike_type());
    const auto& i_mli_pf = syn_pf_to_mli_.step_and_compute(mli_.v_soma(), dt);
    for (size_t i = 0; i < mli_.size(); ++i) mli_.inject_basal(i, i_mli_pf[i]);

    syn_grc_to_golgi_.deliver_spikes(grc_.fired(), grc_.spike_type());
    const auto& i_golgi_grc = syn_grc_to_golgi_.step_and_compute(golgi_.v_soma(), dt);
    for (size_t i = 0; i < golgi_.size(); ++i) golgi_.inject_basal(i, i_golgi_grc[i]);

    // 5. Climbing fiber: inject error signal directly into PC
    if (cf_error_ > 0.01f) {
        float cf_current = cf_error_ * 60.0f;
        for (size_t i = 0; i < pc_.size(); ++i) {
            pc_.inject_basal(i, cf_current);
        }
    }

    // 6. Step MLI, then MLI → PC (inhibition)
    mli_.step(t, dt);
    syn_mli_to_pc_.deliver_spikes(mli_.fired(), mli_.spike_type());
    const auto& i_pc_mli = syn_mli_to_pc_.step_and_compute(pc_.v_soma(), dt);
    for (size_t i = 0; i < pc_.size(); ++i) pc_.inject_basal(i, i_pc_mli[i]);

    // 7. Step PC
    pc_.step(t, dt);

    // 8. Step Golgi, then Golgi → GrC (feedback inhibition, for next step)
    golgi_.step(t, dt);
    syn_golgi_to_grc_.deliver_spikes(golgi_.fired(), golgi_.spike_type());
    const auto& i_grc_golgi = syn_golgi_to_grc_.step_and_compute(grc_.v_soma(), dt);
    for (size_t i = 0; i < grc_.size(); ++i) grc_.inject_basal(i, i_grc_golgi[i]);

    // 9. PC → DCN (inhibitory output)
    syn_pc_to_dcn_.deliver_spikes(pc_.fired(), pc_.spike_type());
    const auto& i_dcn_pc = syn_pc_to_dcn_.step_and_compute(dcn_.v_soma(), dt);
    for (size_t i = 0; i < dcn_.size(); ++i) dcn_.inject_basal(i, i_dcn_pc[i]);

    // 10. Step DCN
    dcn_.step(t, dt);

    // 11. Apply climbing fiber plasticity (PF→PC LTD/LTP)
    apply_climbing_fiber_plasticity(t);

    // 12. Aggregate and reset
    aggregate_firing_state();
    cf_error_ = 0.0f;
}

void Cerebellum::receive_spikes(const std::vector<SpikeEvent>& events) {
    // Arriving spikes → mossy fiber PSP buffer → granule cells
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 25.0f : 15.0f;
        size_t base = evt.neuron_id % psp_grc_.size();
        size_t fan = std::max<size_t>(3, psp_grc_.size() / 10);
        for (size_t k = 0; k < fan; ++k) {
            size_t idx = (base + k) % psp_grc_.size();
            psp_grc_[idx] += current;
        }
    }
}

void Cerebellum::submit_spikes(SpikeBus& bus, int32_t t) {
    // Submit DCN spikes (cerebellum's output to thalamus)
    // We need to map DCN neuron indices to the global region space
    // DCN starts after grc + pc in the aggregate
    size_t dcn_offset = config_.n_granule + config_.n_purkinje;
    std::vector<uint8_t> dcn_in_region(n_neurons_, 0);
    std::vector<int8_t>  dcn_type_region(n_neurons_, 0);
    for (size_t i = 0; i < dcn_.size(); ++i) {
        dcn_in_region[dcn_offset + i] = dcn_.fired()[i];
        dcn_type_region[dcn_offset + i] = dcn_.spike_type()[i];
    }
    bus.submit_spikes(region_id_, dcn_in_region, dcn_type_region, t);
}

void Cerebellum::inject_external(const std::vector<float>& currents) {
    // External currents go to granule cells (mossy fiber pathway)
    for (size_t i = 0; i < std::min(currents.size(), grc_.size()); ++i) {
        grc_.inject_basal(i, currents[i]);
    }
}

void Cerebellum::inject_climbing_fiber(float error_signal) {
    cf_error_ = std::clamp(error_signal, 0.0f, 1.0f);
}

void Cerebellum::inject_mossy_fiber(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), grc_.size()); ++i) {
        grc_.inject_basal(i, currents[i]);
    }
}

void Cerebellum::aggregate_firing_state() {
    size_t offset = 0;

    // Granule cells
    for (size_t i = 0; i < grc_.size(); ++i) {
        fired_[offset + i] = grc_.fired()[i];
        spike_type_[offset + i] = grc_.spike_type()[i];
    }
    offset += grc_.size();

    // Purkinje cells
    for (size_t i = 0; i < pc_.size(); ++i) {
        fired_[offset + i] = pc_.fired()[i];
        spike_type_[offset + i] = pc_.spike_type()[i];
    }
    offset += pc_.size();

    // DCN
    for (size_t i = 0; i < dcn_.size(); ++i) {
        fired_[offset + i] = dcn_.fired()[i];
        spike_type_[offset + i] = dcn_.spike_type()[i];
    }
    offset += dcn_.size();

    // MLI
    for (size_t i = 0; i < mli_.size(); ++i) {
        fired_[offset + i] = mli_.fired()[i];
        spike_type_[offset + i] = mli_.spike_type()[i];
    }
    offset += mli_.size();

    // Golgi
    for (size_t i = 0; i < golgi_.size(); ++i) {
        fired_[offset + i] = golgi_.fired()[i];
        spike_type_[offset + i] = golgi_.spike_type()[i];
    }
}

void Cerebellum::apply_climbing_fiber_plasticity(int32_t t) {
    // Climbing fiber LTD/LTP on PF→PC synapses
    // CF active + GrC active → LTD (weaken wrong movement)
    // GrC active + no CF → LTP (strengthen correct movement)

    bool cf_active = cf_error_ > 0.1f;

    auto& weights = syn_pf_to_pc_.weights();
    const auto& row_ptr = syn_pf_to_pc_.row_ptr();
    const auto& col_idx = syn_pf_to_pc_.col_idx();

    for (size_t pre = 0; pre < grc_.size(); ++pre) {
        if (!grc_.fired()[pre]) continue;  // Only active PFs

        for (size_t j = row_ptr[pre]; j < row_ptr[pre + 1]; ++j) {
            size_t post = col_idx[j];
            bool pc_active = pc_.fired()[post];

            if (cf_active && pc_active) {
                // CF + PF + PC → LTD (heterosynaptic)
                weights[j] -= config_.cf_ltd_rate;
            } else if (!cf_active) {
                // PF alone (no error) → LTP
                weights[j] += config_.cf_ltp_rate;
            }

            // Clamp weights
            weights[j] = std::clamp(weights[j], config_.pf_pc_w_min, config_.pf_pc_w_max);
        }
    }
}

} // namespace wuyun
