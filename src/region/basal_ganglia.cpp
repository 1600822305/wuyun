#include "region/basal_ganglia.h"
#include <random>
#include <algorithm>
#include <climits>

namespace wuyun {

static void build_sparse_connections(
    size_t n_pre, size_t n_post, float prob, float weight,
    std::vector<int32_t>& pre_ids,
    std::vector<int32_t>& post_ids,
    std::vector<float>& weights,
    std::vector<int32_t>& delays,
    unsigned seed = 42
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n_pre; ++i) {
        for (size_t j = 0; j < n_post; ++j) {
            if (dist(rng) < prob) {
                pre_ids.push_back(static_cast<int32_t>(i));
                post_ids.push_back(static_cast<int32_t>(j));
                weights.push_back(weight);
                delays.push_back(1);
            }
        }
    }
}

static SynapseGroup make_empty(size_t n_pre, size_t n_post,
                                const SynapseParams& p, CompartmentType t) {
    return SynapseGroup(n_pre, n_post, {}, {}, {}, {}, p, t);
}

// GPi/GPe tonic firing params: high spontaneous rate, strong inhibitory output
static NeuronParams GPI_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -55.0f;   // depolarized → tonic firing
    p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -52.0f;
    p.somatic.tau_m = 15.0f;
    p.somatic.r_s = 0.8f;
    p.somatic.a = 0.0f;
    p.somatic.b = 0.5f;
    p.somatic.tau_w = 50.0f;
    p.somatic.refractory_period = 2;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

// STN: excitatory, high firing rate
static NeuronParams STN_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -60.0f;
    p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -55.0f;
    p.somatic.tau_m = 12.0f;
    p.somatic.r_s = 1.0f;
    p.somatic.a = 0.01f;
    p.somatic.b = 2.0f;
    p.somatic.tau_w = 100.0f;
    p.somatic.refractory_period = 2;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 2; p.burst_isi = 2;
    return p;
}

BasalGanglia::BasalGanglia(const BasalGangliaConfig& config)
    : BrainRegion(config.name,
                  config.n_d1_msn + config.n_d2_msn +
                  config.n_gpi + config.n_gpe + config.n_stn)
    , config_(config)
    , d1_msn_(config.n_d1_msn, MSN_D1_PARAMS())
    , d2_msn_(config.n_d2_msn, MSN_D2_PARAMS())
    , gpi_(config.n_gpi, GPI_PARAMS())
    , gpe_(config.n_gpe, GPI_PARAMS())   // GPe uses same params as GPi
    , stn_(config.n_stn, STN_PARAMS())
    , syn_d1_to_gpi_(make_empty(config.n_d1_msn, config.n_gpi, GABA_A_PARAMS, CompartmentType::BASAL))
    , syn_d2_to_gpe_(make_empty(config.n_d2_msn, config.n_gpe, GABA_A_PARAMS, CompartmentType::BASAL))
    , syn_gpe_to_stn_(make_empty(config.n_gpe, config.n_stn, GABA_A_PARAMS, CompartmentType::BASAL))
    , syn_stn_to_gpi_(make_empty(config.n_stn, config.n_gpi, AMPA_PARAMS, CompartmentType::BASAL))
    , psp_d1_(config.n_d1_msn, 0.0f)
    , psp_d2_(config.n_d2_msn, 0.0f)
    , psp_stn_(config.n_stn, 0.0f)
    , fired_all_(n_neurons_, 0)
    , spike_type_all_(n_neurons_, 0)
{
    build_synapses();
    // Build input maps for a reasonable max input neuron count
    build_input_maps(256);
}

void BasalGanglia::build_synapses() {
    // D1 → GPi (inhibitory, direct pathway "Go")
    {
        std::vector<int32_t> pre, post; std::vector<float> w; std::vector<int32_t> d;
        build_sparse_connections(config_.n_d1_msn, config_.n_gpi,
                                  config_.p_d1_to_gpi, config_.w_d1_inh, pre, post, w, d, 300);
        syn_d1_to_gpi_ = SynapseGroup(config_.n_d1_msn, config_.n_gpi,
                                        pre, post, w, d, GABA_A_PARAMS, CompartmentType::BASAL);
    }
    // D2 → GPe (inhibitory, indirect pathway)
    {
        std::vector<int32_t> pre, post; std::vector<float> w; std::vector<int32_t> d;
        build_sparse_connections(config_.n_d2_msn, config_.n_gpe,
                                  config_.p_d2_to_gpe, config_.w_d2_inh, pre, post, w, d, 400);
        syn_d2_to_gpe_ = SynapseGroup(config_.n_d2_msn, config_.n_gpe,
                                        pre, post, w, d, GABA_A_PARAMS, CompartmentType::BASAL);
    }
    // GPe → STN (inhibitory)
    {
        std::vector<int32_t> pre, post; std::vector<float> w; std::vector<int32_t> d;
        build_sparse_connections(config_.n_gpe, config_.n_stn,
                                  config_.p_gpe_to_stn, config_.w_gpe_inh, pre, post, w, d, 500);
        syn_gpe_to_stn_ = SynapseGroup(config_.n_gpe, config_.n_stn,
                                         pre, post, w, d, GABA_A_PARAMS, CompartmentType::BASAL);
    }
    // STN → GPi (excitatory, "brake" signal)
    {
        std::vector<int32_t> pre, post; std::vector<float> w; std::vector<int32_t> d;
        build_sparse_connections(config_.n_stn, config_.n_gpi,
                                  config_.p_stn_to_gpi, config_.w_stn_exc, pre, post, w, d, 600);
        syn_stn_to_gpi_ = SynapseGroup(config_.n_stn, config_.n_gpi,
                                         pre, post, w, d, AMPA_PARAMS, CompartmentType::BASAL);
    }
}

void BasalGanglia::build_input_maps(size_t n_input_neurons) {
    input_map_size_ = n_input_neurons;
    ctx_to_d1_map_.resize(n_input_neurons);
    ctx_to_d2_map_.resize(n_input_neurons);
    ctx_to_stn_map_.resize(n_input_neurons);

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < n_input_neurons; ++i) {
        // Cortex → D1: probability p_ctx_to_d1
        for (size_t j = 0; j < d1_msn_.size(); ++j) {
            if (dist(rng) < config_.p_ctx_to_d1)
                ctx_to_d1_map_[i].push_back(static_cast<uint32_t>(j));
        }
        // Cortex → D2: probability p_ctx_to_d2
        for (size_t j = 0; j < d2_msn_.size(); ++j) {
            if (dist(rng) < config_.p_ctx_to_d2)
                ctx_to_d2_map_[i].push_back(static_cast<uint32_t>(j));
        }
        // Cortex → STN (hyperdirect): probability p_ctx_to_stn
        for (size_t j = 0; j < stn_.size(); ++j) {
            if (dist(rng) < config_.p_ctx_to_stn)
                ctx_to_stn_map_[i].push_back(static_cast<uint32_t>(j));
        }
    }

    // Initialize DA-STDP per-connection weights (all start at 1.0)
    if (config_.da_stdp_enabled) {
        ctx_d1_w_.resize(n_input_neurons);
        ctx_d2_w_.resize(n_input_neurons);
        for (size_t i = 0; i < n_input_neurons; ++i) {
            ctx_d1_w_[i].assign(ctx_to_d1_map_[i].size(), 1.0f);
            ctx_d2_w_[i].assign(ctx_to_d2_map_[i].size(), 1.0f);
        }
        input_active_.assign(n_input_neurons, 0);
    }
}

void BasalGanglia::step(int32_t t, float dt) {
    oscillation_.step(dt);
    neuromod_.step(dt);

    // DA modulation: D1 gets tonic excitation proportional to DA
    //                D2 gets tonic excitation inversely proportional to DA
    // Scale is significant: MSN need ~50 to fire, DA should contribute ~10-20
    // Update DA level from spike accumulator (exponential smoothing)
    if (da_source_region_ != UINT32_MAX) {
        // DA firing rate estimate (spikes per step, smoothed)
        da_spike_accum_ *= DA_RATE_TAU;
        da_level_ = std::clamp(0.1f + da_spike_accum_ * 0.05f, 0.0f, 1.0f);
    }

    float da_exc_d1 = da_level_ * 30.0f;   // DA enhances D1
    float da_exc_d2 = (1.0f - da_level_) * 20.0f;  // DA suppresses D2
    for (size_t i = 0; i < d1_msn_.size(); ++i) d1_msn_.inject_basal(i, da_exc_d1);
    for (size_t i = 0; i < d2_msn_.size(); ++i) d2_msn_.inject_basal(i, da_exc_d2);

    // Inject decaying PSP buffers (cross-region synaptic time constant)
    for (size_t i = 0; i < psp_d1_.size(); ++i) {
        if (psp_d1_[i] > 0.5f) d1_msn_.inject_basal(i, psp_d1_[i]);
        psp_d1_[i] *= PSP_DECAY;
    }
    for (size_t i = 0; i < psp_d2_.size(); ++i) {
        if (psp_d2_[i] > 0.5f) d2_msn_.inject_basal(i, psp_d2_[i]);
        psp_d2_[i] *= PSP_DECAY;
    }
    for (size_t i = 0; i < psp_stn_.size(); ++i) {
        if (psp_stn_[i] > 0.5f) stn_.inject_basal(i, psp_stn_[i]);
        psp_stn_[i] *= PSP_DECAY;
    }

    // GPi/GPe get tonic excitation (they fire spontaneously)
    for (size_t i = 0; i < gpi_.size(); ++i) gpi_.inject_basal(i, 8.0f);
    for (size_t i = 0; i < gpe_.size(); ++i) gpe_.inject_basal(i, 6.0f);

    // 1. D1 → GPi (inhibit GPi = allow action)
    syn_d1_to_gpi_.deliver_spikes(d1_msn_.fired(), d1_msn_.spike_type());
    auto i_gpi_d1 = syn_d1_to_gpi_.step_and_compute(gpi_.v_soma(), dt);
    for (size_t i = 0; i < gpi_.size(); ++i) gpi_.inject_basal(i, i_gpi_d1[i]);

    // 2. D2 → GPe
    syn_d2_to_gpe_.deliver_spikes(d2_msn_.fired(), d2_msn_.spike_type());
    auto i_gpe_d2 = syn_d2_to_gpe_.step_and_compute(gpe_.v_soma(), dt);
    for (size_t i = 0; i < gpe_.size(); ++i) gpe_.inject_basal(i, i_gpe_d2[i]);

    // 3. GPe → STN (inhibit STN)
    syn_gpe_to_stn_.deliver_spikes(gpe_.fired(), gpe_.spike_type());
    auto i_stn_gpe = syn_gpe_to_stn_.step_and_compute(stn_.v_soma(), dt);
    for (size_t i = 0; i < stn_.size(); ++i) stn_.inject_basal(i, i_stn_gpe[i]);

    // 4. STN → GPi (excite GPi = brake)
    syn_stn_to_gpi_.deliver_spikes(stn_.fired(), stn_.spike_type());
    auto i_gpi_stn = syn_stn_to_gpi_.step_and_compute(gpi_.v_soma(), dt);
    for (size_t i = 0; i < gpi_.size(); ++i) gpi_.inject_basal(i, i_gpi_stn[i]);

    // Step all populations
    d1_msn_.step(t, dt);
    d2_msn_.step(t, dt);
    gpe_.step(t, dt);
    stn_.step(t, dt);
    gpi_.step(t, dt);

    // DA-STDP: update cortical→MSN weights based on co-activation + DA
    if (config_.da_stdp_enabled) {
        apply_da_stdp(t);
    }

    aggregate_state();
}

void BasalGanglia::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        // DA spikes from VTA → update DA level automatically
        if (evt.region_id == da_source_region_) {
            da_spike_accum_ += 1.0f;
            continue;
        }

        // Cortical spikes → route through pre-built random sparse maps
        float base_current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 25.0f : 15.0f;
        size_t src = evt.neuron_id % input_map_size_;

        // Mark input as active for DA-STDP
        if (config_.da_stdp_enabled && src < input_active_.size()) {
            input_active_[src] = 1;
        }

        for (size_t idx = 0; idx < ctx_to_d1_map_[src].size(); ++idx) {
            uint32_t tgt = ctx_to_d1_map_[src][idx];
            float w = (config_.da_stdp_enabled && src < ctx_d1_w_.size()) ? ctx_d1_w_[src][idx] : 1.0f;
            psp_d1_[tgt] += base_current * w;
        }
        for (size_t idx = 0; idx < ctx_to_d2_map_[src].size(); ++idx) {
            uint32_t tgt = ctx_to_d2_map_[src][idx];
            float w = (config_.da_stdp_enabled && src < ctx_d2_w_.size()) ? ctx_d2_w_[src][idx] : 1.0f;
            psp_d2_[tgt] += base_current * w;
        }
        for (uint32_t tgt : ctx_to_stn_map_[src]) {
            psp_stn_[tgt] += base_current * 0.5f;
        }
    }
}

void BasalGanglia::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_all_, spike_type_all_, t);
}

void BasalGanglia::inject_external(const std::vector<float>& currents) {
    // External = cortical input to D1/D2
    for (size_t i = 0; i < std::min(currents.size(), d1_msn_.size()); ++i) {
        d1_msn_.inject_basal(i, currents[i]);
    }
    for (size_t i = 0; i < std::min(currents.size(), d2_msn_.size()); ++i) {
        d2_msn_.inject_basal(i, currents[i]);
    }
}

void BasalGanglia::inject_cortical_input(const std::vector<float>& d1_cur,
                                          const std::vector<float>& d2_cur) {
    for (size_t i = 0; i < std::min(d1_cur.size(), d1_msn_.size()); ++i) {
        d1_msn_.inject_basal(i, d1_cur[i]);
    }
    for (size_t i = 0; i < std::min(d2_cur.size(), d2_msn_.size()); ++i) {
        d2_msn_.inject_basal(i, d2_cur[i]);
    }
}

void BasalGanglia::set_da_level(float da) {
    da_level_ = std::clamp(da, 0.0f, 1.0f);
}

void BasalGanglia::aggregate_state() {
    size_t off = 0;
    auto copy = [&](const NeuronPopulation& pop) {
        for (size_t i = 0; i < pop.size(); ++i) {
            fired_all_[off + i]      = pop.fired()[i];
            spike_type_all_[off + i] = pop.spike_type()[i];
        }
        off += pop.size();
    };
    copy(d1_msn_);
    copy(d2_msn_);
    copy(gpi_);
    copy(gpe_);
    copy(stn_);
}

void BasalGanglia::apply_da_stdp(int32_t t) {
    // DA reward prediction error (RPE-like)
    float da_error = da_level_ - config_.da_stdp_baseline;
    float lr = config_.da_stdp_lr;

    // Three-factor rule:
    //   D1 (Go):  input active + D1 fired + DA>baseline → strengthen (reinforce Go)
    //             input active + D1 fired + DA<baseline → weaken
    //   D2 (NoGo): OPPOSITE sign — DA>baseline weakens D2, DA<baseline strengthens
    //   This is the biological D1(Gs) vs D2(Gi) receptor asymmetry

    for (size_t src = 0; src < input_active_.size(); ++src) {
        if (!input_active_[src]) continue;

        // D1 pathway: DA+ = reinforce Go
        for (size_t idx = 0; idx < ctx_to_d1_map_[src].size(); ++idx) {
            uint32_t tgt = ctx_to_d1_map_[src][idx];
            if (d1_msn_.fired()[tgt]) {
                // Co-active: pre(cortex) + post(D1) + DA modulation
                ctx_d1_w_[src][idx] += lr * da_error;
                ctx_d1_w_[src][idx] = std::clamp(ctx_d1_w_[src][idx],
                    config_.da_stdp_w_min, config_.da_stdp_w_max);
            }
        }

        // D2 pathway: DA+ = weaken NoGo (opposite sign!)
        for (size_t idx = 0; idx < ctx_to_d2_map_[src].size(); ++idx) {
            uint32_t tgt = ctx_to_d2_map_[src][idx];
            if (d2_msn_.fired()[tgt]) {
                // D2: reverse sign — DA reward weakens NoGo
                ctx_d2_w_[src][idx] -= lr * da_error;
                ctx_d2_w_[src][idx] = std::clamp(ctx_d2_w_[src][idx],
                    config_.da_stdp_w_min, config_.da_stdp_w_max);
            }
        }
    }

    // Clear input activity flags for next step
    std::fill(input_active_.begin(), input_active_.end(), 0);
}

} // namespace wuyun
