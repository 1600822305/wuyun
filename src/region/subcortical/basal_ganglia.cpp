#include "region/subcortical/basal_ganglia.h"
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

    // Build TOPOGRAPHIC sensory→D1 mapping (thalamostriatal pathway)
    // Slots 252-255 = sensory direction channels (UP/DOWN/LEFT/RIGHT)
    // Each maps to ALL D1 neurons in the corresponding action subgroup
    if (config_.da_stdp_enabled) {
        size_t d1_size = d1_msn_.size();
        size_t d1_group = d1_size / 4;
        for (int dir = 0; dir < 4; ++dir) {
            size_t slot = SENSORY_SLOT_BASE + dir;
            if (slot < n_input_neurons) {
                ctx_to_d1_map_[slot].clear();  // Replace random with topographic
                size_t start = dir * d1_group;
                size_t end = (dir < 3) ? (dir + 1) * d1_group : d1_size;
                for (size_t j = start; j < end; ++j) {
                    ctx_to_d1_map_[slot].push_back(static_cast<uint32_t>(j));
                }
                // Also D2: sensory→NoGo for same direction
                ctx_to_d2_map_[slot].clear();
                size_t d2_size = d2_msn_.size();
                size_t d2_group = d2_size / 4;
                size_t d2_start = dir * d2_group;
                size_t d2_end = (dir < 3) ? (dir + 1) * d2_group : d2_size;
                for (size_t j = d2_start; j < d2_end; ++j) {
                    ctx_to_d2_map_[slot].push_back(static_cast<uint32_t>(j));
                }
            }
        }
    }

    // Initialize DA-STDP per-connection weights (all start at 1.0)
    if (config_.da_stdp_enabled) {
        ctx_d1_w_.resize(n_input_neurons);
        ctx_d2_w_.resize(n_input_neurons);
        elig_d1_.resize(n_input_neurons);
        elig_d2_.resize(n_input_neurons);
        for (size_t i = 0; i < n_input_neurons; ++i) {
            ctx_d1_w_[i].assign(ctx_to_d1_map_[i].size(), 1.0f);
            ctx_d2_w_[i].assign(ctx_to_d2_map_[i].size(), 1.0f);
            elig_d1_[i].assign(ctx_to_d1_map_[i].size(), 0.0f);
            elig_d2_[i].assign(ctx_to_d2_map_[i].size(), 0.0f);
        }
        input_active_.assign(n_input_neurons, 0);
    }
}

void BasalGanglia::set_topographic_cortical_source(uint32_t region_id, size_t n_neurons) {
    topo_ctx_rid_ = region_id;
    topo_ctx_n_ = n_neurons;

    // Rebuild ctx→D1/D2 maps for this source's neuron range with topographic bias.
    // Biology: corticostriatal projections from dlPFC maintain partial somatotopy.
    // dlPFC neuron in "channel c" → preferentially connects to D1/D2 subgroup c.
    // channel = (neuron_id × 4) / n_neurons  (proportional spatial mapping)
    size_t d1_size = d1_msn_.size();
    size_t d2_size = d2_msn_.size();
    size_t d1_group = d1_size / 4;
    size_t d2_group = d2_size / 4;

    // Don't touch sensory slots (252-255)
    size_t n_slots = std::min(n_neurons, static_cast<size_t>(SENSORY_SLOT_BASE));
    n_slots = std::min(n_slots, input_map_size_);

    std::mt19937 rng(888);  // Deterministic, different from random maps (seed=777)
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float p_same  = 0.60f;  // 60% connection to matching action subgroup (~4.2 connections)
    float p_other = 0.05f;  // 5% to non-matching subgroups (~1.2 connections, total ~5.4)

    for (size_t i = 0; i < n_slots; ++i) {
        int channel = static_cast<int>((i * 4) / n_neurons);
        if (channel >= 4) channel = 3;

        // Rebuild D1 map for this slot
        ctx_to_d1_map_[i].clear();
        for (size_t j = 0; j < d1_size; ++j) {
            int d1_ch = static_cast<int>(j / d1_group);
            if (d1_ch >= 4) d1_ch = 3;
            float prob = (d1_ch == channel) ? p_same : p_other;
            if (dist(rng) < prob) {
                ctx_to_d1_map_[i].push_back(static_cast<uint32_t>(j));
            }
        }

        // Rebuild D2 map for this slot
        ctx_to_d2_map_[i].clear();
        for (size_t j = 0; j < d2_size; ++j) {
            int d2_ch = static_cast<int>(j / d2_group);
            if (d2_ch >= 4) d2_ch = 3;
            float prob = (d2_ch == channel) ? p_same : p_other;
            if (dist(rng) < prob) {
                ctx_to_d2_map_[i].push_back(static_cast<uint32_t>(j));
            }
        }

        // STN map unchanged (hyperdirect is non-topographic)
    }

    // Rebuild DA-STDP weights and eligibility traces for affected slots
    if (config_.da_stdp_enabled) {
        for (size_t i = 0; i < n_slots; ++i) {
            ctx_d1_w_[i].assign(ctx_to_d1_map_[i].size(), 1.0f);
            ctx_d2_w_[i].assign(ctx_to_d2_map_[i].size(), 1.0f);
            elig_d1_[i].assign(ctx_to_d1_map_[i].size(), 0.0f);
            elig_d2_[i].assign(ctx_to_d2_map_[i].size(), 0.0f);
        }
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
        da_level_ = std::clamp(0.1f + da_spike_accum_ * 0.08f, 0.0f, 1.0f);
    }

    // MSN up-state drive + symmetric DA modulation
    // v26: tonic = up(20) + da_base(15) = 35 (was 40). Moderate reduction.
    // Multiplicative weight gain (3×) does most of the work amplifying weight differences.
    // DA > baseline: D1↑ D2↓ (reward → reinforce Go, suppress NoGo)
    // DA < baseline: D1↓ D2↑ (punishment → suppress Go, reinforce NoGo)
    float up = config_.msn_up_state_drive;
    float da_delta = da_level_ - config_.da_stdp_baseline;  // RPE-like
    float da_base = 15.0f;   // v26: keep at 15 for compatibility
    float da_gain = 50.0f;   // DA modulation strength
    float da_exc_d1 = up + da_base + da_delta * da_gain;   // Go: DA↑ → more
    float da_exc_d2 = up + da_base - da_delta * da_gain;   // NoGo: DA↑ → less
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

    // --- D1/D2 lateral inhibition: MSN collateral GABA competition ---
    // Biology: striatal MSN have GABAergic collateral synapses (~1-3% connectivity)
    // that implement local competition between action channels.
    // Implementation: count recent firing per subgroup, most active subgroup
    // sends inhibitory current to competing subgroups.
    // Effect: "向左走" 被奖励 → D1-LEFT 活跃 → 抑制 D1-RIGHT/UP/DOWN
    //         → 方向选择性在权重中逐渐涌现
    if (config_.lateral_inhibition && d1_msn_.size() >= 4) {
        size_t d1_group = d1_msn_.size() / 4;
        size_t d2_group = d2_msn_.size() / 4;

        // Count fires per subgroup from last step
        int d1_fires[4] = {0, 0, 0, 0};
        int d2_fires[4] = {0, 0, 0, 0};
        for (int g = 0; g < 4; ++g) {
            size_t start = g * d1_group;
            size_t end = (g < 3) ? (g + 1) * d1_group : d1_msn_.size();
            for (size_t j = start; j < end; ++j)
                if (d1_msn_.fired()[j]) d1_fires[g]++;
        }
        for (int g = 0; g < 4; ++g) {
            size_t start = g * d2_group;
            size_t end = (g < 3) ? (g + 1) * d2_group : d2_msn_.size();
            for (size_t j = start; j < end; ++j)
                if (d2_msn_.fired()[j]) d2_fires[g]++;
        }

        // Find max D1 subgroup
        int max_d1 = *std::max_element(d1_fires, d1_fires + 4);
        if (max_d1 > 0) {
            float inh = config_.lateral_inh_strength;
            for (int g = 0; g < 4; ++g) {
                if (d1_fires[g] < max_d1) {
                    // Losing D1 subgroup gets inhibited (GABA: negative current)
                    float suppress = -inh * static_cast<float>(max_d1 - d1_fires[g]);
                    size_t start = g * d1_group;
                    size_t end = (g < 3) ? (g + 1) * d1_group : d1_msn_.size();
                    for (size_t j = start; j < end; ++j)
                        d1_msn_.inject_basal(j, suppress);
                }
            }
        }

        // Same for D2 (losing NoGo channels get suppressed → winner NoGo dominates)
        int max_d2 = *std::max_element(d2_fires, d2_fires + 4);
        if (max_d2 > 0) {
            float inh = config_.lateral_inh_strength;
            for (int g = 0; g < 4; ++g) {
                if (d2_fires[g] < max_d2) {
                    float suppress = -inh * static_cast<float>(max_d2 - d2_fires[g]);
                    size_t start = g * d2_group;
                    size_t end = (g < 3) ? (g + 1) * d2_group : d2_msn_.size();
                    for (size_t j = start; j < end; ++j)
                        d2_msn_.inject_basal(j, suppress);
                }
            }
        }
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
        // L5 corticostriatal axons are among the thickest white matter tracts
        // MSN up-state drive (40) + PSP (30) = 70 → reliable MSN firing from cortical input
        float base_current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 50.0f : 30.0f;
        size_t src = evt.neuron_id % input_map_size_;

        // Mark input as active for DA-STDP
        if (config_.da_stdp_enabled && src < input_active_.size()) {
            input_active_[src] = 1;
            total_cortical_inputs_++;
        }

        for (size_t idx = 0; idx < ctx_to_d1_map_[src].size(); ++idx) {
            uint32_t tgt = ctx_to_d1_map_[src][idx];
            float w = (config_.da_stdp_enabled && src < ctx_d1_w_.size()) ? ctx_d1_w_[src][idx] : 1.0f;
            // v26: multiplicative gain (Surmeier 2007)
            // w=1.0→gain=1.0, w=1.5→gain=2.5, w=0.5→gain=0.25
            // Weight differences are nonlinearly amplified, making learned preferences decisive
            float gain = 1.0f + (w - 1.0f) * config_.weight_gain_factor;
            if (gain < 0.1f) gain = 0.1f;  // Floor: don't go fully silent
            psp_d1_[tgt] += base_current * gain;
        }
        for (size_t idx = 0; idx < ctx_to_d2_map_[src].size(); ++idx) {
            uint32_t tgt = ctx_to_d2_map_[src][idx];
            float w = (config_.da_stdp_enabled && src < ctx_d2_w_.size()) ? ctx_d2_w_[src][idx] : 1.0f;
            float gain = 1.0f + (w - 1.0f) * config_.weight_gain_factor;
            if (gain < 0.1f) gain = 0.1f;
            psp_d2_[tgt] += base_current * gain;
        }
        for (uint32_t tgt : ctx_to_stn_map_[src]) {
            psp_stn_[tgt] += base_current * 0.5f;
        }
    }
}

void BasalGanglia::inject_sensory_context(const float signals[4]) {
    if (!config_.da_stdp_enabled) return;

    size_t d1_size = d1_msn_.size();
    size_t d1_group = d1_size / 4;
    float ctx_psp = 25.0f;  // Sensory context drive strength

    for (int dir = 0; dir < 4; ++dir) {
        if (std::abs(signals[dir]) < 0.01f) continue;

        size_t slot = SENSORY_SLOT_BASE + dir;
        if (slot >= input_active_.size()) continue;

        // Mark sensory slot as active for DA-STDP eligibility trace formation
        input_active_[slot] = 1;

        // Inject current into corresponding D1 subgroup
        // Positive signal = food direction → boost D1 (Go)
        // Negative signal = danger direction → suppress D1
        float current = signals[dir] * ctx_psp;
        size_t start = dir * d1_group;
        size_t end = (dir < 3) ? (dir + 1) * d1_group : d1_size;
        for (size_t j = start; j < end; ++j) {
            psp_d1_[j] += std::max(0.0f, current);
        }

        // For danger: boost D2 (NoGo) instead
        if (signals[dir] < 0.0f) {
            size_t d2_size = d2_msn_.size();
            size_t d2_group = d2_size / 4;
            size_t d2_start = dir * d2_group;
            size_t d2_end = (dir < 3) ? (dir + 1) * d2_group : d2_size;
            for (size_t j = d2_start; j < d2_end; ++j) {
                psp_d2_[j] += std::abs(current);
            }
        }
    }
}

void BasalGanglia::mark_motor_efference(int action_group) {
    if (!config_.da_stdp_enabled) return;
    if (action_group < 0 || action_group >= 4) return;
    size_t slot = SENSORY_SLOT_BASE + action_group;
    if (slot < input_active_.size()) {
        input_active_[slot] = 1;
        total_cortical_inputs_++;
    }
    // Inject PSP through LEARNED topographic weights
    // As DA-STDP potentiates the rewarded direction's weights, PSP grows stronger
    // → D1 fires more for learned directions → BG biases M1 → positive feedback loop
    // 15.0 base × weight: initially 15×1.0=15 (subtle), grows to 15×1.6=24 after learning
    if (slot < ctx_d1_w_.size()) {
        float base_psp = 5.0f;
        for (size_t idx = 0; idx < ctx_to_d1_map_[slot].size(); ++idx) {
            uint32_t tgt = ctx_to_d1_map_[slot][idx];
            float w = ctx_d1_w_[slot][idx];
            psp_d1_[tgt] += base_psp * w;
        }
    }
}

void BasalGanglia::replay_learning_step(int32_t t, float dt) {
    // Lightweight replay: only D1/D2 firing + DA-STDP update.
    // Does NOT step GPi/GPe/STN or process internal synapses.
    // Avoids disrupting BG motor output state during replay.

    // MSN up-state drive + DA modulation (same as normal step)
    float up = config_.msn_up_state_drive;
    float da_delta = da_level_ - config_.da_stdp_baseline;
    float da_base = 15.0f;   // v26: match step() change
    float da_gain = 50.0f;
    float da_exc_d1 = up + da_base + da_delta * da_gain;
    float da_exc_d2 = up + da_base - da_delta * da_gain;
    for (size_t i = 0; i < d1_msn_.size(); ++i) d1_msn_.inject_basal(i, da_exc_d1);
    for (size_t i = 0; i < d2_msn_.size(); ++i) d2_msn_.inject_basal(i, da_exc_d2);

    // Inject decaying PSP from receive_spikes (cortical replay input)
    for (size_t i = 0; i < psp_d1_.size(); ++i) {
        if (psp_d1_[i] > 0.5f) d1_msn_.inject_basal(i, psp_d1_[i]);
        psp_d1_[i] *= PSP_DECAY;
    }
    for (size_t i = 0; i < psp_d2_.size(); ++i) {
        if (psp_d2_[i] > 0.5f) d2_msn_.inject_basal(i, psp_d2_[i]);
        psp_d2_[i] *= PSP_DECAY;
    }

    // Step only D1 and D2 (they need to fire for eligibility trace formation)
    d1_msn_.step(t, dt);
    d2_msn_.step(t, dt);

    // DA-STDP: update weights (replay_mode_ suppresses weight decay)
    if (config_.da_stdp_enabled) {
        apply_da_stdp(t);
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
    // Three-factor learning with eligibility traces:
    //   1. Co-activation (pre=cortex, post=D1/D2) increments eligibility trace
    //   2. DA signal (RPE) modulates weight change proportional to trace
    //   3. Trace decays exponentially (bridges action→reward delay)
    //
    // D1 (Go):  DA>baseline → strengthen (reinforce action)
    // D2 (NoGo): DA>baseline → weaken (reduce inhibition of rewarded action)
    // Biological basis: D1(Gs) vs D2(Gi) receptor asymmetry

    float da_error = da_level_ - config_.da_stdp_baseline;
    float lr = config_.da_stdp_lr;
    float elig_decay = config_.da_stdp_elig_decay;

    // Phase 1: Update eligibility traces from co-activation
    float max_elig = config_.da_stdp_max_elig;
    for (size_t src = 0; src < input_active_.size(); ++src) {
        if (!input_active_[src]) continue;

        for (size_t idx = 0; idx < ctx_to_d1_map_[src].size(); ++idx) {
            uint32_t tgt = ctx_to_d1_map_[src][idx];
            if (d1_msn_.fired()[tgt]) {
                elig_d1_[src][idx] = std::min(elig_d1_[src][idx] + 1.0f, max_elig);
            }
        }
        for (size_t idx = 0; idx < ctx_to_d2_map_[src].size(); ++idx) {
            uint32_t tgt = ctx_to_d2_map_[src][idx];
            if (d2_msn_.fired()[tgt]) {
                elig_d2_[src][idx] = std::min(elig_d2_[src][idx] + 1.0f, max_elig);
            }
        }
    }

    // Phase 2: Apply weight changes = lr * da_error * eligibility_trace
    // Only apply when DA deviates from baseline (RPE ≠ 0)
    if (std::abs(da_error) > 0.001f) {
        for (size_t src = 0; src < elig_d1_.size(); ++src) {
            for (size_t idx = 0; idx < elig_d1_[src].size(); ++idx) {
                if (elig_d1_[src][idx] > 0.001f) {
                    ctx_d1_w_[src][idx] += lr * da_error * elig_d1_[src][idx];
                    ctx_d1_w_[src][idx] = std::clamp(ctx_d1_w_[src][idx],
                        config_.da_stdp_w_min, config_.da_stdp_w_max);
                }
            }
            for (size_t idx = 0; idx < elig_d2_[src].size(); ++idx) {
                if (elig_d2_[src][idx] > 0.001f) {
                    // D2: reverse sign
                    ctx_d2_w_[src][idx] -= lr * da_error * elig_d2_[src][idx];
                    ctx_d2_w_[src][idx] = std::clamp(ctx_d2_w_[src][idx],
                        config_.da_stdp_w_min, config_.da_stdp_w_max);
                }
            }
        }
    }

    // Phase 3: Decay eligibility traces + homeostatic weight decay toward 1.0
    // During replay mode: skip weight decay (prevent over-decay from extra replay steps)
    // but still decay eligibility traces (replay needs fresh traces each pass)
    float w_decay = replay_mode_ ? 0.0f : config_.da_stdp_w_decay;
    for (size_t src = 0; src < elig_d1_.size(); ++src) {
        for (size_t idx = 0; idx < elig_d1_[src].size(); ++idx) {
            elig_d1_[src][idx] *= elig_decay;
        }
        for (size_t idx = 0; idx < elig_d2_[src].size(); ++idx) {
            elig_d2_[src][idx] *= elig_decay;
        }
        // Weight decay: pull toward 1.0 (prevents runaway potentiation/depression)
        if (w_decay > 0.0f) {
            for (size_t idx = 0; idx < ctx_d1_w_[src].size(); ++idx) {
                ctx_d1_w_[src][idx] += w_decay * (1.0f - ctx_d1_w_[src][idx]);
            }
            for (size_t idx = 0; idx < ctx_d2_w_[src].size(); ++idx) {
                ctx_d2_w_[src][idx] += w_decay * (1.0f - ctx_d2_w_[src][idx]);
            }
        }
    }

    // Clear input activity flags for next step
    std::fill(input_active_.begin(), input_active_.end(), 0);
}

} // namespace wuyun
