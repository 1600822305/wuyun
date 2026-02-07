#include "region/cortical_region.h"
#include <algorithm>

namespace wuyun {

CorticalRegion::CorticalRegion(const std::string& name, const ColumnConfig& config)
    : BrainRegion(name, config.n_l4_stellate + config.n_l23_pyramidal +
                        config.n_l5_pyramidal + config.n_l6_pyramidal +
                        config.n_pv_basket + config.n_sst_martinotti +
                        config.n_vip)
    , column_(config)
    , fired_(n_neurons_, 0)
    , spike_type_(n_neurons_, 0)
    , psp_buffer_(config.n_l4_stellate, 0.0f)
    , psp_current_regular_(config.input_psp_regular)
    , psp_current_burst_(config.input_psp_burst)
    , psp_fan_out_(std::max<size_t>(3, static_cast<size_t>(config.n_l4_stellate * config.input_fan_out_frac)))
{}

void CorticalRegion::step(int32_t t, float dt) {
    // Update oscillation and neuromodulation
    oscillation_.step(dt);
    neuromod_.step(dt);

    // NE gain modulation: neuromod system's gain affects all incoming PSP
    float ne_gain = neuromod_.compute_effect().gain;  // 0.5 ~ 2.0

    // Inject decaying PSP buffer into L4 basal (simulates synaptic time constant)
    auto& l4 = column_.l4();
    for (size_t i = 0; i < psp_buffer_.size(); ++i) {
        if (psp_buffer_[i] > 0.5f) {
            l4.inject_basal(i, psp_buffer_[i] * ne_gain);
        }
        psp_buffer_[i] *= PSP_DECAY;  // Exponential decay
    }

    // Step the cortical column
    last_output_ = column_.step(t, dt);

    // Aggregate firing state from all populations
    aggregate_firing_state();
}

void CorticalRegion::receive_spikes(const std::vector<SpikeEvent>& events) {
    // Route arriving spikes to PSP buffer (sustained over multiple steps)
    // Fan-out: each spike activates ~30% of L4 (biological cortico-cortical convergence)
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type))
                        ? psp_current_burst_ : psp_current_regular_;
        size_t base = evt.neuron_id % psp_buffer_.size();
        for (size_t k = 0; k < psp_fan_out_; ++k) {
            size_t idx = (base + k) % psp_buffer_.size();
            psp_buffer_[idx] += current;
        }
    }
}

void CorticalRegion::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void CorticalRegion::inject_external(const std::vector<float>& currents) {
    inject_feedforward(currents);
}

void CorticalRegion::inject_feedforward(const std::vector<float>& currents) {
    column_.inject_feedforward(currents);
}

void CorticalRegion::inject_feedback(const std::vector<float>& currents) {
    // Feedback goes to L2/3 and L5 apical dendrites
    // Split: first n_l23 values -> L2/3, rest -> L5
    auto& l23 = column_.l23();
    auto& l5  = column_.l5();

    std::vector<float> l23_cur(l23.size(), 0.0f);
    std::vector<float> l5_cur(l5.size(), 0.0f);

    for (size_t i = 0; i < std::min(currents.size(), l23.size()); ++i) {
        l23_cur[i] = currents[i];
    }
    for (size_t i = 0; i < l5.size() && (i + l23.size()) < currents.size(); ++i) {
        l5_cur[i] = currents[i + l23.size()];
    }
    column_.inject_feedback(l23_cur, l5_cur);
}

void CorticalRegion::inject_attention(float vip_current) {
    column_.inject_attention(vip_current);
}

void CorticalRegion::aggregate_firing_state() {
    // Merge all population firing states into a single flat vector
    // Order: L4, L23, L5, L6, PV, SST, VIP
    size_t offset = 0;
    auto copy_pop = [&](const NeuronPopulation& pop) {
        const auto& f = pop.fired();
        const auto& s = pop.spike_type();
        for (size_t i = 0; i < pop.size(); ++i) {
            fired_[offset + i]      = f[i];
            spike_type_[offset + i] = s[i];
        }
        offset += pop.size();
    };

    copy_pop(column_.l4());
    copy_pop(column_.l23());
    copy_pop(column_.l5());
    copy_pop(column_.l6());

    // Access inhibitory populations through column internals
    // For now, the remaining slots stay 0 (inhibitory firing not exported)
    // This is fine — SpikeBus only needs excitatory output for cross-region routing
}

} // namespace wuyun
