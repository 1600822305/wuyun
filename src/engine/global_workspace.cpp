#include "engine/global_workspace.h"
#include <cmath>
#include <algorithm>

namespace wuyun {

GlobalWorkspace::GlobalWorkspace(const GWConfig& config)
    : BrainRegion(config.name, config.n_workspace)
    , config_(config)
    , workspace_(config.n_workspace, NeuronParams{})
    , fired_all_(n_neurons_, 0)
    , spike_type_all_(n_neurons_, 0)
{
}

void GlobalWorkspace::register_source(uint32_t region_id, const std::string& name) {
    source_names_[region_id] = name;
}

void GlobalWorkspace::step(int32_t t, float dt) {
    oscillation_.step(dt);
    neuromod_.step(dt);

    // =========================================================
    // 1. Update salience from this step's incoming spikes
    // =========================================================
    // Decay existing salience (prevents lock-in)
    for (auto& [rid, sal] : salience_) {
        sal *= config_.competition_decay;
    }

    // Add new spike contributions
    for (auto& [rid, count] : step_spikes_) {
        salience_[rid] += static_cast<float>(count);
    }
    step_spikes_.clear();

    // =========================================================
    // 2. Competition: find winner
    // =========================================================
    uint32_t winner_id = 0;
    float max_salience = 0.0f;

    for (auto& [rid, sal] : salience_) {
        if (sal > max_salience) {
            max_salience = sal;
            winner_id = rid;
        }
    }
    winning_salience_ = max_salience;

    // =========================================================
    // 3. Ignition check
    // =========================================================
    bool can_ignite = (t - last_ignition_t_) >= config_.min_ignition_gap;

    if (!is_ignited_ && can_ignite && max_salience >= config_.ignition_threshold) {
        // === IGNITION ===
        is_ignited_ = true;
        conscious_content_id_ = static_cast<int32_t>(winner_id);
        if (source_names_.count(winner_id)) {
            conscious_content_name_ = source_names_[winner_id];
        } else {
            conscious_content_name_ = "region_" + std::to_string(winner_id);
        }
        ignition_count_++;
        broadcast_remaining_ = config_.broadcast_duration;
        last_ignition_t_ = t;

        // Strong drive to workspace neurons (ignition burst)
        broadcast_current_ = max_salience * config_.broadcast_gain;
    }

    // =========================================================
    // 4. Broadcast phase: drive workspace neurons
    // =========================================================
    if (broadcast_remaining_ > 0) {
        // Drive workspace neurons with amplified signal
        float drive = broadcast_current_ * 
            (static_cast<float>(broadcast_remaining_) / 
             static_cast<float>(config_.broadcast_duration));
        for (size_t i = 0; i < workspace_.size(); ++i) {
            workspace_.inject_basal(i, drive);
        }
        broadcast_remaining_--;
        if (broadcast_remaining_ == 0) {
            is_ignited_ = false;
        }
    }

    // =========================================================
    // 5. Step workspace neurons
    // =========================================================
    workspace_.step(t, dt);

    aggregate_state();
}

void GlobalWorkspace::receive_spikes(const std::vector<SpikeEvent>& events) {
    // Track per-source-region spike counts for competition
    for (const auto& evt : events) {
        step_spikes_[evt.region_id]++;
    }
}

void GlobalWorkspace::submit_spikes(SpikeBus& bus, int32_t t) {
    // During broadcast, workspace neurons fire → propagates to ILN/CeM → all cortex
    bus.submit_spikes(region_id_, fired_all_, spike_type_all_, t);
}

void GlobalWorkspace::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), workspace_.size()); ++i) {
        workspace_.inject_basal(i, currents[i]);
    }
}

void GlobalWorkspace::aggregate_state() {
    const auto& f = workspace_.fired();
    const auto& s = workspace_.spike_type();
    for (size_t i = 0; i < workspace_.size(); ++i) {
        fired_all_[i]      = f[i];
        spike_type_all_[i] = s[i];
    }
}

} // namespace wuyun
