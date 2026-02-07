#include "engine/simulation_engine.h"
#include <algorithm>

namespace wuyun {

SimulationEngine::SimulationEngine(int32_t max_delay)
    : bus_(max_delay)
{}

void SimulationEngine::add_region(std::unique_ptr<BrainRegion> region) {
    region->register_to_bus(bus_);
    regions_.push_back(std::move(region));
}

BrainRegion* SimulationEngine::find_region(const std::string& name) {
    for (auto& r : regions_) {
        if (r->name() == name) return r.get();
    }
    return nullptr;
}

void SimulationEngine::add_projection(const std::string& src, const std::string& dst,
                                       int32_t delay, const std::string& proj_name) {
    auto* s = find_region(src);
    auto* d = find_region(dst);
    if (s && d) {
        std::string pname = proj_name.empty() ? (src + "->" + dst) : proj_name;
        bus_.add_projection(s->region_id(), d->region_id(), delay, pname);
    }
}

void SimulationEngine::run(int32_t n_steps, float dt) {
    for (int32_t i = 0; i < n_steps; ++i) {
        step(dt);
    }
}

void SimulationEngine::step(float dt) {
    // 1. Deliver arriving spikes to each region
    for (auto& region : regions_) {
        auto events = bus_.get_arriving_spikes(region->region_id(), t_);
        if (!events.empty()) {
            region->receive_spikes(events);
        }
    }

    // 2. Each region steps internally
    for (auto& region : regions_) {
        region->step(t_, dt);
    }

    // 3. Each region submits outgoing spikes
    for (auto& region : regions_) {
        region->submit_spikes(bus_, t_);
    }

    // 4. Advance bus (clear expired slots)
    bus_.advance(t_);

    // 5. Callback
    if (callback_) {
        callback_(t_, *this);
    }

    t_++;
}

SimStats SimulationEngine::stats() const {
    SimStats s;
    s.timestep = t_;
    s.total_regions = regions_.size();
    for (const auto& r : regions_) {
        s.total_neurons += r->n_neurons();
        for (size_t i = 0; i < r->n_neurons(); ++i) {
            if (r->fired()[i]) s.total_spikes++;
        }
    }
    return s;
}

} // namespace wuyun
