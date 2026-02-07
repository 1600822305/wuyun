#include "engine/simulation_engine.h"
#include "region/neuromod/vta_da.h"
#include "region/neuromod/lc_ne.h"
#include "region/neuromod/drn_5ht.h"
#include "region/neuromod/nbm_ach.h"
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

    // 3. Collect neuromodulator levels and broadcast to all regions
    collect_and_broadcast_neuromod();

    // 4. Each region submits outgoing spikes
    for (auto& region : regions_) {
        region->submit_spikes(bus_, t_);
    }

    // 5. Advance bus (clear expired slots)
    bus_.advance(t_);

    // 6. Callback
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

void SimulationEngine::register_neuromod_source(const std::string& region_name,
                                                  NeuromodType type) {
    auto* r = find_region(region_name);
    if (r) {
        neuromod_sources_.push_back({r, type});
    }
}

void SimulationEngine::collect_and_broadcast_neuromod() {
    if (neuromod_sources_.empty()) return;

    // Collect output levels from registered source regions
    for (const auto& src : neuromod_sources_) {
        float level = 0.0f;
        switch (src.type) {
            case NeuromodType::DA: {
                auto* vta = dynamic_cast<VTA_DA*>(src.region);
                if (vta) level = vta->da_output();
                break;
            }
            case NeuromodType::NE: {
                auto* lc = dynamic_cast<LC_NE*>(src.region);
                if (lc) level = lc->ne_output();
                break;
            }
            case NeuromodType::SHT: {
                auto* drn = dynamic_cast<DRN_5HT*>(src.region);
                if (drn) level = drn->sht_output();
                break;
            }
            case NeuromodType::ACh: {
                auto* nbm = dynamic_cast<NBM_ACh*>(src.region);
                if (nbm) level = nbm->ach_output();
                break;
            }
        }

        switch (src.type) {
            case NeuromodType::DA:  global_neuromod_.da  = level; break;
            case NeuromodType::NE:  global_neuromod_.ne  = level; break;
            case NeuromodType::SHT: global_neuromod_.sht = level; break;
            case NeuromodType::ACh: global_neuromod_.ach = level; break;
        }
    }

    // Broadcast to all regions' NeuromodulatorSystem
    for (auto& region : regions_) {
        region->neuromod().set_tonic(global_neuromod_);
    }
}

} // namespace wuyun
