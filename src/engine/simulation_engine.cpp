#include "engine/simulation_engine.h"
#include "region/neuromod/vta_da.h"
#include "region/neuromod/lc_ne.h"
#include "region/neuromod/drn_5ht.h"
#include "region/neuromod/nbm_ach.h"
#include <algorithm>

#ifdef WUYUN_OPENMP
#include <omp.h>
#endif

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

    // 2. Each region steps internally (OpenMP parallel — regions are independent within a step)
    {
        int n_regions = static_cast<int>(regions_.size());
#ifdef WUYUN_OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < n_regions; ++i) {
            regions_[i]->step(t_, dt);
        }
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

// =============================================================================
// v54: 拓扑导出
// =============================================================================

// 脑区分类 (DOT subgraph 分组)
static const char* classify_region(const std::string& name) {
    // 皮层
    if (name == "V1" || name == "V2" || name == "V4" || name == "IT" ||
        name == "dlPFC" || name == "M1" || name == "FPC" || name == "ACC" ||
        name == "vmPFC" || name == "LGN")
        return "cortical";
    // 价值/前额叶
    if (name == "OFC")
        return "cortical";
    // 皮层下
    if (name == "BG" || name == "MotorThal" || name == "SC" ||
        name == "Cerebellum" || name == "NAcc")
        return "subcortical";
    // 边缘
    if (name == "Hippocampus" || name == "Amygdala" || name == "Hypothalamus" ||
        name == "PAG" || name == "LHb" || name == "SeptalNucleus" ||
        name == "MammillaryBody")
        return "limbic";
    // 神经调质
    if (name == "VTA" || name == "SNc" || name == "LC" ||
        name == "DRN" || name == "NBM")
        return "neuromod";
    return "other";
}

std::string SimulationEngine::export_dot() const {
    std::string dot;
    dot += "digraph Brain {\n";
    dot += "  rankdir=LR;\n";
    dot += "  bgcolor=\"#1a1a2e\";\n";
    dot += "  node [fontname=\"Arial\", fontcolor=white, color=white];\n";
    dot += "  edge [color=\"#888888\", fontcolor=\"#aaaaaa\", fontsize=9];\n";
    dot += "\n";

    // 分组收集区域
    std::vector<size_t> cortical, subcortical, limbic, neuromod, other;
    for (size_t i = 0; i < regions_.size(); ++i) {
        const auto& r = *regions_[i];
        std::string cls = classify_region(r.name());
        if (cls == "cortical")       cortical.push_back(i);
        else if (cls == "subcortical") subcortical.push_back(i);
        else if (cls == "limbic")      limbic.push_back(i);
        else if (cls == "neuromod")    neuromod.push_back(i);
        else                           other.push_back(i);
    }

    // 节点大小: 根据神经元数量缩放
    auto node_attrs = [&](size_t idx) -> std::string {
        const auto& r = *regions_[idx];
        float w = 0.3f + static_cast<float>(r.n_neurons()) * 0.01f;
        w = std::min(w, 1.5f);
        char buf[256];
        snprintf(buf, sizeof(buf),
            "    \"%s\" [label=\"%s\\n%zun\", width=%.2f, height=%.2f, fixedsize=true, shape=ellipse",
            r.name().c_str(), r.name().c_str(), r.n_neurons(), w, w * 0.7f);
        return std::string(buf);
    };

    // 皮层 subgraph
    dot += "  subgraph cluster_cortical {\n";
    dot += "    label=\"Cortical\"; fontcolor=\"#6699cc\"; color=\"#334466\";\n";
    for (size_t i : cortical) {
        dot += node_attrs(i) + ", fillcolor=\"#2a4a7f\", style=filled];\n";
    }
    dot += "  }\n\n";

    // 皮层下
    dot += "  subgraph cluster_subcortical {\n";
    dot += "    label=\"Subcortical\"; fontcolor=\"#66cc99\"; color=\"#336644\";\n";
    for (size_t i : subcortical) {
        dot += node_attrs(i) + ", fillcolor=\"#2a6f4f\", style=filled];\n";
    }
    dot += "  }\n\n";

    // 边缘
    dot += "  subgraph cluster_limbic {\n";
    dot += "    label=\"Limbic\"; fontcolor=\"#cc9966\"; color=\"#664433\";\n";
    for (size_t i : limbic) {
        dot += node_attrs(i) + ", fillcolor=\"#7f5a2a\", style=filled];\n";
    }
    dot += "  }\n\n";

    // 调质
    dot += "  subgraph cluster_neuromod {\n";
    dot += "    label=\"Neuromodulatory\"; fontcolor=\"#cc6666\"; color=\"#663333\";\n";
    for (size_t i : neuromod) {
        dot += node_attrs(i) + ", fillcolor=\"#7f2a2a\", style=filled, shape=diamond];\n";
    }
    dot += "  }\n\n";

    // 其他
    for (size_t i : other) {
        dot += node_attrs(i) + ", fillcolor=\"#555555\", style=filled];\n";
    }
    dot += "\n";

    // 投射边
    const auto& projs = bus_.projections();
    for (const auto& p : projs) {
        const std::string& src = bus_.region_name(p.src_region);
        const std::string& dst = bus_.region_name(p.dst_region);
        char buf[256];
        snprintf(buf, sizeof(buf), "  \"%s\" -> \"%s\" [label=\"d=%d\"];\n",
                 src.c_str(), dst.c_str(), p.delay);
        dot += buf;
    }

    dot += "}\n";
    return dot;
}

std::string SimulationEngine::export_topology_summary() const {
    std::string s;
    char buf[256];

    snprintf(buf, sizeof(buf),
        "=== Brain Topology (%zu regions, %zu projections) ===\n\n",
        regions_.size(), bus_.num_projections());
    s += buf;

    // 区域列表
    s += "  #   Name                Neurons  Type\n";
    s += "  --- ------------------- -------- -----------\n";
    for (size_t i = 0; i < regions_.size(); ++i) {
        const auto& r = *regions_[i];
        snprintf(buf, sizeof(buf), "  %3zu %-20s %5zu    %s\n",
                 i, r.name().c_str(), r.n_neurons(), classify_region(r.name()));
        s += buf;
    }

    s += "\n";

    // 投射列表
    s += "  #   Source -> Dest                Delay\n";
    s += "  --- ------------------------------ -----\n";
    const auto& projs = bus_.projections();
    for (size_t i = 0; i < projs.size(); ++i) {
        const auto& p = projs[i];
        std::string arrow = bus_.region_name(p.src_region) + " -> " + bus_.region_name(p.dst_region);
        snprintf(buf, sizeof(buf), "  %3zu %-30s %3d\n",
                 i, arrow.c_str(), p.delay);
        s += buf;
    }

    // 统计
    size_t total_neurons = 0;
    for (size_t i = 0; i < regions_.size(); ++i)
        total_neurons += regions_[i]->n_neurons();
    snprintf(buf, sizeof(buf), "\n  Total: %zu neurons, %zu projections\n",
             total_neurons, bus_.num_projections());
    s += buf;

    return s;
}

} // namespace wuyun
