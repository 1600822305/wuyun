/**
 * pywuyun — Python bindings for WuYun brain simulation engine
 *
 * Exposes SimulationEngine, BrainRegion, SpikeBus, and all region types
 * to Python via pybind11. Enables interactive experimentation and visualization.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core/neuromodulator.h"
#include "engine/simulation_engine.h"
#include "region/cortical_region.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/neuromod/vta_da.h"
#include "region/neuromod/lc_ne.h"
#include "region/neuromod/drn_5ht.h"
#include "region/neuromod/nbm_ach.h"
#include "region/limbic/hippocampus.h"
#include "region/limbic/amygdala.h"
#include "region/limbic/septal_nucleus.h"
#include "region/limbic/mammillary_body.h"
#include "region/limbic/hypothalamus.h"
#include "engine/global_workspace.h"
#include "region/subcortical/cerebellum.h"

namespace py = pybind11;
using namespace wuyun;

// Helper: convert fired vector to numpy array
static py::array_t<uint8_t> fired_to_numpy(const std::vector<uint8_t>& fired) {
    return py::array_t<uint8_t>(
        {static_cast<py::ssize_t>(fired.size())},
        fired.data()
    );
}

// Helper: collect spike raster for a region over multiple steps
struct SpikeRecorder {
    std::vector<std::vector<uint32_t>> timesteps;  // per-step: list of neuron IDs that fired

    void record(const BrainRegion& region, int32_t t) {
        std::vector<uint32_t> spikes;
        const auto& f = region.fired();
        for (size_t i = 0; i < f.size(); ++i) {
            if (f[i]) spikes.push_back(static_cast<uint32_t>(i));
        }
        timesteps.push_back(std::move(spikes));
    }

    size_t total_spikes() const {
        size_t n = 0;
        for (const auto& ts : timesteps) n += ts.size();
        return n;
    }

    // Return (times, neuron_ids) numpy arrays for raster plot
    std::pair<py::array_t<int32_t>, py::array_t<uint32_t>> to_raster() const {
        size_t total = total_spikes();
        std::vector<int32_t> times;
        std::vector<uint32_t> neurons;
        times.reserve(total);
        neurons.reserve(total);

        for (size_t t = 0; t < timesteps.size(); ++t) {
            for (uint32_t nid : timesteps[t]) {
                times.push_back(static_cast<int32_t>(t));
                neurons.push_back(nid);
            }
        }

        return {
            py::array_t<int32_t>({static_cast<py::ssize_t>(times.size())}, times.data()),
            py::array_t<uint32_t>({static_cast<py::ssize_t>(neurons.size())}, neurons.data())
        };
    }

    void clear() { timesteps.clear(); }
};

PYBIND11_MODULE(pywuyun, m) {
    m.doc() = "WuYun brain simulation engine Python bindings";

    // =========================================================================
    // SpikeRecorder (helper for visualization)
    // =========================================================================
    py::class_<SpikeRecorder>(m, "SpikeRecorder",
        "Record spike raster data for visualization")
        .def(py::init<>())
        .def("record", &SpikeRecorder::record, "Record firing at current step",
             py::arg("region"), py::arg("t"))
        .def("total_spikes", &SpikeRecorder::total_spikes)
        .def("to_raster", &SpikeRecorder::to_raster,
             "Return (times, neuron_ids) numpy arrays")
        .def("clear", &SpikeRecorder::clear)
        .def("__len__", [](const SpikeRecorder& r) { return r.timesteps.size(); });

    // =========================================================================
    // NeuromodulatorLevels
    // =========================================================================
    py::class_<NeuromodulatorLevels>(m, "NeuromodulatorLevels")
        .def(py::init<>())
        .def_readwrite("da",  &NeuromodulatorLevels::da)
        .def_readwrite("ne",  &NeuromodulatorLevels::ne)
        .def_readwrite("sht", &NeuromodulatorLevels::sht)
        .def_readwrite("ach", &NeuromodulatorLevels::ach)
        .def("__repr__", [](const NeuromodulatorLevels& l) {
            char buf[128];
            snprintf(buf, sizeof(buf), "NeuromodulatorLevels(da=%.3f, ne=%.3f, sht=%.3f, ach=%.3f)",
                     l.da, l.ne, l.sht, l.ach);
            return std::string(buf);
        });

    // =========================================================================
    // NeuromodulatorSystem
    // =========================================================================
    py::class_<NeuromodulatorSystem>(m, "NeuromodulatorSystem")
        .def("set_tonic", &NeuromodulatorSystem::set_tonic)
        .def("current", &NeuromodulatorSystem::current)
        .def("compute_effect", &NeuromodulatorSystem::compute_effect);

    // =========================================================================
    // BrainRegion (base)
    // =========================================================================
    py::class_<BrainRegion>(m, "BrainRegion",
        "Brain region base class")
        .def("name", &BrainRegion::name)
        .def("n_neurons", &BrainRegion::n_neurons)
        .def("region_id", &BrainRegion::region_id)
        .def("fired", [](const BrainRegion& r) { return fired_to_numpy(r.fired()); },
             "Return fired state as numpy uint8 array")
        .def("spike_count", [](const BrainRegion& r) {
            size_t n = 0;
            for (auto f : r.fired()) if (f) n++;
            return n;
        }, "Number of neurons that fired this step")
        .def("inject_external", &BrainRegion::inject_external,
             "Inject external current", py::arg("currents"))
        .def("neuromod", static_cast<NeuromodulatorSystem& (BrainRegion::*)()>(&BrainRegion::neuromod),
             py::return_value_policy::reference);

    // =========================================================================
    // ColumnConfig
    // =========================================================================
    py::class_<ColumnConfig>(m, "ColumnConfig", "Cortical column config")
        .def(py::init<>())
        .def_readwrite("name", &ColumnConfig::name)
        .def_readwrite("n_l4_stellate", &ColumnConfig::n_l4_stellate)
        .def_readwrite("n_l23_pyramidal", &ColumnConfig::n_l23_pyramidal)
        .def_readwrite("n_l5_pyramidal", &ColumnConfig::n_l5_pyramidal)
        .def_readwrite("n_l6_pyramidal", &ColumnConfig::n_l6_pyramidal)
        .def_readwrite("n_pv_basket", &ColumnConfig::n_pv_basket)
        .def_readwrite("n_sst_martinotti", &ColumnConfig::n_sst_martinotti)
        .def_readwrite("n_vip", &ColumnConfig::n_vip)
        .def_readwrite("input_psp_regular", &ColumnConfig::input_psp_regular)
        .def_readwrite("input_psp_burst", &ColumnConfig::input_psp_burst)
        .def_readwrite("stdp_enabled", &ColumnConfig::stdp_enabled);

    // =========================================================================
    // CorticalRegion
    // =========================================================================
    py::class_<CorticalRegion, BrainRegion>(m, "CorticalRegion",
        "Cortical region (V1, dlPFC, M1, OFC, etc.)")
        .def(py::init<const std::string&, const ColumnConfig&>(),
             py::arg("name"), py::arg("config"))
        .def("enable_predictive_coding", &CorticalRegion::enable_predictive_coding)
        .def("add_feedback_source", &CorticalRegion::add_feedback_source)
        .def("prediction_error", &CorticalRegion::prediction_error)
        .def("precision_sensory", &CorticalRegion::precision_sensory)
        .def("precision_prior", &CorticalRegion::precision_prior)
        .def("predictive_coding_enabled", &CorticalRegion::predictive_coding_enabled)
        .def("enable_working_memory", &CorticalRegion::enable_working_memory)
        .def("working_memory_enabled", &CorticalRegion::working_memory_enabled)
        .def("wm_persistence", &CorticalRegion::wm_persistence)
        .def("wm_da_gain", &CorticalRegion::wm_da_gain)
        .def("set_attention_gain", &CorticalRegion::set_attention_gain,
             py::arg("gain"), "Set top-down attention gain (1.0=normal, >1.0=attend, <1.0=ignore)")
        .def("attention_gain", &CorticalRegion::attention_gain);

    // =========================================================================
    // ThalamicConfig + ThalamicRelay
    // =========================================================================
    py::class_<ThalamicConfig>(m, "ThalamicConfig")
        .def(py::init<>())
        .def_readwrite("name", &ThalamicConfig::name)
        .def_readwrite("n_relay", &ThalamicConfig::n_relay)
        .def_readwrite("n_trn", &ThalamicConfig::n_trn);

    py::class_<ThalamicRelay, BrainRegion>(m, "ThalamicRelay", "Thalamic relay")
        .def(py::init<const ThalamicConfig&>(), py::arg("config"))
        .def("inject_external", &ThalamicRelay::inject_external);

    // =========================================================================
    // BasalGangliaConfig + BasalGanglia
    // =========================================================================
    py::class_<BasalGangliaConfig>(m, "BasalGangliaConfig")
        .def(py::init<>())
        .def_readwrite("name", &BasalGangliaConfig::name)
        .def_readwrite("n_d1_msn", &BasalGangliaConfig::n_d1_msn)
        .def_readwrite("n_d2_msn", &BasalGangliaConfig::n_d2_msn)
        .def_readwrite("n_gpi", &BasalGangliaConfig::n_gpi)
        .def_readwrite("n_gpe", &BasalGangliaConfig::n_gpe)
        .def_readwrite("n_stn", &BasalGangliaConfig::n_stn);

    py::class_<BasalGanglia, BrainRegion>(m, "BasalGanglia", "Basal ganglia")
        .def(py::init<const BasalGangliaConfig&>(), py::arg("config"))
        .def("set_da_level", &BasalGanglia::set_da_level)
        .def("set_da_source_region", &BasalGanglia::set_da_source_region);

    // =========================================================================
    // VTA_DA
    // =========================================================================
    py::class_<VTAConfig>(m, "VTAConfig")
        .def(py::init<>())
        .def_readwrite("name", &VTAConfig::name);

    py::class_<VTA_DA, BrainRegion>(m, "VTA_DA", "VTA dopamine region")
        .def(py::init<const VTAConfig&>(), py::arg("config"));

    // =========================================================================
    // LC_NE / DRN_5HT / NBM_ACh
    // =========================================================================
    py::class_<LCConfig>(m, "LCConfig").def(py::init<>());
    py::class_<LC_NE, BrainRegion>(m, "LC_NE", "Locus coeruleus NE")
        .def(py::init<const LCConfig&>(), py::arg("config"))
        .def("ne_output", &LC_NE::ne_output)
        .def("inject_arousal", &LC_NE::inject_arousal);

    py::class_<DRNConfig>(m, "DRNConfig").def(py::init<>());
    py::class_<DRN_5HT, BrainRegion>(m, "DRN_5HT", "Dorsal raphe 5-HT")
        .def(py::init<const DRNConfig&>(), py::arg("config"))
        .def("sht_output", &DRN_5HT::sht_output);

    py::class_<NBMConfig>(m, "NBMConfig").def(py::init<>());
    py::class_<NBM_ACh, BrainRegion>(m, "NBM_ACh", "Basal forebrain ACh")
        .def(py::init<const NBMConfig&>(), py::arg("config"))
        .def("ach_output", &NBM_ACh::ach_output);

    // =========================================================================
    // Hippocampus
    // =========================================================================
    py::class_<HippocampusConfig>(m, "HippocampusConfig")
        .def(py::init<>())
        .def_readwrite("name", &HippocampusConfig::name);

    py::class_<Hippocampus, BrainRegion>(m, "Hippocampus", "Hippocampus")
        .def(py::init<const HippocampusConfig&>(), py::arg("config"));

    // =========================================================================
    // Amygdala
    // =========================================================================
    py::class_<AmygdalaConfig>(m, "AmygdalaConfig")
        .def(py::init<>())
        .def_readwrite("name", &AmygdalaConfig::name);

    py::class_<Amygdala, BrainRegion>(m, "Amygdala", "Amygdala")
        .def(py::init<const AmygdalaConfig&>(), py::arg("config"))
        .def("set_pfc_source_region", &Amygdala::set_pfc_source_region);

    // =========================================================================
    // SeptalNucleus
    // =========================================================================
    py::class_<SeptalConfig>(m, "SeptalConfig")
        .def(py::init<>())
        .def_readwrite("name", &SeptalConfig::name);

    py::class_<SeptalNucleus, BrainRegion>(m, "SeptalNucleus", "Septal nucleus theta pacemaker")
        .def(py::init<const SeptalConfig&>(), py::arg("config"))
        .def("ach_output", &SeptalNucleus::ach_output)
        .def("theta_phase", &SeptalNucleus::theta_phase);

    // =========================================================================
    // MammillaryBody
    // =========================================================================
    py::class_<MammillaryConfig>(m, "MammillaryConfig")
        .def(py::init<>())
        .def_readwrite("name", &MammillaryConfig::name);

    py::class_<MammillaryBody, BrainRegion>(m, "MammillaryBody", "Mammillary body (Papez circuit)")
        .def(py::init<const MammillaryConfig&>(), py::arg("config"));

    // =========================================================================
    // Cerebellum
    // =========================================================================
    py::class_<CerebellumConfig>(m, "CerebellumConfig")
        .def(py::init<>())
        .def_readwrite("name", &CerebellumConfig::name);

    py::class_<Cerebellum, BrainRegion>(m, "Cerebellum", "Cerebellum")
        .def(py::init<const CerebellumConfig&>(), py::arg("config"))
        .def("inject_mossy_fiber", &Cerebellum::inject_mossy_fiber)
        .def("inject_climbing_fiber", &Cerebellum::inject_climbing_fiber);

    // =========================================================================
    // Hypothalamus
    // =========================================================================
    py::class_<HypothalamusConfig>(m, "HypothalamusConfig")
        .def(py::init<>())
        .def_readwrite("name", &HypothalamusConfig::name)
        .def_readwrite("n_scn", &HypothalamusConfig::n_scn)
        .def_readwrite("n_vlpo", &HypothalamusConfig::n_vlpo)
        .def_readwrite("n_orexin", &HypothalamusConfig::n_orexin)
        .def_readwrite("n_pvn", &HypothalamusConfig::n_pvn)
        .def_readwrite("n_lh", &HypothalamusConfig::n_lh)
        .def_readwrite("n_vmh", &HypothalamusConfig::n_vmh)
        .def_readwrite("circadian_period", &HypothalamusConfig::circadian_period)
        .def_readwrite("homeostatic_sleep_pressure", &HypothalamusConfig::homeostatic_sleep_pressure)
        .def_readwrite("stress_level", &HypothalamusConfig::stress_level)
        .def_readwrite("hunger_level", &HypothalamusConfig::hunger_level)
        .def_readwrite("satiety_level", &HypothalamusConfig::satiety_level);

    py::class_<Hypothalamus, BrainRegion>(m, "Hypothalamus", "Hypothalamus internal drive system")
        .def(py::init<const HypothalamusConfig&>(), py::arg("config"))
        .def("wake_level", &Hypothalamus::wake_level)
        .def("circadian_phase", &Hypothalamus::circadian_phase)
        .def("is_sleeping", &Hypothalamus::is_sleeping)
        .def("stress_output", &Hypothalamus::stress_output)
        .def("hunger_output", &Hypothalamus::hunger_output)
        .def("satiety_output", &Hypothalamus::satiety_output)
        .def("set_sleep_pressure", &Hypothalamus::set_sleep_pressure)
        .def("set_stress_level", &Hypothalamus::set_stress_level)
        .def("set_hunger_level", &Hypothalamus::set_hunger_level)
        .def("set_satiety_level", &Hypothalamus::set_satiety_level);

    // =========================================================================
    // GlobalWorkspace
    // =========================================================================
    py::class_<GWConfig>(m, "GWConfig")
        .def(py::init<>())
        .def_readwrite("name", &GWConfig::name)
        .def_readwrite("n_workspace", &GWConfig::n_workspace)
        .def_readwrite("ignition_threshold", &GWConfig::ignition_threshold)
        .def_readwrite("competition_decay", &GWConfig::competition_decay)
        .def_readwrite("min_ignition_gap", &GWConfig::min_ignition_gap)
        .def_readwrite("broadcast_gain", &GWConfig::broadcast_gain)
        .def_readwrite("broadcast_duration", &GWConfig::broadcast_duration);

    py::class_<GlobalWorkspace, BrainRegion>(m, "GlobalWorkspace",
        "Global Workspace Theory (Baars/Dehaene) - consciousness via competition/ignition/broadcast")
        .def(py::init<const GWConfig&>(), py::arg("config"))
        .def("is_ignited", &GlobalWorkspace::is_ignited)
        .def("conscious_content_id", &GlobalWorkspace::conscious_content_id)
        .def("conscious_content_name", &GlobalWorkspace::conscious_content_name)
        .def("ignition_count", &GlobalWorkspace::ignition_count)
        .def("broadcast_remaining", &GlobalWorkspace::broadcast_remaining)
        .def("winning_salience", &GlobalWorkspace::winning_salience)
        .def("register_source", &GlobalWorkspace::register_source,
             py::arg("region_id"), py::arg("name"));

    // =========================================================================
    // SpikeBus
    // =========================================================================
    py::class_<SpikeBus>(m, "SpikeBus")
        .def("num_projections", &SpikeBus::num_projections);

    // =========================================================================
    // SimulationEngine
    // =========================================================================
    py::class_<SimulationEngine>(m, "SimulationEngine",
        "Simulation engine - global clock + SpikeBus orchestration")
        .def(py::init<int32_t>(), py::arg("max_delay") = 10)
        .def("add_cortical", [](SimulationEngine& eng, const std::string& name,
                                const ColumnConfig& cfg) -> CorticalRegion* {
            eng.add_region(std::make_unique<CorticalRegion>(name, cfg));
            return dynamic_cast<CorticalRegion*>(eng.find_region(name));
        }, py::return_value_policy::reference, "Add cortical region and return reference")
        .def("add_thalamic", [](SimulationEngine& eng, const ThalamicConfig& cfg) -> ThalamicRelay* {
            eng.add_region(std::make_unique<ThalamicRelay>(cfg));
            return dynamic_cast<ThalamicRelay*>(eng.find_region(cfg.name));
        }, py::return_value_policy::reference)
        .def("add_basal_ganglia", [](SimulationEngine& eng, const BasalGangliaConfig& cfg) -> BasalGanglia* {
            eng.add_region(std::make_unique<BasalGanglia>(cfg));
            return dynamic_cast<BasalGanglia*>(eng.find_region(cfg.name));
        }, py::return_value_policy::reference)
        .def("add_vta", [](SimulationEngine& eng, const VTAConfig& cfg) -> VTA_DA* {
            eng.add_region(std::make_unique<VTA_DA>(cfg));
            return dynamic_cast<VTA_DA*>(eng.find_region(cfg.name));
        }, py::return_value_policy::reference)
        .def("add_lc", [](SimulationEngine& eng, const LCConfig& cfg) -> LC_NE* {
            eng.add_region(std::make_unique<LC_NE>(cfg));
            return dynamic_cast<LC_NE*>(eng.find_region("LC"));
        }, py::return_value_policy::reference)
        .def("add_drn", [](SimulationEngine& eng, const DRNConfig& cfg) -> DRN_5HT* {
            eng.add_region(std::make_unique<DRN_5HT>(cfg));
            return dynamic_cast<DRN_5HT*>(eng.find_region("DRN"));
        }, py::return_value_policy::reference)
        .def("add_nbm", [](SimulationEngine& eng, const NBMConfig& cfg) -> NBM_ACh* {
            eng.add_region(std::make_unique<NBM_ACh>(cfg));
            return dynamic_cast<NBM_ACh*>(eng.find_region("NBM"));
        }, py::return_value_policy::reference)
        .def("add_hippocampus", [](SimulationEngine& eng, const HippocampusConfig& cfg) -> Hippocampus* {
            eng.add_region(std::make_unique<Hippocampus>(cfg));
            return dynamic_cast<Hippocampus*>(eng.find_region(cfg.name));
        }, py::return_value_policy::reference)
        .def("add_amygdala", [](SimulationEngine& eng, const AmygdalaConfig& cfg) -> Amygdala* {
            eng.add_region(std::make_unique<Amygdala>(cfg));
            return dynamic_cast<Amygdala*>(eng.find_region(cfg.name));
        }, py::return_value_policy::reference)
        .def("add_cerebellum", [](SimulationEngine& eng, const CerebellumConfig& cfg) -> Cerebellum* {
            eng.add_region(std::make_unique<Cerebellum>(cfg));
            return dynamic_cast<Cerebellum*>(eng.find_region(cfg.name));
        }, py::return_value_policy::reference)
        .def("add_hypothalamus", [](SimulationEngine& eng, const HypothalamusConfig& cfg) -> Hypothalamus* {
            eng.add_region(std::make_unique<Hypothalamus>(cfg));
            return dynamic_cast<Hypothalamus*>(eng.find_region(cfg.name));
        }, py::return_value_policy::reference)
        .def("add_global_workspace", [](SimulationEngine& eng, const GWConfig& cfg) -> GlobalWorkspace* {
            eng.add_region(std::make_unique<GlobalWorkspace>(cfg));
            return dynamic_cast<GlobalWorkspace*>(eng.find_region(cfg.name));
        }, py::return_value_policy::reference)
        .def("add_projection", &SimulationEngine::add_projection,
             py::arg("src"), py::arg("dst"), py::arg("delay"),
             py::arg("proj_name") = std::string(""))
        .def("step", &SimulationEngine::step, py::arg("dt") = 1.0f)
        .def("run", &SimulationEngine::run, py::arg("steps"), py::arg("dt") = 1.0f)
        .def("current_time", &SimulationEngine::current_time)
        .def("num_regions", &SimulationEngine::num_regions)
        .def("find_region", &SimulationEngine::find_region,
             py::return_value_policy::reference, py::arg("name"))
        .def("region", [](SimulationEngine& eng, size_t idx) -> BrainRegion& {
            return eng.region(idx);
        }, py::return_value_policy::reference)
        .def("bus", &SimulationEngine::bus, py::return_value_policy::reference)
        .def("register_neuromod_source", [](SimulationEngine& eng,
                const std::string& name, int type) {
            eng.register_neuromod_source(name,
                static_cast<SimulationEngine::NeuromodType>(type));
        }, py::arg("name"), py::arg("type"),
           "Register neuromod source (type: 0=DA, 1=NE, 2=5HT, 3=ACh)")
        // Convenience: build standard 21-region brain
        .def("build_standard_brain", [](SimulationEngine& eng) {
            // LGN
            ThalamicConfig lgn_cfg;
            lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
            eng.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

            auto add_ctx = [&](const std::string& name, size_t l4, size_t l23,
                               size_t l5, size_t l6, size_t pv, size_t sst, size_t vip) {
                ColumnConfig c;
                c.n_l4_stellate = l4; c.n_l23_pyramidal = l23;
                c.n_l5_pyramidal = l5; c.n_l6_pyramidal = l6;
                c.n_pv_basket = pv; c.n_sst_martinotti = sst; c.n_vip = vip;
                eng.add_region(std::make_unique<CorticalRegion>(name, c));
            };

            // === Visual cortex (ventral what + dorsal where) ===
            add_ctx("V1",    50, 100, 50, 40, 15, 10, 5);  // Primary visual
            add_ctx("V2",    40,  80, 40, 30, 12,  8, 4);  // Secondary visual
            add_ctx("V4",    30,  60, 30, 25, 10,  6, 3);  // Color/form
            add_ctx("IT",    20,  50, 25, 20,  8,  5, 2);  // Object recognition
            add_ctx("MT",    35,  70, 35, 25, 10,  7, 3);  // Motion
            add_ctx("PPC",   30,  65, 35, 25, 10,  6, 3);  // Spatial/action

            // === Somatosensory cortex ===
            add_ctx("S1",    40,  80, 40, 30, 12,  8, 4);  // Primary somatosensory
            add_ctx("S2",    25,  50, 25, 20,  8,  5, 2);  // Secondary somatosensory

            // === Auditory cortex ===
            add_ctx("A1",    35,  70, 35, 25, 10,  7, 3);  // Primary auditory

            // === Chemical senses ===
            add_ctx("Gustatory", 15, 35, 18, 12, 5, 3, 2); // Taste (anterior insula)
            add_ctx("Piriform",  15, 35, 18, 12, 5, 3, 2); // Olfactory cortex

            // === Prefrontal / Decision ===
            add_ctx("OFC",   25,  60, 30, 20,  8,  5, 3);  // Orbitofrontal (value)
            add_ctx("vmPFC", 20,  55, 30, 20,  8,  5, 2);  // Ventromedial PFC
            add_ctx("ACC",   20,  50, 30, 20,  8,  5, 2);  // Anterior cingulate
            add_ctx("dlPFC", 30,  80, 40, 30, 10,  8, 4);  // Dorsolateral PFC
            add_ctx("FEF",   20,  45, 25, 18,  7,  4, 2);  // Frontal eye fields

            // === Motor cortex ===
            add_ctx("PMC",   25,  55, 35, 20,  8,  5, 3);  // Premotor cortex
            add_ctx("SMA",   20,  45, 30, 18,  7,  4, 2);  // Supplementary motor
            add_ctx("M1",    30,  60, 40, 20, 10,  6, 3);  // Primary motor

            // === Association cortex ===
            add_ctx("PCC",   18,  45, 25, 18,  6,  4, 2);  // Posterior cingulate
            add_ctx("Insula",20,  50, 25, 18,  8,  5, 2);  // Interoception
            add_ctx("TPJ",   20,  50, 25, 18,  7,  5, 2);  // Theory of mind
            add_ctx("Broca", 20,  50, 30, 20,  8,  5, 2);  // Speech production
            add_ctx("Wernicke",18, 45, 25, 18, 7,  4, 2);  // Speech comprehension

            BasalGangliaConfig bg;
            bg.name = "BG"; bg.n_d1_msn = 50; bg.n_d2_msn = 50;
            bg.n_gpi = 15; bg.n_gpe = 15; bg.n_stn = 10;
            eng.add_region(std::make_unique<BasalGanglia>(bg));

            // === Thalamic nuclei ===
            auto add_thal = [&](const std::string& name, size_t relay, size_t trn) {
                ThalamicConfig tc;
                tc.name = name; tc.n_relay = relay; tc.n_trn = trn;
                eng.add_region(std::make_unique<ThalamicRelay>(tc));
            };
            add_thal("MotorThal", 30, 10);   // VA/VL motor relay
            add_thal("VPL",  25,  8);         // Somatosensory relay (body)
            add_thal("MGN",  20,  6);         // Auditory relay
            add_thal("MD",   25,  8);         // Mediodorsal → PFC
            add_thal("VA",   20,  6);         // Ventral anterior → motor planning
            add_thal("LP",   18,  6);         // Lateral posterior → PPC
            add_thal("LD",   15,  5);         // Lateral dorsal → cingulate/hipp
            add_thal("Pulvinar", 30, 10);     // Visual attention/association
            add_thal("CeM",  15,  5);         // Centromedian → arousal
            add_thal("ILN",  12,  4);         // Intralaminar (CL/CM/Pf) → consciousness

            eng.add_region(std::make_unique<VTA_DA>(VTAConfig{}));

            // Hippocampus with Presubiculum + HATA
            HippocampusConfig hipp_cfg;
            hipp_cfg.n_presub = 25;  // Head direction cells
            hipp_cfg.n_hata   = 15;  // Hipp-Amyg transition
            eng.add_region(std::make_unique<Hippocampus>(hipp_cfg));

            // Amygdala with MeA/CoA/AB
            AmygdalaConfig amyg_cfg;
            amyg_cfg.n_mea = 20;  // Social/olfactory
            amyg_cfg.n_coa = 15;  // Olfactory
            amyg_cfg.n_ab  = 20;  // Multimodal
            eng.add_region(std::make_unique<Amygdala>(amyg_cfg));

            eng.add_region(std::make_unique<Cerebellum>(CerebellumConfig{}));
            eng.add_region(std::make_unique<LC_NE>(LCConfig{}));
            eng.add_region(std::make_unique<DRN_5HT>(DRNConfig{}));
            eng.add_region(std::make_unique<NBM_ACh>(NBMConfig{}));

            // Septal Nucleus (theta pacemaker)
            eng.add_region(std::make_unique<SeptalNucleus>(SeptalConfig{}));

            // Mammillary Body (Papez circuit relay)
            eng.add_region(std::make_unique<MammillaryBody>(MammillaryConfig{}));

            // Anterior Thalamic Nucleus (Papez circuit)
            add_thal("ATN", 20, 8);

            // Hypothalamus (internal drive system)
            eng.add_region(std::make_unique<Hypothalamus>(HypothalamusConfig{}));

            // Global Workspace (consciousness)
            eng.add_region(std::make_unique<GlobalWorkspace>(GWConfig{}));

            // ============================================================
            // PROJECTIONS (~90 anatomical connections)
            // ============================================================

            // --- Visual hierarchy (ventral what) ---
            eng.add_projection("LGN", "V1", 2);
            eng.add_projection("V1", "V2", 2);
            eng.add_projection("V2", "V4", 2);
            eng.add_projection("V4", "IT", 2);
            eng.add_projection("V2", "V1", 3);   // feedback
            eng.add_projection("V4", "V2", 3);
            eng.add_projection("IT", "V4", 3);

            // --- Visual hierarchy (dorsal where) ---
            eng.add_projection("V1", "MT", 2);
            eng.add_projection("V2", "MT", 2);
            eng.add_projection("MT", "PPC", 2);
            eng.add_projection("PPC", "MT", 3);
            eng.add_projection("PPC", "IT", 3);   // dorsal→ventral
            eng.add_projection("IT", "PPC", 3);   // ventral→dorsal
            eng.add_projection("MT", "FEF", 2);   // motion→saccade
            eng.add_projection("FEF", "V4", 3);   // attention feedback
            eng.add_projection("FEF", "MT", 3);

            // --- Pulvinar visual attention hub ---
            eng.add_projection("V1", "Pulvinar", 2);
            eng.add_projection("Pulvinar", "V2", 2);
            eng.add_projection("Pulvinar", "V4", 2);
            eng.add_projection("Pulvinar", "MT", 2);
            eng.add_projection("Pulvinar", "PPC", 2);
            eng.add_projection("FEF", "Pulvinar", 2); // top-down attention

            // --- Somatosensory ---
            eng.add_projection("VPL", "S1", 2);    // thalamocortical
            eng.add_projection("S1", "S2", 2);
            eng.add_projection("S2", "S1", 3);     // feedback
            eng.add_projection("S1", "M1", 2);     // sensorimotor
            eng.add_projection("S2", "PPC", 2);    // multimodal
            eng.add_projection("S1", "Insula", 2); // interoception

            // --- Auditory ---
            eng.add_projection("MGN", "A1", 2);    // thalamocortical
            eng.add_projection("A1", "Wernicke", 2); // speech comprehension
            eng.add_projection("A1", "TPJ", 2);    // social/voice

            // --- Chemical senses ---
            eng.add_projection("Gustatory", "Insula", 2);  // taste→interoception
            eng.add_projection("Gustatory", "OFC", 2);     // taste→value
            eng.add_projection("Piriform", "Amygdala", 2); // smell→emotion
            eng.add_projection("Piriform", "OFC", 2);      // smell→value
            eng.add_projection("Piriform", "Hippocampus", 2); // smell→memory

            // --- Prefrontal / Decision ---
            eng.add_projection("IT", "OFC", 3);
            eng.add_projection("OFC", "vmPFC", 2);
            eng.add_projection("vmPFC", "BG", 2);
            eng.add_projection("vmPFC", "Amygdala", 3);
            eng.add_projection("ACC", "dlPFC", 2);
            eng.add_projection("ACC", "LC", 2);     // conflict→arousal
            eng.add_projection("dlPFC", "ACC", 2);
            eng.add_projection("IT", "dlPFC", 3);
            eng.add_projection("PPC", "dlPFC", 3);
            eng.add_projection("dlPFC", "FEF", 2);  // executive→saccade
            eng.add_projection("Insula", "ACC", 2); // interoception→conflict
            eng.add_projection("Insula", "Amygdala", 2); // interoception→emotion
            eng.add_projection("OFC", "Insula", 2); // value→interoception

            // --- MD thalamus → PFC reciprocal ---
            eng.add_projection("MD", "dlPFC", 2);
            eng.add_projection("MD", "OFC", 2);
            eng.add_projection("MD", "ACC", 2);
            eng.add_projection("dlPFC", "MD", 3);

            // --- Motor hierarchy ---
            eng.add_projection("PPC", "PMC", 2);    // spatial→premotor
            eng.add_projection("dlPFC", "PMC", 2);  // executive→premotor
            eng.add_projection("PMC", "M1", 2);     // premotor→primary
            eng.add_projection("SMA", "M1", 2);     // supplementary→primary
            eng.add_projection("SMA", "PMC", 2);    // SMA→PMC
            eng.add_projection("dlPFC", "SMA", 2);  // executive→SMA
            eng.add_projection("BG", "VA", 2);      // BG→VA motor planning
            eng.add_projection("VA", "PMC", 2);     // VA→premotor
            eng.add_projection("VA", "SMA", 2);     // VA→SMA
            eng.add_projection("dlPFC", "BG", 2);
            eng.add_projection("BG", "MotorThal", 2);
            eng.add_projection("MotorThal", "M1", 2);
            eng.add_projection("M1", "Cerebellum", 2);
            eng.add_projection("Cerebellum", "MotorThal", 2);
            eng.add_projection("PPC", "M1", 3);     // visuomotor

            // --- Language ---
            eng.add_projection("Wernicke", "Broca", 2); // arcuate fasciculus
            eng.add_projection("Broca", "PMC", 2);      // speech→motor
            eng.add_projection("Broca", "dlPFC", 2);    // syntax→executive
            eng.add_projection("Wernicke", "TPJ", 2);   // comprehension→social
            eng.add_projection("Wernicke", "IT", 3);    // semantic
            eng.add_projection("dlPFC", "Broca", 2);    // executive→speech

            // --- Default mode / Social ---
            eng.add_projection("PCC", "vmPFC", 2);  // DMN core
            eng.add_projection("vmPFC", "PCC", 2);
            eng.add_projection("PCC", "Hippocampus", 2); // episodic memory
            eng.add_projection("TPJ", "PCC", 2);   // social→DMN
            eng.add_projection("PCC", "TPJ", 2);
            eng.add_projection("TPJ", "dlPFC", 2);  // social→executive

            // --- LP / LD thalamic connections ---
            eng.add_projection("LP", "PPC", 2);     // association→parietal
            eng.add_projection("PPC", "LP", 3);
            eng.add_projection("LD", "PCC", 2);     // limbic→cingulate
            eng.add_projection("LD", "Hippocampus", 2);

            // --- CeM / ILN arousal/consciousness ---
            eng.add_projection("CeM", "BG", 2);    // arousal→striatum
            eng.add_projection("CeM", "ACC", 2);   // arousal→conflict
            eng.add_projection("ILN", "dlPFC", 2);  // consciousness→PFC
            eng.add_projection("ILN", "ACC", 2);    // consciousness→ACC
            eng.add_projection("ACC", "CeM", 2);   // salience→arousal

            // --- Limbic connections (existing + new) ---
            eng.add_projection("V1", "Amygdala", 2);
            eng.add_projection("dlPFC", "Amygdala", 2);
            eng.add_projection("Amygdala", "OFC", 2);
            eng.add_projection("dlPFC", "Hippocampus", 3);
            eng.add_projection("Hippocampus", "dlPFC", 3);
            eng.add_projection("Amygdala", "VTA", 2);
            eng.add_projection("Amygdala", "Hippocampus", 2);
            eng.add_projection("Amygdala", "Insula", 2); // emotion→interoception
            eng.add_projection("VTA", "BG", 1);

            // --- Papez circuit ---
            eng.add_projection("Hippocampus", "MammillaryBody", 2);
            eng.add_projection("MammillaryBody", "ATN", 2);
            eng.add_projection("ATN", "ACC", 2);

            // --- Septal → Hippocampus ---
            eng.add_projection("SeptalNucleus", "Hippocampus", 1);

            // --- Global Workspace broadcast ---
            eng.add_projection("V1", "GW", 2);     // Visual→GW competition
            eng.add_projection("IT", "GW", 2);     // Object→GW
            eng.add_projection("PPC", "GW", 2);    // Spatial→GW
            eng.add_projection("dlPFC", "GW", 2);  // Executive→GW
            eng.add_projection("ACC", "GW", 2);    // Conflict→GW
            eng.add_projection("OFC", "GW", 2);    // Value→GW
            eng.add_projection("Insula", "GW", 2); // Interoception→GW
            eng.add_projection("A1", "GW", 2);     // Auditory→GW
            eng.add_projection("S1", "GW", 2);     // Somatosensory→GW
            eng.add_projection("GW", "ILN", 1);    // GW→ILN broadcast hub
            eng.add_projection("GW", "CeM", 1);    // GW→CeM arousal

            // --- Hypothalamus drives ---
            eng.add_projection("Hypothalamus", "LC", 2);    // Orexin→LC (wake→arousal)
            eng.add_projection("Hypothalamus", "DRN", 2);   // Orexin→DRN (wake→serotonin)
            eng.add_projection("Hypothalamus", "NBM", 2);   // Orexin→NBM (wake→ACh)
            eng.add_projection("Hypothalamus", "VTA", 2);   // LH→VTA (hunger→motivation)
            eng.add_projection("Hypothalamus", "Amygdala", 2); // PVN→CeA (stress→fear)
            eng.add_projection("Amygdala", "Hypothalamus", 2); // CeA→PVN (fear→stress)
            eng.add_projection("Insula", "Hypothalamus", 2);   // Interoception→drives
            eng.add_projection("Hypothalamus", "ACC", 2);   // Drive signals→conflict

            // Register GW source names for readable conscious content
            auto* gw_ptr = dynamic_cast<GlobalWorkspace*>(eng.find_region("GW"));
            if (gw_ptr) {
                const char* gw_sources[] = {"V1","IT","PPC","dlPFC","ACC","OFC","Insula","A1","S1"};
                for (auto name : gw_sources) {
                    auto* r = eng.find_region(name);
                    if (r) gw_ptr->register_source(r->region_id(), name);
                }
            }

            // Neuromod sources
            using NM = SimulationEngine::NeuromodType;
            eng.register_neuromod_source("VTA", NM::DA);
            eng.register_neuromod_source("LC",  NM::NE);
            eng.register_neuromod_source("DRN", NM::SHT);
            eng.register_neuromod_source("NBM", NM::ACh);

            // Wire DA/PFC sources
            auto* bg_ptr = dynamic_cast<BasalGanglia*>(eng.find_region("BG"));
            auto* vta_ptr = eng.find_region("VTA");
            if (bg_ptr && vta_ptr) bg_ptr->set_da_source_region(vta_ptr->region_id());
            auto* amyg_ptr = dynamic_cast<Amygdala*>(eng.find_region("Amygdala"));
            auto* pfc_ptr = eng.find_region("dlPFC");
            if (amyg_ptr && pfc_ptr) amyg_ptr->set_pfc_source_region(pfc_ptr->region_id());
        }, "Build standard brain with all regions, projections and neuromodulators");

    // =========================================================================
    // NeuromodType enum
    // =========================================================================
    py::enum_<SimulationEngine::NeuromodType>(m, "NeuromodType")
        .value("DA",  SimulationEngine::NeuromodType::DA)
        .value("NE",  SimulationEngine::NeuromodType::NE)
        .value("SHT", SimulationEngine::NeuromodType::SHT)
        .value("ACh", SimulationEngine::NeuromodType::ACh);

    // =========================================================================
    // Module-level convenience
    // =========================================================================
    m.def("version", []() { return "0.4.0"; });
}
