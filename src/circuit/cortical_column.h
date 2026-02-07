#pragma once
/**
 * CorticalColumn - 6 layers cortical column template
 *
 * The core computational unit of neocortex.
 * Contains excitatory + inhibitory populations and intra-column synapses.
 *
 * Layer structure (02_neuron_system_design.md S6):
 *   L1:   Feedback target (axons only, no cell bodies here)
 *   L2/3: Prediction error output (regular) + match signal (burst)
 *   L4:   Feedforward input (from thalamus / lower areas)
 *   L5:   Drive output (burst only -> BG/brainstem)
 *   L6:   Prediction generation -> thalamus / lower L1
 *
 * Predictive coding flow:
 *   Feedforward -> L4 stellate -> L2/3 basal (I_basal)
 *   Feedback   -> L1 -> L2/3 & L5 apical (I_apical)
 *   L2/3 regular -> higher area L4  (prediction error)
 *   L5   burst   -> subcortical      (drive)
 *   L6   output  -> lower area L1    (prediction)
 *
 * Inhibitory microcircuit (attention gating):
 *   PV+  (basket)     -> soma      : direct inhibition
 *   SST+ (Martinotti)  -> apical    : blocks burst
 *   VIP               -> SST       : disinhibition = attention gate
 */

#include "core/types.h"
#include "core/population.h"
#include "core/synapse_group.h"
#include "plasticity/homeostatic.h"
#include <vector>
#include <cstddef>
#include <string>
#include <memory>

namespace wuyun {

// =============================================================================
// Column configuration
// =============================================================================

struct ColumnConfig {
    std::string name = "column";

    // --- Neuron counts per population ---
    size_t n_l4_stellate     = 100;   // L4 excitatory
    size_t n_l23_pyramidal   = 200;   // L2/3 excitatory (largest)
    size_t n_l5_pyramidal    = 100;   // L5 excitatory
    size_t n_l6_pyramidal    = 80;    // L6 excitatory

    size_t n_pv_basket       = 30;    // PV+ inhibitory (fast, soma-targeting)
    size_t n_sst_martinotti  = 20;    // SST+ inhibitory (apical-targeting)
    size_t n_vip             = 10;    // VIP inhibitory (disinhibition)

    // --- Excitatory connection probabilities ---
    float p_l4_to_l23        = 0.3f;  // L4 -> L2/3 basal
    float p_l23_to_l5        = 0.2f;  // L2/3 -> L5 basal
    float p_l5_to_l6         = 0.2f;  // L5 -> L6 basal
    float p_l6_to_l4         = 0.15f; // L6 -> L4 (internal prediction loop)
    float p_l23_recurrent    = 0.1f;  // L2/3 -> L2/3 lateral recurrent

    // --- Inhibitory connection probabilities ---
    float p_pv_to_l23        = 0.4f;  // PV -> L2/3 soma (GABA_A)
    float p_pv_to_l4         = 0.3f;  // PV -> L4 soma
    float p_pv_to_l5         = 0.3f;  // PV -> L5 soma
    float p_pv_to_l6         = 0.2f;  // PV -> L6 soma
    float p_sst_to_l23_api   = 0.3f;  // SST -> L2/3 apical (GABA_B)
    float p_sst_to_l5_api    = 0.3f;  // SST -> L5 apical (GABA_B)
    float p_vip_to_sst       = 0.5f;  // VIP -> SST (disinhibition)
    float p_exc_to_pv        = 0.3f;  // Excitatory -> PV
    float p_exc_to_sst       = 0.2f;  // Excitatory -> SST
    float p_exc_to_vip       = 0.15f; // Excitatory -> VIP

    // --- Initial synapse weights ---
    float w_exc              = 0.5f;  // Excitatory AMPA weight
    float w_nmda             = 0.3f;  // Excitatory NMDA weight (weaker)
    float w_inh              = 0.5f;  // Inhibitory weight
    float w_l6_to_l4         = 0.3f;  // Prediction loop (weaker initially)
    float w_recurrent        = 0.2f;  // L2/3 recurrent (weak)

    // --- Cross-region PSP input parameters ---
    float input_psp_regular  = 35.0f;  // PSP current per regular spike
    float input_psp_burst    = 55.0f;  // PSP current per burst spike
    float input_fan_out_frac = 0.3f;   // Fraction of L4 activated per spike

    // --- Cortical STDP (online learning) ---
    bool  stdp_enabled       = false;  // Enable STDP on excitatory synapses
    float stdp_a_plus        = 0.01f;  // LTP amplitude (standard cortical)
    float stdp_a_minus       = -0.012f;// LTD amplitude
    float stdp_tau           = 20.0f;  // Time window (ms)
    float stdp_w_max         = 1.5f;   // Max weight
};

// =============================================================================
// Column output struct
// =============================================================================

/** Aggregated column output after each step */
struct ColumnOutput {
    // L2/3 regular spikes -> prediction error (to higher area L4)
    std::vector<uint8_t> l23_regular;
    // L2/3 burst spikes -> match signal (learning/attention)
    std::vector<uint8_t> l23_burst;
    // L5 burst spikes -> drive (to subcortical)
    std::vector<uint8_t> l5_burst;
    // L6 output -> prediction (to lower area L1 / thalamus)
    std::vector<uint8_t> l6_fired;

    size_t n_regular = 0;
    size_t n_burst   = 0;
    size_t n_drive   = 0;
};

// =============================================================================
// CorticalColumn
// =============================================================================

class CorticalColumn {
public:
    CorticalColumn(const ColumnConfig& config);

    /**
     * Run one timestep of the column
     *
     * @param t   Current timestep
     * @param dt  Time delta (ms)
     * @return    Column output (prediction errors, bursts, drive)
     */
    ColumnOutput step(int t, float dt = 1.0f);

    /** Enable STDP on cortical excitatory synapses (called after construction) */
    void enable_stdp();
    bool has_stdp() const { return stdp_active_; }

    /** v26: ACh modulation of STDP learning rate (Froemke et al. 2007)
     *  Biology: NBM ACh release during salient events widens STDP window
     *  and enhances LTP, making reward-relevant features learned faster.
     *  gain=1.0 normal; >1.0 enhanced learning; <1.0 suppressed */
    void set_ach_stdp_gain(float gain) { ach_stdp_gain_ = gain; }
    float ach_stdp_gain() const { return ach_stdp_gain_; }

    /** Enable homeostatic plasticity (synaptic scaling on feedforward excitatory synapses) */
    void enable_homeostatic(const HomeostaticParams& params = {});
    bool has_homeostatic() const { return homeo_active_; }

    /** Mean firing rate of each excitatory population (for diagnostics) */
    float l4_mean_rate()  const { return homeo_l4_  ? homeo_l4_->mean_rate()  : 0.0f; }
    float l23_mean_rate() const { return homeo_l23_ ? homeo_l23_->mean_rate() : 0.0f; }
    float l5_mean_rate()  const { return homeo_l5_  ? homeo_l5_->mean_rate()  : 0.0f; }
    float l6_mean_rate()  const { return homeo_l6_  ? homeo_l6_->mean_rate()  : 0.0f; }

    // --- External input injection ---

    /** Feedforward input -> L4 stellate basal dendrites */
    void inject_feedforward(const std::vector<float>& currents);

    /** Feedback input -> L2/3 & L5 apical dendrites (via L1) */
    void inject_feedback(const std::vector<float>& currents_l23,
                         const std::vector<float>& currents_l5);

    /** VIP activation signal (attention gate from PFC) */
    void inject_attention(float vip_drive);

    // --- Accessors ---
    const std::string& name() const { return config_.name; }
    size_t total_neurons() const;
    size_t total_synapses() const;

    NeuronPopulation& l4()  { return l4_stellate_; }
    NeuronPopulation& l23() { return l23_pyramidal_; }
    NeuronPopulation& l5()  { return l5_pyramidal_; }
    NeuronPopulation& l6()  { return l6_pyramidal_; }

    const NeuronPopulation& l4()  const { return l4_stellate_; }
    const NeuronPopulation& l23() const { return l23_pyramidal_; }
    const NeuronPopulation& l5()  const { return l5_pyramidal_; }
    const NeuronPopulation& l6()  const { return l6_pyramidal_; }

private:
    void build_populations();
    void build_synapses();

    /** Deliver spikes from pre-population through synapse group to post-population */
    void deliver_and_inject(
        const NeuronPopulation& pre,
        SynapseGroup& syn,
        NeuronPopulation& post,
        float dt
    );

    /** Classify L2/3 and L5 output into regular/burst categories */
    void classify_output(ColumnOutput& out);

    ColumnConfig config_;

    // === Excitatory populations ===
    NeuronPopulation l4_stellate_;
    NeuronPopulation l23_pyramidal_;
    NeuronPopulation l5_pyramidal_;
    NeuronPopulation l6_pyramidal_;

    // === Inhibitory populations ===
    NeuronPopulation pv_basket_;
    NeuronPopulation sst_martinotti_;
    NeuronPopulation vip_;

    // === Excitatory AMPA synapses ===
    SynapseGroup syn_l4_to_l23_;      // L4 -> L2/3 basal (AMPA)
    SynapseGroup syn_l23_to_l5_;      // L2/3 -> L5 basal (AMPA)
    SynapseGroup syn_l5_to_l6_;       // L5 -> L6 basal (AMPA)
    SynapseGroup syn_l6_to_l4_;       // L6 -> L4 basal (AMPA, prediction loop)
    SynapseGroup syn_l23_recurrent_;  // L2/3 -> L2/3 lateral (AMPA)

    // === Excitatory NMDA synapses (parallel slow channel) ===
    SynapseGroup syn_l4_to_l23_nmda_; // L4 -> L2/3 (NMDA)
    SynapseGroup syn_l23_to_l5_nmda_; // L2/3 -> L5 (NMDA)
    SynapseGroup syn_l23_rec_nmda_;   // L2/3 recurrent (NMDA)

    // === Excitatory -> Inhibitory ===
    SynapseGroup syn_exc_to_pv_;      // L2/3 -> PV (AMPA)
    SynapseGroup syn_exc_to_sst_;     // L2/3 -> SST (AMPA)
    SynapseGroup syn_exc_to_vip_;     // L2/3 -> VIP (AMPA)

    // === PV -> all excitatory soma (GABA_A fast) ===
    SynapseGroup syn_pv_to_l23_;      // PV -> L2/3 soma
    SynapseGroup syn_pv_to_l4_;       // PV -> L4 soma
    SynapseGroup syn_pv_to_l5_;       // PV -> L5 soma
    SynapseGroup syn_pv_to_l6_;       // PV -> L6 soma

    // === SST -> apical (GABA_B slow, blocks burst!) ===
    SynapseGroup syn_sst_to_l23_api_; // SST -> L2/3 apical
    SynapseGroup syn_sst_to_l5_api_;  // SST -> L5 apical

    // === VIP -> SST (GABA_A, disinhibition) ===
    SynapseGroup syn_vip_to_sst_;     // VIP -> SST soma

    // === STDP state ===
    bool stdp_active_ = false;
    float ach_stdp_gain_ = 1.0f;  // v26: ACh modulation of STDP rate

    // === Homeostatic plasticity state ===
    bool homeo_active_ = false;
    uint32_t homeo_step_count_ = 0;
    uint32_t homeo_interval_ = 100;
    std::unique_ptr<SynapticScaler> homeo_l4_;
    std::unique_ptr<SynapticScaler> homeo_l23_;
    std::unique_ptr<SynapticScaler> homeo_l5_;
    std::unique_ptr<SynapticScaler> homeo_l6_;

    void apply_homeostatic_scaling();
};

} // namespace wuyun
