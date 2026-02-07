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
#include <vector>
#include <cstddef>
#include <string>

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

    // --- Connection probabilities (intra-column) ---
    float p_l4_to_l23        = 0.3f;  // L4 -> L2/3 basal
    float p_l23_to_l5        = 0.2f;  // L2/3 -> L5 basal
    float p_l5_to_l6         = 0.2f;  // L5 -> L6 basal
    float p_l6_to_l4         = 0.15f; // L6 -> L4 (internal prediction loop)

    // Inhibitory connections
    float p_pv_to_exc        = 0.4f;  // PV -> all excitatory (soma, GABA_A)
    float p_sst_to_apical    = 0.3f;  // SST -> L2/3 & L5 apical (GABA_B slow)
    float p_vip_to_sst       = 0.5f;  // VIP -> SST (disinhibition)
    float p_exc_to_pv        = 0.3f;  // Excitatory -> PV (drive inhibition)
    float p_exc_to_sst       = 0.2f;  // Excitatory -> SST
    float p_exc_to_vip       = 0.15f; // Excitatory -> VIP

    // --- Initial synapse weights ---
    float w_exc              = 0.5f;  // Excitatory synapse weight
    float w_inh              = 0.5f;  // Inhibitory synapse weight
    float w_l6_to_l4         = 0.3f;  // Prediction loop (weaker initially)
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

    // === Intra-column excitatory synapses ===
    SynapseGroup syn_l4_to_l23_;     // L4 -> L2/3 basal (AMPA)
    SynapseGroup syn_l23_to_l5_;     // L2/3 -> L5 basal (AMPA)
    SynapseGroup syn_l5_to_l6_;      // L5 -> L6 basal (AMPA)
    SynapseGroup syn_l6_to_l4_;      // L6 -> L4 basal (AMPA, prediction loop)

    // === Excitatory -> Inhibitory synapses ===
    SynapseGroup syn_exc_to_pv_;     // Excitatory -> PV (AMPA)
    SynapseGroup syn_exc_to_sst_;    // Excitatory -> SST (AMPA)
    SynapseGroup syn_exc_to_vip_;    // Excitatory -> VIP (AMPA)

    // === Inhibitory synapses ===
    SynapseGroup syn_pv_to_exc_;     // PV -> excitatory soma (GABA_A)
    SynapseGroup syn_sst_to_apical_; // SST -> L2/3 & L5 apical (GABA_B)
    SynapseGroup syn_vip_to_sst_;    // VIP -> SST (GABA_A, disinhibition)
};

} // namespace wuyun
