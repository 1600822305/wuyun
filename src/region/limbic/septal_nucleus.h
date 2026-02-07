#pragma once
/**
 * SeptalNucleus — 隔核 (内侧隔核-斜角带复合体, MS-DBB)
 *
 * 海马 theta 节律的起搏器:
 *   - 胆碱能神经元 → Hippocampus (ACh 调制)
 *   - GABA能节律神经元 → Hippocampus 抑制性中间神经元 (theta 起搏)
 *
 * Theta 节律 (~4-8 Hz):
 *   GABA 神经元以 theta 频率发放 (每 125-250ms 一个 burst)
 *   → 驱动海马 DG/CA3/CA1 basket cells
 *   → 通过 E-I 循环产生 theta 震荡
 *
 * 生物学参考:
 *   - Buzsáki 2002: Theta oscillations in the hippocampus
 *   - Stewart & Fox 1990: Septal pacemaker of hippocampal theta
 */

#include "region/brain_region.h"
#include "core/population.h"
#include "core/synapse_group.h"

namespace wuyun {

struct SeptalConfig {
    std::string name = "SeptalNucleus";

    size_t n_ach   = 15;   // 胆碱能神经元 (→Hipp ACh)
    size_t n_gaba  = 20;   // GABA能起搏神经元 (theta rhythm)

    // Theta pacemaker parameters
    float theta_period = 150.0f;  // ~6.7 Hz (150ms period)
    float theta_drive  = 25.0f;   // Pacemaker drive current amplitude

    // Internal connectivity
    float p_gaba_to_ach = 0.3f;   // GABA → ACh (phase coordination)
    float w_gaba_ach    = 1.0f;

    // ACh output level
    float ach_output = 0.25f;     // Tonic ACh output (modulates Hipp)
};

class SeptalNucleus : public BrainRegion {
public:
    explicit SeptalNucleus(const SeptalConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_all_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_all_; }

    /** ACh output level for broadcast (隔核→海马 胆碱能调制) */
    float ach_output() const { return ach_output_; }

    /** Current theta phase (0~1) */
    float theta_phase() const { return theta_phase_; }

    const NeuronPopulation& ach_pop()  const { return ach_; }
    const NeuronPopulation& gaba_pop() const { return gaba_; }

private:
    void aggregate_state();

    SeptalConfig config_;

    NeuronPopulation ach_;     // 胆碱能
    NeuronPopulation gaba_;    // GABA能起搏

    SynapseGroup syn_gaba_to_ach_;  // Phase coordination

    // PSP buffer
    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_ach_;

    // Theta pacemaker state
    float theta_phase_ = 0.0f;    // 0~1 cyclic phase
    float ach_output_  = 0.0f;

    std::vector<uint8_t> fired_all_;
    std::vector<int8_t>  spike_type_all_;
};

} // namespace wuyun
