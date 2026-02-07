#pragma once
/**
 * Hypothalamus — 下丘脑内驱力系统
 *
 * 6个核团:
 *   SCN  — 视交叉上核: 昼夜节律起搏器 (~24h 正弦振荡)
 *   VLPO — 腹外侧视前区: 睡眠促进 (GABA/galanin → 抑制觉醒中枢)
 *   Orexin — 外侧下丘脑orexin神经元: 觉醒稳定 (→LC/DRN/NBM)
 *   PVN  — 室旁核: 应激反应 (CRH → HPA轴)
 *   LH   — 外侧下丘脑: 摄食/饥饿驱力
 *   VMH  — 腹内侧核: 饱腹/能量平衡
 *
 * 关键回路:
 *   Sleep-wake flip-flop (Saper 2005):
 *     VLPO ⟷ Orexin/LC/DRN 互相抑制
 *     SCN → VLPO (昼夜门控)
 *     Orexin 稳定开关 (防止嗜睡发作)
 *
 *   Stress:
 *     PVN → CeA (应激→恐惧), PVN → VTA (应激→DA)
 *
 *   Feeding:
 *     LH ⟷ VMH 互相抑制 (饥饿↔饱腹平衡)
 *     LH → VTA (饥饿→动机)
 *
 * 生物学参考:
 *   - Saper et al. (2005) Hypothalamic regulation of sleep and circadian rhythms
 *   - Sakurai (2007) The neural circuit of orexin (hypocretin)
 *   - Ulrich-Lai & Herman (2009) Neural regulation of endocrine stress responses
 */

#include "region/brain_region.h"
#include "core/population.h"
#include "core/synapse_group.h"
#include <algorithm>

namespace wuyun {

struct HypothalamusConfig {
    std::string name = "Hypothalamus";

    // === Population sizes ===
    size_t n_scn    = 20;   // SCN circadian pacemaker
    size_t n_vlpo   = 15;   // VLPO sleep-promoting
    size_t n_orexin = 15;   // Orexin/Hypocretin wake-promoting
    size_t n_pvn    = 15;   // PVN stress response
    size_t n_lh     = 12;   // LH hunger drive
    size_t n_vmh    = 12;   // VMH satiety

    // === SCN circadian parameters ===
    float circadian_period = 24000.0f;  // ~24h in timestep units
    float scn_drive_amp    = 20.0f;     // SCN pacemaker amplitude

    // === Sleep-wake flip-flop ===
    float p_vlpo_to_orexin = 0.4f;  // VLPO→Orexin inhibition
    float w_vlpo_orexin    = 1.2f;
    float p_orexin_to_vlpo = 0.4f;  // Orexin→VLPO inhibition
    float w_orexin_vlpo    = 1.2f;
    float p_scn_to_vlpo    = 0.3f;  // SCN→VLPO circadian gate
    float w_scn_vlpo       = 0.8f;

    // === LH ⟷ VMH feeding balance ===
    float p_lh_to_vmh = 0.3f;
    float w_lh_vmh    = 1.0f;
    float p_vmh_to_lh = 0.3f;
    float w_vmh_lh    = 1.0f;

    // === Internal drive levels (0~1, settable externally) ===
    float homeostatic_sleep_pressure = 0.3f;  // Adenosine-like accumulator
    float stress_level    = 0.1f;  // Cortisol/CRH baseline
    float hunger_level    = 0.3f;  // Ghrelin-like hunger signal
    float satiety_level   = 0.3f;  // Leptin-like satiety signal
};

class Hypothalamus : public BrainRegion {
public:
    explicit Hypothalamus(const HypothalamusConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_all_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_all_; }

    // === State accessors ===

    /** Sleep-wake state: >0.5 = awake, <0.5 = sleep */
    float wake_level() const { return wake_level_; }

    /** Circadian phase (0~1, 0=midnight, 0.5=noon) */
    float circadian_phase() const { return circadian_phase_; }

    /** Is the system in sleep state? */
    bool is_sleeping() const { return wake_level_ < 0.5f; }

    /** Stress level (PVN output, 0~1) */
    float stress_output() const { return stress_output_; }

    /** Hunger drive (LH output, 0~1) */
    float hunger_output() const { return hunger_output_; }

    /** Satiety level (VMH output, 0~1) */
    float satiety_output() const { return satiety_output_; }

    // === External drive control ===

    /** Set homeostatic sleep pressure (accumulates with wake time) */
    void set_sleep_pressure(float p) { config_.homeostatic_sleep_pressure = std::clamp(p, 0.0f, 1.0f); }

    /** Set stress level */
    void set_stress_level(float s) { config_.stress_level = std::clamp(s, 0.0f, 1.0f); }

    /** Set hunger level */
    void set_hunger_level(float h) { config_.hunger_level = std::clamp(h, 0.0f, 1.0f); }

    /** Set satiety level */
    void set_satiety_level(float s) { config_.satiety_level = std::clamp(s, 0.0f, 1.0f); }

    // === Population access ===
    const NeuronPopulation& scn_pop()    const { return scn_; }
    const NeuronPopulation& vlpo_pop()   const { return vlpo_; }
    const NeuronPopulation& orexin_pop() const { return orexin_; }
    const NeuronPopulation& pvn_pop()    const { return pvn_; }
    const NeuronPopulation& lh_pop()     const { return lh_; }
    const NeuronPopulation& vmh_pop()    const { return vmh_; }

private:
    void aggregate_state();

    HypothalamusConfig config_;

    // === Populations ===
    NeuronPopulation scn_;       // Circadian pacemaker
    NeuronPopulation vlpo_;      // Sleep (GABA)
    NeuronPopulation orexin_;    // Wake (excitatory)
    NeuronPopulation pvn_;       // Stress (CRH)
    NeuronPopulation lh_;        // Hunger
    NeuronPopulation vmh_;       // Satiety

    // === Internal synapses ===
    SynapseGroup syn_vlpo_to_orexin_;  // VLPO→Orexin (GABA, sleep→wake inhibition)
    SynapseGroup syn_orexin_to_vlpo_;  // Orexin→VLPO (inhibitory, wake→sleep inhibition)
    SynapseGroup syn_scn_to_vlpo_;     // SCN→VLPO (excitatory, circadian gate)
    SynapseGroup syn_lh_to_vmh_;       // LH→VMH (GABA, hunger inhibits satiety)
    SynapseGroup syn_vmh_to_lh_;       // VMH→LH (GABA, satiety inhibits hunger)

    // === PSP buffers ===
    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_vlpo_;
    std::vector<float> psp_orexin_;
    std::vector<float> psp_pvn_;

    // === State variables ===
    float circadian_phase_ = 0.0f;     // 0~1 (0=midnight)
    float wake_level_      = 0.8f;     // 0=deep sleep, 1=full wake
    float stress_output_   = 0.1f;
    float hunger_output_   = 0.3f;
    float satiety_output_  = 0.3f;

    // Aggregated firing
    std::vector<uint8_t> fired_all_;
    std::vector<int8_t>  spike_type_all_;
};

} // namespace wuyun
