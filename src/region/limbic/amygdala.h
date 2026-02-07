#pragma once
/**
 * Amygdala — 杏仁核情感系统
 *
 * 实现杏仁核的核心恐惧/奖赏条件化通路:
 *   感觉 → La(外侧核,输入) → BLA(基底外侧核,学习) → CeA(中央核,输出)
 *                                    ↕
 *                              ITC(闰核,PFC门控) — 恐惧消退
 *
 * 关键特性 (按 01 文档 §2.1.4):
 *   - La: 多模态感觉汇入站
 *   - BLA: CS-US 关联学习 (DA-STDP 驱动)
 *   - CeA: 恐惧/应激行为输出 → 下丘脑, PAG, LC
 *   - ITC: BLA→CeA 的抑制性门控, PFC 调控恐惧消退
 *
 * 遵守 00 文档反作弊原则:
 *   - 价值学习存在于 BLA 突触权重中, 不是标签/字典
 *   - 恐惧消退是 ITC 门控 + 突触可塑性的结果, 不是 IF 逻辑
 *
 * 设计文档: docs/01_brain_region_plan.md §2.1.4
 */

#include "region/brain_region.h"
#include "core/population.h"
#include "core/synapse_group.h"
#include <climits>

namespace wuyun {

struct AmygdalaConfig {
    std::string name = "Amygdala";

    // --- Population sizes ---
    size_t n_la   = 50;   // 外侧核 (sensory input)
    size_t n_bla  = 80;   // 基底外侧核 (CS-US learning)
    size_t n_cea  = 30;   // 中央核 (output)
    size_t n_itc  = 20;   // 闰核 (inhibitory gate, PFC-controlled)
    size_t n_mea  = 0;    // 内侧核 (social/olfactory, optional)
    size_t n_coa  = 0;    // 皮质核 (olfactory, optional)
    size_t n_ab   = 0;    // 副基底核 (multimodal, optional)

    // --- Connection probabilities ---
    float p_la_to_bla    = 0.20f;   // La → BLA
    float p_bla_to_cea   = 0.25f;   // BLA → CeA
    float p_la_to_cea    = 0.10f;   // La → CeA (direct fast path)
    float p_bla_to_itc   = 0.15f;   // BLA → ITC
    float p_itc_to_cea   = 0.30f;   // ITC → CeA (inhibitory gate)
    float p_bla_to_bla   = 0.05f;   // BLA recurrent
    // Optional nuclei connections (only active if n_mea/n_coa/n_ab > 0)
    float p_la_to_mea     = 0.25f;   // La → MeA
    float p_la_to_coa     = 0.20f;   // La → CoA
    float p_bla_to_ab     = 0.20f;   // BLA → AB
    float p_ab_to_cea     = 0.15f;   // AB → CeA
    float p_mea_to_cea    = 0.15f;   // MeA → CeA

    // --- Synapse weights ---
    float w_la_bla       = 0.6f;
    float w_bla_cea      = 0.7f;
    float w_la_cea       = 0.4f;    // Direct fast path (weaker)
    float w_bla_itc      = 0.5f;
    float w_itc_cea      = 2.0f;    // Inhibitory gate (positive; GABA e_rev handles sign)
    float w_bla_rec      = 0.2f;
    float w_mea           = 0.8f;
    float w_coa           = 0.7f;
    float w_ab            = 0.7f;

    // --- Fear conditioning STDP (La→BLA, one-shot learning) ---
    // Biology: BLA LTP is NMDA-dependent, gated by US (pain/danger).
    // Very fast: a single CS-US pairing can establish fear memory.
    // (LeDoux 2000, Maren 2001)
    bool  fear_stdp_enabled = true;
    float fear_stdp_a_plus  = 0.10f;  // Very fast LTP (10x cortical, one-shot)
    float fear_stdp_a_minus = -0.03f; // Weak LTD (fear is hard to extinguish)
    float fear_stdp_tau     = 25.0f;
    float fear_stdp_w_max   = 3.0f;   // High ceiling: strong fear associations
};

class Amygdala : public BrainRegion {
public:
    explicit Amygdala(const AmygdalaConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_all_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_all_; }

    // --- Amygdala-specific interface ---

    /** Inject sensory input to La (感觉→外侧核) */
    void inject_sensory(const std::vector<float>& currents);

    /** Inject PFC top-down to ITC (PFC→闰核, 恐惧消退调控) */
    void inject_pfc_to_itc(const std::vector<float>& currents);

    /** Set PFC source region ID (for routing PFC spikes → ITC in receive_spikes) */
    void set_pfc_source_region(uint32_t rid) { pfc_source_region_ = rid; }

    /** Get CeA output (fear/stress response readout) */
    const NeuronPopulation& cea() const { return cea_; }
    const NeuronPopulation& bla() const { return bla_; }
    const NeuronPopulation& la()  const { return la_; }
    const NeuronPopulation& itc() const { return itc_; }
    const NeuronPopulation& mea() const { return mea_; }
    const NeuronPopulation& coa() const { return coa_; }
    const NeuronPopulation& ab()  const { return ab_; }
    bool has_mea() const { return config_.n_mea > 0; }
    bool has_coa() const { return config_.n_coa > 0; }
    bool has_ab()  const { return config_.n_ab > 0; }

    // --- Fear conditioning closed-loop interface ---

    /** Inject unconditioned stimulus (pain/danger) to BLA.
     *  Biology: US (e.g. foot shock) directly activates BLA neurons.
     *  When paired with sensory CS (via La), La→BLA STDP strengthens
     *  the CS→BLA association, establishing fear memory.
     *  (LeDoux 2000: amygdala is essential for fear conditioning) */
    void inject_us(float magnitude);

    /** Get CeA fear output level [0,1].
     *  High = strong fear response → should drive VTA/LHb for DA pause.
     *  Biology: CeA is the main output of the amygdala,
     *  projecting to VTA, PAG, hypothalamus for defensive behaviors. */
    float fear_output() const;

    /** Get CeA fear output as VTA inhibition signal.
     *  Scaled version of fear_output() for direct VTA/LHb injection.
     *  Biology: CeA → VTA/RMTg → DA pause (aversive prediction) */
    float cea_vta_drive() const;

private:
    void build_synapses();
    void aggregate_state();

    AmygdalaConfig config_;

    // --- 4 populations ---
    NeuronPopulation la_;    // 外侧核 (input)
    NeuronPopulation bla_;   // 基底外侧核 (learning)
    NeuronPopulation cea_;   // 中央核 (output)
    NeuronPopulation itc_;   // 闰核 (inhibitory gate)
    NeuronPopulation mea_;   // 内侧核 (optional, social/olfactory)
    NeuronPopulation coa_;   // 皮质核 (optional, olfactory)
    NeuronPopulation ab_;    // 副基底核 (optional, multimodal)

    // --- Synapses ---
    SynapseGroup syn_la_to_bla_;    // La → BLA
    SynapseGroup syn_bla_to_cea_;   // BLA → CeA (fear expression)
    SynapseGroup syn_la_to_cea_;    // La → CeA (direct fast path)
    SynapseGroup syn_bla_to_itc_;   // BLA → ITC
    SynapseGroup syn_itc_to_cea_;   // ITC → CeA (inhibitory gate)
    SynapseGroup syn_bla_rec_;      // BLA → BLA recurrent
    SynapseGroup syn_la_to_mea_;    // La → MeA (optional)
    SynapseGroup syn_la_to_coa_;    // La → CoA (optional)
    SynapseGroup syn_bla_to_ab_;    // BLA → AB (optional)
    SynapseGroup syn_ab_to_cea_;    // AB → CeA (optional)
    SynapseGroup syn_mea_to_cea_;   // MeA → CeA (optional)

    // PSP buffer for cross-region input
    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_la_;      // sensory → La
    std::vector<float> psp_itc_;     // PFC → ITC
    uint32_t pfc_source_region_ = UINT32_MAX;  // PFC region ID (for routing)

    // --- Fear conditioning state ---
    float us_strength_ = 0.0f;       // Current US drive (decays)
    static constexpr float US_DECAY = 0.85f;

    std::vector<uint8_t> fired_all_;
    std::vector<int8_t>  spike_type_all_;
};

} // namespace wuyun
