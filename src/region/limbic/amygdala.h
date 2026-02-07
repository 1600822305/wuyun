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

    // --- Connection probabilities ---
    float p_la_to_bla    = 0.20f;   // La → BLA
    float p_bla_to_cea   = 0.25f;   // BLA → CeA
    float p_la_to_cea    = 0.10f;   // La → CeA (direct fast path)
    float p_bla_to_itc   = 0.15f;   // BLA → ITC
    float p_itc_to_cea   = 0.30f;   // ITC → CeA (inhibitory gate)
    float p_bla_to_bla   = 0.05f;   // BLA recurrent

    // --- Synapse weights ---
    float w_la_bla       = 0.6f;
    float w_bla_cea      = 0.7f;
    float w_la_cea       = 0.4f;    // Direct fast path (weaker)
    float w_bla_itc      = 0.5f;
    float w_itc_cea      = 2.0f;    // Inhibitory gate (positive; GABA e_rev handles sign)
    float w_bla_rec      = 0.2f;
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

private:
    void build_synapses();
    void aggregate_state();

    AmygdalaConfig config_;

    // --- 4 populations ---
    NeuronPopulation la_;    // 外侧核 (input)
    NeuronPopulation bla_;   // 基底外侧核 (learning)
    NeuronPopulation cea_;   // 中央核 (output)
    NeuronPopulation itc_;   // 闰核 (inhibitory gate)

    // --- Synapses ---
    SynapseGroup syn_la_to_bla_;    // La → BLA
    SynapseGroup syn_bla_to_cea_;   // BLA → CeA (fear expression)
    SynapseGroup syn_la_to_cea_;    // La → CeA (direct fast path)
    SynapseGroup syn_bla_to_itc_;   // BLA → ITC
    SynapseGroup syn_itc_to_cea_;   // ITC → CeA (inhibitory gate)
    SynapseGroup syn_bla_rec_;      // BLA → BLA recurrent

    // PSP buffer for cross-region input
    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_la_;      // sensory → La
    std::vector<float> psp_itc_;     // PFC → ITC
    uint32_t pfc_source_region_ = UINT32_MAX;  // PFC region ID (for routing)

    std::vector<uint8_t> fired_all_;
    std::vector<int8_t>  spike_type_all_;
};

} // namespace wuyun
