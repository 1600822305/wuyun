#pragma once
/**
 * ThalamicRelay — 丘脑中继回路
 *
 * 结构:
 *   Relay neurons (中继神经元) — Tonic/Burst 双模式
 *   TRN neurons (网状核) — 纯抑制, 门控中继
 *
 * 信号流:
 *   感觉输入 → Relay → 皮层 L4
 *   皮层 L6  → Relay (反馈调制)
 *   Relay ↔ TRN (互相连接: Relay激活TRN, TRN抑制Relay)
 *
 * 门控机制:
 *   TRN 对 Relay 施加侧向抑制 → 注意力选择
 *   PFC→TRN 调制 → 自上而下注意力控制
 *
 * 设计文档: docs/02_neuron_system_design.md §5.3
 */

#include "region/brain_region.h"
#include "core/population.h"
#include "core/synapse_group.h"

namespace wuyun {

struct ThalamicConfig {
    std::string name = "thalamus";
    size_t n_relay = 100;    // 中继神经元数
    size_t n_trn   = 30;     // TRN 神经元数
    bool   burst_mode = false; // true=burst mode (睡眠/静息), false=tonic (清醒)

    // 连接概率
    float p_input_to_relay = 0.3f;   // 感觉输入→relay
    float p_relay_to_trn   = 0.4f;   // relay→TRN
    float p_trn_to_relay   = 0.5f;   // TRN→relay (抑制)
    float p_cortical_fb    = 0.2f;   // 皮层L6→relay反馈

    // 突触权重
    float w_input     = 0.8f;
    float w_relay_trn = 0.5f;
    float w_trn_inh   = 0.6f;
    float w_cortical  = 0.3f;
};

class ThalamicRelay : public BrainRegion {
public:
    ThalamicRelay(const ThalamicConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_all_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_all_; }

    // --- 丘脑特有接口 ---

    /** 注入皮层反馈到 relay apical (调制) */
    void inject_cortical_feedback(const std::vector<float>& currents);

    /** 注入 PFC→TRN 注意力控制信号 */
    void inject_trn_modulation(const std::vector<float>& currents);

    /** 切换 Tonic/Burst 模式 */
    void set_mode(bool burst_mode);

    NeuronPopulation& relay() { return relay_; }
    NeuronPopulation& trn()   { return trn_; }

private:
    void build_synapses();

    ThalamicConfig config_;

    NeuronPopulation relay_;   // 中继神经元
    NeuronPopulation trn_;     // 网状核 (抑制)

    // Relay→TRN (兴奋)
    SynapseGroup syn_relay_to_trn_;
    // TRN→Relay (抑制)
    SynapseGroup syn_trn_to_relay_;

    // 聚合状态
    std::vector<uint8_t> fired_all_;
    std::vector<int8_t>  spike_type_all_;

    void aggregate_state();
};

} // namespace wuyun
