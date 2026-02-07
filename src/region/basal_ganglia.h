#pragma once
/**
 * BasalGanglia — 基底节回路
 *
 * 动作选择通路:
 *   Direct  (Go):   Cortex → D1 MSN → GPi (抑制) → Thalamus (去抑制) → 动作
 *   Indirect(NoGo): Cortex → D2 MSN → GPe → STN → GPi (兴奋) → Thalamus (抑制) → 停止
 *   Hyperdirect:    Cortex → STN → GPi (快速刹车)
 *
 * DA 调制:
 *   DA → D1: 增强 Go (LTP)
 *   DA → D2: 减弱 NoGo (LTD)
 *   → 净效应: DA↑ = 更容易行动
 *
 * 设计文档: docs/01_brain_region_plan.md BG-01~04
 */

#include "region/brain_region.h"
#include "core/population.h"
#include "core/synapse_group.h"

namespace wuyun {

struct BasalGangliaConfig {
    std::string name = "basal_ganglia";
    size_t n_d1_msn  = 100;   // D1 中棘神经元 (Go)
    size_t n_d2_msn  = 100;   // D2 中棘神经元 (NoGo)
    size_t n_gpi     = 30;    // 内苍白球 (输出核, 持续抑制)
    size_t n_gpe     = 30;    // 外苍白球
    size_t n_stn     = 20;    // 丘脑底核

    // 连接概率
    float p_ctx_to_d1  = 0.2f;
    float p_ctx_to_d2  = 0.2f;
    float p_ctx_to_stn = 0.15f;  // hyperdirect
    float p_d1_to_gpi  = 0.3f;
    float p_d2_to_gpe  = 0.3f;
    float p_gpe_to_stn = 0.4f;
    float p_stn_to_gpi = 0.4f;

    // 权重
    float w_ctx_exc  = 0.5f;
    float w_d1_inh   = 0.8f;   // D1→GPi 强抑制 (Go)
    float w_d2_inh   = 0.6f;
    float w_gpe_inh  = 0.5f;
    float w_stn_exc  = 0.7f;   // STN→GPi 强兴奋 (刹车)
};

class BasalGanglia : public BrainRegion {
public:
    BasalGanglia(const BasalGangliaConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_all_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_all_; }

    // --- BG 特有接口 ---

    /** 注入皮层输入到 D1/D2/STN */
    void inject_cortical_input(const std::vector<float>& d1_cur,
                                const std::vector<float>& d2_cur);

    /** 设置 DA 水平 (影响 D1/D2 兴奋性) */
    void set_da_level(float da);

    /** 获取 GPi 输出 (持续抑制 - 去抑制 = 动作选择) */
    const NeuronPopulation& gpi() const { return gpi_; }

    NeuronPopulation& d1()  { return d1_msn_; }
    NeuronPopulation& d2()  { return d2_msn_; }
    NeuronPopulation& stn() { return stn_; }

private:
    void build_synapses();
    void aggregate_state();

    BasalGangliaConfig config_;
    float da_level_ = 0.1f;  // DA tonic baseline

    // 5 populations
    NeuronPopulation d1_msn_;    // Go pathway
    NeuronPopulation d2_msn_;    // NoGo pathway
    NeuronPopulation gpi_;       // Output (tonic inhibition)
    NeuronPopulation gpe_;       // Indirect pathway relay
    NeuronPopulation stn_;       // Subthalamic nucleus (excitatory)

    // Direct: D1 → GPi (inhibitory)
    SynapseGroup syn_d1_to_gpi_;
    // Indirect: D2 → GPe (inhibitory)
    SynapseGroup syn_d2_to_gpe_;
    // Indirect: GPe → STN (inhibitory)
    SynapseGroup syn_gpe_to_stn_;
    // Indirect + Hyperdirect: STN → GPi (excitatory)
    SynapseGroup syn_stn_to_gpi_;

    std::vector<uint8_t> fired_all_;
    std::vector<int8_t>  spike_type_all_;
};

} // namespace wuyun
