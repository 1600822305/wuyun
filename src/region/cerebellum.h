#pragma once
/**
 * Cerebellum — 小脑运动学习系统
 *
 * 架构 (与皮层完全不同的计算范式):
 *   苔藓纤维(MF) → 颗粒细胞(GrC, 扩展层) → 平行纤维(PF)
 *   → 浦肯野细胞(PC, 收敛/输出) → 深部核团(DCN, 最终输出)
 *
 * 学习规则:
 *   攀爬纤维(CF, 来自下橄榄IO) 传递误差信号
 *   CF + PF同时激活 → PF→PC LTD (减弱错误运动)
 *   PF单独激活(无CF) → PF→PC LTP (强化正确运动)
 *
 * 信号流:
 *   输入: 皮层(M1/PFC) → 桥核(Pontine) → MF → GrC
 *   误差: 下橄榄(IO) → CF → PC (1:1)
 *   输出: PC(GABA,抑制) → DCN → 丘脑 → M1
 *
 * 功能:
 *   - 运动时序精确控制 (timing)
 *   - 运动误差在线校正
 *   - 经典条件反射 (眨眼反射等)
 *
 * 设计文档: docs/01_brain_region_plan.md CB-01~04
 */

#include "region/brain_region.h"
#include "core/population.h"
#include "core/synapse_group.h"

namespace wuyun {

struct CerebellumConfig {
    std::string name = "Cerebellum";

    // Population sizes
    size_t n_granule    = 200;  // 颗粒细胞 (扩展层, 生物上最多)
    size_t n_purkinje   = 30;   // 浦肯野细胞 (输出层)
    size_t n_dcn        = 20;   // 深部核团 (最终输出)
    size_t n_mli        = 15;   // 分子层中间神经元 (stellate/basket)
    size_t n_golgi      = 10;   // 高尔基细胞 (颗粒层反馈抑制)

    // Connectivity
    float p_mf_to_grc   = 0.15f;  // 苔藓纤维→颗粒 (稀疏, 4:1扩展)
    float p_pf_to_pc    = 0.40f;  // 平行纤维→浦肯野 (广泛汇聚)
    float p_pf_to_mli   = 0.20f;  // 平行纤维→分子层抑制
    float p_mli_to_pc   = 0.30f;  // 分子层→浦肯野 (前馈抑制)
    float p_pc_to_dcn   = 0.35f;  // 浦肯野→深核 (抑制性, 调制而非沉默)
    float p_golgi_to_grc= 0.20f;  // 高尔基→颗粒 (反馈抑制)
    float p_grc_to_golgi= 0.15f;  // 颗粒→高尔基

    // Synaptic weights
    float w_mf_grc      = 0.8f;
    float w_pf_pc       = 0.5f;   // 初始PF→PC权重 (会被LTD/LTP修改)
    float w_pf_mli      = 0.5f;
    float w_mli_pc      = 0.6f;   // 抑制性
    float w_pc_dcn      = 0.4f;   // 抑制性 (PC是GABA能, DCN有强内源驱动)
    float w_golgi_grc   = 0.5f;   // 抑制性
    float w_grc_golgi   = 0.4f;

    // Climbing fiber LTD/LTP parameters
    float cf_ltd_rate    = 0.02f;  // CF+PF → LTD
    float cf_ltp_rate    = 0.005f; // PF alone → LTP
    float pf_pc_w_min    = 0.1f;
    float pf_pc_w_max    = 1.5f;
};

class Cerebellum : public BrainRegion {
public:
    Cerebellum(const CerebellumConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_; }

    // --- 小脑特有接口 ---

    /** 注入攀爬纤维误差信号 (来自下橄榄, 0=无误差, 1=最大误差) */
    void inject_climbing_fiber(float error_signal);

    /** 注入苔藓纤维输入 (来自桥核/皮层, 直接到颗粒细胞) */
    void inject_mossy_fiber(const std::vector<float>& currents);

    /** 获取 DCN 输出 */
    NeuronPopulation& dcn() { return dcn_; }
    NeuronPopulation& granule() { return grc_; }
    NeuronPopulation& purkinje() { return pc_; }

    /** 获取 CF 误差信号 */
    float last_cf_error() const { return cf_error_; }

private:
    CerebellumConfig config_;

    // Populations
    NeuronPopulation grc_;     // 颗粒细胞
    NeuronPopulation pc_;      // 浦肯野细胞
    NeuronPopulation dcn_;     // 深部核团
    NeuronPopulation mli_;     // 分子层中间神经元
    NeuronPopulation golgi_;   // 高尔基细胞

    // Excitatory synapses
    SynapseGroup syn_mf_to_grc_;    // 苔藓→颗粒
    SynapseGroup syn_pf_to_pc_;     // 平行纤维→浦肯野 (LTD/LTP target)
    SynapseGroup syn_pf_to_mli_;    // 平行纤维→分子层
    SynapseGroup syn_grc_to_golgi_; // 颗粒→高尔基

    // Inhibitory synapses
    SynapseGroup syn_mli_to_pc_;    // 分子层→浦肯野 (前馈抑制)
    SynapseGroup syn_pc_to_dcn_;    // 浦肯野→深核 (主抑制输出)
    SynapseGroup syn_golgi_to_grc_; // 高尔基→颗粒 (反馈抑制)

    // Climbing fiber state
    float cf_error_ = 0.0f;

    // Cross-region PSP buffer
    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_grc_;  // PSP for granule cells (from SpikeBus)

    // Aggregate firing state
    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;

    void aggregate_firing_state();
    void apply_climbing_fiber_plasticity(int32_t t);
};

} // namespace wuyun
