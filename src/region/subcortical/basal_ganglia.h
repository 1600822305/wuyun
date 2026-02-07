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

    // MSN up-state drive: brings MSN from down state (-80mV) closer to threshold
    // Biological basis: MSN exhibit bistable up/down states (Wilson & Kawaguchi 1996)
    // v26: keep tonic=40 (original), rely on multiplicative weight gain (3×) to amplify differences
    // Surmeier 2007: D1 enhances cortical INPUT gain, not tonic drive
    float msn_up_state_drive = 25.0f;

    // Cortical weight gain amplification (Surmeier et al. 2007)
    // Biology: D1 receptors enhance NMDA/Ca2+ channels, amplifying cortical input gain
    // Effect: weight differences are nonlinearly amplified, not just linearly added
    // gain = 1 + (w - 1) * factor; w=1.5→gain=2.5, w=0.5→gain=0.25(clamped)
    float weight_gain_factor = 3.0f;

    // --- DA-STDP (three-factor reinforcement learning) ---
    bool  da_stdp_enabled  = false;   // Enable DA-STDP on cortical→MSN
    float da_stdp_lr       = 0.005f;  // Learning rate
    float da_stdp_baseline = 0.3f;    // DA baseline (~4Hz VTA tonic, above=reward, below=punishment)
    float da_stdp_w_min    = 0.1f;    // Min connection weight
    float da_stdp_w_max    = 3.0f;    // Max connection weight
    float da_stdp_elig_decay = 0.98f; // Eligibility trace decay per step (~50 step window, 0.98^15=0.74)
    float da_stdp_max_elig = 50.0f;  // Per-synapse elig ceiling (prevents Δw explosion: 0.03×0.5×50=0.75)
    float da_stdp_w_decay  = 0.003f;  // Weight decay toward 1.0 per step (recovery in ~67 steps)

    // D1/D2 lateral inhibition (MSN collateral GABA, Humphries et al. 2009)
    // Biology: MSN→MSN collateral synapses provide ~1-3% lateral connectivity,
    // creating competition between action channels. The winning channel (most active D1 subgroup)
    // suppresses competing channels → direction selectivity emerges.
    // Without this, all D1 subgroups receive similar cortical input and converge to same weights.
    bool  lateral_inhibition = true;    // Enable D1/D2 inter-subgroup competition
    float lateral_inh_strength = 8.0f; // GABA-mediated inhibitory current to losing subgroups
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

    /** 设置 DA 水平 (影响 D1/D2 兴奋性) — 仅用于直接测试, 正式仿真由脉冲自动推算 */
    void set_da_level(float da);

    /** 设置 DA 源区域 ID (来自 VTA 的脉冲将自动更新 DA 水平) */
    void set_da_source_region(uint32_t region_id) { da_source_region_ = region_id; }

    /** 获取 GPi 输出 (持续抑制 - 去抑制 = 动作选择) */
    const NeuronPopulation& gpi() const { return gpi_; }

    NeuronPopulation& d1()  { return d1_msn_; }
    NeuronPopulation& d2()  { return d2_msn_; }
    NeuronPopulation& stn() { return stn_; }

    /** Sensory context injection (thalamostriatal pathway)
     *  signals[4] = {UP, DOWN, LEFT, RIGHT} attractiveness
     *  Positive = food direction, Negative = danger direction
     *  Sets input_active_ for dedicated sensory slots → topographic D1 mapping */
    void inject_sensory_context(const float signals[4]);
    /** Motor efference copy: mark action as active for elig trace, NO PSP injection */
    void mark_motor_efference(int action_group);

    /** Register a cortical source for topographic (channel-aligned) corticostriatal mapping.
     *  Rebuilds ctx→D1/D2 maps so that neurons from this source connect preferentially
     *  to the corresponding D1/D2 action subgroup (proportional spatial mapping).
     *  Biology: corticostriatal projections maintain partial somatotopy/retinotopy */
    void set_topographic_cortical_source(uint32_t region_id, size_t n_neurons);

    /** Awake SWR replay mode: suppress weight decay during replay steps */
    void set_replay_mode(bool m) { replay_mode_ = m; }
    bool replay_mode() const { return replay_mode_; }

    /** Lightweight replay step: only D1/D2 firing + DA-STDP, no GPi/GPe/STN.
     *  Call receive_spikes() first to inject cortical spikes, then this. */
    void replay_learning_step(int32_t t, float dt = 1.0f);

    /** DA-STDP 权重诊断 */
    size_t d1_weight_count() const { return ctx_d1_w_.size(); }
    const std::vector<float>& d1_weights_for(size_t src) const { return ctx_d1_w_[src]; }
    size_t d2_weight_count() const { return ctx_d2_w_.size(); }
    const std::vector<float>& d2_weights_for(size_t src) const { return ctx_d2_w_[src]; }
    float da_level() const { return da_level_; }
    float da_spike_accum() const { return da_spike_accum_; }

    /** Eligibility trace diagnostics */
    float total_elig_d1() const {
        float s = 0; for (auto& v : elig_d1_) for (float e : v) s += e; return s;
    }
    float total_elig_d2() const {
        float s = 0; for (auto& v : elig_d2_) for (float e : v) s += e; return s;
    }
    size_t input_active_count() const {
        size_t c = 0; for (auto a : input_active_) c += a; return c;
    }
    size_t total_cortical_inputs() const { return total_cortical_inputs_; }

private:
    void build_synapses();
    void aggregate_state();

    BasalGangliaConfig config_;
    float da_level_ = 0.3f;      // DA tonic baseline (matches VTA tonic_rate)
    uint32_t da_source_region_ = UINT32_MAX;  // VTA region ID (UINT32_MAX = not set)
    float da_spike_accum_ = 0.0f; // DA spike accumulator for rate estimation
    size_t total_cortical_inputs_ = 0;  // Cumulative cortical spike events (never cleared)
    static constexpr float DA_RATE_TAU = 0.98f; // exponential smoothing (slower decay, DA persists longer)
    static constexpr size_t SENSORY_SLOT_BASE = 252; // Input slots 252-255 = sensory direction channels

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

    // 跨区域输入随机映射表 (构造时生成, 替代 id%5 硬编码)
    // cortex_to_d1_map_[i] = 第i个连接的目标 D1 神经元
    std::vector<std::vector<uint32_t>> ctx_to_d1_map_;  // per-input neuron → D1 targets
    std::vector<std::vector<uint32_t>> ctx_to_d2_map_;  // per-input neuron → D2 targets
    std::vector<std::vector<uint32_t>> ctx_to_stn_map_; // per-input neuron → STN targets (hyperdirect)
    size_t input_map_size_ = 0;
    void build_input_maps(size_t n_input_neurons);

    // Topographic cortical source (dlPFC → D1/D2 channel-aligned mapping)
    uint32_t topo_ctx_rid_ = UINT32_MAX;  // Registered topographic source region ID
    size_t   topo_ctx_n_   = 0;           // Source neuron count

    // PSP 缓冲 (模拟突触时间常数)
    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_d1_;
    std::vector<float> psp_d2_;
    std::vector<float> psp_stn_;

    std::vector<uint8_t> fired_all_;
    std::vector<int8_t>  spike_type_all_;

    // --- DA-STDP online learning ---
    // Per-connection weights (parallel to ctx_to_d1/d2_map_)
    std::vector<std::vector<float>> ctx_d1_w_;  // [src][idx]
    std::vector<std::vector<float>> ctx_d2_w_;  // [src][idx]
    std::vector<uint8_t> input_active_;  // flags: which input slots fired this step

    // Eligibility traces (Izhikevich 2007, Frémaux & Gerstner 2016)
    // Bridges temporal gap between action (cortex→BG co-activation) and reward (DA)
    std::vector<std::vector<float>> elig_d1_;  // [src][idx] decaying trace
    std::vector<std::vector<float>> elig_d2_;  // [src][idx]
    void apply_da_stdp(int32_t t);

    bool replay_mode_ = false;  // Suppress weight decay during awake replay
};

} // namespace wuyun
