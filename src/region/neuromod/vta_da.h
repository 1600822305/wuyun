#pragma once
/**
 * VTA_DA — 腹侧被盖区多巴胺系统
 *
 * 核心功能: 奖励预测误差 (RPE) 信号
 *   RPE > 0: 实际奖励 > 预期 → DA phasic burst → 强化行为
 *   RPE = 0: 符合预期 → DA tonic → 维持
 *   RPE < 0: 实际奖励 < 预期 → DA pause → 削弱行为
 *
 * 投射目标:
 *   → 纹状体 (D1/D2 MSN): 动作选择学习
 *   → PFC: 工作记忆/执行功能调制
 *
 * 设计文档: docs/01_brain_region_plan.md MB-03/04
 */

#include "region/brain_region.h"
#include "core/population.h"

namespace wuyun {

struct VTAConfig {
    std::string name = "VTA";
    size_t n_da_neurons = 50;    // DA 神经元数
    float  tonic_rate   = 0.3f;  // tonic baseline DA (~4Hz VTA firing, 归一化)
    float  phasic_gain  = 0.5f;  // RPE → phasic DA 增益
};

class VTA_DA : public BrainRegion {
public:
    VTA_DA(const VTAConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_; }

    // --- DA 特有接口 ---

    /** 获取当前 DA 输出水平 (tonic + phasic) */
    float da_output() const { return da_level_; }

    /** 获取最近的 RPE (now computed from spike rates, not injected scalars) */
    float last_rpe() const { return last_rpe_; }

    /** 注入LHb抑制信号 (0~1, LHb firing → RMTg GABA → VTA DA pause)
     *  This is ModulationBus (volume transmission), acceptable per 02 design doc. */
    void inject_lhb_inhibition(float inhibition);

    /** v46: Register hedonic source region (Hypothalamus LH → VTA excitation)
     *  Spikes from this region = "actual reward arrived" → DA burst
     *  Biology: LH glutamatergic → VTA (Nieh et al. 2015) */
    void register_hedonic_source(uint32_t region_id) { hedonic_source_id_ = region_id; has_hedonic_source_ = true; }

    /** v46: Register prediction source region (OFC → VTA inhibition)
     *  Spikes from this region = "expected reward" → suppress DA (no surprise)
     *  Biology: OFC/striatum → VTA GABAergic interneurons (Takahashi et al. 2011) */
    void register_prediction_source(uint32_t region_id) { prediction_source_id_ = region_id; has_prediction_source_ = true; }

    NeuronPopulation& neurons() { return da_neurons_; }

private:
    VTAConfig config_;
    NeuronPopulation da_neurons_;

    float last_rpe_        = 0.0f;
    float da_level_        = 0.1f;  // tonic + phasic

    float lhb_inhibition_   = 0.0f;  // LHb → RMTg → VTA inhibition (0~1)
    float lhb_inh_psp_      = 0.0f;  // Sustained LHb inhibition (exponential decay)
    static constexpr float LHB_INH_PSP_DECAY = 0.85f;

    // v46: Spike-driven RPE (replaces inject_reward / set_expected_reward)
    // Hedonic source (Hypothalamus LH): actual reward signal → excites DA neurons
    // Prediction source (OFC): expected value → inhibits DA neurons (no surprise)
    // RPE = hedonic_rate - prediction_rate (Schultz 1997)
    uint32_t hedonic_source_id_ = 0;
    uint32_t prediction_source_id_ = 0;
    bool has_hedonic_source_ = false;
    bool has_prediction_source_ = false;
    float hedonic_psp_    = 0.0f;   // Accumulated hedonic spikes (actual reward)
    float prediction_psp_ = 0.0f;   // Accumulated prediction spikes (expected value)
    static constexpr float HEDONIC_PSP_DECAY = 0.85f;
    static constexpr float PREDICTION_PSP_DECAY = 0.85f;

    // v37: Track tonic firing rate for firing-rate-based DA computation
    float tonic_firing_smooth_ = 0.0f;
    int   step_count_          = 0;
    static constexpr int WARMUP_STEPS = 50;

    // PSP buffer for cross-region input (general cortical/striatal modulation)
    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_da_;

    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;
};

} // namespace wuyun
