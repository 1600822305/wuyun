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
    float  tonic_rate   = 0.1f;  // tonic baseline DA (归一化)
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

    /** 注入奖励信号 (正=奖励, 负=惩罚) */
    void inject_reward(float reward);

    /** 注入预期奖励 (来自纹状体/PFC 的预测) */
    void set_expected_reward(float expected);

    /** 获取当前 DA 输出水平 (tonic + phasic) */
    float da_output() const { return da_level_; }

    /** 获取最近的 RPE */
    float last_rpe() const { return last_rpe_; }

    NeuronPopulation& neurons() { return da_neurons_; }

private:
    VTAConfig config_;
    NeuronPopulation da_neurons_;

    float reward_input_    = 0.0f;
    float expected_reward_ = 0.0f;
    float last_rpe_        = 0.0f;
    float da_level_        = 0.1f;  // tonic + phasic

    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;
};

} // namespace wuyun
