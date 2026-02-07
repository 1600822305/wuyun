#pragma once
/**
 * NBM_ACh — 基底核乙酰胆碱系统 (Nucleus Basalis of Meynert)
 *
 * 核心功能: 学习模式切换 / 注意力调制
 *   ACh↑ → 自下而上学习 (basal主导, 新异刺激驱动)
 *   ACh↓ → 自上而下预测 (apical主导, 内部模型驱动)
 *
 * 输入驱动:
 *   - 不确定性/意外信号 → ACh phasic burst
 *   - Amygdala → 情绪相关注意力
 *
 * 投射目标: 皮层 (volume transmission)
 *   → 皮层: 学习率/模式切换
 *   → 海马: 编码vs回忆模式
 */

#include "region/brain_region.h"
#include "core/population.h"

namespace wuyun {

struct NBMConfig {
    std::string name = "NBM";
    size_t n_ach_neurons = 15;
    float  tonic_rate    = 0.2f;   // ACh baseline
    float  phasic_gain   = 0.4f;
};

class NBM_ACh : public BrainRegion {
public:
    NBM_ACh(const NBMConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_; }

    /** 注入不确定性/意外信号 */
    void inject_surprise(float surprise);

    /** 获取当前 ACh 输出水平 (0~1) */
    float ach_output() const { return ach_level_; }

    NeuronPopulation& neurons() { return ach_neurons_; }

private:
    NBMConfig config_;
    NeuronPopulation ach_neurons_;

    float surprise_input_ = 0.0f;
    float ach_level_       = 0.2f;

    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_ach_;

    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;
};

} // namespace wuyun
