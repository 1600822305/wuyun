#pragma once
/**
 * DRN_5HT — 背侧缝核血清素系统 (Dorsal Raphe Nucleus)
 *
 * 核心功能: 冲动控制 / 耐心 / 情绪调节
 *   5-HT↑ → 折扣因子↑ (更耐心, 更重视远期奖励)
 *   5-HT↓ → 折扣因子↓ (更冲动, 偏好即时奖励)
 *
 * 输入驱动:
 *   - 奖励历史 (持续奖励 → 5-HT↑, "一切顺利")
 *   - PFC 前额叶抑制控制
 *
 * 投射目标: 全脑 (volume transmission)
 *   → BG: 折扣/耐心调制
 *   → 杏仁核: 情绪调节
 *   → 皮层: 抑制控制
 */

#include "region/brain_region.h"
#include "core/population.h"

namespace wuyun {

struct DRNConfig {
    std::string name = "DRN";
    size_t n_5ht_neurons = 20;
    float  tonic_rate    = 0.3f;   // 5-HT baseline (moderate patience)
    float  phasic_gain   = 0.3f;
};

class DRN_5HT : public BrainRegion {
public:
    DRN_5HT(const DRNConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_; }

    /** 注入"一切顺利"信号 (持续正奖励) */
    void inject_wellbeing(float wellbeing);

    /** 获取当前 5-HT 输出水平 (0~1) */
    float sht_output() const { return sht_level_; }

    NeuronPopulation& neurons() { return sht_neurons_; }

private:
    DRNConfig config_;
    NeuronPopulation sht_neurons_;

    float wellbeing_input_ = 0.0f;
    float sht_level_       = 0.3f;

    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_5ht_;

    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;
};

} // namespace wuyun
