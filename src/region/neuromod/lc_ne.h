#pragma once
/**
 * LC_NE — 蓝斑核去甲肾上腺素系统 (Locus Coeruleus)
 *
 * 核心功能: 全脑增益调节 + 警觉/唤醒
 *   NE↑ → 信号增益↑ (信噪比提高, 更敏锐)
 *   NE↓ → 信号增益↓ (更放松, 默认模式)
 *
 * 输入驱动:
 *   - Amygdala CeA → 威胁/应激 → NE phasic burst
 *   - 新异刺激 → 短暂NE升高 → 探索模式
 *
 * 投射目标: 全脑 (volume transmission)
 *   → 皮层: 增益调制 (PSP放大)
 *   → BG: 探索噪声
 *   → 海马: 编码促进
 *
 * 设计文档: docs/02_neuron_system_design.md §6.2
 */

#include "region/brain_region.h"
#include "core/population.h"

namespace wuyun {

struct LCConfig {
    std::string name = "LC";
    size_t n_ne_neurons = 15;     // NE 神经元数 (蓝斑核很小)
    float  tonic_rate   = 0.2f;   // tonic baseline NE
    float  phasic_gain  = 0.4f;   // firing rate → phasic NE 增益
};

class LC_NE : public BrainRegion {
public:
    LC_NE(const LCConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_; }

    // --- NE 特有接口 ---

    /** 注入应激/唤醒信号 */
    void inject_arousal(float arousal);

    /** 获取当前 NE 输出水平 (0~1) */
    float ne_output() const { return ne_level_; }

    NeuronPopulation& neurons() { return ne_neurons_; }

private:
    LCConfig config_;
    NeuronPopulation ne_neurons_;

    float arousal_input_ = 0.0f;
    float ne_level_      = 0.2f;

    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_ne_;

    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;
};

} // namespace wuyun
