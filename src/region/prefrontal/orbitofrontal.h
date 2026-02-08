#pragma once
/**
 * OrbitofrontalCortex — 眶额皮层 (OFC, BA11/47)
 *
 * 核心功能: 刺激-结果关联 (stimulus-outcome learning)
 *   不同于 BG 的动作-结果学习, OFC 编码"看到什么→预期什么奖赏"
 *   当奖赏预期改变时 (reversal), OFC 快速更新
 *
 * 在 GridWorld 中的作用:
 *   1. IT "food-like" 视觉模式 → OFC 正价值神经元活跃 → dlPFC 趋近偏置
 *   2. IT "danger-like" 模式 → OFC 负价值神经元活跃 → dlPFC 回避偏置
 *   3. VTA DA 到达 → OFC 更新价值表征 (类似 STDP: IT+DA 共现 → 学习)
 *
 * 与其他区域的分工:
 *   - BG: 动作→结果 (哪个方向好?) → DA-STDP
 *   - OFC: 刺激→结果 (这个东西好不好?) → 价值预测
 *   - Amygdala: 刺激→情绪 (这个东西可怕吗?) → 快速, 难消退
 *   - vmPFC: 价值→决策 (综合评估后要不要做?) → 情绪调节
 *
 * 生物学:
 *   - Rolls 2000: OFC 编码主观价值和期望奖赏
 *   - Wallis 2007: OFC → dlPFC 价值信号引导决策
 *   - OFC 损伤 → 反转学习障碍, 持续追逐旧奖赏
 *
 * 设计文档: docs/01_brain_region_plan.md A-02
 */

#include "region/brain_region.h"
#include "core/population.h"

namespace wuyun {

struct OFCConfig {
    std::string name = "OFC";
    size_t n_value_pos = 4;  // Positive value neurons (food-predicting)
    size_t n_value_neg = 4;  // Negative value neurons (danger-predicting)
    size_t n_inh       = 4;  // Inhibitory (E/I balance + value competition)
};

class OrbitofrontalCortex : public BrainRegion {
public:
    OrbitofrontalCortex(const OFCConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_; }

    /** DA modulation — volume transmission from VTA (ModulationBus, not SpikeBus)
     *  High DA → strengthen current value associations
     *  Low DA → weaken (reversal learning) */
    void set_da_level(float da) { da_level_ = da; }

    /** Diagnostic: net value signal (positive = expect reward, negative = expect punishment) */
    float value_signal() const { return value_signal_; }

    NeuronPopulation& value_pos() { return value_pos_; }
    NeuronPopulation& value_neg() { return value_neg_; }

private:
    OFCConfig config_;

    NeuronPopulation value_pos_;  // Positive expected value
    NeuronPopulation value_neg_;  // Negative expected value
    NeuronPopulation inh_;        // Inhibitory interneurons

    std::vector<float> psp_pos_;
    std::vector<float> psp_neg_;
    std::vector<float> psp_inh_;

    float da_level_ = 0.3f;       // Baseline DA
    float value_signal_ = 0.0f;   // Net value output (diagnostic)

    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;

    static constexpr float PSP_DECAY = 0.85f;

    void aggregate_state();
};

} // namespace wuyun
