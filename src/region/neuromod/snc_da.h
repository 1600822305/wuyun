#pragma once
/**
 * SNc_DA — 黑质致密部多巴胺系统 (习惯学习通路)
 *
 * 与 VTA 的分工:
 *   VTA (mesolimbic):  phasic RPE → NAcc/PFC → 目标导向学习 (新行为)
 *   SNc (nigrostriatal): tonic DA → 背侧纹状体 → 习惯维持 (已学行为)
 *
 * 核心功能:
 *   1. 维持稳定的 tonic DA → 背侧 BG 已学权重不退化
 *   2. 当行为被反复奖励时，SNc tonic 逐渐升高 → 习惯巩固
 *   3. SNc 不像 VTA 那样对单次 RPE 敏感 → 抗波动
 *
 * 生物学:
 *   - SNc DA 神经元投射到背侧纹状体 (caudate/putamen)
 *   - 帕金森病 = SNc 退化 → 运动障碍 (而非奖励障碍)
 *   - SNc DA 对运动执行至关重要 (Haber 2003)
 *   - 习惯形成: 目标导向(VTA/ventral) → 习惯(SNc/dorsal) 转移 (Yin & Knowlton 2006)
 *
 * 设计文档: docs/01_brain_region_plan.md MB-04
 */

#include "region/brain_region.h"
#include "core/population.h"

namespace wuyun {

struct SNcConfig {
    std::string name = "SNc";
    size_t n_da_neurons = 4;
    float  tonic_rate   = 0.3f;   // Baseline tonic DA (same as VTA)
    float  habit_lr     = 0.002f; // Slow habit consolidation rate
};

class SNc_DA : public BrainRegion {
public:
    SNc_DA(const SNcConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_; }

    /** Current DA output level (tonic-dominant, stable) */
    float da_output() const { return da_level_; }

    /** Inject reward signal (for slow habit consolidation)
     *  Unlike VTA: SNc responds slowly to repeated rewards, not single events */
    void inject_reward_history(float avg_reward) { reward_history_ = avg_reward; }

    /** Inject BG D1 activity (dorsal striatal feedback)
     *  Well-learned actions → high D1 activity → SNc tonic maintenance */
    void inject_d1_activity(float d1_rate) { d1_activity_ = d1_rate; }

private:
    SNcConfig config_;
    NeuronPopulation da_pop_;

    float da_level_;
    float tonic_baseline_;     // Slowly adapting baseline
    float reward_history_ = 0.0f;
    float d1_activity_ = 0.0f;

    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;
    std::vector<float>   psp_buf_;

    static constexpr float PSP_DECAY = 0.85f;
};

} // namespace wuyun
