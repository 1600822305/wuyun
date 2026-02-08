#pragma once
/**
 * SuperiorColliculus — 上丘 (皮层下快速显著性检测)
 *
 * 核心功能: 视网膜→上丘→Pulvinar 的快速通道 (~60ms)
 *   比皮层通路 (LGN→V1→...→dlPFC, ~14步) 快得多
 *   编码视觉显著性 (亮度变化、运动、突然出现的物体)
 *   不编码物体身份 (那是皮层的工作)
 *
 * 与丘脑纹状体通路 (Step 38) 的区别:
 *   Step 38 LGN→BG: 粗糙显著性直接驱动 MSN up-state
 *   SC: 更精细的显著性检测 + 注意力定向 + Pulvinar 增强
 *
 * 在 GridWorld 中的作用:
 *   1. 快速检测 danger (不等皮层处理完)
 *   2. 通过 Pulvinar 增强视觉层级中的显著刺激
 *   3. 提供显著性信号给 BG (补充皮层慢通路)
 *
 * 生物学:
 *   - 视网膜 → SC 浅层 (视觉地图)
 *   - SC 深层 → 动眼 + 头转向 + Pulvinar + BG
 *   - Ingle 1973: SC 是两栖类的"整个视觉大脑"
 *   - Krauzlis 2013: SC 在灵长类中负责注意力定向
 *
 * 设计文档: docs/01_brain_region_plan.md MB-01
 */

#include "region/brain_region.h"
#include "core/population.h"

namespace wuyun {

struct SCConfig {
    std::string name = "SC";
    size_t n_superficial = 4;  // Superficial layer (visual map, retinotopic)
    size_t n_deep        = 4;  // Deep layer (multimodal integration, motor output)
};

class SuperiorColliculus : public BrainRegion {
public:
    SuperiorColliculus(const SCConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_; }

    /** Saliency output: how salient is the current visual scene?
     *  High saliency → Pulvinar enhancement + BG arousal */
    float saliency_output() const { return saliency_; }

    NeuronPopulation& superficial() { return superficial_; }
    NeuronPopulation& deep()        { return deep_; }

private:
    SCConfig config_;

    NeuronPopulation superficial_;  // Visual map (retinotopic)
    NeuronPopulation deep_;         // Multimodal integration + output

    std::vector<float> psp_sup_;
    std::vector<float> psp_deep_;

    // Saliency tracking: detects change in input pattern
    float saliency_ = 0.0f;
    float prev_input_level_ = 0.0f;

    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;

    static constexpr float PSP_DECAY = 0.8f;  // Fast decay (SC is fast processing)

    void aggregate_state();
};

} // namespace wuyun
