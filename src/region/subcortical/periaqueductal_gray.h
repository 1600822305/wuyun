#pragma once
/**
 * PeriaqueductalGray — 导水管周围灰质 (PAG)
 *
 * 核心功能: 不经过 BG 的应急防御行为输出
 *   CeA → PAG → 脑干运动核 → 冻结/逃跑
 *   比 BG DA-STDP 通路快得多 (硬连线, 无需学习)
 *
 * PAG 柱状组织 (Bandler & Shipley 1994):
 *   dlPAG (背外侧): 主动应对 — 逃跑、攻击 (fight-or-flight)
 *   vlPAG (腹外侧): 被动应对 — 冻结、不动 (freezing)
 *
 * 在 GridWorld 中的作用:
 *   1. CeA 高活跃 → PAG 激活 → 直接偏置 M1 运动输出 (不走 BG)
 *   2. 第一次遇到 danger 的即时反应 (BG 还没学会回避)
 *   3. 与 BG 习得性回避互补: PAG = 本能反射, BG = 习得策略
 *
 * 生物学:
 *   - CeA 是 PAG 的主要兴奋性输入 (LeDoux 1996)
 *   - PAG→SC 调制显著性 (恐惧增强视觉注意)
 *   - PAG→LC 增强 NE 释放 (恐惧→警觉)
 *   - PAG→VTA 调制 DA (防御行为抑制奖赏寻求)
 *
 * 设计文档: docs/01_brain_region_plan.md MB-07
 */

#include "region/brain_region.h"
#include "core/population.h"

namespace wuyun {

struct PAGConfig {
    std::string name = "PAG";
    size_t n_dlpag = 4;   // Dorsolateral: active coping (flight/fight)
    size_t n_vlpag = 4;   // Ventrolateral: passive coping (freeze)
};

class PeriaqueductalGray : public BrainRegion {
public:
    PeriaqueductalGray(const PAGConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_; }

    /** Inject CeA fear drive (amygdala output → PAG activation)
     *  High fear → dlPAG flight OR vlPAG freeze depending on threat proximity */
    void inject_fear(float cea_drive);

    /** Defense output: how strongly PAG is driving defensive motor behavior
     *  > 0: active defense (flight) — biases M1 away from threat
     *  Used by ClosedLoopAgent to inject emergency motor bias */
    float defense_output() const { return defense_level_; }

    /** Freeze output: how strongly PAG is suppressing movement
     *  > 0: passive defense — suppresses all motor output */
    float freeze_output() const { return freeze_level_; }

    /** Arousal drive to LC (fear → NE ↑ → heightened alertness) */
    float arousal_drive() const { return arousal_; }

    NeuronPopulation& dlpag() { return dlpag_; }
    NeuronPopulation& vlpag() { return vlpag_; }

private:
    PAGConfig config_;

    NeuronPopulation dlpag_;  // Active defense (flight)
    NeuronPopulation vlpag_;  // Passive defense (freeze)

    std::vector<float> psp_dl_;
    std::vector<float> psp_vl_;

    float fear_input_ = 0.0f;
    float defense_level_ = 0.0f;
    float freeze_level_ = 0.0f;
    float arousal_ = 0.0f;

    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;

    static constexpr float PSP_DECAY = 0.8f;
    // Threshold: fear must exceed this to activate PAG (prevents noise)
    static constexpr float FEAR_THRESHOLD = 0.03f;

    void aggregate_state();
};

} // namespace wuyun
