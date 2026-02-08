#pragma once
/**
 * NucleusAccumbens — 伏隔核 (腹侧纹状体)
 *
 * 核心功能: 动机/奖赏整合，独立于背侧纹状体的运动选择
 *   VTA→NAcc (中脑边缘通路): 奖赏预测 → 趋近动机
 *   Amygdala→NAcc: 情绪价值 → 回避动机
 *   Hippocampus→NAcc: 空间上下文 → 情境依赖动机
 *   NAcc→VP (腹侧苍白球): 动机 → 调制 BG 运动活力
 *
 * 与背侧 BG 的分工:
 *   NAcc (ventral): "多想动" (motivation/vigor)
 *   dStr (dorsal):  "往哪动" (action selection)
 *
 * 生物学: Mogenson 1980 "limbic-motor interface"
 *   NAcc Core: 工具性行为，趋近
 *   NAcc Shell: 新奇、环境变化检测
 *
 * 设计文档: docs/01_brain_region_plan.md BG-03
 */

#include "region/brain_region.h"
#include "core/population.h"

namespace wuyun {

struct NAccConfig {
    std::string name = "NAcc";
    size_t n_core_d1 = 4;    // Core D1 MSN (approach motivation)
    size_t n_core_d2 = 4;    // Core D2 MSN (avoidance motivation)
    size_t n_shell   = 4;    // Shell neurons (novelty/context change)
    size_t n_vp      = 4;    // Ventral Pallidum output
};

class NucleusAccumbens : public BrainRegion {
public:
    NucleusAccumbens(const NAccConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_; }

    /** Set DA level from VTA (mesolimbic pathway) */
    void set_da_level(float da) { da_level_ = da; }

    /** Motivation output: approach vs avoidance balance
     *  > 0: approach (core D1 dominant → increase motor vigor)
     *  < 0: avoidance (core D2 dominant → suppress motor output)
     *  Used by ClosedLoopAgent to modulate BG exploration/drive */
    float motivation_output() const { return motivation_; }

    /** Novelty signal from shell (high when unexpected input pattern)
     *  Drives ACh release via VP→NBM pathway */
    float novelty_signal() const { return novelty_; }

    NeuronPopulation& core_d1() { return core_d1_; }
    NeuronPopulation& core_d2() { return core_d2_; }
    NeuronPopulation& shell()   { return shell_; }
    NeuronPopulation& vp()      { return vp_; }

private:
    NAccConfig config_;
    float da_level_ = 0.3f;    // VTA DA (mesolimbic)
    float motivation_ = 0.0f;  // approach-avoidance balance
    float novelty_ = 0.0f;     // shell novelty detection

    // Populations
    NeuronPopulation core_d1_;  // Go/approach
    NeuronPopulation core_d2_;  // NoGo/avoidance
    NeuronPopulation shell_;    // Novelty/context
    NeuronPopulation vp_;       // Ventral pallidum output

    // PSP buffers
    std::vector<float> psp_d1_;
    std::vector<float> psp_d2_;
    std::vector<float> psp_shell_;
    std::vector<float> psp_vp_;

    // Shell activity tracking (for novelty detection)
    float shell_activity_smooth_ = 0.0f;

    // Aggregated firing state
    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;

    static constexpr float PSP_DECAY = 0.85f;
    static constexpr float DA_BASELINE = 0.3f;
    // D1: DA enhances approach (Surmeier 2007)
    static constexpr float D1_DA_GAIN = 40.0f;
    // D2: low DA enhances avoidance
    static constexpr float D2_DA_GAIN = 30.0f;
    // Shell novelty EMA
    static constexpr float NOVELTY_TAU = 0.95f;

    void aggregate_state();
};

} // namespace wuyun
