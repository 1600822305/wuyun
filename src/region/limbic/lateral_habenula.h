#pragma once
/**
 * LateralHabenula — 外侧缰核 (负奖励预测误差中心)
 *
 * 核心功能 (Matsumoto & Hikosaka 2007):
 *   - 编码 **负RPE**: 预期奖励未出现 或 遭遇惩罚 → LHb 兴奋
 *   - LHb → RMTg(GABA) → VTA(DA): 抑制VTA多巴胺释放 → DA pause
 *   - 与VTA互补: VTA编码正RPE, LHb编码负RPE
 *   - 对强化学习至关重要: 没有负信号就无法学会"回避"
 *
 * 投射:
 *   LHb → VTA (抑制, 经RMTg中继, 此处简化为直接抑制)
 *   LHb → DRN (5-HT调制)
 *
 * 输入:
 *   - 惩罚/厌恶刺激 (danger事件)
 *   - 期望落空信号 (expected reward - actual reward, when negative)
 *   - 基底节GPb(边界苍白球) → LHb (此处由agent直接注入简化)
 *
 * 设计文档: docs/01_brain_region_plan.md §2.2.3 ET-01
 */

#include "region/brain_region.h"
#include "core/population.h"

namespace wuyun {

struct LHbConfig {
    std::string name = "LHb";
    size_t n_neurons = 25;      // LHb神经元数 (小核团)

    float  punishment_gain = 1.5f;  // 惩罚信号 → LHb兴奋增益
    float  frustration_gain = 1.0f; // 期望落空 → LHb兴奋增益
    float  tonic_drive = 8.0f;      // 基线驱动 (低频自发活动 ~2Hz)

    // LHb输出 → VTA抑制强度
    // Biology: LHb → RMTg(GABA中间神经元) → VTA DA neurons
    // 简化: LHb firing rate × inhibition_gain = VTA DA抑制量
    float  vta_inhibition_gain = 0.8f;
};

class LateralHabenula : public BrainRegion {
public:
    explicit LateralHabenula(const LHbConfig& config = {});

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_; }

    // --- LHb-specific interface ---

    /** 注入惩罚信号 (danger碰撞等厌恶刺激) */
    void inject_punishment(float punishment);

    /** 注入期望落空信号 (frustrative non-reward: expected > actual) */
    void inject_frustration(float frustration);

    /** 获取当前LHb输出水平 (用于抑制VTA) */
    float output_level() const { return output_level_; }

    /** 获取对VTA的抑制量 (0~1, 越高抑制越强) */
    float vta_inhibition() const { return vta_inhibition_; }

    NeuronPopulation& neurons() { return neurons_; }

private:
    LHbConfig config_;
    NeuronPopulation neurons_;

    float punishment_input_  = 0.0f;
    float frustration_input_ = 0.0f;
    float output_level_      = 0.0f;  // LHb firing rate (normalized)
    float vta_inhibition_    = 0.0f;  // Computed inhibition for VTA

    // Sustained drive buffer (exponential decay, like VTA reward_psp_)
    float aversive_psp_ = 0.0f;
    static constexpr float AVERSIVE_PSP_DECAY = 0.88f;

    // PSP buffer for cross-region input
    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_;

    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;
};

} // namespace wuyun
