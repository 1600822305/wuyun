#pragma once
/**
 * SuperiorColliculus — 上丘 (皮层下快速显著性检测 + 定向反射弧)
 *
 * 核心功能: 视网膜→上丘→脑干运动核 的快速通道 (~2-3步)
 *   比皮层通路 (LGN→V1→...→dlPFC, ~14步) 快得多
 *   编码视觉显著性 (亮度变化、运动、突然出现的物体)
 *   不编码物体身份 (那是皮层的工作)
 *
 * v52 反射弧升级:
 *   SC 浅层: 视网膜输入 → 视觉地图 (retinotopic)
 *   SC 深层: 方向性运动地图 → 定向反射
 *     深层神经元有偏好方向 (像 M1 群体向量)
 *     inject_visual_patch() 计算显著性质心 → 方向性深层激活
 *     深层发放 → agent 读取群体向量 → M1 注入 = 趋近反射
 *
 *   生物学: SC 深层 = 运动地图 + 视觉地图对齐
 *     Stein & Meredith 1993: SC 深层编码朝向运动方向
 *     Ingle 1973: 蛙 SC = 整个视觉大脑, 直接驱动转向
 *     Krauzlis 2013: 灵长类 SC 深层 → saccade + 注意力转移
 *
 *   先天回路 (不学习):
 *     视觉刺激出现 → SC 计算方位 → 定向朝向 → 趋近
 *     = "看到东西就走过去看看" 的本能
 *     这条通路的存在本身就是先验, 不需要注册表
 *
 * 设计文档: docs/01_brain_region_plan.md MB-01
 */

#include "region/brain_region.h"
#include "core/population.h"
#include <vector>

namespace wuyun {

struct SCConfig {
    std::string name = "SC";
    size_t n_superficial = 4;  // Superficial layer (visual map, retinotopic)
    size_t n_deep        = 4;  // Deep layer (motor map, directional orientation)
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

    // --- v52: 视觉定向反射 ---

    /** 注入视觉视野 (每个 brain step 调用)
     *  计算视觉显著性质心 → 方向 + 强度
     *  注入方向性电流到深层神经元 (偏好方向匹配)
     *  生物学: 视网膜 → SC 浅层 (retinotopic) → SC 深层 (运动地图)
     *  @param pixels  视觉 patch (row-major, N×N)
     *  @param width   patch 宽度
     *  @param height  patch 高度
     *  @param gain    注入增益 (AgentConfig.sc_approach_gain) */
    void inject_visual_patch(const std::vector<float>& pixels,
                             int width, int height, float gain);

    /** 显著性方向角 (弧度, 0=RIGHT, π/2=UP) */
    float saliency_direction() const { return saliency_direction_; }
    /** 显著性强度 (0 = 无刺激, >1 = 强刺激) */
    float saliency_magnitude() const { return saliency_magnitude_; }

    /** 深层神经元偏好方向 (与 M1 群体向量对齐) */
    const std::vector<float>& deep_preferred_dir() const { return deep_preferred_dir_; }

    NeuronPopulation& superficial() { return superficial_; }
    NeuronPopulation& deep()        { return deep_; }

private:
    SCConfig config_;

    NeuronPopulation superficial_;  // Visual map (retinotopic)
    NeuronPopulation deep_;         // Motor map (directional)

    std::vector<float> psp_sup_;
    std::vector<float> psp_deep_;

    // v52: 深层运动地图 — 每个神经元有偏好方向
    std::vector<float> deep_preferred_dir_;

    // Saliency tracking: detects change in input pattern
    float saliency_ = 0.0f;
    float prev_input_level_ = 0.0f;

    // v52: 方向性显著性 (从 inject_visual_patch 计算)
    float saliency_direction_ = 0.0f;   // angle (radians)
    float saliency_magnitude_ = 0.0f;   // strength

    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;

    static constexpr float PSP_DECAY = 0.8f;  // Fast decay (SC is fast processing)

    void aggregate_state();
};

} // namespace wuyun
