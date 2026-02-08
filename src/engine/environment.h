#pragma once
/**
 * Environment — 抽象环境接口
 *
 * 定义大脑与外部世界的交互协议:
 *   - observe()  : 感觉输入 (视觉 patch → LGN)
 *   - step()     : 运动输出 (M1 群体向量 → 位移) + 奖赏反馈
 *   - pos/world  : 空间信息 (→ 海马/认知地图)
 *
 * 设计原则:
 *   - 只暴露大脑需要的信息, 不暴露环境内部结构
 *   - observe() 返回通用 float 向量 (视觉/任何 2D 传感器)
 *   - 空间信息独立于视觉 (海马不需要知道"格子")
 *   - 统计用 positive/negative 而非 food/danger (语义无关)
 *
 * 实现:
 *   - GridWorldEnv : 10×10 格子世界 (食物/危险/墙壁)
 *   - (future) MultiRoomEnv, ContinuousArena, ...
 */

#include <vector>
#include <cstdint>

namespace wuyun {

class Environment {
public:
    virtual ~Environment() = default;

    // --- Lifecycle ---
    virtual void reset() = 0;
    virtual void reset_with_seed(uint32_t seed) = 0;

    // --- Sensory ---
    /** 获取当前观测 (视觉 patch, row-major float[vis_width * vis_height]) */
    virtual std::vector<float> observe() const = 0;
    virtual size_t vis_width() const = 0;
    virtual size_t vis_height() const = 0;

    // --- Motor ---
    struct Result {
        float reward         = 0.0f;
        bool  positive_event = false;   // food-like reward event
        bool  negative_event = false;   // danger-like punishment event
        float pos_x          = 0.0f;    // agent position after step
        float pos_y          = 0.0f;
    };
    /** 执行连续运动 (dx, dy 浮点位移), 返回结果 */
    virtual Result step(float dx, float dy) = 0;

    // --- Spatial (for hippocampus / cognitive map) ---
    virtual float pos_x() const = 0;
    virtual float pos_y() const = 0;
    virtual float world_width() const = 0;
    virtual float world_height() const = 0;

    // --- Statistics (for evolution fitness) ---
    virtual uint32_t positive_count() const = 0;
    virtual uint32_t negative_count() const = 0;
    virtual uint32_t step_count() const = 0;
};

} // namespace wuyun
