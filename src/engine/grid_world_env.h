#pragma once
/**
 * GridWorldEnv — GridWorld 环境适配器
 *
 * 将 GridWorld 包装为通用 Environment 接口。
 * GridWorld 本身零修改, 所有方法一行代理。
 *
 * 使用方式:
 *   auto env = std::make_unique<GridWorldEnv>(world_config);
 *   ClosedLoopAgent agent(std::move(env), agent_config);
 *
 * GridWorld 特有功能 (迷宫/可视化) 通过 grid_world() 下转型访问。
 */

#include "engine/environment.h"
#include "engine/grid_world.h"

namespace wuyun {

class GridWorldEnv : public Environment {
public:
    explicit GridWorldEnv(const GridWorldConfig& cfg);

    // --- Environment interface ---
    void reset() override;
    void reset_with_seed(uint32_t seed) override;

    std::vector<float> observe() const override;
    size_t vis_width() const override;
    size_t vis_height() const override;

    Result step(float dx, float dy) override;

    float pos_x() const override;
    float pos_y() const override;
    float world_width() const override;
    float world_height() const override;

    uint32_t positive_count() const override;
    uint32_t negative_count() const override;
    uint32_t step_count() const override;

    // --- GridWorld-specific access (tests/visualization can downcast) ---
    GridWorld& grid_world() { return world_; }
    const GridWorld& grid_world() const { return world_; }

private:
    GridWorld world_;
    size_t vis_w_;   // cached vision_side()
    size_t vis_h_;
};

} // namespace wuyun
