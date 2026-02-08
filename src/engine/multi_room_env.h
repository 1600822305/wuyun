#pragma once
/**
 * MultiRoomEnv — 多房间迷宫环境
 *
 * 完全独立于 GridWorld, 验证 Environment 抽象接口。
 *
 * 结构:
 *   n_rooms_x × n_rooms_y 个房间, 由墙壁分隔, 门道连接。
 *   食物/危险随机分布在房间内。
 *   Agent 必须穿越门道才能到达其他房间的食物。
 *
 * 挑战 (vs GridWorld open field):
 *   1. 导航: 必须找到并穿过门道
 *   2. 空间记忆: 记住哪个房间有食物
 *   3. 探索: 系统性搜索多个房间
 *
 * 内部用 grid 表示 (方便碰撞+观测), 但这是实现细节,
 * 对外只暴露 Environment 接口。
 */

#include "engine/environment.h"
#include <vector>
#include <cstdint>
#include <random>
#include <string>

namespace wuyun {

struct MultiRoomConfig {
    size_t n_rooms_x    = 2;       // 房间列数
    size_t n_rooms_y    = 2;       // 房间行数
    size_t room_w       = 4;       // 每个房间内部宽度 (不含墙)
    size_t room_h       = 4;       // 每个房间内部高度 (不含墙)
    size_t n_food       = 4;       // 食物数量
    size_t n_danger     = 2;       // 危险数量
    int    vision_radius = 2;      // 视野半径
    uint32_t seed       = 42;

    // 视觉编码值 (与 GridWorldConfig 兼容)
    float vis_empty  = 0.0f;
    float vis_food   = 0.9f;
    float vis_danger = 0.3f;
    float vis_wall   = 0.1f;
    float vis_agent  = 0.6f;
    float vis_door   = 0.05f;     // 门道略亮于墙壁

    // 计算总网格尺寸: rooms * (room_size + 1) + 1 (墙壁占 1 格)
    size_t grid_width()  const { return n_rooms_x * (room_w + 1) + 1; }
    size_t grid_height() const { return n_rooms_y * (room_h + 1) + 1; }
    size_t vision_side() const { return static_cast<size_t>(2 * vision_radius + 1); }
};

class MultiRoomEnv : public Environment {
public:
    explicit MultiRoomEnv(const MultiRoomConfig& cfg = {});

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

    // --- MultiRoom-specific ---
    std::string to_string() const;
    size_t grid_w() const { return grid_w_; }
    size_t grid_h() const { return grid_h_; }

private:
    enum class Cell : uint8_t { EMPTY = 0, FOOD = 1, DANGER = 2, WALL = 3 };

    void build_rooms();
    void place_random(Cell type, size_t count);
    void respawn_food();
    float cell_visual(int x, int y) const;

    MultiRoomConfig cfg_;
    size_t grid_w_, grid_h_;
    std::vector<Cell> grid_;       // row-major [y * grid_w_ + x]

    float agent_fx_ = 1.5f, agent_fy_ = 1.5f;
    int   agent_ix_ = 1,    agent_iy_ = 1;

    std::mt19937 rng_;
    uint32_t food_collected_ = 0;
    uint32_t danger_hits_    = 0;
    uint32_t step_count_     = 0;

    size_t idx(int x, int y) const { return static_cast<size_t>(y) * grid_w_ + x; }
    bool in_bounds(int x, int y) const {
        return x >= 0 && x < (int)grid_w_ && y >= 0 && y < (int)grid_h_;
    }
    bool is_passable(int x, int y) const {
        return in_bounds(x, y) && grid_[idx(x, y)] != Cell::WALL;
    }
};

} // namespace wuyun
