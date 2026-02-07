#pragma once
/**
 * GridWorld — 简单2D格子世界
 *
 * 功能:
 *   - NxN 网格, Agent 可移动
 *   - 3x3 局部视野 (编码为 9 像素灰度)
 *   - 食物 (奖励 +1), 危险 (惩罚 -1), 墙壁 (无法通过)
 *   - 食物被吃后在随机位置重生
 *
 * 格子类型:
 *   EMPTY=0, FOOD=1, DANGER=2, WALL=3
 *
 * 视觉编码:
 *   EMPTY=0.0, FOOD=0.8, DANGER=0.3, WALL=0.1, AGENT=0.5 (自身位置)
 *   → 3x3 patch → VisualInput center-surround → LGN
 */

#include <vector>
#include <cstdint>
#include <random>
#include <string>

namespace wuyun {

enum class CellType : uint8_t {
    EMPTY  = 0,
    FOOD   = 1,
    DANGER = 2,
    WALL   = 3
};

enum class Action : uint8_t {
    UP    = 0,
    DOWN  = 1,
    LEFT  = 2,
    RIGHT = 3,
    STAY  = 4   // no-op (when M1 is silent)
};

struct GridWorldConfig {
    size_t width       = 10;
    size_t height      = 10;
    size_t n_food      = 3;     // 食物数量
    size_t n_danger    = 2;     // 危险数量
    uint32_t seed      = 42;    // 随机种子

    // 视觉编码值
    float vis_empty    = 0.0f;
    float vis_food     = 0.9f;
    float vis_danger   = 0.3f;
    float vis_wall     = 0.1f;
    float vis_agent    = 0.6f;
};

struct StepResult {
    float  reward    = 0.0f;   // 本步奖励 (+1 food, -1 danger, 0 otherwise)
    bool   got_food  = false;
    bool   hit_danger= false;
    bool   hit_wall  = false;
    int    agent_x   = 0;
    int    agent_y   = 0;
};

class GridWorld {
public:
    explicit GridWorld(const GridWorldConfig& config = {});

    /** 重置世界 (Agent回到起点, 重新放置食物/危险) */
    void reset();

    /** Agent 执行动作, 返回结果 */
    StepResult act(Action action);

    /** 获取 3x3 局部视野 (长度=9, 行优先) */
    std::vector<float> observe() const;

    /** 获取完整视野 (长度=width*height, 用于可视化) */
    std::vector<float> full_observation() const;

    // --- 访问器 ---
    int agent_x() const { return agent_x_; }
    int agent_y() const { return agent_y_; }
    size_t width()  const { return config_.width; }
    size_t height() const { return config_.height; }
    CellType cell(int x, int y) const;

    /** 累计统计 */
    uint32_t total_food_collected() const { return food_collected_; }
    uint32_t total_danger_hits()    const { return danger_hits_; }
    uint32_t total_steps()          const { return step_count_; }

    /** 获取文本表示 (调试) */
    std::string to_string() const;

private:
    GridWorldConfig config_;
    std::vector<CellType> grid_;  // row-major [y * width + x]
    int agent_x_ = 0, agent_y_ = 0;
    std::mt19937 rng_;

    uint32_t food_collected_ = 0;
    uint32_t danger_hits_    = 0;
    uint32_t step_count_     = 0;

    size_t idx(int x, int y) const { return static_cast<size_t>(y) * config_.width + x; }
    bool in_bounds(int x, int y) const {
        return x >= 0 && x < (int)config_.width && y >= 0 && y < (int)config_.height;
    }
    void place_random(CellType type, size_t count);
    void respawn_food();
    float cell_to_visual(int x, int y) const;
};

} // namespace wuyun
