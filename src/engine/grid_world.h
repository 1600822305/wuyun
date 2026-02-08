#pragma once
/**
 * GridWorld — 简单2D格子世界
 *
 * 功能:
 *   - NxN 网格 (默认 10×10), Agent 可移动
 *   - 可配置局部视野 (默认 5×5, vision_radius=2)
 *   - 食物 (奖励 +1), 危险 (惩罚 -1), 墙壁 (无法通过)
 *   - 食物被吃后在随机位置重生
 *
 * 格子类型:
 *   EMPTY=0, FOOD=1, DANGER=2, WALL=3
 *
 * 视觉编码:
 *   EMPTY=0.0, FOOD=0.9, DANGER=0.3, WALL=0.1, AGENT=0.6 (自身位置)
 *   → NxN patch → VisualInput center-surround → LGN
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

/** v48: Maze layout presets */
enum class MazeType : uint8_t {
    OPEN_FIELD  = 0,   // Default: no internal walls, random food/danger
    T_MAZE      = 1,   // T-shaped choice point (Packard & McGaugh 1996)
    CORRIDOR    = 2,   // Straight corridor, food at end (delayed reward)
    SIMPLE_MAZE = 3,   // 7x7 maze with turns
};

struct GridWorldConfig {
    size_t width       = 10;
    size_t height      = 10;
    size_t n_food      = 5;     // 食物数量 (v21: 3→5, 更丰富的觅食环境)
    size_t n_danger    = 3;     // 危险数量 (v21: 2→3, 3%密度, 可学习的回避挑战)
    uint32_t seed      = 42;    // 随机种子
    int    vision_radius = 2;   // 视野半径 (v21: 1→2, 3×3→5×5, 释放PC/睡眠/空间记忆)
    MazeType maze_type = MazeType::OPEN_FIELD;  // v48: maze layout

    // 视觉编码值
    float vis_empty    = 0.0f;
    float vis_food     = 0.9f;
    float vis_danger   = 0.3f;
    float vis_wall     = 0.1f;
    float vis_agent    = 0.6f;

    // 计算视野尺寸
    size_t vision_side() const { return static_cast<size_t>(2 * vision_radius + 1); }
    size_t vision_pixels() const { return vision_side() * vision_side(); }
};

struct StepResult {
    float  reward    = 0.0f;   // 本步奖励 (+1 food, -1 danger, 0 otherwise)
    bool   got_food  = false;
    bool   hit_danger= false;
    bool   hit_wall  = false;
    int    agent_x   = 0;
    int    agent_y   = 0;
    // v55: continuous position (same as int when using discrete act())
    float  agent_fx  = 0.0f;
    float  agent_fy  = 0.0f;
};

class GridWorld {
public:
    explicit GridWorld(const GridWorldConfig& config = {});

    /** 重置世界 (Agent回到起点, 重新放置食物/危险) */
    void reset();

    /** v53: 换种子重置 (反转学习: 保持大脑, 换世界布局)
     *  改变 RNG 种子 → 食物/危险位置完全不同 */
    void reset_with_seed(uint32_t new_seed);

    /** Agent 执行动作 (离散: ±1格移动), 返回结果 */
    StepResult act(Action action);

    /** v55: Agent 连续移动 (dx,dy 浮点位移), 碰撞检测用 floor 格子
     *  Biology: 真实运动是连续的, 格子只是底层基板
     *  @param dx, dy  位移 (|displacement| ≤ 1.0 per step 典型)
     *  @return StepResult 含浮点坐标 */
    StepResult act_continuous(float dx, float dy);

    /** 获取 NxN 局部视野 (N=2*vision_radius+1, 行优先) */
    std::vector<float> observe() const;

    /** 获取完整视野 (长度=width*height, 用于可视化) */
    std::vector<float> full_observation() const;

    // --- 访问器 ---
    int agent_x() const { return agent_x_; }
    int agent_y() const { return agent_y_; }
    // v55: continuous position accessors
    float agent_fx() const { return agent_fx_; }
    float agent_fy() const { return agent_fy_; }
    size_t width()  const { return config_.width; }
    size_t height() const { return config_.height; }
    CellType cell(int x, int y) const;

    /** v48: Set a specific cell type at position (for maze building) */
    void set_cell(int x, int y, CellType type);

    /** v48: Set agent start position */
    void set_agent_pos(int x, int y);

    /** 累计统计 */
    uint32_t total_food_collected() const { return food_collected_; }
    uint32_t total_danger_hits()    const { return danger_hits_; }
    uint32_t total_steps()          const { return step_count_; }

    /** 获取文本表示 (调试) */
    std::string to_string() const;

private:
    /** v48: Load predefined maze layout */
    void load_maze(MazeType type);
    GridWorldConfig config_;
    std::vector<CellType> grid_;  // row-major [y * width + x]
    int agent_x_ = 0, agent_y_ = 0;
    float agent_fx_ = 0.0f, agent_fy_ = 0.0f;  // v55: continuous position
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
