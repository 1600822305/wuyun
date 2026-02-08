#include "engine/grid_world.h"
#include <algorithm>
#include <sstream>

namespace wuyun {

GridWorld::GridWorld(const GridWorldConfig& config)
    : config_(config)
    , grid_(config.width * config.height, CellType::EMPTY)
    , rng_(config.seed)
{
    reset();
}

void GridWorld::reset() {
    food_collected_ = 0;
    danger_hits_    = 0;
    step_count_     = 0;

    if (config_.maze_type != MazeType::OPEN_FIELD) {
        // v48: Load predefined maze layout (sets grid, agent pos, food)
        load_maze(config_.maze_type);
    } else {
        // Original: open field with random food/danger
        std::fill(grid_.begin(), grid_.end(), CellType::EMPTY);
        agent_x_ = static_cast<int>(config_.width / 2);
        agent_y_ = static_cast<int>(config_.height / 2);
        agent_fx_ = static_cast<float>(agent_x_) + 0.5f;
        agent_fy_ = static_cast<float>(agent_y_) + 0.5f;
        place_random(CellType::FOOD,   config_.n_food);
        place_random(CellType::DANGER, config_.n_danger);
    }
}

void GridWorld::reset_with_seed(uint32_t new_seed) {
    config_.seed = new_seed;
    rng_.seed(new_seed);
    reset();
}

void GridWorld::set_cell(int x, int y, CellType type) {
    if (in_bounds(x, y)) {
        grid_[idx(x, y)] = type;
    }
}

void GridWorld::set_agent_pos(int x, int y) {
    if (in_bounds(x, y)) {
        agent_x_ = x;
        agent_y_ = y;
        agent_fx_ = static_cast<float>(x) + 0.5f;
        agent_fy_ = static_cast<float>(y) + 0.5f;
    }
}

void GridWorld::load_maze(MazeType type) {
    // Resize grid if maze requires different dimensions
    auto set_size = [&](size_t w, size_t h) {
        config_.width = w;
        config_.height = h;
        grid_.resize(w * h);
    };

    std::fill(grid_.begin(), grid_.end(), CellType::EMPTY);

    switch (type) {

    case MazeType::T_MAZE: {
        // T-maze (5x4): minimal choice paradigm
        // Agent at junction, food visible, 3 steps to reach
        // Tests: can the brain learn "go left then up" when food is visible?
        //
        //  ##### y=0
        //  #F.E# y=1   F=food(1,1), E=empty(3,1)
        //  #.#.# y=2   Wall(2,2) forces left/right choice
        //  #.A.# y=3   Agent(2,3)
        //  ##### y=4   (outer wall added by boundary check)
        //
        // Left path: (2,3)→(1,3)→(1,2)→(1,1)=FOOD (3 steps)
        // Right path: (2,3)→(3,3)→(3,2)→(3,1)=EMPTY (3 steps, no reward)
        // 5x5 vision from (2,3) sees entire grid → food VISIBLE from start
        set_size(5, 5);
        // Border walls
        for (int x = 0; x < 5; ++x) {
            grid_[idx(x, 0)] = CellType::WALL;
            grid_[idx(x, 4)] = CellType::WALL;
        }
        for (int y = 0; y < 5; ++y) {
            grid_[idx(0, y)] = CellType::WALL;
            grid_[idx(4, y)] = CellType::WALL;
        }
        // Central divider (forces left/right choice)
        grid_[idx(2, 2)] = CellType::WALL;

        // Food in left arm only (no danger — pure choice task)
        grid_[idx(1, 1)] = CellType::FOOD;

        // Agent at bottom center
        agent_x_ = 2;
        agent_y_ = 3;
        break;
    }

    case MazeType::CORRIDOR: {
        // Corridor (10x3): straight path, food at end
        // Tests delayed reward credit assignment over 8 steps
        //
        //  ########## y=0
        //  #A......F# y=1
        //  ########## y=2
        set_size(10, 3);
        for (int x = 0; x < 10; ++x) {
            grid_[idx(x, 0)] = CellType::WALL;
            grid_[idx(x, 2)] = CellType::WALL;
        }
        grid_[idx(0, 1)] = CellType::WALL;
        grid_[idx(9, 1)] = CellType::WALL;
        // Food at right end
        grid_[idx(8, 1)] = CellType::FOOD;
        // Agent at left
        agent_x_ = 1;
        agent_y_ = 1;
        break;
    }

    case MazeType::SIMPLE_MAZE: {
        // Simple maze (7x7) with two turns
        //
        //  ####### y=0
        //  #A.#..# y=1
        //  ##.#.## y=2
        //  #.....# y=3
        //  #.###.# y=4
        //  #....F# y=5
        //  ####### y=6
        set_size(7, 7);
        // Border walls
        for (int x = 0; x < 7; ++x) {
            grid_[idx(x, 0)] = CellType::WALL;
            grid_[idx(x, 6)] = CellType::WALL;
        }
        for (int y = 0; y < 7; ++y) {
            grid_[idx(0, y)] = CellType::WALL;
            grid_[idx(6, y)] = CellType::WALL;
        }
        // Internal walls
        grid_[idx(3, 1)] = CellType::WALL;
        grid_[idx(3, 2)] = CellType::WALL;
        grid_[idx(1, 2)] = CellType::WALL;
        grid_[idx(5, 2)] = CellType::WALL;
        grid_[idx(2, 4)] = CellType::WALL;
        grid_[idx(3, 4)] = CellType::WALL;
        grid_[idx(4, 4)] = CellType::WALL;

        // Food at bottom right
        grid_[idx(5, 5)] = CellType::FOOD;
        // Agent at top left
        agent_x_ = 1;
        agent_y_ = 1;
        break;
    }

    default:
        // OPEN_FIELD handled in reset() directly
        break;
    }
}

void GridWorld::place_random(CellType type, size_t count) {
    std::uniform_int_distribution<int> dx(0, static_cast<int>(config_.width - 1));
    std::uniform_int_distribution<int> dy(0, static_cast<int>(config_.height - 1));

    for (size_t i = 0; i < count; ++i) {
        for (int attempt = 0; attempt < 100; ++attempt) {
            int x = dx(rng_);
            int y = dy(rng_);
            // Don't place on agent or occupied cell
            if (x == agent_x_ && y == agent_y_) continue;
            if (grid_[idx(x, y)] != CellType::EMPTY) continue;
            grid_[idx(x, y)] = type;
            break;
        }
    }
}

void GridWorld::respawn_food() {
    if (config_.maze_type != MazeType::OPEN_FIELD) {
        // v48: In maze mode, reset the maze layout to respawn food at its fixed position
        // This also resets agent to start position (trial-based learning)
        load_maze(config_.maze_type);
    } else {
        place_random(CellType::FOOD, 1);
    }
}

CellType GridWorld::cell(int x, int y) const {
    if (!in_bounds(x, y)) return CellType::WALL;
    return grid_[idx(x, y)];
}

StepResult GridWorld::act(Action action) {
    StepResult result;
    step_count_++;

    int nx = agent_x_, ny = agent_y_;
    switch (action) {
        case Action::UP:    ny--; break;
        case Action::DOWN:  ny++; break;
        case Action::LEFT:  nx--; break;
        case Action::RIGHT: nx++; break;
        case Action::STAY:  break;
    }

    // Check bounds
    if (!in_bounds(nx, ny)) {
        result.hit_wall = true;
        result.reward = -0.1f;  // Small penalty for hitting wall
    } else {
        CellType target = grid_[idx(nx, ny)];

        if (target == CellType::WALL) {
            result.hit_wall = true;
            result.reward = -0.1f;
        } else {
            // Move
            agent_x_ = nx;
            agent_y_ = ny;

            if (target == CellType::FOOD) {
                result.got_food = true;
                result.reward = 1.0f;
                food_collected_++;
                grid_[idx(nx, ny)] = CellType::EMPTY;
                respawn_food();  // Food respawns elsewhere
            } else if (target == CellType::DANGER) {
                result.hit_danger = true;
                result.reward = -1.0f;
                danger_hits_++;
                // Danger stays (persistent hazard)
            } else {
                // Small negative reward for each step (encourages efficiency)
                result.reward = -0.01f;
            }
        }
    }

    // v55: sync float position with integer position in discrete mode
    agent_fx_ = static_cast<float>(agent_x_) + 0.5f;
    agent_fy_ = static_cast<float>(agent_y_) + 0.5f;
    result.agent_x = agent_x_;
    result.agent_y = agent_y_;
    result.agent_fx = agent_fx_;
    result.agent_fy = agent_fy_;
    return result;
}

// v55: Continuous movement — agent moves by (dx, dy) float displacement.
// Collision detection uses the grid cell at floor(new_position).
// Biology: real movement is continuous; the grid is just the substrate for
// placing food/danger/walls. Agent position is (fx, fy) in [0, width)×[0, height).
StepResult GridWorld::act_continuous(float dx, float dy) {
    StepResult result;
    step_count_++;

    float w = static_cast<float>(config_.width);
    float h = static_cast<float>(config_.height);

    // Proposed new position
    float nfx = agent_fx_ + dx;
    float nfy = agent_fy_ + dy;

    // Clamp to world bounds (with small margin to stay inside)
    nfx = std::clamp(nfx, 0.01f, w - 0.01f);
    nfy = std::clamp(nfy, 0.01f, h - 0.01f);

    // Grid cell of new position
    int nx = static_cast<int>(std::floor(nfx));
    int ny = static_cast<int>(std::floor(nfy));
    nx = std::clamp(nx, 0, static_cast<int>(config_.width) - 1);
    ny = std::clamp(ny, 0, static_cast<int>(config_.height) - 1);

    // Check what's in the target cell
    CellType target = grid_[idx(nx, ny)];

    if (target == CellType::WALL) {
        // Bounce back: don't move into wall
        result.hit_wall = true;
        result.reward = -0.1f;
        // Stay at current position
    } else {
        // Move
        agent_fx_ = nfx;
        agent_fy_ = nfy;
        agent_x_ = nx;
        agent_y_ = ny;

        if (target == CellType::FOOD) {
            result.got_food = true;
            result.reward = 1.0f;
            food_collected_++;
            grid_[idx(nx, ny)] = CellType::EMPTY;
            respawn_food();
        } else if (target == CellType::DANGER) {
            result.hit_danger = true;
            result.reward = -1.0f;
            danger_hits_++;
        } else {
            result.reward = -0.01f;
        }
    }

    result.agent_x = agent_x_;
    result.agent_y = agent_y_;
    result.agent_fx = agent_fx_;
    result.agent_fy = agent_fy_;
    return result;
}

float GridWorld::cell_to_visual(int x, int y) const {
    if (!in_bounds(x, y)) return config_.vis_wall;
    if (x == agent_x_ && y == agent_y_) return config_.vis_agent;
    switch (grid_[idx(x, y)]) {
        case CellType::EMPTY:  return config_.vis_empty;
        case CellType::FOOD:   return config_.vis_food;
        case CellType::DANGER: return config_.vis_danger;
        case CellType::WALL:   return config_.vis_wall;
    }
    return config_.vis_empty;
}

std::vector<float> GridWorld::observe() const {
    int r = config_.vision_radius;
    int side = 2 * r + 1;
    std::vector<float> obs(static_cast<size_t>(side * side));
    int k = 0;
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            obs[k++] = cell_to_visual(agent_x_ + dx, agent_y_ + dy);
        }
    }
    return obs;
}

std::vector<float> GridWorld::full_observation() const {
    std::vector<float> obs(config_.width * config_.height);
    for (int y = 0; y < (int)config_.height; ++y) {
        for (int x = 0; x < (int)config_.width; ++x) {
            obs[idx(x, y)] = cell_to_visual(x, y);
        }
    }
    return obs;
}

std::string GridWorld::to_string() const {
    std::ostringstream ss;
    for (int y = 0; y < (int)config_.height; ++y) {
        for (int x = 0; x < (int)config_.width; ++x) {
            if (x == agent_x_ && y == agent_y_) {
                ss << 'A';
            } else {
                switch (grid_[idx(x, y)]) {
                    case CellType::EMPTY:  ss << '.'; break;
                    case CellType::FOOD:   ss << 'F'; break;
                    case CellType::DANGER: ss << 'D'; break;
                    case CellType::WALL:   ss << '#'; break;
                }
            }
        }
        ss << '\n';
    }
    return ss.str();
}

} // namespace wuyun
