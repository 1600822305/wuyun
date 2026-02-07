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
    std::fill(grid_.begin(), grid_.end(), CellType::EMPTY);

    // Agent starts at center
    agent_x_ = static_cast<int>(config_.width / 2);
    agent_y_ = static_cast<int>(config_.height / 2);

    // Place food and danger
    place_random(CellType::FOOD,   config_.n_food);
    place_random(CellType::DANGER, config_.n_danger);

    food_collected_ = 0;
    danger_hits_    = 0;
    step_count_     = 0;
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
    place_random(CellType::FOOD, 1);
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

    result.agent_x = agent_x_;
    result.agent_y = agent_y_;
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
    // 3x3 patch centered on agent
    std::vector<float> obs(9);
    int k = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
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
