#include "engine/multi_room_env.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace wuyun {

MultiRoomEnv::MultiRoomEnv(const MultiRoomConfig& cfg)
    : cfg_(cfg)
    , grid_w_(cfg.grid_width())
    , grid_h_(cfg.grid_height())
    , grid_(grid_w_ * grid_h_, Cell::WALL)
    , rng_(cfg.seed)
{
    build_rooms();
    place_random(Cell::FOOD, cfg_.n_food);
    place_random(Cell::DANGER, cfg_.n_danger);
    // Start agent in first room center
    agent_ix_ = static_cast<int>(cfg_.room_w / 2 + 1);
    agent_iy_ = static_cast<int>(cfg_.room_h / 2 + 1);
    agent_fx_ = static_cast<float>(agent_ix_) + 0.5f;
    agent_fy_ = static_cast<float>(agent_iy_) + 0.5f;
    // Clear agent's starting cell
    grid_[idx(agent_ix_, agent_iy_)] = Cell::EMPTY;
}

// =============================================================================
// Room generation
// =============================================================================

void MultiRoomEnv::build_rooms() {
    // Fill everything with WALL first (done in constructor)
    // Then carve out rooms and add doorways

    size_t rw = cfg_.room_w;
    size_t rh = cfg_.room_h;

    // Carve rooms: room (rx, ry) occupies grid cells:
    //   x: [rx*(rw+1)+1 .. rx*(rw+1)+rw]
    //   y: [ry*(rh+1)+1 .. ry*(rh+1)+rh]
    for (size_t ry = 0; ry < cfg_.n_rooms_y; ++ry) {
        for (size_t rx = 0; rx < cfg_.n_rooms_x; ++rx) {
            int x0 = static_cast<int>(rx * (rw + 1) + 1);
            int y0 = static_cast<int>(ry * (rh + 1) + 1);
            for (int dy = 0; dy < (int)rh; ++dy) {
                for (int dx = 0; dx < (int)rw; ++dx) {
                    grid_[idx(x0 + dx, y0 + dy)] = Cell::EMPTY;
                }
            }
        }
    }

    // Add doorways between adjacent rooms
    // Horizontal doors: between room (rx, ry) and (rx+1, ry)
    for (size_t ry = 0; ry < cfg_.n_rooms_y; ++ry) {
        for (size_t rx = 0; rx + 1 < cfg_.n_rooms_x; ++rx) {
            // Wall column between rooms rx and rx+1
            int wall_x = static_cast<int>((rx + 1) * (rw + 1));
            // Door at random y within room height
            int y0 = static_cast<int>(ry * (rh + 1) + 1);
            std::uniform_int_distribution<int> dist(0, static_cast<int>(rh) - 1);
            int door_y = y0 + dist(rng_);
            grid_[idx(wall_x, door_y)] = Cell::EMPTY;
        }
    }

    // Vertical doors: between room (rx, ry) and (rx, ry+1)
    for (size_t ry = 0; ry + 1 < cfg_.n_rooms_y; ++ry) {
        for (size_t rx = 0; rx < cfg_.n_rooms_x; ++rx) {
            int wall_y = static_cast<int>((ry + 1) * (rh + 1));
            int x0 = static_cast<int>(rx * (rw + 1) + 1);
            std::uniform_int_distribution<int> dist(0, static_cast<int>(rw) - 1);
            int door_x = x0 + dist(rng_);
            grid_[idx(door_x, wall_y)] = Cell::EMPTY;
        }
    }
}

void MultiRoomEnv::place_random(Cell type, size_t count) {
    // Collect empty cells
    std::vector<size_t> empty;
    for (size_t i = 0; i < grid_.size(); ++i) {
        if (grid_[i] == Cell::EMPTY) {
            // Don't place on agent start
            int x = static_cast<int>(i % grid_w_);
            int y = static_cast<int>(i / grid_w_);
            if (x != agent_ix_ || y != agent_iy_) {
                empty.push_back(i);
            }
        }
    }
    std::shuffle(empty.begin(), empty.end(), rng_);
    size_t n = std::min(count, empty.size());
    for (size_t i = 0; i < n; ++i) {
        grid_[empty[i]] = type;
    }
}

void MultiRoomEnv::respawn_food() {
    // Count existing food
    size_t existing = 0;
    for (auto c : grid_) if (c == Cell::FOOD) existing++;
    if (existing < cfg_.n_food) {
        place_random(Cell::FOOD, cfg_.n_food - existing);
    }
}

// =============================================================================
// Environment interface
// =============================================================================

void MultiRoomEnv::reset() {
    rng_.seed(cfg_.seed);
    grid_.assign(grid_w_ * grid_h_, Cell::WALL);
    build_rooms();
    agent_ix_ = static_cast<int>(cfg_.room_w / 2 + 1);
    agent_iy_ = static_cast<int>(cfg_.room_h / 2 + 1);
    agent_fx_ = static_cast<float>(agent_ix_) + 0.5f;
    agent_fy_ = static_cast<float>(agent_iy_) + 0.5f;
    grid_[idx(agent_ix_, agent_iy_)] = Cell::EMPTY;
    place_random(Cell::FOOD, cfg_.n_food);
    place_random(Cell::DANGER, cfg_.n_danger);
    food_collected_ = 0;
    danger_hits_ = 0;
    step_count_ = 0;
}

void MultiRoomEnv::reset_with_seed(uint32_t seed) {
    cfg_.seed = seed;
    reset();
}

// --- Sensory ---

float MultiRoomEnv::cell_visual(int x, int y) const {
    if (!in_bounds(x, y)) return cfg_.vis_wall;
    switch (grid_[idx(x, y)]) {
        case Cell::FOOD:   return cfg_.vis_food;
        case Cell::DANGER: return cfg_.vis_danger;
        case Cell::WALL:   return cfg_.vis_wall;
        default:           return cfg_.vis_empty;
    }
}

std::vector<float> MultiRoomEnv::observe() const {
    int r = cfg_.vision_radius;
    size_t side = cfg_.vision_side();
    std::vector<float> patch(side * side, cfg_.vis_wall);

    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            int wx = agent_ix_ + dx;
            int wy = agent_iy_ + dy;
            size_t pi = static_cast<size_t>((dy + r) * (int)side + (dx + r));
            if (dx == 0 && dy == 0) {
                patch[pi] = cfg_.vis_agent;
            } else {
                patch[pi] = cell_visual(wx, wy);
            }
        }
    }
    return patch;
}

size_t MultiRoomEnv::vis_width()  const { return cfg_.vision_side(); }
size_t MultiRoomEnv::vis_height() const { return cfg_.vision_side(); }

// --- Motor ---

Environment::Result MultiRoomEnv::step(float dx, float dy) {
    step_count_++;

    int prev_ix = agent_ix_, prev_iy = agent_iy_;

    // Continuous movement with collision detection (same logic as GridWorld)
    float nx = agent_fx_ + dx;
    float ny = agent_fy_ + dy;

    // Clamp to world bounds
    nx = std::max(0.01f, std::min(nx, static_cast<float>(grid_w_) - 0.01f));
    ny = std::max(0.01f, std::min(ny, static_cast<float>(grid_h_) - 0.01f));

    int new_ix = static_cast<int>(std::floor(nx));
    int new_iy = static_cast<int>(std::floor(ny));

    // Collision: can't move into walls
    if (!is_passable(new_ix, new_iy)) {
        // Try sliding along axes
        int slide_x = static_cast<int>(std::floor(agent_fx_ + dx));
        if (is_passable(slide_x, agent_iy_)) {
            agent_fx_ += dx;
            agent_ix_ = slide_x;
        } else {
            int slide_y = static_cast<int>(std::floor(agent_fy_ + dy));
            if (is_passable(agent_ix_, slide_y)) {
                agent_fy_ += dy;
                agent_iy_ = slide_y;
            }
            // else: stuck, no movement
        }
    } else {
        agent_fx_ = nx;
        agent_fy_ = ny;
        agent_ix_ = new_ix;
        agent_iy_ = new_iy;
    }

    // Check cell events ONLY on cell transition (v55 danger trap fix)
    bool cell_changed = (agent_ix_ != prev_ix || agent_iy_ != prev_iy);
    Result result{};
    result.pos_x = agent_fx_;
    result.pos_y = agent_fy_;

    if (cell_changed) {
        Cell& cell = grid_[idx(agent_ix_, agent_iy_)];
        if (cell == Cell::FOOD) {
            result.reward = 1.0f;
            result.positive_event = true;
            food_collected_++;
            cell = Cell::EMPTY;
            respawn_food();
        } else if (cell == Cell::DANGER) {
            result.reward = -1.0f;
            result.negative_event = true;
            danger_hits_++;
        }
    }

    return result;
}

// --- Spatial ---

float MultiRoomEnv::pos_x() const { return agent_fx_; }
float MultiRoomEnv::pos_y() const { return agent_fy_; }
float MultiRoomEnv::world_width()  const { return static_cast<float>(grid_w_); }
float MultiRoomEnv::world_height() const { return static_cast<float>(grid_h_); }

// --- Statistics ---

uint32_t MultiRoomEnv::positive_count() const { return food_collected_; }
uint32_t MultiRoomEnv::negative_count() const { return danger_hits_; }
uint32_t MultiRoomEnv::step_count()     const { return step_count_; }

// --- Debug ---

std::string MultiRoomEnv::to_string() const {
    std::ostringstream ss;
    for (int y = 0; y < (int)grid_h_; ++y) {
        for (int x = 0; x < (int)grid_w_; ++x) {
            if (x == agent_ix_ && y == agent_iy_) { ss << 'A'; continue; }
            switch (grid_[idx(x, y)]) {
                case Cell::EMPTY:  ss << '.'; break;
                case Cell::FOOD:   ss << 'F'; break;
                case Cell::DANGER: ss << 'D'; break;
                case Cell::WALL:   ss << '#'; break;
            }
        }
        ss << '\n';
    }
    return ss.str();
}

} // namespace wuyun
