#include "engine/grid_world_env.h"

namespace wuyun {

GridWorldEnv::GridWorldEnv(const GridWorldConfig& cfg)
    : world_(cfg)
    , vis_w_(cfg.vision_side())
    , vis_h_(cfg.vision_side())
{}

// --- Lifecycle ---

void GridWorldEnv::reset() {
    world_.reset();
}

void GridWorldEnv::reset_with_seed(uint32_t seed) {
    world_.reset_with_seed(seed);
}

// --- Sensory ---

std::vector<float> GridWorldEnv::observe() const {
    return world_.observe();
}

size_t GridWorldEnv::vis_width() const { return vis_w_; }
size_t GridWorldEnv::vis_height() const { return vis_h_; }

// --- Motor ---

Environment::Result GridWorldEnv::step(float dx, float dy) {
    StepResult r = world_.act_continuous(dx, dy);
    return Result{
        r.reward,
        r.got_food,       // positive_event
        r.hit_danger,     // negative_event
        r.agent_fx,       // pos_x
        r.agent_fy        // pos_y
    };
}

// --- Spatial ---

float GridWorldEnv::pos_x() const { return world_.agent_fx(); }
float GridWorldEnv::pos_y() const { return world_.agent_fy(); }
float GridWorldEnv::world_width()  const { return static_cast<float>(world_.width()); }
float GridWorldEnv::world_height() const { return static_cast<float>(world_.height()); }

// --- Statistics ---

uint32_t GridWorldEnv::positive_count() const { return world_.total_food_collected(); }
uint32_t GridWorldEnv::negative_count() const { return world_.total_danger_hits(); }
uint32_t GridWorldEnv::step_count()     const { return world_.total_steps(); }

} // namespace wuyun
