/**
 * test_multi_room.cpp — MultiRoomEnv 环境测试
 *
 * 验证:
 * 1. 多房间生成 (墙壁/房间/门道)
 * 2. 连续移动 + 碰撞检测
 * 3. 食物/危险交互 + 重生
 * 4. observe() 视觉 patch 格式正确
 * 5. 与 ClosedLoopAgent 闭环运行 (Environment 接口验证)
 */

#include "engine/multi_room_env.h"
#include "engine/closed_loop_agent.h"

#include <cstdio>
#include <cmath>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace wuyun;

static int g_pass = 0, g_fail = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
        g_fail++; return; \
    } \
} while(0)

// =========================================================================
// Test 1: 多房间生成
// =========================================================================
static void test_room_generation() {
    printf("\n--- Test 1: Room generation ---\n");

    MultiRoomConfig cfg;
    cfg.n_rooms_x = 2; cfg.n_rooms_y = 2;
    cfg.room_w = 4; cfg.room_h = 4;
    cfg.n_food = 4; cfg.n_danger = 2;
    cfg.seed = 42;

    MultiRoomEnv env(cfg);

    // Grid size: 2*(4+1)+1 = 11 × 11
    TEST_ASSERT(env.grid_w() == 11, "Grid width = 11");
    TEST_ASSERT(env.grid_h() == 11, "Grid height = 11");

    printf("  Grid: %zux%zu\n", env.grid_w(), env.grid_h());
    printf("%s", env.to_string().c_str());

    // Agent should be in first room
    TEST_ASSERT(env.pos_x() > 0.5f && env.pos_x() < 5.0f, "Agent in first room X");
    TEST_ASSERT(env.pos_y() > 0.5f && env.pos_y() < 5.0f, "Agent in first room Y");
    printf("  Agent at (%.1f, %.1f)\n", env.pos_x(), env.pos_y());

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 2: 移动 + 碰撞
// =========================================================================
static void test_movement_collision() {
    printf("\n--- Test 2: Movement + collision ---\n");

    MultiRoomConfig cfg;
    cfg.n_rooms_x = 2; cfg.n_rooms_y = 1;
    cfg.room_w = 3; cfg.room_h = 3;
    cfg.n_food = 0; cfg.n_danger = 0;
    cfg.seed = 42;

    MultiRoomEnv env(cfg);
    float start_x = env.pos_x();
    float start_y = env.pos_y();

    // Move right within room
    env.step(0.5f, 0.0f);
    TEST_ASSERT(env.pos_x() > start_x, "Move right increases X");

    // Try to move into wall (top boundary of room 1 is y=0)
    float y_before = env.pos_y();
    for (int i = 0; i < 20; ++i) env.step(0.0f, -0.5f);
    // Should be stopped by wall
    TEST_ASSERT(env.pos_y() >= 0.5f, "Wall collision stops movement");

    printf("  Start: (%.1f, %.1f), After moves: (%.1f, %.1f)\n",
           start_x, start_y, env.pos_x(), env.pos_y());
    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 3: 食物交互 + 重生
// =========================================================================
static void test_food_interaction() {
    printf("\n--- Test 3: Food interaction ---\n");

    MultiRoomConfig cfg;
    cfg.n_rooms_x = 1; cfg.n_rooms_y = 1;
    cfg.room_w = 5; cfg.room_h = 5;
    cfg.n_food = 3; cfg.n_danger = 1;
    cfg.seed = 42;

    MultiRoomEnv env(cfg);

    // Run many steps, should eventually find food
    int food_found = 0;
    int danger_hit = 0;
    for (int i = 0; i < 500; ++i) {
        // Random walk
        float dx = (i % 3 == 0) ? 0.6f : ((i % 3 == 1) ? -0.6f : 0.0f);
        float dy = (i % 5 < 2) ? 0.6f : ((i % 5 < 4) ? -0.6f : 0.0f);
        auto r = env.step(dx, dy);
        if (r.positive_event) food_found++;
        if (r.negative_event) danger_hit++;
    }

    printf("  500 steps: food=%d, danger=%d\n", food_found, danger_hit);
    TEST_ASSERT(env.positive_count() == (uint32_t)food_found, "positive_count matches");
    TEST_ASSERT(env.negative_count() == (uint32_t)danger_hit, "negative_count matches");
    TEST_ASSERT(env.step_count() == 500, "step_count = 500");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 4: 观测格式
// =========================================================================
static void test_observation() {
    printf("\n--- Test 4: Observation format ---\n");

    MultiRoomConfig cfg;
    cfg.n_rooms_x = 2; cfg.n_rooms_y = 2;
    cfg.room_w = 4; cfg.room_h = 4;
    cfg.vision_radius = 2;
    cfg.seed = 42;

    MultiRoomEnv env(cfg);

    auto obs = env.observe();
    size_t expected = cfg.vision_side() * cfg.vision_side();  // 5×5=25
    TEST_ASSERT(obs.size() == expected, "Observation size = 25");
    TEST_ASSERT(env.vis_width() == 5, "vis_width = 5");
    TEST_ASSERT(env.vis_height() == 5, "vis_height = 5");

    // Center should be agent
    size_t center = obs.size() / 2;
    TEST_ASSERT(std::abs(obs[center] - cfg.vis_agent) < 0.01f, "Center = agent");

    printf("  5x5 patch center=%.1f (agent=%.1f)\n", obs[center], cfg.vis_agent);
    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 5: 与 ClosedLoopAgent 闭环运行 (Environment 接口验证)
// =========================================================================
static void test_agent_with_multiroom() {
    printf("\n--- Test 5: ClosedLoopAgent + MultiRoomEnv ---\n");

    MultiRoomConfig room_cfg;
    room_cfg.n_rooms_x = 2; room_cfg.n_rooms_y = 2;
    room_cfg.room_w = 4; room_cfg.room_h = 4;
    room_cfg.n_food = 4; room_cfg.n_danger = 2;
    room_cfg.vision_radius = 2;
    room_cfg.seed = 42;

    AgentConfig agent_cfg;
    agent_cfg.brain_scale = 1;
    agent_cfg.fast_eval = true;

    auto env = std::make_unique<MultiRoomEnv>(room_cfg);
    ClosedLoopAgent agent(std::move(env), agent_cfg);

    // Run 100 agent steps without crash
    for (int i = 0; i < 100; ++i) {
        agent.agent_step();
    }

    printf("  100 agent steps completed (no crash)\n");
    printf("  V1: %zu neurons, M1: %zu neurons\n",
           agent.v1()->n_neurons(), agent.m1()->n_neurons());

    TEST_ASSERT(agent.v1() != nullptr, "V1 exists with MultiRoomEnv");
    TEST_ASSERT(agent.m1() != nullptr, "M1 exists with MultiRoomEnv");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Test 6: reset 验证
// =========================================================================
static void test_reset() {
    printf("\n--- Test 6: Reset ---\n");

    MultiRoomConfig cfg;
    cfg.n_rooms_x = 2; cfg.n_rooms_y = 2;
    cfg.room_w = 3; cfg.room_h = 3;
    cfg.n_food = 2; cfg.n_danger = 1;
    cfg.seed = 42;

    MultiRoomEnv env(cfg);

    // Run some steps
    for (int i = 0; i < 50; ++i) env.step(0.3f, 0.2f);
    TEST_ASSERT(env.step_count() == 50, "50 steps before reset");

    // Reset
    env.reset();
    TEST_ASSERT(env.step_count() == 0, "step_count = 0 after reset");
    TEST_ASSERT(env.positive_count() == 0, "positive_count = 0 after reset");
    TEST_ASSERT(env.negative_count() == 0, "negative_count = 0 after reset");

    // Reset with different seed
    env.reset_with_seed(999);
    for (int i = 0; i < 30; ++i) env.step(0.3f, 0.2f);
    TEST_ASSERT(env.step_count() == 30, "30 steps after reset_with_seed");

    printf("  [PASS]\n"); g_pass++;
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("=== MultiRoomEnv Tests ===\n");

    test_room_generation();
    test_movement_collision();
    test_food_interaction();
    test_observation();
    test_agent_with_multiroom();
    test_reset();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
