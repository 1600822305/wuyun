/**
 * benchmark_multiroom.cpp — MultiRoomEnv 觅食表现
 */

#include "engine/closed_loop_agent.h"
#include "engine/multi_room_env.h"
#include <cstdio>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace wuyun;

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    uint32_t seeds[] = {42, 77, 123, 256, 789};
    int steps = 300;

    printf("=== MultiRoomEnv Benchmark (2x2 rooms, %d steps) ===\n\n", steps);

    int total_food = 0, total_danger = 0;

    for (auto seed : seeds) {
        MultiRoomConfig mcfg;
        mcfg.n_rooms_x = 2; mcfg.n_rooms_y = 2;
        mcfg.room_w = 4; mcfg.room_h = 4;
        mcfg.n_food = 5; mcfg.n_danger = 3;
        mcfg.seed = seed;

        AgentConfig acfg;
        acfg.brain_scale = 1;
        acfg.fast_eval = true;

        auto env_ptr = std::make_unique<MultiRoomEnv>(mcfg);
        MultiRoomEnv* env_raw = env_ptr.get();
        ClosedLoopAgent agent(std::move(env_ptr), acfg);

        int food = 0, danger = 0;
        int early_food = 0, early_danger = 0;
        int late_food = 0, late_danger = 0;
        int half = steps / 2;

        for (int i = 0; i < steps; ++i) {
            auto r = agent.agent_step();
            if (r.positive_event) { food++; if (i < half) early_food++; else late_food++; }
            if (r.negative_event) { danger++; if (i < half) early_danger++; else late_danger++; }
        }

        float early_safety = (float)early_food / (std::max)(1, early_food + early_danger);
        float late_safety = (float)late_food / (std::max)(1, late_food + late_danger);

        printf("  seed=%3u | food=%2d danger=%2d | early=%.0f%% late=%.0f%% | pos=(%.1f,%.1f)\n",
               seed, food, danger, early_safety*100, late_safety*100,
               env_raw->pos_x(), env_raw->pos_y());

        total_food += food;
        total_danger += danger;
    }

    printf("\n  Avg: food=%.1f  danger=%.1f  (5 seeds x %d steps)\n",
           total_food / 5.0f, total_danger / 5.0f, steps);

    // Show one map
    printf("\n--- Sample map (seed=42) ---\n");
    MultiRoomConfig show_cfg;
    show_cfg.n_rooms_x = 2; show_cfg.n_rooms_y = 2;
    show_cfg.room_w = 4; show_cfg.room_h = 4;
    show_cfg.n_food = 5; show_cfg.n_danger = 3;
    show_cfg.seed = 42;
    MultiRoomEnv show_env(show_cfg);
    printf("%s", show_env.to_string().c_str());

    return 0;
}
