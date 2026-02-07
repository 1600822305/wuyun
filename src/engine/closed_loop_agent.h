#pragma once
/**
 * ClosedLoopAgent — 闭环智能体
 *
 * 将 GridWorld 环境与 WuYun 大脑连接形成完整的感知-决策-行动循环:
 *
 *   GridWorld.observe() → VisualInput → LGN → V1 → ... → dlPFC → BG → MotorThal → M1
 *                                                                                   ↓
 *   GridWorld.act(action) ← decode_action() ← M1 L5 fired pattern
 *                ↓
 *   reward → VTA.inject_reward() → DA → BG DA-STDP → 学习
 *
 * 动作解码:
 *   M1 L5 神经元分为4组 (UP/DOWN/LEFT/RIGHT), 统计各组发放数, winner-take-all
 *   如果全部沉默 → STAY
 *
 * 生物学基础:
 *   - 运动皮层 M1 L5 锥体细胞直接投射到脊髓 (皮质脊髓束)
 *   - BG Go/NoGo 通路选择动作
 *   - VTA RPE 信号驱动强化学习
 */

#include "engine/grid_world.h"
#include "engine/simulation_engine.h"
#include "engine/sensory_input.h"
#include "region/cortical_region.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/neuromod/vta_da.h"
#include "region/limbic/hippocampus.h"
#include "plasticity/homeostatic.h"
#include <memory>
#include <vector>
#include <functional>
#include <random>

namespace wuyun {

struct AgentConfig {
    // Brain scale
    int brain_scale = 1;    // 1=default, 3=large

    // Perception
    size_t vision_width  = 3;   // 3x3 local patch
    size_t vision_height = 3;

    // Action decoding
    size_t brain_steps_per_action = 15;  // 每个环境步的脑步数 (LGN→V1→dlPFC→BG需7步延迟)
    size_t reward_processing_steps = 5;  // 奖励处理步数 (DA传播到BG)

    // Reward scaling
    float reward_scale = 1.0f;  // reward → VTA inject_reward multiplier

    // Exploration
    float exploration_noise = 55.0f;  // M1 L5 noise amplitude (motor exploration, needs >50 for 10-step firing)
    size_t exploration_anneal_steps = 0;  // Steps over which noise reduces (0=no anneal, let BG override)

    // Learning
    bool enable_da_stdp     = true;   // BG DA-STDP
    float da_stdp_lr        = 0.005f; // DA-STDP learning rate (0.005×0.5×50=0.125 per food event)
    bool enable_homeostatic = true;   // Homeostatic plasticity
    bool enable_cortical_stdp = true; // V1+dlPFC online STDP (experience-dependent representation)
    float cortical_stdp_a_plus  = 0.005f;  // LTP (half of default 0.01, slower online learning)
    float cortical_stdp_a_minus = -0.006f; // LTD (slightly stronger → competitive selectivity)
    float cortical_stdp_w_max   = 1.5f;    // Max synaptic weight

    // GridWorld
    GridWorldConfig world_config;
};

/** 每步回调: agent_step, action, reward, agent_x, agent_y */
using AgentStepCallback = std::function<void(int, Action, float, int, int)>;

class ClosedLoopAgent {
public:
    explicit ClosedLoopAgent(const AgentConfig& config = {});

    // Non-copyable, non-movable (contains SimulationEngine with unique_ptrs + cached raw pointers)
    ClosedLoopAgent(const ClosedLoopAgent&) = delete;
    ClosedLoopAgent& operator=(const ClosedLoopAgent&) = delete;
    ClosedLoopAgent(ClosedLoopAgent&&) = delete;
    ClosedLoopAgent& operator=(ClosedLoopAgent&&) = delete;

    /** 重置环境 (大脑保持不变, 只重置GridWorld) */
    void reset_world();

    /**
     * 执行一个环境步:
     *   1. observe → encode → inject LGN
     *   2. run brain N steps
     *   3. decode M1 → action
     *   4. world.act(action)
     *   5. reward → VTA
     * @return StepResult
     */
    StepResult agent_step();

    /** 运行 n 个环境步 */
    void run(int n_steps);

    /** 设置每步回调 */
    void set_callback(AgentStepCallback cb) { callback_ = std::move(cb); }

    // --- 访问器 ---
    GridWorld&         world()  { return world_; }
    SimulationEngine&  brain()  { return engine_; }

    int    agent_step_count() const { return agent_step_count_; }
    Action last_action()      const { return last_action_; }
    float  last_reward()      const { return last_reward_; }

    /** 最近 N 步的平均奖励 (滑动窗口) */
    float avg_reward(size_t window = 100) const;

    /** 最近 N 步的食物收集率 */
    float food_rate(size_t window = 100) const;

    // --- 诊断 ---
    CorticalRegion* v1()    const { return v1_; }
    CorticalRegion* dlpfc() const { return dlpfc_; }
    CorticalRegion* m1()    const { return m1_; }
    BasalGanglia*   bg()    const { return bg_; }
    VTA_DA*         vta()   const { return vta_; }
    Hippocampus*    hipp()  const { return hipp_; }

private:
    AgentConfig config_;

    GridWorld world_;
    SimulationEngine engine_;
    VisualInput visual_encoder_;

    // Cached region pointers
    BrainRegion*    lgn_   = nullptr;
    CorticalRegion* v1_    = nullptr;
    CorticalRegion* dlpfc_ = nullptr;
    CorticalRegion* m1_    = nullptr;
    BasalGanglia*   bg_    = nullptr;
    VTA_DA*         vta_   = nullptr;
    Hippocampus*    hipp_  = nullptr;

    // State
    int    agent_step_count_ = 0;
    Action last_action_      = Action::STAY;
    float  last_reward_      = 0.0f;
    float  pending_reward_   = 0.0f;  // Reward to inject at start of next step
    bool   has_pending_reward_ = false;

    // Reward history (ring buffer)
    std::vector<float> reward_history_;
    std::vector<int>   food_history_;    // 1 if got food, 0 otherwise
    size_t history_idx_ = 0;

    AgentStepCallback callback_;
    std::mt19937 motor_rng_{12345};

    void build_brain();
    Action decode_m1_action(const std::vector<int>& l5_accum) const;
    void inject_observation();
    void inject_reward(float reward);
};

} // namespace wuyun
