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
#include "engine/episode_buffer.h"
#include "region/cortical_region.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/neuromod/vta_da.h"
#include "region/limbic/lateral_habenula.h"
#include "region/limbic/hippocampus.h"
#include "region/limbic/amygdala.h"
#include "region/subcortical/cerebellum.h"
#include "engine/sleep_cycle.h"
#include "plasticity/homeostatic.h"
#include <memory>
#include <vector>
#include <functional>
#include <random>

namespace wuyun {

struct AgentConfig {
    // Brain scale
    int brain_scale = 1;  // scale=1 default (scale=3 暴露 D2 过度激活问题)

    // Perception (auto-computed from world_config.vision_radius in constructor)
    size_t vision_width  = 5;   // v21: default 5x5 local patch (vision_radius=2)
    size_t vision_height = 5;

    // Action decoding
    // v30b: Baldwin 100gen×60pop evolved (gen87, fitness=2.53)
    size_t brain_steps_per_action = 14;
    size_t reward_processing_steps = 10;

    float reward_scale = 5.0f;

    float exploration_noise = 70.0f;
    size_t exploration_anneal_steps = 0;  // Steps over which noise reduces (0=no anneal, let BG override)

    // Learning
    // Learning — Baldwin 100gen evolved (gen87)
    bool enable_da_stdp     = true;
    float da_stdp_lr        = 0.039f;
    bool enable_homeostatic = true;
    bool enable_cortical_stdp = true;
    float cortical_stdp_a_plus  = 0.005f;
    float cortical_stdp_a_minus = -0.013f;
    float cortical_stdp_w_max   = 2.2f;

    float lgn_gain           = 500.0f;
    float lgn_baseline       = 19.0f;
    float lgn_noise_amp      = 0.5f;

    float bg_to_m1_gain      = 5.7f;
    float attractor_drive_ratio  = 0.47f;
    float background_drive_ratio = 0.30f;

    float ne_food_scale      = 2.6f;
    float ne_floor           = 0.84f;

    float homeostatic_target_rate = 14.4f;
    float homeostatic_eta    = 0.0067f;

    // Brain size factors (multiplied on base neuron counts)
    float v1_size_factor     = 1.0f;
    float dlpfc_size_factor  = 1.0f;
    float bg_size_factor     = 1.0f;

    // Predictive coding (dlPFC → V1 attentional feedback)
    // v21: enabled by default — 5×5 vision field has enough redundancy for PC benefit.
    // Step 15-B verified: PC provides +0.121 improvement advantage in 5×5 vision,
    // reduces danger by 40%. Only harmful in tiny 3×3 scenes (反馈=噪声).
    bool  enable_predictive_coding = true;

    // LHb negative RPE (punishment learning via DA pause)
    bool  enable_lhb         = true;   // Enable LHb for negative RPE
    float lhb_punishment_gain = 1.5f;  // Punishment signal → LHb excitation gain
    float lhb_frustration_gain = 1.0f; // Frustrative non-reward → LHb excitation gain

    // Amygdala (fear conditioning)
    bool  enable_amygdala    = true;   // Enable amygdala fear circuit
    float amyg_us_gain       = 1.5f;   // US magnitude scaling for BLA injection

    // Awake SWR Replay (experience replay via hippocampal sharp-wave ripples)
    bool  enable_replay      = true;
    int   replay_passes      = 5;
    float replay_da_scale    = 0.76f;
    size_t replay_buffer_size = 50;    // Max episodes in buffer (v21: 30→50, 10×10 has 100 positions)

    // Negative experience replay (LHb-controlled avoidance learning)
    // Previously disabled: D2 over-strengthening without LHb control.
    // Now safe: LHb provides graded DA pause → controlled D2 learning.
    bool  enable_negative_replay = true;  // Enable replay of danger episodes
    int   negative_replay_passes = 2;     // Conservative: fewer passes than positive (5)
    float negative_replay_da_scale = 0.3f; // DA dip scale (baseline - |reward|×this)

    // Sleep consolidation (periodic offline replay)
    // Biology: NREM SWR replays recent experiences for BG+cortical consolidation.
    // Agent runs wake_steps, then sleeps for sleep_nrem_steps, then wakes.
    // During sleep: no environment interaction, replay all buffered episodes.
    // v21: enabled for 10×10 environment — more positions to remember = more forgetting
    //      = sleep consolidation combats weight decay effectively.
    //      In 3×3 it was harmful (awake replay sufficient, over-consolidation).
    //      Tuning: very light naps, long intervals, gentle DA — prevent over-consolidation
    //      while combating forgetting in 100-cell grid.
    bool   enable_sleep_consolidation = true;   // v21: enabled for larger environments
    size_t wake_steps_before_sleep    = 800;   // v21: long interval, light touch
    size_t sleep_nrem_steps           = 15;    // v21: very light consolidation per bout
    int    sleep_replay_passes        = 1;     // Single pass (prevent over-consolidation)
    float  sleep_positive_da          = 0.35f; // v21: barely above baseline (0.3), gentle nudge

    // v30: Cerebellum forward model (Yoshida 2025: CB-BG synergistic RL)
    // M1 efference copy + visual context → predict next sensory state
    // Prediction error → climbing fiber → PF-PC LTD → fast correction
    bool  enable_cerebellum = true;

    // v27: Developmental period (critical period for visual feature learning)
    // Biology: infant visual cortex spends ~6 months self-organizing via Hebbian STDP
    // before goal-directed behavior begins. Agent random-walks during dev period,
    // visual STDP + predictive coding learn features, no DA-STDP reward learning.
    size_t dev_period_steps = 2000;   // Steps of random exploration before reward learning
    bool   enable_predictive_learning = true;  // L6 prediction + error-gated STDP

    // Evolution fast-eval mode
    bool fast_eval = false;  // Skip hippocampus + cortical STDP for ~40% speedup

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
    BrainRegion*    lgn()   const { return lgn_; }
    CorticalRegion* v1()    const { return v1_; }
    CorticalRegion* v2()    const { return v2_; }
    CorticalRegion* v4()    const { return v4_; }
    CorticalRegion* it_ctx() const { return it_; }  // "it" is C++ keyword-adjacent, use it_ctx
    CorticalRegion* dlpfc() const { return dlpfc_; }
    CorticalRegion* m1()    const { return m1_; }
    BasalGanglia*   bg()    const { return bg_; }
    VTA_DA*         vta()   const { return vta_; }
    Hippocampus*    hipp()  const { return hipp_; }
    LateralHabenula* lhb()  const { return lhb_; }
    Amygdala*       amyg()  const { return amyg_; }
    Cerebellum*     cb()    const { return cb_; }

private:
    AgentConfig config_;

    GridWorld world_;
    SimulationEngine engine_;
    VisualInput visual_encoder_;

    // Cached region pointers
    BrainRegion*    lgn_   = nullptr;
    CorticalRegion* v1_    = nullptr;
    CorticalRegion* v2_    = nullptr;   // Step 24: visual hierarchy
    CorticalRegion* v4_    = nullptr;   // Step 24: visual hierarchy
    CorticalRegion* it_    = nullptr;   // Step 24: invariant object recognition
    CorticalRegion* dlpfc_ = nullptr;
    CorticalRegion* m1_    = nullptr;
    BasalGanglia*   bg_    = nullptr;
    VTA_DA*         vta_   = nullptr;
    Hippocampus*    hipp_  = nullptr;
    LateralHabenula* lhb_   = nullptr;
    Amygdala*       amyg_   = nullptr;
    Cerebellum*     cb_     = nullptr;   // v30: forward model

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

    // --- Frustration tracking (expected reward not received) ---
    float expected_reward_level_ = 0.0f;  // Tracks recent food rate → expected reward

    // --- Awake SWR replay ---
    EpisodeBuffer replay_buffer_;
    void run_awake_replay(float reward);
    void run_negative_replay(float reward);
    void capture_dlpfc_spikes(int action_group);

    // --- Sleep consolidation ---
    SleepCycleManager sleep_mgr_;
    size_t wake_step_counter_ = 0;
    void run_sleep_consolidation();
};

} // namespace wuyun
