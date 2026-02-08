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
#include "region/neuromod/lc_ne.h"
#include "region/neuromod/nbm_ach.h"
#include "region/neuromod/drn_5ht.h"
#include "region/limbic/lateral_habenula.h"
#include "region/limbic/hippocampus.h"
#include "region/limbic/amygdala.h"
#include "region/subcortical/cerebellum.h"
#include "region/anterior_cingulate.h"
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
    // Step 33 best: 30gen×40pop Baldwin (gen26, fitness=2.05, consolidation+dev100)
    size_t brain_steps_per_action = 12;
    size_t reward_processing_steps = 7;

    float reward_scale = 1.43f;

    float exploration_noise = 83.0f;
    size_t exploration_anneal_steps = 0;  // Steps over which noise reduces (0=no anneal, let BG override)

    // Learning
    // Step 33 best (30gen Baldwin, consolidation+dev100, gen26 fitness=2.05)
    bool enable_da_stdp     = true;
    float da_stdp_lr        = 0.014f;
    bool enable_homeostatic = true;
    bool enable_cortical_stdp = true;
    float cortical_stdp_a_plus  = 0.004f;
    float cortical_stdp_a_minus = -0.006f;  // 1.5× LTD (mild)
    float cortical_stdp_w_max   = 1.31f;

    float lgn_gain           = 469.0f;
    float lgn_baseline       = 8.72f;
    float lgn_noise_amp      = 0.50f;

    float bg_to_m1_gain      = 10.76f;  // v33: evolution found BG/noise balance
    float attractor_drive_ratio  = 0.34f;
    float background_drive_ratio = 0.22f;

    float ne_food_scale      = 1.0f;
    float ne_floor           = 0.67f;

    float homeostatic_target_rate = 10.94f;
    float homeostatic_eta    = 0.00073f;  // v33: 9× lower (was erasing learned diffs!)

    // Brain size factors (multiplied on base neuron counts)
    float v1_size_factor     = 1.01f;
    float dlpfc_size_factor  = 2.17f;  // v33: larger dlPFC for better decisions
    float bg_size_factor     = 1.33f;

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
    bool  enable_amygdala    = true;   // v33: 修复过度泛化，不禁用
    float amyg_us_gain       = 1.5f;   // US magnitude scaling for BLA injection

    // Synaptic consolidation / metaplasticity (v33: prevents catastrophic forgetting)
    // Biology: STC (Frey & Morris 1997) — well-learned synapses resist decay + opposing updates
    bool  enable_synaptic_consolidation = true;

    // Awake SWR Replay (experience replay via hippocampal sharp-wave ripples)
    bool  enable_replay      = true;
    int   replay_passes      = 7;
    float replay_da_scale    = 0.74f;
    size_t replay_buffer_size = 50;    // Max episodes in buffer (v21: 30→50, 10×10 has 100 positions)
    bool  enable_interleaved_replay = true;  // v33: mix positive+negative episodes during replay

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
    bool   enable_sleep_consolidation = true;  // v31: 修复后重新启用 (DA=baseline, engine.step)
    size_t wake_steps_before_sleep    = 800;   // v21: long interval, light touch
    size_t sleep_nrem_steps           = 15;    // v21: very light consolidation per bout
    int    sleep_replay_passes        = 1;     // Single pass (prevent over-consolidation)
    float  sleep_positive_da          = 0.30f; // v31: =baseline (NREM DA is LOW, no new learning)

    // v34: 神经调质系统接入 (LC-NE, NBM-ACh, DRN-5HT)
    // 替换手工计算的探索噪声和ACh boost，用真实神经元动态驱动
    bool  enable_lc_ne    = true;   // LC蓝斑: NE驱动探索/利用平衡
    bool  enable_nbm_ach  = true;   // NBM基底核: ACh驱动STDP注意力门控
    bool  enable_drn_5ht  = true;   // DRN缝核: 5-HT驱动耐心/学习率调制

    // v30: Cerebellum forward model (Yoshida 2025: CB-BG synergistic RL)
    // M1 efference copy + visual context → predict next sensory state
    // Prediction error → climbing fiber → PF-PC LTD → fast correction
    bool  enable_cerebellum = false; // ablation: +0.18 有害 (CF-LTD过度抑制)

    // v35: ACC 前扣带回 (冲突监测 + 惊讶检测 + 波动性 + 觅食决策)
    // 替代硬编码 ne_floor, 让探索/利用平衡由神经动力学驱动
    // Biology: dACC冲突→LC-NE↑探索, PRO模型惊讶→注意力↑
    //          波动性→学习率调制, 觅食→策略切换
    // Refs: Botvinick 2001, Alexander & Brown 2011, Behrens 2007, Shenhav 2013
    bool  enable_acc = true;

    // v27: Developmental period (critical period for visual feature learning)
    // Biology: infant visual cortex spends ~6 months self-organizing via Hebbian STDP
    // before goal-directed behavior begins. Agent random-walks during dev period,
    // visual STDP + predictive coding learn features, no DA-STDP reward learning.
    size_t dev_period_steps = 100;    // v33: 2000→100, 快速发展期后进入奖励学习
    bool   enable_predictive_learning = true;  // L6 prediction + error-gated STDP

    // Evolution fast-eval mode
    bool fast_eval = false;

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
    AnteriorCingulate* acc() const { return acc_; }

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
    LC_NE*          lc_     = nullptr;   // v34: NE exploration
    NBM_ACh*        nbm_    = nullptr;   // v34: ACh attention
    DRN_5HT*        drn_    = nullptr;   // v34: 5-HT patience
    AnteriorCingulate* acc_  = nullptr;   // v35: ACC conflict/surprise/volatility

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
