#pragma once
/**
 * ClosedLoopAgent — 闭环智能体
 *
 * 将 Environment 环境与 WuYun 大脑连接形成完整的感知-决策-行动循环:
 *
 *   Environment.observe() → VisualInput → LGN → V1 → ... → dlPFC → BG → MotorThal → M1
 *                                                                                      ↓
 *   Environment.step(dx,dy) ← decode_m1_continuous() ← M1 L5 population vector
 *                ↓
 *   reward → Hypothalamus → VTA DA → BG DA-STDP → 学习
 *
 * 动作解码:
 *   M1 L5 群体向量编码 (Georgopoulos 1986), 连续位移 (dx, dy)
 *
 * 生物学基础:
 *   - 运动皮层 M1 L5 锥体细胞直接投射到脊髓 (皮质脊髓束)
 *   - BG Go/NoGo 通路选择动作
 *   - VTA RPE 信号驱动强化学习
 */

#include "engine/environment.h"
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
#include "region/subcortical/nucleus_accumbens.h"
#include "region/neuromod/snc_da.h"
#include "region/subcortical/superior_colliculus.h"
#include "region/subcortical/periaqueductal_gray.h"
#include "region/limbic/hypothalamus.h"
#include "region/anterior_cingulate.h"
#include "region/prefrontal/orbitofrontal.h"
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

    // Perception (auto-computed from Environment::vis_width/height in constructor)
    size_t vision_width  = 5;   // v21: default 5x5 local patch (vision_radius=2)
    size_t vision_height = 5;

    // Action decoding
    // Step 33 best: 30gen×40pop Baldwin (gen26, fitness=2.05, consolidation+dev100)
    // v39: 12→20 to accommodate 5-level visual hierarchy (14-step propagation delay)
    // With 20 steps: cortical signals reach BG at step ~14, leaving 6 steps of
    // cortical→BG overlap for direction-specific eligibility trace building.
    // Previous: 12 steps < 14 delay → cortical spikes never reached BG within one step.
    // v47: brain_steps 20→12 (evolved: spike-driven RPE propagates faster)
    size_t brain_steps_per_action = 12;
    size_t reward_processing_steps = 9;   // v47: Baldwin 10→9

    float reward_scale = 2.39f;  // v47: Baldwin 3.50→2.39 (VTA RPE is spike-driven, less amplification)

    float exploration_noise = 35.6f;  // v47: Baldwin 48→36 (pop vector more efficient)
    size_t exploration_anneal_steps = 0;

    // Learning
    // v47: Baldwin re-evolution for pop vector + VTA internal RPE architecture
    bool enable_da_stdp     = true;
    float da_stdp_lr        = 0.080f;  // v47: Baldwin 0.022→0.080 (spike RPE weaker → bigger LR)
    bool enable_homeostatic = true;
    bool enable_cortical_stdp = true;
    float cortical_stdp_a_plus  = 0.0015f; // v47: Baldwin 0.001→0.0015
    float cortical_stdp_a_minus = -0.013f; // v47: Baldwin -0.010→-0.013
    float cortical_stdp_w_max   = 1.42f;   // v47: Baldwin 0.81→1.42

    float lgn_gain           = 234.0f;  // v47: Baldwin 394→234 (Hypo added signal, less LGN needed)
    float lgn_baseline       = 17.3f;   // v47: Baldwin 16→17.3
    float lgn_noise_amp      = 4.9f;    // v47: Baldwin 2.0→4.9

    float bg_to_m1_gain      = 6.09f;   // v47: Baldwin 7.08→6.09 (cos similarity gentler)
    float attractor_drive_ratio  = 0.55f; // v47: Baldwin 0.50→0.55
    float background_drive_ratio = 0.02f; // v47: Baldwin 0.26→0.02 (near zero! attractor dominates)

    float ne_food_scale      = 6.55f;  // v47: Baldwin 2.36→6.55
    float ne_floor           = 0.65f;   // v47: Baldwin 0.81→0.65

    float homeostatic_target_rate = 11.0f;  // v47: Baldwin 3.2→11.0 (higher target firing rate)
    float homeostatic_eta    = 0.0079f;     // v47: Baldwin 0.0022→0.0079

    // Brain size factors (multiplied on base neuron counts)
    float v1_size_factor     = 1.75f;  // v47: Baldwin 1.12→1.75 (bigger V1 for richer signal)
    float dlpfc_size_factor  = 1.09f;  // v47: Baldwin 2.31→1.09 (smaller dlPFC)
    float bg_size_factor     = 0.96f;  // v47: Baldwin 0.77→0.96

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
    int   replay_passes      = 11;     // v47: Baldwin 14→11
    float replay_da_scale    = 0.61f;   // v47: Baldwin 0.31→0.61 (stronger replay DA for spike RPE)
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

    // v40: NAcc 伏隔核 (ventral striatum) — 动机/奖赏整合
    // Biology: VTA→NAcc (mesolimbic) 独立于 SNc→dStr (nigrostriatal)
    //   NAcc Core: approach motivation (D1) vs avoidance (D2)
    //   NAcc Shell: novelty/context change detection
    //   NAcc→VP→BG: motivation modulates motor vigor
    // Ref: Mogenson 1980 "limbic-motor interface"
    bool  enable_nacc = true;

    // v40: SNc 黑质致密部 (nigrostriatal DA) — 习惯学习通路
    // Biology: SNc tonic DA → 背侧纹状体, 维持已学习惯 (Yin & Knowlton 2006)
    //   VTA phasic: 新学习 | SNc tonic: 已学维持
    //   习惯形成后 SNc 维持 BG 权重, VTA 波动不再影响已学行为
    bool  enable_snc = true;

    // v40: Superior Colliculus 上丘 — 皮层下快速显著性检测
    // Biology: 视网膜→SC→Pulvinar, 60ms 快速通道 (Krauzlis 2013)
    //   SC 浅层: 视觉地图 | SC 深层: 多模态整合+注意力定向
    //   危险物体快速反应, 不等皮层处理完
    bool  enable_sc = true;

    // v41: PAG 导水管周围灰质 — CeA→PAG 应急反射
    // Biology: CeA→PAG→脑干, 不经 BG 的硬连线防御 (LeDoux 1996)
    //   dlPAG: 主动防御 (flight) | vlPAG: 被动防御 (freeze)
    //   第一次遇到 danger 的即时反应 (BG 还没学会回避)
    bool  enable_pag = true;

    // v52: 皮层下反射弧 (先天硬连线, 不需要学习)
    // SC 趋近反射: SC 深层 → M1 方向性激活
    //   生物学: SC 深层运动地图与视觉地图对齐 (Stein & Meredith 1993)
    //   看到亮/显著刺激 → SC 计算方位 → 驱动 M1 朝向 = "天生好奇"
    //   这条通路 2-3 步出结果, 皮层慢通路 14 步
    float sc_approach_gain = 25.0f;   // SC 深层→M1 趋近增益 (要压过噪声~35)

    // PAG 冻结反射: PAG dlPAG → M1 全局抑制
    //   生物学: PAG→脑干运动核抑制 = 冻结反应 (LeDoux 1996)
    //   CeA 高活性 (恐惧) → PAG 激活 → 抑制所有 M1 输出 → STAY
    //   v43 教训: PAG→M1 激活(驱动运动)是错的(PAG 没方向信息)
    //   v52 正确: PAG→M1 抑制(压制运动) — 冻结不需要方向
    float pag_freeze_gain = 30.0f;    // PAG dlPAG→M1 冻结抑制增益 (要压过趋近+噪声)

    // v59: 墙壁回避反射 — 视觉 patch 检测前方墙壁 → M1 偏离墙壁方向
    //   生物学: 前庭触须系统 / 视动反射 (Goodale 2011)
    //   看到墙壁在前方 → 计算墙壁质心方向 → M1 cos 驱动反方向
    float wall_avoid_gain = 20.0f;

    // v59: 探索饥饿重置 — 连续 N 步无奖赏 → noise 翻倍
    //   生物学: LC NE burst mode 在长时间无奖赏时触发 (Aston-Jones 2005)
    //   防止 agent 在同一区域徘徊
    size_t starvation_threshold = 30;    // 多少步无奖赏后加大探索
    float  starvation_noise_boost = 2.0f; // 饥饿时 noise 乘数

    // v52b: 新奇性一次学习 (Phase B)
    // 生物学: 新奇刺激 → VTA DA burst 远大于熟悉刺激 (Ljungberg 1992)
    //   第一次碰到食物: DA burst × novelty_boost → 一次成型
    //   第 N 次: DA burst × 1.0 (已熟悉, 不需要再放大)
    //   第一次碰到危险: Amygdala STDP 已有 one-shot (a_plus=0.10)
    //     + 新奇性放大 → 更强的恐惧记忆 + 更多回放
    //   生物机制: 海马 CA1 新奇检测 → VTA (Lisman & Grace 2005)
    float novelty_da_boost = 5.0f;    // 第一次奖励的 DA 放大倍数

    // v41: FPC 前额极皮层 (BA10) — 元认知/多步规划
    // Biology: 人脑层级最高的前额叶区域 (Koechlin 2003)
    //   维持长期目标, 多任务协调, 前瞻推理
    //   FPC → dlPFC top-down 调制 (目标→计划→动作)
    bool  enable_fpc = true;

    // v42: OFC 眶额皮层 (BA11/47) — 刺激-结果关联 (价值预测)
    // Biology: OFC 编码"看到X→预期Y奖赏", DA 调制更新 (Rolls 2000)
    //   pos value: 食物相关刺激 | neg value: 危险相关刺激
    //   OFC → dlPFC/NAcc 价值信号引导决策
    bool  enable_ofc = true;

    // v42: vmPFC 腹内侧前额叶 (BA14/25) — 情绪调节/恐惧消退
    // Biology: vmPFC → Amygdala ITC "安全信号", 驱动恐惧消退 (Milad & Quirk 2002)
    //   综合 OFC 价值 + Hippocampus 上下文 → 安全评估
    bool  enable_vmpfc = true;

    // v27: Developmental period (critical period for visual feature learning)
    // Biology: infant visual cortex spends ~6 months self-organizing via Hebbian STDP
    // before goal-directed behavior begins. Agent random-walks during dev period,
    // visual STDP + predictive coding learn features, no DA-STDP reward learning.
    size_t dev_period_steps = 100;    // v33: 2000→100, 快速发展期后进入奖励学习
    bool   enable_predictive_learning = true;  // L6 prediction + error-gated STDP

    // Evolution fast-eval mode
    bool fast_eval = false;

    // v55: Continuous movement — the ONLY motor output mode
    // Biology: M1 population vector (angle + coherence) → float displacement.
    // There is no discrete 4-direction mode; real brains don't have one.
    float continuous_step_size = 0.8f;  // max displacement per step (≤1.0 to avoid skipping cells)
};

/** 每步回调: agent_step, action, reward, agent_x, agent_y */
using AgentStepCallback = std::function<void(int, Action, float, int, int)>;

class ClosedLoopAgent {
public:
    /** 构造闭环智能体
     *  @param env    环境 (所有权转移给 Agent)
     *  @param config 大脑/学习参数 (vision_width/height 自动从 env 推导) */
    ClosedLoopAgent(std::unique_ptr<Environment> env, const AgentConfig& config = {});

    // Non-copyable, non-movable (contains SimulationEngine with unique_ptrs + cached raw pointers)
    ClosedLoopAgent(const ClosedLoopAgent&) = delete;
    ClosedLoopAgent& operator=(const ClosedLoopAgent&) = delete;
    ClosedLoopAgent(ClosedLoopAgent&&) = delete;
    ClosedLoopAgent& operator=(ClosedLoopAgent&&) = delete;

    /** 重置环境 (大脑保持不变, 只重置环境) */
    void reset_world();

    /** v53: 换种子重置环境 (反转学习: 大脑保留, 世界换布局)
     *  食物/危险位置完全改变, 但大脑权重保留
     *  → 测试: 旧策略失效时能否快速适应 */
    void reset_world_with_seed(uint32_t seed);

    /**
     * 执行一个环境步:
     *   1. observe → encode → inject LGN
     *   2. run brain N steps
     *   3. decode M1 → action
     *   4. env.step(dx, dy)
     *   5. reward → Hypothalamus → VTA
     * @return Environment::Result
     */
    Environment::Result agent_step();

    /** 运行 n 个环境步 */
    void run(int n_steps);

    /** 设置每步回调 */
    void set_callback(AgentStepCallback cb) { callback_ = std::move(cb); }

    // --- 访问器 ---
    Environment&       env()    { return *env_; }
    const Environment& env() const { return *env_; }
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
    NucleusAccumbens* nacc() const { return nacc_; }
    SNc_DA*           snc()  const { return snc_; }
    SuperiorColliculus* sc()   const { return sc_; }
    PeriaqueductalGray*  pag()  const { return pag_; }
    CorticalRegion*     fpc()  const { return fpc_; }
    OrbitofrontalCortex* ofc()  const { return ofc_; }
    CorticalRegion*     vmpfc() const { return vmpfc_; }
    Hypothalamus*       hypo()  const { return hypo_; }

private:
    AgentConfig config_;

    std::unique_ptr<Environment> env_;
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
    NucleusAccumbens*  nacc_  = nullptr;   // v40: NAcc motivation/reward
    SNc_DA*            snc_   = nullptr;   // v40: SNc habit DA
    SuperiorColliculus* sc_    = nullptr;   // v40: SC saliency
    PeriaqueductalGray*  pag_   = nullptr;   // v41: PAG defense
    CorticalRegion*     fpc_   = nullptr;   // v41: FPC planning
    OrbitofrontalCortex* ofc_   = nullptr;   // v42: OFC value
    CorticalRegion*     vmpfc_  = nullptr;   // v42: vmPFC emotion regulation
    Hypothalamus*       hypo_   = nullptr;   // v46: hedonic sensory interface

    // v45: Population vector encoding (Georgopoulos 1986)
    // Each M1 L5 neuron has a random preferred direction angle θ ∈ [0, 2π)
    // Each BG D1 neuron has a random preferred direction angle
    // Action = population vector of fired L5 neurons, mapped to closest cardinal direction
    std::vector<float> m1_preferred_dir_;   // M1 L5 preferred direction angles
    std::vector<float> d1_preferred_dir_;   // BG D1 preferred direction angles

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
    // v55: Continuous decode — returns (dx, dy) displacement from population vector
    std::pair<float, float> decode_m1_continuous(const std::vector<int>& l5_accum) const;
    void inject_observation();
    void inject_reward(float reward);

    // --- Frustration tracking (expected reward not received) ---
    float expected_reward_level_ = 0.0f;  // Tracks recent food rate → expected reward

    // --- v52b: 新奇性跟踪 (一次学习) ---
    // 每次遇到食物/危险, 新奇性减半 (habituation)
    // 新奇性高 → DA 放大 + 回放加倍
    // 生物学: 第一次 → 巨大 DA burst, 第 N 次 → 正常 DA burst
    float food_novelty_   = 1.0f;   // 1.0 = 从没见过, 0.0 = 完全熟悉
    float danger_novelty_ = 1.0f;

    // v59: 探索饥饿计数器
    size_t steps_since_reward_ = 0;

    // --- v36: Spatial value map (cognitive map / Tolman 1948) ---
    // Records reward outcomes at each position → value gradient for navigation.
    // Biology: hippocampal place cells + OFC value coding = spatial value memory.
    // Updated on reward events, decays slowly → persistent spatial knowledge.
    std::vector<float> spatial_value_map_;   // [world_width × world_height], init 0
    int spatial_map_w_ = 0;  // discretized map width  (= world_width)
    int spatial_map_h_ = 0;  // discretized map height (= world_height)
    // v37: asymmetric decay — positive values (food) persist, negative (danger) extinguish
    // Biology: fear extinction is faster than reward memory (Milad & Quirk 2002)
    // Previous: uniform 0.999 decay (half-life 693 steps) → negative values never cleared
    //   → entire map became negative → all-direction NoGo → behavioral collapse
    static constexpr float SPATIAL_VALUE_DECAY_POS = 0.998f;  // food: half-life ~346 steps
    static constexpr float SPATIAL_VALUE_DECAY_NEG = 0.990f;  // danger: half-life ~69 steps (fast extinction)
    static constexpr float SPATIAL_VALUE_LR    = 0.3f;   // learning rate
    static constexpr float SPATIAL_VALUE_MAX   = 1.0f;   // cap positive values
    static constexpr float SPATIAL_VALUE_MIN   = -0.5f;  // cap negative (prevent runaway)
    void update_spatial_value_map(float reward);

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
