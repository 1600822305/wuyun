#pragma once
/**
 * AnteriorCingulate — 前扣带回皮层 (ACC)
 *
 * 整合多个经典 ACC 计算模型:
 *
 * 1. 冲突监测 (Botvinick et al. 2001):
 *    - 检测 BG D1 子群之间的动作竞争冲突
 *    - conflict = Σ_{i≠j} rate_i × rate_j (能量函数)
 *    - 高冲突 → 增加认知控制 (ACC→LC→NE↑)
 *
 * 2. 预测结果模型 PRO (Alexander & Brown 2011):
 *    - ACC 预测行动-结果的概率
 *    - 实际结果与预测不符 → 惊讶信号 (不分正负效价)
 *    - surprise = |actual_outcome - predicted_outcome|
 *
 * 3. 环境波动性检测 (Behrens et al. 2007):
 *    - 追踪奖励率的变化速度
 *    - 高波动 → 提高学习率 (快速适应)
 *    - 低波动 → 降低学习率 (保持稳定)
 *
 * 4. 努力/控制价值 EVC (Shenhav et al. 2013, Verguts et al. 2015):
 *    - 计算施加认知控制的期望价值
 *    - ACC→LC: 冲突→NE boosting (替代硬编码 ne_floor)
 *    - ACC→dlPFC: 注意力/努力分配
 *
 * 5. 觅食决策 (Kolling et al. 2012, Hayden et al. 2011):
 *    - 追踪局部 vs 全局奖励率
 *    - 局部 < 全局 → 切换策略 (leave patch)
 *
 * 解剖学连接 (StatPearls, Neuroanatomy Cingulate Cortex):
 *   输入: dlPFC(上下文), BG(动作竞争), VTA-DA(RPE), Amygdala-CeA(威胁)
 *   输出: LC(唤醒/探索), dlPFC(控制/注意), VTA(惊讶调制)
 *
 * 子区域:
 *   dACC (背侧): 认知/冲突/觅食 — Brodmann 24b, 32'
 *   vACC (腹侧): 情绪/自主神经 — Brodmann 25, 32
 *   CMA  (扣带运动区): 内在状态→运动指令
 *
 * 遵守 00 文档反作弊原则:
 *   - 冲突信号来自神经元活动模式，不是 IF 逻辑
 *   - 预测存在于突触权重中，不是变量
 *   - 波动性是统计量的涌现，不是硬编码
 *
 * 生物学参考:
 *   - Botvinick et al. (2001) Conflict monitoring and cognitive control
 *   - Alexander & Brown (2011) Medial prefrontal cortex as action-outcome predictor
 *   - Behrens et al. (2007) Learning the value of information in an uncertain world
 *   - Shenhav et al. (2013) The expected value of control
 *   - Verguts et al. (2015) Adaptive effort allocation
 *   - Holroyd & Coles (2002) The neural basis of human error processing (RL-ERN)
 *   - Kolling et al. (2012) Neural mechanisms of foraging
 */

#include "region/brain_region.h"
#include "core/population.h"
#include "core/synapse_group.h"
#include "core/types.h"
#include <array>

namespace wuyun {

struct ACCConfig {
    std::string name = "ACC";

    // === 神经元群体 ===
    // dACC: 背侧ACC — 冲突监测 + 觅食决策 + 认知控制
    size_t n_dacc     = 12;   // dACC 锥体神经元 (冲突/觅食/波动性)
    // vACC: 腹侧ACC — 情绪评价 + 自主神经调控
    size_t n_vacc     = 8;    // vACC 锥体神经元 (情绪/动机/惊讶)
    // 抑制性中间神经元 (E/I 平衡)
    size_t n_inh      = 6;    // PV+ basket cells (快速抑制)

    // === 内部连接 ===
    float p_dacc_to_vacc = 0.25f;  // dACC → vACC (认知→情绪)
    float p_vacc_to_dacc = 0.20f;  // vACC → dACC (情绪→认知)
    float p_exc_to_inh   = 0.30f;  // Exc → Inh
    float p_inh_to_exc   = 0.40f;  // Inh → Exc (E/I balance)
    float w_exc          = 0.5f;
    float w_inh          = 0.8f;

    // === 冲突监测参数 (Botvinick 2001) ===
    // conflict = Σ_{i≠j} rate_i × rate_j / max(Σrate², ε)
    // 高冲突 = 多个D1子群同等活跃 (动作竞争)
    // 低冲突 = 单个D1子群主导 (明确选择)
    float conflict_decay      = 0.85f;   // 冲突积分器衰减
    float conflict_gain       = 3.0f;    // 冲突 → dACC兴奋增益

    // === PRO模型: 预测与惊讶 (Alexander & Brown 2011) ===
    // 预测: 缓慢跟踪最近奖励率 (作为对下一步结果的预测)
    // 惊讶: |实际结果 - 预测| (不分正负效价!)
    // "ACC doesn't care about good or bad, only if it was expected"
    float prediction_tau      = 0.97f;   // 预测器衰减 (慢速跟踪)
    float surprise_gain       = 2.0f;    // 惊讶 → vACC兴奋增益
    float surprise_decay      = 0.80f;   // 惊讶信号衰减

    // === 波动性检测 (Behrens 2007) ===
    // 追踪奖励率方差的变化速度
    // 高波动 → 环境在变 → 需要更快学习 → 提高学习率
    float volatility_fast_tau = 0.90f;   // 快速奖励率追踪
    float volatility_slow_tau = 0.99f;   // 慢速奖励率追踪 (基线)
    float volatility_gain     = 2.0f;    // 波动性信号增益

    // === 觅食决策 (Kolling 2012) ===
    // 局部奖励率(当前策略) vs 全局奖励率(长期平均)
    // 局部 < 全局 → 应该切换策略 → ACC activation↑
    float foraging_tau        = 0.95f;   // 全局奖励率追踪

    // === 输出增益 ===
    float acc_to_lc_gain      = 1.2f;    // ACC→LC: 冲突+惊讶 → 唤醒/探索
    float acc_to_dlpfc_gain   = 1.0f;    // ACC→dlPFC: 控制信号 → 注意力
    float acc_to_vta_gain     = 0.5f;    // ACC→VTA: 惊讶 → DA调制
};

class AnteriorCingulate : public BrainRegion {
public:
    explicit AnteriorCingulate(const ACCConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_all_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_all_; }

    // === ACC 特有接口 ===

    // --- 输入注入 ---

    /** 注入 BG D1 子群发放率 (4个方向组)
     *  Biology: 纹状体→ACC投射 (cortico-striatal loop)
     *  用于冲突监测: 多组同等活跃 = 高冲突 */
    void inject_d1_rates(const std::array<float, 4>& d1_group_rates);

    /** 注入奖励结果 (用于PRO预测误差 + 波动性计算)
     *  Biology: VTA DA → ACC (RPE信号)
     *  ACC 不区分正负效价, 只关注"是否预期到了" */
    void inject_outcome(float reward);

    /** 注入威胁/应激信号 (来自 Amygdala CeA)
     *  Biology: CeA → vACC (情绪唤醒, 高紧迫性) */
    void inject_threat(float threat_level);

    // --- 输出读取 ---

    /** 冲突水平 [0,1]: 多高的动作竞争
     *  高冲突 → 应提高探索 (ACC→LC→NE↑)
     *  Biology: dACC conflict signal drives LC phasic burst */
    float conflict_level() const { return conflict_level_; }

    /** 惊讶水平 [0,1]: 结果有多出乎预料
     *  高惊讶 → 需要更多注意力 (ACC→LC→NE↑)
     *  Biology: unexpected outcomes (both + and -) → ACC activation */
    float surprise_level() const { return surprise_level_; }

    /** 波动性水平 [0,1]: 环境变化有多快
     *  高波动 → 应提高学习率
     *  Biology: ACC adjusts learning rate based on reward volatility
     *  (Behrens 2007: optimal Bayesian learner adjusts learning rate) */
    float volatility_level() const { return volatility_level_; }

    /** 觅食切换信号 [0,1]: 是否应该切换策略
     *  高 → 当前策略不如平均 → 应该改变
     *  Biology: ACC foraging value = global_rate - local_rate */
    float foraging_signal() const { return foraging_signal_; }

    /** 综合控制信号 [0,1]: ACC→LC 唤醒/探索驱动
     *  = conflict + surprise + foraging 的加权组合
     *  替代硬编码 ne_floor, 让 LC-NE 动态响应认知需求 */
    float arousal_drive() const { return arousal_drive_; }

    /** 注意力控制信号 [0,1]: ACC→dlPFC 认知控制
     *  高冲突/惊讶时增强dlPFC注意力 → 更精确的决策 */
    float attention_signal() const { return attention_signal_; }

    /** 学习率调制因子 [0.5, 2.0]: ACC 波动性 → DA-STDP 学习率缩放
     *  高波动 → 因子>1 → 学得快; 低波动 → 因子<1 → 学得慢 */
    float learning_rate_modulation() const { return lr_modulation_; }

    // --- 内部状态诊断 ---
    float predicted_reward() const { return predicted_reward_; }
    float reward_rate_fast() const { return reward_rate_fast_; }
    float reward_rate_slow() const { return reward_rate_slow_; }

    const NeuronPopulation& dacc() const { return dacc_; }
    const NeuronPopulation& vacc() const { return vacc_; }

private:
    void build_synapses();
    void aggregate_state();
    void update_conflict(int32_t t);
    void update_surprise(int32_t t);
    void update_volatility(int32_t t);
    void update_foraging(int32_t t);
    void compute_outputs(int32_t t);

    ACCConfig config_;

    // === 神经元群体 ===
    NeuronPopulation dacc_;     // dACC (背侧, 冲突/觅食/认知)
    NeuronPopulation vacc_;     // vACC (腹侧, 情绪/惊讶/动机)
    NeuronPopulation inh_;      // PV+ basket cells (E/I balance)

    // === 内部突触 ===
    SynapseGroup syn_dacc_to_vacc_;  // dACC → vACC
    SynapseGroup syn_vacc_to_dacc_;  // vACC → dACC
    SynapseGroup syn_dacc_to_inh_;   // dACC → Inh
    SynapseGroup syn_vacc_to_inh_;   // vACC → Inh
    SynapseGroup syn_inh_to_dacc_;   // Inh → dACC
    SynapseGroup syn_inh_to_vacc_;   // Inh → vACC

    // === PSP 缓冲 (跨区域输入) ===
    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_dacc_;
    std::vector<float> psp_vacc_;

    // === 冲突监测状态 (Botvinick 2001) ===
    std::array<float, 4> d1_rates_ = {0.0f, 0.0f, 0.0f, 0.0f};
    float conflict_raw_     = 0.0f;   // 原始冲突能量
    float conflict_level_   = 0.0f;   // 平滑冲突水平 [0,1]

    // === PRO 预测状态 (Alexander & Brown 2011) ===
    float predicted_reward_ = 0.0f;   // 预期奖励 (缓慢跟踪)
    float last_outcome_     = 0.0f;   // 最近实际结果
    float surprise_raw_     = 0.0f;   // 原始惊讶
    float surprise_level_   = 0.0f;   // 平滑惊讶水平 [0,1]

    // === 波动性状态 (Behrens 2007) ===
    float reward_rate_fast_ = 0.0f;   // 快速奖励率追踪
    float reward_rate_slow_ = 0.0f;   // 慢速奖励率追踪 (基线)
    float volatility_raw_   = 0.0f;   // |fast - slow| = 变化速度
    float volatility_level_ = 0.0f;   // 平滑波动性 [0,1]

    // === 觅食状态 (Kolling 2012) ===
    float local_reward_rate_  = 0.0f; // 近期奖励率 (当前"patch")
    float global_reward_rate_ = 0.0f; // 长期平均奖励率
    float foraging_signal_    = 0.0f; // 切换信号 [0,1]

    // === 输出信号 ===
    float arousal_drive_    = 0.0f;   // ACC→LC (综合唤醒)
    float attention_signal_ = 0.0f;   // ACC→dlPFC (认知控制)
    float lr_modulation_    = 1.0f;   // 波动性→学习率调制

    // === 威胁输入 ===
    float threat_input_     = 0.0f;
    static constexpr float THREAT_DECAY = 0.85f;

    // === 聚合发放状态 ===
    std::vector<uint8_t> fired_all_;
    std::vector<int8_t>  spike_type_all_;
};

} // namespace wuyun
