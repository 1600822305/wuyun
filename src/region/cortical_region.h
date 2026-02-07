#pragma once
/**
 * CorticalRegion — 皮层区域 (CorticalColumn 的 BrainRegion 包装)
 *
 * 将已有的 CorticalColumn (6层模板) 适配到 BrainRegion 接口,
 * 使其可以接入 SpikeBus / 振荡 / 调质 系统。
 *
 * V1, dlPFC, M1 等皮层区域都通过此类实例化 (不同参数)。
 */

#include "region/brain_region.h"
#include "circuit/cortical_column.h"
#include <set>
#include <unordered_map>

namespace wuyun {

class CorticalRegion : public BrainRegion {
public:
    /**
     * @param name    区域名称 (如 "V1", "dlPFC", "M1")
     * @param config  皮层柱配置
     */
    CorticalRegion(const std::string& name, const ColumnConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_; }

    // --- 皮层特有接口 ---

    /** 注入前馈输入到 L4 basal */
    void inject_feedforward(const std::vector<float>& currents);

    /** 注入反馈预测到 L2/3 + L5 apical */
    void inject_feedback(const std::vector<float>& currents);

    /** 注入注意力信号到 VIP */
    void inject_attention(float vip_current);

    // --- 注意力接口 ---

    /** 设置顶下注意力增益 (PFC→感觉区选择性放大)
     *  gain=1.0 正常; >1.0 注意; <1.0 忽略
     *  效果: (1) L4 PSP乘以gain (2) VIP驱动→SST去抑制→L2/3 burst增强 */
    void set_attention_gain(float gain) { attention_gain_ = gain; }
    float attention_gain() const { return attention_gain_; }

    /** 获取内部皮层柱 */
    CorticalColumn&       column()       { return column_; }
    const CorticalColumn& column() const { return column_; }

    /** 上一步的输出 */
    const ColumnOutput& output() const { return last_output_; }

    // --- 预测编码接口 ---

    /** 启用预测编码 (L6预测 + L2/3误差 + 精度加权) */
    void enable_predictive_coding();
    bool predictive_coding_enabled() const { return pc_enabled_; }

    /** 标记反馈来源区域 (这些区域的脉冲→prediction_buffer_) */
    void add_feedback_source(uint32_t region_id);

    /** 当前预测误差强度 (指数平滑) */
    float prediction_error() const { return pc_error_smooth_; }

    /** 精度参数 (NE↑→sensory↑, ACh↑→prior↓) */
    float precision_sensory() const { return pc_precision_sensory_; }
    float precision_prior()   const { return pc_precision_prior_; }

    // --- 工作记忆接口 ---

    /** 启用工作记忆 (L2/3循环自持 + DA稳定) */
    void enable_working_memory();
    bool working_memory_enabled() const { return wm_enabled_; }

    /** 当前工作记忆持续性 (活跃L2/3比例, 0~1) */
    float wm_persistence() const;

    /** DA对工作记忆的增益 */
    float wm_da_gain() const { return wm_da_gain_; }

    // --- 拓扑输入映射 ---

    /** 注册拓扑输入源: 该源的spikes使用比例映射(preserves spatial structure)
     *  而非默认的模取余映射(scrambles spatial info).
     *  Biology: V1→V2→V4→IT维持partial retinotopy */
    void add_topographic_input(uint32_t source_region_id, size_t source_n_neurons);

    // --- 稳态可塑性接口 ---

    /** 启用稳态可塑性 (突触缩放, 维持E/I平衡) */
    void enable_homeostatic(const HomeostaticParams& params = {}) { column_.enable_homeostatic(params); }
    bool homeostatic_enabled() const { return column_.has_homeostatic(); }

    /** 各层平均发放率 (诊断) */
    float l4_mean_rate()  const { return column_.l4_mean_rate(); }
    float l23_mean_rate() const { return column_.l23_mean_rate(); }
    float l5_mean_rate()  const { return column_.l5_mean_rate(); }
    float l6_mean_rate()  const { return column_.l6_mean_rate(); }

    // --- 睡眠慢波接口 ---

    /** 设置睡眠模式 (NREM慢波 up/down 状态交替) */
    void set_sleep_mode(bool sleep) { sleep_mode_ = sleep; rem_mode_ = false; slow_wave_phase_ = 0.0f; }
    bool is_sleep_mode() const { return sleep_mode_; }

    /** 当前是否处于 up state (慢波上升期, 神经元可兴奋) */
    bool is_up_state() const { return sleep_mode_ && slow_wave_phase_ < UP_DUTY_CYCLE; }

    /** 慢波相位 (0~1, 0~UP_DUTY=up, UP_DUTY~1=down) */
    float slow_wave_phase() const { return slow_wave_phase_; }

    // --- REM睡眠接口 ---

    /** 设置REM模式 (去同步化 + PGO波 + 运动抑制) */
    void set_rem_mode(bool rem);
    bool is_rem_mode() const { return rem_mode_; }

    /** 注入PGO波 (梦境视觉激活, 由SleepCycleManager触发) */
    void inject_pgo_wave(float amplitude);

    /** 设置运动抑制 (REM肌肉弛缓, 用于M1等运动区) */
    void set_motor_atonia(bool atonia) { motor_atonia_ = atonia; }
    bool is_motor_atonia() const { return motor_atonia_; }

private:
    CorticalColumn column_;
    ColumnOutput   last_output_;

    // 聚合发放状态 (所有群体合并)
    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;

    // PSP 输入缓冲: 模拟跨区域突触后电位的时间常数
    // 每个到达脉冲维持数步的电流注入 (指数衰减)
    static constexpr float PSP_DECAY = 0.7f;  // 每步衰减为 70%
    std::vector<float> psp_buffer_;  // 每个 L4 神经元的残余 PSP 电流
    float  psp_current_regular_;     // PSP current per regular spike
    float  psp_current_burst_;       // PSP current per burst spike
    size_t psp_fan_out_;             // Number of L4 neurons per incoming spike

    void aggregate_firing_state();

    // --- 拓扑输入源 (region_id → source_n_neurons) ---
    std::unordered_map<uint32_t, size_t> topo_sources_;

    // --- 预测编码状态 ---
    bool pc_enabled_ = false;
    std::set<uint32_t> pc_feedback_sources_;  // 反馈来源区域ID
    std::vector<float> pc_prediction_buf_;    // L2/3 sized, 来自高级区L6的预测
    float pc_precision_sensory_ = 1.0f;       // NE调制: 感觉精度
    float pc_precision_prior_   = 1.0f;       // ACh调制: 先验精度
    float pc_error_smooth_      = 0.0f;       // 指数平滑的预测误差
    static constexpr float PC_ERROR_SMOOTH = 0.1f;  // 平滑率
    static constexpr float PC_PRED_DECAY   = 0.7f;  // 预测缓冲衰减

    // --- 注意力状态 ---
    float attention_gain_ = 1.0f;            // 顶下注意力增益
    static constexpr float VIP_ATT_DRIVE = 15.0f;  // VIP驱动强度 per unit gain above 1.0

    // --- 工作记忆状态 ---
    bool wm_enabled_ = false;
    std::vector<float> wm_recurrent_buf_;    // L2/3 sized 循环缓冲
    float wm_da_gain_ = 1.0f;               // DA调制增益
    static constexpr float WM_RECURRENT_STR = 12.0f; // 基础循环电流 (NMDA-mediated)
    static constexpr float WM_DECAY         = 0.95f; // NMDA 时间常数 (半衰期~14步, 真实约100ms)
    static constexpr float WM_DA_SENSITIVITY= 3.0f;  // D1增强NMDA电流 (3-5x in PFC)
    static constexpr float WM_FAN_OUT       = 5.0f;  // 循环扩散范围

    // --- 睡眠慢波状态 ---
    bool  sleep_mode_       = false;
    float slow_wave_phase_  = 0.0f;
    static constexpr float SLOW_WAVE_FREQ  = 0.001f;  // ~1Hz at 1000 steps/sec
    static constexpr float UP_DUTY_CYCLE   = 0.4f;    // 40% up, 60% down
    static constexpr float DOWN_STATE_INH  = -8.0f;   // Inhibitory current during down state

    // --- REM睡眠状态 ---
    bool  rem_mode_         = false;
    bool  motor_atonia_     = false;
    static constexpr float REM_NOISE_AMP   = 30.0f;   // 去同步化噪声幅度 (bias=18+jitter=12)
    static constexpr float ATONIA_INH      = -20.0f;  // 运动抑制电流
};

} // namespace wuyun
