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

    // --- 预测编码状态 ---
    bool pc_enabled_ = false;
    std::set<uint32_t> pc_feedback_sources_;  // 反馈来源区域ID
    std::vector<float> pc_prediction_buf_;    // L2/3 sized, 来自高级区L6的预测
    float pc_precision_sensory_ = 1.0f;       // NE调制: 感觉精度
    float pc_precision_prior_   = 1.0f;       // ACh调制: 先验精度
    float pc_error_smooth_      = 0.0f;       // 指数平滑的预测误差
    static constexpr float PC_ERROR_SMOOTH = 0.1f;  // 平滑率
    static constexpr float PC_PRED_DECAY   = 0.7f;  // 预测缓冲衰减
};

} // namespace wuyun
