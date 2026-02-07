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
};

} // namespace wuyun
