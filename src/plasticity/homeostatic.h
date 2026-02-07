#pragma once
/**
 * Homeostatic Plasticity — 稳态可塑性
 *
 * 突触缩放 (Synaptic Scaling):
 *   目标: 维持神经元发放率在合理范围内
 *   机制: 如果发放率偏离目标, 整体缩放所有输入突触权重
 *
 *   scale = target_rate / actual_rate
 *   w_i → w_i · (1 + η · (target_rate - actual_rate) / target_rate)
 *
 * 时间尺度: 非常慢 (秒~分钟级), 用滑动平均估计发放率
 *
 * 设计文档: docs/02_neuron_system_design.md §3.1
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wuyun {

struct HomeostaticParams {
    float target_rate   = 5.0f;    // 目标发放率 (Hz)
    float eta           = 0.001f;  // 缩放学习率 (非常慢)
    float tau_rate      = 5000.0f; // 发放率估计时间常数 (ms)
    float w_min         = 0.01f;   // 权重下限 (不允许降到0)
    float w_max         = 2.0f;    // 权重上限
};

/**
 * 突触缩放处理器
 *
 * 每个神经元群体一个, 追踪发放率并周期性缩放输入权重。
 */
class SynapticScaler {
public:
    /**
     * @param n_neurons  神经元数量
     * @param params     稳态参数
     */
    SynapticScaler(size_t n_neurons, const HomeostaticParams& params);

    /**
     * 更新发放率估计 (每步调用)
     *
     * @param fired  发放标志数组 (size = n_neurons)
     * @param dt     时间步长 (ms)
     */
    void update_rates(const bool* fired, float dt = 1.0f);

    /**
     * 对一组突触权重应用缩放
     *
     * @param weights    权重数组 (in/out)
     * @param n_synapses 突触数量
     * @param post_ids   每个突触的突触后 ID
     */
    void apply_scaling(float* weights, size_t n_synapses, const int32_t* post_ids);

    // 访问器
    float rate(size_t idx) const { return rates_[idx]; }
    size_t size() const { return n_; }

private:
    size_t n_;
    HomeostaticParams params_;
    std::vector<float> rates_;   // 滑动平均发放率估计 (Hz)
};

} // namespace wuyun
