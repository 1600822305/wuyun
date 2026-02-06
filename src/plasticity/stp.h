#pragma once
/**
 * STP — Short-Term Plasticity (Tsodyks-Markram 模型)
 *
 * 短时程可塑性分两种:
 *   STD (Short-Term Depression): 高频发放→资源耗竭→突触减弱
 *   STF (Short-Term Facilitation): 高频发放→Ca²⁺积累→突触增强
 *
 * 模型方程:
 *   dx/dt = (1 - x) / tau_D - u · x · δ(t - t_spike)
 *   du/dt = (U - u) / tau_F + U · (1 - u) · δ(t - t_spike)
 *   有效权重 = w · u · x
 *
 * 设计文档: docs/02_neuron_system_design.md §4
 */

#include <cstddef>

namespace wuyun {

struct STPParams {
    float U     = 0.2f;     // 基线释放概率
    float tau_D = 200.0f;   // 抑压恢复时间常数 (ms)
    float tau_F = 50.0f;    // 易化衰减时间常数 (ms)
};

/** 单突触 STP 状态 */
struct STPState {
    float x = 1.0f;   // 可用资源 (0~1, 1=满)
    float u = 0.2f;   // 释放概率 (动态变化)
};

/**
 * 更新 STP 状态 (每个时间步调用)
 *
 * @param state   STP 状态 (in/out)
 * @param params  STP 参数
 * @param spiked  本步是否发放
 * @param dt      时间步长 (ms)
 * @return 有效增益 u · x (乘到突触权重上)
 */
float stp_step(STPState& state, const STPParams& params, bool spiked, float dt = 1.0f);

/**
 * 批量 STP 更新
 *
 * @param states    STP 状态数组 (长度 = n)
 * @param n         数量
 * @param params    STP 参数
 * @param fired     发放标志数组
 * @param gains     输出: 有效增益数组 (长度 = n)
 * @param dt        时间步长
 */
void stp_step_batch(
    STPState* states,
    size_t n,
    const STPParams& params,
    const bool* fired,
    float* gains,
    float dt = 1.0f
);

// 预定义参数: STD 为主 (皮层兴奋性突触)
constexpr STPParams STP_DEPRESSION  {0.5f,  200.0f, 20.0f};
// 预定义参数: STF 为主 (皮层抑制性突触)
constexpr STPParams STP_FACILITATION{0.1f,  100.0f, 500.0f};

} // namespace wuyun
