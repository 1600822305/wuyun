#pragma once
/**
 * iSTDP — 抑制性 STDP (对称型)
 *
 * 维护 E/I 平衡:
 *   |Δt| < τ_window → 增强抑制 (相关 = 需要更强抑制)
 *   |Δt| ≥ τ_window → 减弱抑制 (不相关 = 抑制过强)
 *
 * 方程:
 *   Δw = A · exp(-|Δt| / τ)    当 |Δt| < τ_window
 *   Δw = -B                     当 |Δt| ≥ τ_window
 *
 * 设计文档: docs/02_neuron_system_design.md §3.2
 */

#include <cstddef>
#include <cstdint>

namespace wuyun {

struct ISTDPParams {
    float a_corr    = 0.005f;   // 相关时增强幅度
    float b_uncorr  = 0.001f;   // 不相关时减弱幅度
    float tau       = 20.0f;    // 时间窗衰减常数 (ms)
    float tau_window= 30.0f;    // 相关/不相关判定边界 (ms)
    float w_min     = 0.0f;
    float w_max     = 2.0f;     // 抑制权重上限可以较高
};

/** 计算单对 iSTDP 权重更新 */
float istdp_delta_w(float t_pre, float t_post, const ISTDPParams& params);

/** 批量 iSTDP 更新 */
void istdp_update_batch(
    float* weights,
    size_t n_synapses,
    const float* pre_times,
    const float* post_times,
    const int32_t* pre_ids,
    const int32_t* post_ids,
    const ISTDPParams& params
);

} // namespace wuyun
