#pragma once
/**
 * STDP — Spike-Timing Dependent Plasticity
 *
 * 经典 STDP 规则:
 *   Δw = A+ · exp(-Δt/τ+)  if Δt > 0 (pre before post → LTP)
 *   Δw = A- · exp(+Δt/τ-)  if Δt < 0 (post before pre → LTD)
 *
 * 其中 Δt = t_post - t_pre
 *
 * 设计文档: docs/02_neuron_system_design.md §5
 */

#include <cstddef>

namespace wuyun {

struct STDPParams {
    float a_plus   = 0.01f;    // LTP 幅度
    float a_minus  = -0.012f;  // LTD 幅度 (负数)
    float tau_plus  = 20.0f;   // LTP 时间窗 (ms)
    float tau_minus = 20.0f;   // LTD 时间窗 (ms)
    float w_min     = 0.0f;    // 权重下限
    float w_max     = 1.0f;    // 权重上限
};

/**
 * 计算单对 pre-post 的 STDP 权重更新
 *
 * @param t_pre   突触前脉冲时间
 * @param t_post  突触后脉冲时间
 * @param params  STDP 参数
 * @return 权重变化量 Δw
 */
float stdp_delta_w(float t_pre, float t_post, const STDPParams& params);

/**
 * 批量 STDP 更新 — 对一组突触权重应用 STDP
 *
 * @param weights     突触权重数组 (in/out)
 * @param n_synapses  突触数量
 * @param pre_times   突触前最近发放时间
 * @param post_times  突触后最近发放时间
 * @param pre_ids     每个突触的突触前神经元ID
 * @param post_ids    每个突触的突触后神经元ID
 * @param params      STDP 参数
 */
void stdp_update_batch(
    float* weights,
    size_t n_synapses,
    const float* pre_times,
    const float* post_times,
    const int* pre_ids,
    const int* post_ids,
    const STDPParams& params
);

} // namespace wuyun
