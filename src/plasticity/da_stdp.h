#pragma once
/**
 * DA-STDP — 三因子多巴胺调制 STDP
 *
 * 核心机制:
 *   1. STDP 产生资格痕迹 (eligibility trace), 而非直接改权重
 *   2. 资格痕迹以 τ_e 衰减
 *   3. 当 DA 信号到达时, 资格痕迹 × DA 浓度 → 实际权重变化
 *   4. 解决信用分配问题 (奖励延迟)
 *
 * 方程:
 *   de/dt = -e / τ_e + STDP(Δt)
 *   Δw = DA_signal · e
 *
 * 设计文档: docs/02_neuron_system_design.md §3.2
 */

#include "stdp.h"
#include <vector>
#include <cstddef>

namespace wuyun {

struct DASTDPParams {
    STDPParams stdp;                // 基础 STDP 参数
    float tau_eligibility = 1000.0f; // 资格痕迹衰减时间 (ms)
    float da_baseline     = 0.0f;   // DA 基线 (tonic level)
    float w_min           = 0.0f;
    float w_max           = 1.0f;
};

/**
 * DA-STDP 状态管理器
 *
 * 每个突触维护一个资格痕迹 (eligibility trace)。
 * 每步: 更新资格痕迹 → 当 DA 信号到达时转化为权重变化。
 */
class DASTDPProcessor {
public:
    /**
     * @param n_synapses  突触数量
     * @param params      DA-STDP 参数
     */
    DASTDPProcessor(size_t n_synapses, const DASTDPParams& params);

    /**
     * 更新资格痕迹 (每步调用)
     *
     * @param pre_times   突触前最近发放时间 (per neuron)
     * @param post_times  突触后最近发放时间 (per neuron)
     * @param pre_ids     每个突触的突触前 ID
     * @param post_ids    每个突触的突触后 ID
     * @param dt          时间步长 (ms)
     */
    void update_traces(
        const float* pre_times,
        const float* post_times,
        const int32_t* pre_ids,
        const int32_t* post_ids,
        float dt = 1.0f
    );

    /**
     * 应用 DA 调制的权重更新
     *
     * @param weights     突触权重数组 (in/out)
     * @param da_signal   当前 DA 浓度 (相对于基线)
     */
    void apply_da_modulation(float* weights, float da_signal);

    // 访问器
    size_t size() const { return n_; }
    const std::vector<float>& traces() const { return eligibility_; }

private:
    size_t n_;
    DASTDPParams params_;
    std::vector<float> eligibility_;  // 资格痕迹 (每突触)
};

} // namespace wuyun
