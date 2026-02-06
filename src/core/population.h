#pragma once
/**
 * NeuronPopulation — SoA 向量化双区室 AdLIF+ 神经元群体
 *
 * 将 Python 原型 wuyun/core/population.py 翻译为 C++。
 * 数学方程完全等价，数据布局改为 Struct of Arrays (SoA)。
 *
 * 胞体:
 *   τ_m · dV_s/dt = -(V_s - V_rest) + R_s · I_total - w + κ · (V_a - V_s)
 *   τ_w · dw/dt   = a · (V_s - V_rest) - w
 *   发放: V_s ≥ V_threshold → reset, w += b, 进入不应期
 *
 * 顶端树突:
 *   τ_a · dV_a/dt = -(V_a - V_rest) + R_a · I_apical + κ_back · (V_s - V_a)
 *   Ca²⁺ 脉冲: V_a ≥ V_ca_threshold → ca_spike=True, V_a += ca_boost
 *
 * Burst 判定:
 *   fired & ca_spike → BURST_START
 *   fired & ~ca_spike → REGULAR
 *
 * 设计文档: docs/02_neuron_system_design.md §1
 */

#include "types.h"
#include <vector>
#include <cstddef>

namespace wuyun {

class NeuronPopulation {
public:
    /**
     * @param n      群体中的神经元数量
     * @param params 神经元参数 (同构群体)
     */
    NeuronPopulation(size_t n, const NeuronParams& params);

    /** 推进一个时间步, 返回发放的神经元数量 */
    size_t step(int t, float dt = 1.0f);

    // --- 外部注入电流 (每步清零) ---
    void inject_basal(size_t idx, float current);
    void inject_apical(size_t idx, float current);
    void inject_soma(size_t idx, float current);

    // --- 访问器 ---
    size_t size() const { return n_; }
    bool   has_apical() const { return has_apical_; }

    const std::vector<float>&   v_soma()     const { return v_soma_; }
    const std::vector<float>&   v_apical()   const { return v_apical_; }
    const std::vector<float>&   w_adapt()    const { return w_adapt_; }
    const std::vector<int8_t>&  spike_type() const { return spike_type_; }
    const std::vector<uint8_t>& fired()      const { return fired_; }

    // 可写访问 (用于突触电流注入)
    std::vector<float>& i_basal()  { return i_basal_; }
    std::vector<float>& i_apical() { return i_apical_; }
    std::vector<float>& i_soma()   { return i_soma_; }

private:
    void update_apical(float dt);
    void continue_burst(size_t idx, float dt);
    void update_soma_and_fire(size_t idx, int t, float dt);
    void clear_inputs();

    size_t n_;
    bool   has_apical_;

    // --- 参数向量 (SoA) ---
    std::vector<float> v_rest_;
    std::vector<float> v_threshold_;
    std::vector<float> v_reset_;
    std::vector<float> tau_m_;
    std::vector<float> r_s_;
    std::vector<float> a_adapt_;
    std::vector<float> b_adapt_;
    std::vector<float> tau_w_;
    std::vector<int>   refrac_period_;

    std::vector<float> kappa_;
    std::vector<float> kappa_back_;
    std::vector<float> tau_a_;
    std::vector<float> r_a_;
    std::vector<float> v_ca_thresh_;
    std::vector<float> ca_boost_val_;
    std::vector<int>   ca_dur_;
    std::vector<int>   burst_spike_count_;
    std::vector<int>   burst_isi_val_;

    // --- 动态状态 (SoA) ---
    std::vector<float>   v_soma_;
    std::vector<float>   v_apical_;
    std::vector<float>   w_adapt_;
    std::vector<int>     refrac_count_;
    std::vector<uint8_t> ca_spike_;
    std::vector<int>     ca_timer_;
    std::vector<int>     burst_remain_;
    std::vector<int>     burst_isi_ct_;

    // --- 输入 (每步清零) ---
    std::vector<float> i_basal_;
    std::vector<float> i_apical_;
    std::vector<float> i_soma_;

    // --- 输出 ---
    std::vector<uint8_t> fired_;
    std::vector<int8_t>  spike_type_;
};

} // namespace wuyun
