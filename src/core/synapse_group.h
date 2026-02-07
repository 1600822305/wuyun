#pragma once
/**
 * SynapseGroup — CSR 稀疏突触组
 *
 * 存储一组突触前→突触后连接，使用 Compressed Sparse Row (CSR) 格式。
 * 对应 Python 原型: wuyun/core/synapse_group.py
 *
 * 突触电流模型:
 *   I_syn = g_max * w * s * (V_post - E_rev)
 *   ds/dt = -s / tau_decay  (on spike: s += 1)
 *
 * 设计文档: docs/02_neuron_system_design.md §2
 */

#include "types.h"
#include "../plasticity/stp.h"
#include "../plasticity/stdp.h"
#include <vector>
#include <cstdint>

namespace wuyun {

class SynapseGroup {
public:
    /**
     * 构造 CSR 稀疏突触组
     *
     * @param n_pre       突触前神经元数量
     * @param n_post      突触后神经元数量
     * @param pre_ids     突触前神经元 ID 数组 (长度 = n_synapses)
     * @param post_ids    突触后神经元 ID 数组 (长度 = n_synapses)
     * @param weights     初始权重数组
     * @param delays      传导延迟数组 (steps)
     * @param syn_params  突触类型参数
     * @param target      目标区室
     */
    SynapseGroup(
        size_t n_pre,
        size_t n_post,
        const std::vector<int32_t>& pre_ids,
        const std::vector<int32_t>& post_ids,
        const std::vector<float>& weights,
        const std::vector<int32_t>& delays,
        const SynapseParams& syn_params,
        CompartmentType target = CompartmentType::BASAL
    );

    /** 接收突触前脉冲 (无延迟版本, 立即投递) */
    void deliver_spikes(const std::vector<uint8_t>& pre_fired,
                        const std::vector<int8_t>& pre_spike_type);

    /**
     * 更新门控变量并计算突触电流
     *
     * @param v_post  突触后膜电位数组
     * @param dt      时间步长
     * @return 聚合到每个 post 神经元的突触电流
     */
    std::vector<float> step_and_compute(const std::vector<float>& v_post, float dt = 1.0f);

    // --- 访问器 ---
    size_t n_synapses() const { return col_idx_.size(); }
    size_t n_pre()      const { return n_pre_; }
    size_t n_post()     const { return n_post_; }
    CompartmentType target() const { return target_; }

    const std::vector<float>& weights() const { return weights_; }
    std::vector<float>& weights() { return weights_; }

    /** 启用 STP (Tsodyks-Markram 短时程可塑性), 每个突触前神经元一个 STPState */
    void enable_stp(const STPParams& params);
    bool has_stp() const { return stp_enabled_; }

    /** 启用 STDP (长时程可塑性) */
    void enable_stdp(const STDPParams& params);
    bool has_stdp() const { return stdp_enabled_; }

    /**
     * 应用 STDP 权重更新 (在 step 后调用)
     * @param pre_fired   突触前神经元发放状态
     * @param post_fired  突触后神经元发放状态
     * @param t           当前时间步
     */
    void apply_stdp(const std::vector<uint8_t>& pre_fired,
                    const std::vector<uint8_t>& post_fired,
                    int32_t t);

private:
    size_t n_pre_;
    size_t n_post_;
    CompartmentType target_;

    // CSR 格式
    std::vector<int32_t> row_ptr_;    // 长度 = n_pre + 1
    std::vector<int32_t> col_idx_;    // 长度 = n_synapses (post neuron IDs)
    std::vector<float>   weights_;    // 长度 = n_synapses
    std::vector<int32_t> delays_;     // 长度 = n_synapses

    // 突触参数
    float tau_decay_;
    float e_rev_;
    float g_max_;
    float mg_conc_;   // Mg²⁺ 浓度, >0 时启用 NMDA 电压门控 B(V)

    // 门控变量
    std::vector<float> g_;            // 长度 = n_synapses

    // STP (optional, per pre-neuron)
    bool stp_enabled_ = false;
    STPParams stp_params_;
    std::vector<STPState> stp_states_; // 长度 = n_pre (enabled 时)

    // STDP (optional, online weight plasticity)
    bool stdp_enabled_ = false;
    STDPParams stdp_params_;
    std::vector<float> last_spike_pre_;   // 长度 = n_pre
    std::vector<float> last_spike_post_;  // 长度 = n_post

    // 聚合输出缓冲
    std::vector<float> i_post_;       // 长度 = n_post
};

} // namespace wuyun
