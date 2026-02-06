#pragma once
/**
 * Layer 0: 基础类型定义
 *
 * 悟韵系统最底层的"原子"定义，不依赖任何其他模块。
 * 对应 Python 原型: wuyun/spike/signal_types.py
 */

#include <cstdint>

namespace wuyun {

// =============================================================================
// 脉冲类型枚举
// =============================================================================

/**
 * 双区室神经元的发放模式直接编码预测编码信息:
 *
 * | 基底树突(前馈) | 顶端树突(反馈) | 发放模式    | 含义       |
 * |---------------|---------------|------------|-----------|
 * | ✔ 激活        | ✖ 未激活      | REGULAR    | 预测误差   |
 * | ✔ 激活        | ✔ 激活        | BURST      | 预测匹配   |
 * | ✖ 未激活      | ✔ 激活        | NONE       | 无事发生   |
 * | ✖ 未激活      | ✖ 未激活      | NONE       | 沉默       |
 */
enum class SpikeType : int8_t {
    NONE           = 0,
    REGULAR        = 1,
    BURST_START    = 2,
    BURST_CONTINUE = 3,
    BURST_END      = 4,
};

inline bool is_burst(SpikeType t) {
    return t == SpikeType::BURST_START ||
           t == SpikeType::BURST_CONTINUE ||
           t == SpikeType::BURST_END;
}

inline bool is_active(SpikeType t) {
    return t != SpikeType::NONE;
}

// =============================================================================
// 区室类型枚举
// =============================================================================

/** 突触目标区室 — 决定电流注入位置 (预测编码的硬件基础) */
enum class CompartmentType : int8_t {
    SOMA   = 0,   // 胞体 (直接驱动)
    BASAL  = 1,   // 基底树突 (前馈输入)
    APICAL = 2,   // 顶端树突 (反馈输入)
};

// =============================================================================
// 突触类型枚举
// =============================================================================

enum class SynapseType : int8_t {
    AMPA   = 0,
    NMDA   = 1,
    GABA_A = 2,
    GABA_B = 3,
};

// =============================================================================
// 神经元类型枚举
// =============================================================================

enum class NeuronType : int8_t {
    // 兴奋性 (双区室, κ > 0)
    L23_PYRAMIDAL  = 0,
    L5_PYRAMIDAL   = 1,
    L6_PYRAMIDAL   = 2,
    L4_STELLATE    = 3,
    GRANULE        = 4,   // 海马DG/小脑

    // 抑制性 (单区室, κ = 0)
    PV_BASKET      = 10,
    SST_MARTINOTTI = 11,
    VIP_INTERNEURON= 12,
    CHANDELIER     = 13,
    NGF            = 14,  // 慢抑制 GABA_B

    // 特化型
    THALAMIC_RELAY = 20,
    TRN            = 21,
    MEDIUM_SPINY_D1= 22,
    MEDIUM_SPINY_D2= 23,
    PURKINJE       = 24,
    DOPAMINERGIC   = 25,
    SEROTONERGIC   = 26,
};

// =============================================================================
// 突触参数结构体
// =============================================================================

/** 突触类型参数 — 生物物理常数 (允许硬编码) */
struct SynapseParams {
    float tau_decay = 2.0f;     // 衰减时间常数 (ms)
    float tau_rise  = 0.5f;     // 上升时间常数 (ms)
    float e_rev     = 0.0f;     // 反转电位 (mV)
    float g_max     = 1.0f;     // 最大电导 (nS)
};

// 预定义参数集
constexpr SynapseParams AMPA_PARAMS   {2.0f,   0.5f,   0.0f,  1.0f};
constexpr SynapseParams NMDA_PARAMS   {100.0f, 5.0f,   0.0f,  0.5f};
constexpr SynapseParams GABA_A_PARAMS {6.0f,   0.5f, -70.0f,  1.0f};
constexpr SynapseParams GABA_B_PARAMS {200.0f, 5.0f, -95.0f,  0.3f};

// =============================================================================
// 神经元参数结构体
// =============================================================================

/** 胞体区室参数 */
struct SomaticParams {
    float v_rest     = -65.0f;
    float v_threshold= -50.0f;
    float v_reset    = -60.0f;
    float tau_m      = 20.0f;   // 膜时间常数 (ms)
    float r_s        = 1.0f;    // 膜输入阻抗
    float a          = 0.01f;   // 亚阈值适应耦合
    float b          = 5.0f;    // 脉冲后适应增量
    float tau_w      = 200.0f;  // 适应时间常数 (ms)
    int   refractory_period = 3; // 不应期 (ms/steps)
};

/** 顶端树突区室参数 */
struct ApicalParams {
    float tau_a         = 20.0f;
    float r_a           = 1.0f;
    float v_ca_threshold= -40.0f;
    float ca_boost      = 20.0f;
    int   ca_duration   = 5;     // Ca²⁺ 脉冲持续步数
};

/** 完整神经元参数 */
struct NeuronParams {
    SomaticParams somatic;
    ApicalParams  apical;
    float kappa          = 0.3f;  // apical→soma 正向耦合
    float kappa_backward = 0.1f;  // soma→apical 反向耦合
    int   burst_spike_count = 3;  // burst 中脉冲数
    int   burst_isi      = 3;     // burst 内脉冲间隔 (steps)
};

// 预定义神经元参数
inline NeuronParams L23_PYRAMIDAL_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 20.0f;
    p.somatic.r_s = 1.0f; p.somatic.a = 0.01f;
    p.somatic.b = 5.0f; p.somatic.tau_w = 200.0f;
    p.somatic.refractory_period = 3;
    p.apical.tau_a = 20.0f; p.apical.r_a = 1.0f;
    p.apical.v_ca_threshold = -40.0f; p.apical.ca_boost = 20.0f;
    p.apical.ca_duration = 5;
    p.kappa = 0.3f; p.kappa_backward = 0.1f;
    p.burst_spike_count = 3; p.burst_isi = 3;
    return p;
}

inline NeuronParams L5_PYRAMIDAL_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -55.0f; p.somatic.tau_m = 15.0f;
    p.somatic.r_s = 1.2f; p.somatic.a = 0.02f;
    p.somatic.b = 8.0f; p.somatic.tau_w = 150.0f;
    p.somatic.refractory_period = 2;
    p.apical.tau_a = 15.0f; p.apical.r_a = 1.2f;
    p.apical.v_ca_threshold = -35.0f; p.apical.ca_boost = 25.0f;
    p.apical.ca_duration = 7;
    p.kappa = 0.6f; p.kappa_backward = 0.15f;
    p.burst_spike_count = 4; p.burst_isi = 2;
    return p;
}

inline NeuronParams PV_BASKET_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -45.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 10.0f;
    p.somatic.r_s = 0.8f; p.somatic.a = 0.1f;
    p.somatic.b = 0.0f; p.somatic.tau_w = 50.0f;
    p.somatic.refractory_period = 1;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;  // 单区室
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

inline NeuronParams THALAMIC_RELAY_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 20.0f;
    p.somatic.r_s = 1.0f; p.somatic.a = 0.02f;
    p.somatic.b = 6.0f; p.somatic.tau_w = 100.0f;
    p.somatic.refractory_period = 2;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 3; p.burst_isi = 2;
    return p;
}

} // namespace wuyun
