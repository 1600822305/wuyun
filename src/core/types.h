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
    float mg_conc   = 0.0f;     // Mg²⁺ 浓度 (mM), >0 启用电压门控 (NMDA)
};

// 预定义参数集
constexpr SynapseParams AMPA_PARAMS   {2.0f,   0.5f,   0.0f,  1.0f, 0.0f};
constexpr SynapseParams NMDA_PARAMS   {100.0f, 5.0f,   0.0f,  0.5f, 1.0f};  // Mg²⁺=1.0mM
constexpr SynapseParams GABA_A_PARAMS {6.0f,   0.5f, -70.0f,  1.0f, 0.0f};
constexpr SynapseParams GABA_B_PARAMS {200.0f, 5.0f, -95.0f,  0.3f, 0.0f};

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

// 丘脑中继: Tonic 模式 (κ=0.3, 中等耦合, 忠实中继前馈信号)
inline NeuronParams THALAMIC_RELAY_TONIC_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 20.0f;
    p.somatic.r_s = 1.0f; p.somatic.a = 0.02f;
    p.somatic.b = 3.0f; p.somatic.tau_w = 100.0f;
    p.somatic.refractory_period = 2;
    p.kappa = 0.3f; p.kappa_backward = 0.1f;  // 中等耦合: tonic模式
    p.burst_spike_count = 1; p.burst_isi = 1;  // tonic: 单脉冲
    return p;
}

// 丘脑中继: Burst 模式 (κ=0.5, 强耦合, 低T-type Ca²⁺通道激活 → burst)
// T-type Ca²⁺ 阈值 ~-50mV, 比皮层 HVA Ca²⁺ (-35~-40mV) 更低
inline NeuronParams THALAMIC_RELAY_BURST_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -70.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -65.0f; p.somatic.tau_m = 20.0f;
    p.somatic.r_s = 1.0f; p.somatic.a = 0.05f;
    p.somatic.b = 8.0f; p.somatic.tau_w = 80.0f;
    p.somatic.refractory_period = 3;
    p.apical.tau_a = 15.0f; p.apical.r_a = 1.5f;
    p.apical.v_ca_threshold = -50.0f; p.apical.ca_boost = 30.0f;
    p.apical.ca_duration = 8;
    p.kappa = 0.5f; p.kappa_backward = 0.2f;
    p.burst_spike_count = 4; p.burst_isi = 2;
    return p;
}

// 丘脑网状核 TRN: 纯抑制, 门控丘脑中继 (单区室)
inline NeuronParams TRN_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -45.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 15.0f;
    p.somatic.r_s = 0.9f; p.somatic.a = 0.1f;
    p.somatic.b = 0.5f; p.somatic.tau_w = 50.0f;
    p.somatic.refractory_period = 2;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 3; p.burst_isi = 2;
    return p;
}

// 纹状体中棘神经元 D1 (直接通路, DA增强LTP)
inline NeuronParams MSN_D1_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -80.0f; p.somatic.v_threshold = -50.0f;  // 高阈值, 需强输入
    p.somatic.v_reset = -65.0f; p.somatic.tau_m = 25.0f;
    p.somatic.r_s = 0.8f; p.somatic.a = 0.01f;
    p.somatic.b = 3.0f; p.somatic.tau_w = 300.0f;
    p.somatic.refractory_period = 3;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;  // 单区室
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

// 纹状体中棘神经元 D2 (间接通路, DA增强LTD)
inline NeuronParams MSN_D2_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -80.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -65.0f; p.somatic.tau_m = 25.0f;
    p.somatic.r_s = 0.8f; p.somatic.a = 0.01f;
    p.somatic.b = 3.0f; p.somatic.tau_w = 300.0f;
    p.somatic.refractory_period = 3;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

// 海马颗粒细胞 DG (高阈值, 极稀疏编码)
inline NeuronParams GRANULE_CELL_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -75.0f; p.somatic.v_threshold = -45.0f;  // 非常高阈值
    p.somatic.v_reset = -65.0f; p.somatic.tau_m = 20.0f;
    p.somatic.r_s = 0.7f; p.somatic.a = 0.02f;
    p.somatic.b = 2.0f; p.somatic.tau_w = 200.0f;
    p.somatic.refractory_period = 3;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

// 小脑浦肯野细胞 (高频自发放, 特殊树突)
inline NeuronParams PURKINJE_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -60.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -55.0f; p.somatic.tau_m = 10.0f;
    p.somatic.r_s = 1.5f; p.somatic.a = 0.0f;  // 无亚阈值适应
    p.somatic.b = 0.5f; p.somatic.tau_w = 50.0f;
    p.somatic.refractory_period = 1;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;  // 特殊树突不用双区室模型
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

// 多巴胺能神经元 VTA/SNc (低频自发放, tonic ~4Hz)
inline NeuronParams DOPAMINE_NEURON_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -60.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -55.0f; p.somatic.tau_m = 30.0f;
    p.somatic.r_s = 0.8f; p.somatic.a = 0.02f;
    p.somatic.b = 5.0f; p.somatic.tau_w = 500.0f;  // 非常慢的适应
    p.somatic.refractory_period = 4;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 3; p.burst_isi = 3;  // phasic burst
    return p;
}

// =============================================================================
// 海马特化神经元
// =============================================================================

// 海马位置细胞 CA1/CA3 (place cell, 双区室, theta调制)
inline NeuronParams PLACE_CELL_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 20.0f;
    p.somatic.r_s = 1.0f; p.somatic.a = 0.01f;
    p.somatic.b = 5.0f; p.somatic.tau_w = 200.0f;
    p.somatic.refractory_period = 3;
    p.kappa = 0.3f; p.kappa_backward = 0.1f;  // 双区室: theta相位进动
    p.burst_spike_count = 3; p.burst_isi = 2;
    return p;
}

// 内嗅皮层网格细胞 (grid cell, 双区室, 弱耦合)
inline NeuronParams GRID_CELL_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 20.0f;
    p.somatic.r_s = 1.0f; p.somatic.a = 0.01f;
    p.somatic.b = 4.0f; p.somatic.tau_w = 200.0f;
    p.somatic.refractory_period = 3;
    p.kappa = 0.2f; p.kappa_backward = 0.1f;
    p.burst_spike_count = 2; p.burst_isi = 3;
    return p;
}

// 头朝向细胞 (head direction cell, 弱耦合, 持续发放)
inline NeuronParams HD_CELL_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -60.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -55.0f; p.somatic.tau_m = 15.0f;
    p.somatic.r_s = 1.0f; p.somatic.a = 0.0f;
    p.somatic.b = 1.0f; p.somatic.tau_w = 100.0f;
    p.somatic.refractory_period = 2;
    p.kappa = 0.1f; p.kappa_backward = 0.05f;
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

// 海马苔藓细胞 DG hilus (mossy cell, 单区室, 高兴奋性)
inline NeuronParams MOSSY_CELL_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -60.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -55.0f; p.somatic.tau_m = 15.0f;
    p.somatic.r_s = 1.2f; p.somatic.a = 0.01f;
    p.somatic.b = 3.0f; p.somatic.tau_w = 150.0f;
    p.somatic.refractory_period = 2;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 2; p.burst_isi = 2;
    return p;
}

// =============================================================================
// 抑制性特化神经元
// =============================================================================

// 枝形烛台细胞 Chandelier (PV+, 靶向轴突起始段 AIS, 最强单突触抑制)
inline NeuronParams CHANDELIER_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -45.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 10.0f;
    p.somatic.r_s = 0.8f; p.somatic.a = 0.1f;
    p.somatic.b = 0.0f; p.somatic.tau_w = 50.0f;
    p.somatic.refractory_period = 1;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

// 神经胶质形态细胞 NGF (neurogliaform, 慢GABA_B体积释放)
inline NeuronParams NGF_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -65.0f; p.somatic.v_threshold = -50.0f;
    p.somatic.v_reset = -60.0f; p.somatic.tau_m = 25.0f;  // 慢
    p.somatic.r_s = 0.7f; p.somatic.a = 0.02f;
    p.somatic.b = 1.0f; p.somatic.tau_w = 300.0f;  // 非常慢适应
    p.somatic.refractory_period = 3;
    p.kappa = 0.0f; p.kappa_backward = 0.0f;
    p.burst_spike_count = 1; p.burst_isi = 1;
    return p;
}

} // namespace wuyun
