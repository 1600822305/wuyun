#pragma once
/**
 * 单神经元辅助函数 — 供调试和单元测试使用
 *
 * 生产仿真使用 NeuronPopulation (向量化)。
 * 此文件提供单个神经元的 step 函数，方便验证数学方程。
 */

#include "types.h"

namespace wuyun {

/** 单神经元状态 (用于测试/调试) */
struct NeuronState {
    float v_soma    = -65.0f;
    float v_apical  = -65.0f;
    float w_adapt   = 0.0f;
    int   refrac_count = 0;
    bool  ca_spike  = false;
    int   ca_timer  = 0;
    int   burst_remain = 0;
    int   burst_isi_ct = 0;
    SpikeType last_spike = SpikeType::NONE;
};

/**
 * 单神经元 step — 用于单元测试验证
 *
 * @param state   神经元状态 (in/out)
 * @param params  神经元参数
 * @param i_basal  基底树突输入电流
 * @param i_apical 顶端树突输入电流
 * @param i_soma   胞体直接输入电流
 * @param t        当前时间步
 * @param dt       时间步长 (ms)
 * @return 本步脉冲类型
 */
SpikeType neuron_step(
    NeuronState& state,
    const NeuronParams& params,
    float i_basal,
    float i_apical,
    float i_soma,
    int t,
    float dt = 1.0f
);

} // namespace wuyun
