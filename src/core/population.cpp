#include "core/population.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

NeuronPopulation::NeuronPopulation(size_t n, const NeuronParams& params)
    : n_(n)
    , has_apical_(params.kappa > 0.0f)
    // Parameters (broadcast scalar to vector)
    , v_rest_(n, params.somatic.v_rest)
    , v_threshold_(n, params.somatic.v_threshold)
    , v_reset_(n, params.somatic.v_reset)
    , tau_m_(n, params.somatic.tau_m)
    , r_s_(n, params.somatic.r_s)
    , a_adapt_(n, params.somatic.a)
    , b_adapt_(n, params.somatic.b)
    , tau_w_(n, params.somatic.tau_w)
    , refrac_period_(n, params.somatic.refractory_period)
    , kappa_(n, params.kappa)
    , kappa_back_(n, params.kappa_backward)
    , tau_a_(n, params.apical.tau_a)
    , r_a_(n, params.apical.r_a)
    , v_ca_thresh_(n, params.apical.v_ca_threshold)
    , ca_boost_val_(n, params.apical.ca_boost)
    , ca_dur_(n, params.apical.ca_duration)
    , burst_spike_count_(n, params.burst_spike_count)
    , burst_isi_val_(n, params.burst_isi)
    // Dynamic state
    , v_soma_(n, params.somatic.v_rest)
    , v_apical_(n, params.somatic.v_rest)
    , w_adapt_(n, 0.0f)
    , refrac_count_(n, 0)
    , ca_spike_(n, 0)
    , ca_timer_(n, 0)
    , burst_remain_(n, 0)
    , burst_isi_ct_(n, 0)
    // Inputs
    , i_basal_(n, 0.0f)
    , i_apical_(n, 0.0f)
    , i_soma_(n, 0.0f)
    // Outputs
    , fired_(n, 0)
    , spike_type_(n, static_cast<int8_t>(SpikeType::NONE))
{
}

void NeuronPopulation::inject_basal(size_t idx, float current) {
    if (idx < n_) i_basal_[idx] += current;
}

void NeuronPopulation::inject_apical(size_t idx, float current) {
    if (idx < n_) i_apical_[idx] += current;
}

void NeuronPopulation::inject_soma(size_t idx, float current) {
    if (idx < n_) i_soma_[idx] += current;
}

// =============================================================================
// Step 1: 顶端树突更新 + Ca²⁺ 脉冲检测
// =============================================================================

void NeuronPopulation::update_apical(float dt) {
    for (size_t i = 0; i < n_; ++i) {
        // τ_a · dV_a/dt = -(V_a - V_rest) + R_a · I_apical + κ_back · (V_s - V_a)
        float leak    = -(v_apical_[i] - v_rest_[i]);
        float inp     = r_a_[i] * i_apical_[i];
        float coupling= kappa_back_[i] * (v_soma_[i] - v_apical_[i]);
        float dv      = (leak + inp + coupling) / tau_a_[i] * dt;
        v_apical_[i] += dv;

        // Ca²⁺ 脉冲状态机
        if (ca_timer_[i] > 0) {
            ca_timer_[i] -= 1;
            if (ca_timer_[i] == 0) {
                ca_spike_[i] = 0;
            }
        } else if (v_apical_[i] >= v_ca_thresh_[i]) {
            ca_spike_[i]  = 1;
            ca_timer_[i]  = ca_dur_[i];
            v_apical_[i] += ca_boost_val_[i];
        }
    }
}

// =============================================================================
// Step 2: Burst 状态机 — 正在 burst 中的神经元
// =============================================================================

void NeuronPopulation::continue_burst(size_t i, float dt) {
    // ISI 倒计时
    burst_isi_ct_[i] -= 1;

    // 胞体更新 (含不应期)
    if (refrac_count_[i] > 0) {
        refrac_count_[i] -= 1;
    } else {
        float total_input = i_basal_[i] + i_soma_[i];
        float v   = v_soma_[i];
        float v_a = has_apical_ ? v_apical_[i] : v_rest_[i];

        float leak    = -(v - v_rest_[i]);
        float inp     = r_s_[i] * total_input;
        float coupling= kappa_[i] * (v_a - v);
        float dv      = (leak + inp - w_adapt_[i] + coupling) / tau_m_[i] * dt;
        v_soma_[i] += dv;

        float dw = (a_adapt_[i] * (v_soma_[i] - v_rest_[i]) - w_adapt_[i]) / tau_w_[i] * dt;
        w_adapt_[i] += dw;
    }

    // ISI 到期 → 发放 burst 脉冲
    if (burst_isi_ct_[i] <= 0) {
        burst_remain_[i] -= 1;
        burst_isi_ct_[i] = burst_isi_val_[i];

        // 强制重置胞体
        v_soma_[i]  = v_reset_[i];
        w_adapt_[i]+= b_adapt_[i] * 0.5f;  // burst 内适应较弱

        if (burst_remain_[i] <= 0) {
            spike_type_[i] = static_cast<int8_t>(SpikeType::BURST_END);
        } else {
            spike_type_[i] = static_cast<int8_t>(SpikeType::BURST_CONTINUE);
        }
        fired_[i] = 1;
    }
}

// =============================================================================
// Step 3: 胞体更新 + 发放检测 — 非 burst 中的神经元
// =============================================================================

void NeuronPopulation::update_soma_and_fire(size_t i, int /*t*/, float dt) {
    // 不应期
    if (refrac_count_[i] > 0) {
        refrac_count_[i] -= 1;
        return;
    }

    // τ_m · dV_s/dt = -(V_s - V_rest) + R_s · I_total - w + κ · (V_a - V_s)
    float total_input = i_basal_[i] + i_soma_[i];
    float v   = v_soma_[i];
    float v_a = has_apical_ ? v_apical_[i] : v_rest_[i];

    float leak    = -(v - v_rest_[i]);
    float inp     = r_s_[i] * total_input;
    float coupling= kappa_[i] * (v_a - v);
    float dv      = (leak + inp - w_adapt_[i] + coupling) / tau_m_[i] * dt;
    v_soma_[i] += dv;

    // τ_w · dw/dt = a · (V_s - V_rest) - w
    float dw = (a_adapt_[i] * (v_soma_[i] - v_rest_[i]) - w_adapt_[i]) / tau_w_[i] * dt;
    w_adapt_[i] += dw;

    // 发放检测
    if (v_soma_[i] >= v_threshold_[i]) {
        v_soma_[i]      = v_reset_[i];
        w_adapt_[i]    += b_adapt_[i];
        refrac_count_[i]= refrac_period_[i];

        // Burst vs Regular 判定
        if (has_apical_ && ca_spike_[i]) {
            // BURST_START: 前馈 + 反馈同时激活
            spike_type_[i] = static_cast<int8_t>(SpikeType::BURST_START);
            burst_remain_[i] = burst_spike_count_[i] - 1;  // 第一个已经发了
            burst_isi_ct_[i] = burst_isi_val_[i];
        } else {
            // REGULAR: 只有前馈
            spike_type_[i] = static_cast<int8_t>(SpikeType::REGULAR);
        }
        fired_[i] = 1;
    }
}

// =============================================================================
// 主循环
// =============================================================================

size_t NeuronPopulation::step(int t, float dt) {
    // 清零输出
    std::fill(fired_.begin(), fired_.end(), static_cast<uint8_t>(0));
    std::fill(spike_type_.begin(), spike_type_.end(),
              static_cast<int8_t>(SpikeType::NONE));

    // Step 1: 顶端树突更新
    if (has_apical_) {
        update_apical(dt);
    }

    // Step 2-3: 对每个神经元，根据状态选择路径
    for (size_t i = 0; i < n_; ++i) {
        if (burst_remain_[i] > 0) {
            continue_burst(i, dt);
        } else {
            update_soma_and_fire(i, t, dt);
        }
    }

    // 清空输入
    clear_inputs();

    // 统计发放数
    size_t fire_count = 0;
    for (size_t i = 0; i < n_; ++i) {
        fire_count += fired_[i];
    }
    return fire_count;
}

void NeuronPopulation::clear_inputs() {
    std::fill(i_basal_.begin(), i_basal_.end(), 0.0f);
    std::fill(i_apical_.begin(), i_apical_.end(), 0.0f);
    std::fill(i_soma_.begin(), i_soma_.end(), 0.0f);
}

} // namespace wuyun
