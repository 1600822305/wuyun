#include "core/neuron.h"

namespace wuyun {

SpikeType neuron_step(
    NeuronState& s,
    const NeuronParams& p,
    float i_basal,
    float i_apical,
    float i_soma,
    int /*t*/,
    float dt
) {
    bool has_apical = (p.kappa > 0.0f);

    // === Step 1: 顶端树突更新 ===
    if (has_apical) {
        float leak     = -(s.v_apical - p.somatic.v_rest);
        float inp      = p.apical.r_a * i_apical;
        float coupling = p.kappa_backward * (s.v_soma - s.v_apical);
        float dv       = (leak + inp + coupling) / p.apical.tau_a * dt;
        s.v_apical += dv;

        // Ca²⁺ 脉冲状态机
        if (s.ca_timer > 0) {
            s.ca_timer -= 1;
            if (s.ca_timer == 0) s.ca_spike = false;
        } else if (s.v_apical >= p.apical.v_ca_threshold) {
            s.ca_spike   = true;
            s.ca_timer   = p.apical.ca_duration;
            s.v_apical  += p.apical.ca_boost;
        }
    }

    // === Step 2: Burst 状态机 ===
    if (s.burst_remain > 0) {
        s.burst_isi_ct -= 1;

        // 胞体更新
        if (s.refrac_count > 0) {
            s.refrac_count -= 1;
        } else {
            float total = i_basal + i_soma;
            float v_a   = has_apical ? s.v_apical : p.somatic.v_rest;
            float leak  = -(s.v_soma - p.somatic.v_rest);
            float inp   = p.somatic.r_s * total;
            float coup  = p.kappa * (v_a - s.v_soma);
            s.v_soma   += (leak + inp - s.w_adapt + coup) / p.somatic.tau_m * dt;

            float dw = (p.somatic.a * (s.v_soma - p.somatic.v_rest) - s.w_adapt) / p.somatic.tau_w * dt;
            s.w_adapt += dw;
        }

        if (s.burst_isi_ct <= 0) {
            s.burst_remain -= 1;
            s.burst_isi_ct  = p.burst_isi;
            s.v_soma        = p.somatic.v_reset;
            s.w_adapt      += p.somatic.b * 0.5f;

            SpikeType result = (s.burst_remain <= 0)
                ? SpikeType::BURST_END
                : SpikeType::BURST_CONTINUE;
            s.last_spike = result;
            return result;
        }

        s.last_spike = SpikeType::NONE;
        return SpikeType::NONE;
    }

    // === Step 3: 胞体更新 + 发放检测 ===
    if (s.refrac_count > 0) {
        s.refrac_count -= 1;
        s.last_spike = SpikeType::NONE;
        return SpikeType::NONE;
    }

    float total = i_basal + i_soma;
    float v_a   = has_apical ? s.v_apical : p.somatic.v_rest;
    float leak  = -(s.v_soma - p.somatic.v_rest);
    float inp   = p.somatic.r_s * total;
    float coup  = p.kappa * (v_a - s.v_soma);
    s.v_soma   += (leak + inp - s.w_adapt + coup) / p.somatic.tau_m * dt;

    float dw = (p.somatic.a * (s.v_soma - p.somatic.v_rest) - s.w_adapt) / p.somatic.tau_w * dt;
    s.w_adapt += dw;

    if (s.v_soma >= p.somatic.v_threshold) {
        s.v_soma      = p.somatic.v_reset;
        s.w_adapt    += p.somatic.b;
        s.refrac_count= p.somatic.refractory_period;

        if (has_apical && s.ca_spike) {
            s.burst_remain = p.burst_spike_count - 1;
            s.burst_isi_ct = p.burst_isi;
            s.last_spike   = SpikeType::BURST_START;
            return SpikeType::BURST_START;
        } else {
            s.last_spike = SpikeType::REGULAR;
            return SpikeType::REGULAR;
        }
    }

    s.last_spike = SpikeType::NONE;
    return SpikeType::NONE;
}

} // namespace wuyun
