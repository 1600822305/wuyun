#pragma once
/**
 * OscillationTracker — 振荡相位追踪器
 *
 * 追踪脑区级振荡节律的相位, 用于:
 *   - Theta-gamma 耦合 (海马: gamma burst 嵌套在 theta 谷)
 *   - 相位编码 (发放时的振荡相位携带信息)
 *   - 跨区域同步 (相位锁定 = 功能连接)
 *
 * 振荡频段 (02 文档 §4.1):
 *   Delta:  0.5-4 Hz   (深睡)
 *   Theta:  4-8 Hz     (海马导航/记忆编码)
 *   Alpha:  8-13 Hz    (清醒静息/注意力抑制)
 *   Beta:   13-30 Hz   (运动准备/状态维持)
 *   Gamma:  30-100 Hz  (局部处理/特征绑定)
 *
 * 设计文档: docs/02_neuron_system_design.md §4
 */

#include <cstdint>
#include <cmath>

namespace wuyun {

enum class OscBand : uint8_t {
    DELTA = 0,   // 0.5-4 Hz
    THETA = 1,   // 4-8 Hz
    ALPHA = 2,   // 8-13 Hz
    BETA  = 3,   // 13-30 Hz
    GAMMA = 4,   // 30-100 Hz
    NUM_BANDS = 5
};

/** 单频段振荡器 */
struct Oscillator {
    float frequency = 6.0f;   // Hz
    float phase     = 0.0f;   // 当前相位 [0, 2π)
    float amplitude = 1.0f;   // 振幅 (0~1)

    /** 推进一步 */
    void step(float dt_ms) {
        float dt_s = dt_ms * 0.001f;
        phase += 2.0f * 3.14159265f * frequency * dt_s;
        if (phase >= 2.0f * 3.14159265f) {
            phase -= 2.0f * 3.14159265f;
        }
    }

    /** 当前值 [-amplitude, +amplitude] */
    float value() const {
        return amplitude * std::sin(phase);
    }

    /** 是否在谷附近 (phase ∈ [π-w, π+w]), 用于 theta-gamma 耦合 */
    bool at_trough(float width = 0.5f) const {
        constexpr float PI = 3.14159265f;
        float d = std::abs(phase - PI);
        return d < width || d > (2.0f * PI - width);
    }

    /** 是否在峰附近 (phase ∈ [-w, +w]) */
    bool at_peak(float width = 0.5f) const {
        constexpr float PI = 3.14159265f;
        return phase < width || phase > (2.0f * PI - width);
    }
};

/**
 * 多频段振荡追踪器
 *
 * 每个脑区一个, 追踪多个频段的相位。
 */
class OscillationTracker {
public:
    OscillationTracker();

    /** 设置某频段的频率和振幅 */
    void set_band(OscBand band, float freq_hz, float amplitude = 1.0f);

    /** 推进所有频段一步 */
    void step(float dt_ms = 1.0f);

    /** 获取某频段的当前相位 [0, 2π) */
    float phase(OscBand band) const;

    /** 获取某频段的当前值 [-amp, +amp] */
    float value(OscBand band) const;

    /** 获取某频段的振荡器 */
    const Oscillator& oscillator(OscBand band) const;
    Oscillator& oscillator(OscBand band);

    /** Theta-gamma 耦合: gamma振幅是否应增强 (theta谷时) */
    bool theta_gamma_coupling() const;

private:
    Oscillator bands_[static_cast<size_t>(OscBand::NUM_BANDS)];
};

} // namespace wuyun
