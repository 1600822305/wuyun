#pragma once
/**
 * SleepCycleManager — 睡眠周期管理器
 *
 * 管理 AWAKE → NREM → REM → NREM → REM ... 的完整睡眠周期。
 *
 * 生物学基础:
 *   - 人类睡眠: ~90min 周期 (NREM + REM)
 *   - 前半夜: NREM 主导 (慢波 + SWR 记忆巩固)
 *   - 后半夜: REM 增长 (theta + 创造性重组 + 梦境)
 *   - VLPO (NREM-on) ↔ PnO (REM-on) flip-flop
 *
 * 使用方式:
 *   1. 创建 SleepCycleManager
 *   2. enter_sleep() 开始睡眠
 *   3. 每步调用 step() 自动推进 NREM↔REM
 *   4. 查询 stage() 获取当前阶段
 *   5. wake_up() 唤醒
 *
 * 参考文献:
 *   - Saper et al (2005) Hypothalamic regulation of sleep and circadian rhythms
 *   - Hobson & Pace-Schott (2002) The cognitive neuroscience of sleep
 */

#include <cstdint>
#include <cstddef>

namespace wuyun {

enum class SleepStage : uint8_t {
    AWAKE = 0,
    NREM  = 1,    // NREM 慢波 (SWR replay, cortical slow oscillation)
    REM   = 2     // REM (theta, PGO waves, dreaming, motor atonia)
};

struct SleepCycleConfig {
    // Cycle timing (in simulation steps)
    size_t nrem_duration     = 600;   // NREM 阶段持续步数 (前半夜较长)
    size_t rem_duration      = 200;   // REM 阶段持续步数 (后半夜增长)
    size_t nrem_growth       = 0;     // 每周期 NREM 缩短量
    size_t rem_growth        = 50;    // 每周期 REM 增长量 (模拟自然规律)
    size_t max_rem_duration  = 500;   // REM 最大持续
    size_t min_nrem_duration = 200;   // NREM 最小持续

    // REM parameters
    float rem_theta_freq     = 0.006f;  // Theta ~6Hz (at 1000 steps/sec)
    float rem_pgo_prob       = 0.02f;   // PGO burst probability per step
    float rem_pgo_amplitude  = 25.0f;   // PGO burst current amplitude
    float rem_motor_inhibit  = -15.0f;  // Motor cortex inhibition during REM (atonia)
    float rem_cortex_noise   = 8.0f;    // Desynchronized cortical noise during REM
    float rem_theta_amp      = 10.0f;   // Hippocampal theta modulation amplitude
};

class SleepCycleManager {
public:
    explicit SleepCycleManager(const SleepCycleConfig& config = {});

    /** Advance one step. Call every simulation step during sleep. */
    void step();

    /** Enter sleep (starts with NREM) */
    void enter_sleep();

    /** Wake up (return to AWAKE) */
    void wake_up();

    // --- State queries ---

    SleepStage stage() const { return stage_; }
    bool is_sleeping() const { return stage_ != SleepStage::AWAKE; }
    bool is_nrem()     const { return stage_ == SleepStage::NREM; }
    bool is_rem()      const { return stage_ == SleepStage::REM; }

    /** Current cycle number (0-indexed, increments at each NREM→REM transition) */
    uint32_t cycle_count() const { return cycle_count_; }

    /** Steps in current stage */
    size_t stage_timer() const { return stage_timer_; }

    /** Total sleep duration (all stages combined) */
    size_t total_sleep_steps() const { return total_sleep_steps_; }

    /** REM theta phase [0, 1) */
    float rem_theta_phase() const { return theta_phase_; }

    /** Is a PGO wave active this step? */
    bool pgo_active() const { return pgo_active_; }

    /** Current NREM/REM durations for this cycle */
    size_t current_nrem_duration() const { return current_nrem_dur_; }
    size_t current_rem_duration()  const { return current_rem_dur_; }

    const SleepCycleConfig& config() const { return config_; }

private:
    SleepCycleConfig config_;
    SleepStage stage_ = SleepStage::AWAKE;

    size_t   stage_timer_       = 0;
    uint32_t cycle_count_       = 0;
    size_t   total_sleep_steps_ = 0;

    // Cycle durations (evolve across cycles)
    size_t current_nrem_dur_ = 0;
    size_t current_rem_dur_  = 0;

    // REM state
    float theta_phase_ = 0.0f;
    bool  pgo_active_  = false;

    void transition_to_nrem();
    void transition_to_rem();
};

} // namespace wuyun
