#include "engine/sleep_cycle.h"
#include <random>
#include <cmath>
#include <algorithm>

namespace wuyun {

SleepCycleManager::SleepCycleManager(const SleepCycleConfig& config)
    : config_(config)
{
}

void SleepCycleManager::enter_sleep() {
    if (stage_ != SleepStage::AWAKE) return;
    cycle_count_ = 0;
    total_sleep_steps_ = 0;
    transition_to_nrem();
}

void SleepCycleManager::wake_up() {
    stage_ = SleepStage::AWAKE;
    stage_timer_ = 0;
    theta_phase_ = 0.0f;
    pgo_active_ = false;
}

void SleepCycleManager::step() {
    if (stage_ == SleepStage::AWAKE) return;

    ++stage_timer_;
    ++total_sleep_steps_;

    if (stage_ == SleepStage::NREM) {
        // NREM → REM transition
        if (stage_timer_ >= current_nrem_dur_) {
            transition_to_rem();
        }
    } else if (stage_ == SleepStage::REM) {
        // Advance theta phase
        theta_phase_ += config_.rem_theta_freq;
        if (theta_phase_ >= 1.0f) theta_phase_ -= 1.0f;

        // PGO wave generation (stochastic)
        static std::mt19937 pgo_rng(77777);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        pgo_active_ = (dist(pgo_rng) < config_.rem_pgo_prob);

        // REM → NREM transition (new cycle)
        if (stage_timer_ >= current_rem_dur_) {
            ++cycle_count_;
            transition_to_nrem();
        }
    }
}

void SleepCycleManager::transition_to_nrem() {
    stage_ = SleepStage::NREM;
    stage_timer_ = 0;
    theta_phase_ = 0.0f;
    pgo_active_ = false;

    // Compute NREM duration for this cycle (shrinks over night)
    size_t shrink = config_.nrem_growth * cycle_count_;
    current_nrem_dur_ = (config_.nrem_duration > shrink)
        ? config_.nrem_duration - shrink
        : config_.min_nrem_duration;
    current_nrem_dur_ = std::max(current_nrem_dur_, config_.min_nrem_duration);

    // Compute REM duration for this cycle (grows over night)
    current_rem_dur_ = std::min(
        config_.rem_duration + config_.rem_growth * cycle_count_,
        config_.max_rem_duration);
}

void SleepCycleManager::transition_to_rem() {
    stage_ = SleepStage::REM;
    stage_timer_ = 0;
    theta_phase_ = 0.0f;
    pgo_active_ = false;
}

} // namespace wuyun
