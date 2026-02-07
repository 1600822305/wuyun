#include "core/neuromodulator.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

NeuromodulatorSystem::NeuromodulatorSystem()
    : tonic_{}
    , phasic_{0.0f, 0.0f, 0.0f, 0.0f}
{}

void NeuromodulatorSystem::set_tonic(const NeuromodulatorLevels& levels) {
    tonic_ = levels;
}

void NeuromodulatorSystem::inject_phasic(float d_da, float d_ne, float d_sht, float d_ach) {
    phasic_.da  += d_da;
    phasic_.ne  += d_ne;
    phasic_.sht += d_sht;
    phasic_.ach += d_ach;
}

void NeuromodulatorSystem::step(float dt) {
    // Phasic components decay toward zero
    phasic_.da  -= phasic_.da  * (dt / tau_da_);
    phasic_.ne  -= phasic_.ne  * (dt / tau_ne_);
    phasic_.sht -= phasic_.sht * (dt / tau_sht_);
    phasic_.ach -= phasic_.ach * (dt / tau_ach_);
}

NeuromodulatorLevels NeuromodulatorSystem::current() const {
    return {
        std::clamp(tonic_.da  + phasic_.da,  0.0f, 1.0f),
        std::clamp(tonic_.ne  + phasic_.ne,  0.0f, 1.0f),
        std::clamp(tonic_.sht + phasic_.sht, 0.0f, 1.0f),
        std::clamp(tonic_.ach + phasic_.ach, 0.0f, 1.0f)
    };
}

ModulationEffect NeuromodulatorSystem::compute_effect() const {
    auto cur = current();
    ModulationEffect eff;

    // NE → gain: 0.5 (low alertness) ~ 2.0 (high alertness)
    eff.gain = 0.5f + 1.5f * cur.ne;

    // DA → learning rate: 0.1 (low DA) ~ 3.0 (high DA, phasic burst)
    eff.learning_rate = 0.1f + 2.9f * cur.da;

    // 5-HT → discount factor: 0.8 (impulsive) ~ 0.99 (patient)
    eff.discount = 0.8f + 0.19f * cur.sht;

    // ACh → basal weight: high ACh = bottom-up (basal), low ACh = top-down (apical)
    eff.basal_weight = cur.ach;

    return eff;
}

} // namespace wuyun
