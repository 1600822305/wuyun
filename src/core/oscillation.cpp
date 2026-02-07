#include "core/oscillation.h"

namespace wuyun {

OscillationTracker::OscillationTracker() {
    // Default frequencies for each band
    bands_[static_cast<size_t>(OscBand::DELTA)] = {2.0f,  0.0f, 1.0f};
    bands_[static_cast<size_t>(OscBand::THETA)] = {6.0f,  0.0f, 1.0f};
    bands_[static_cast<size_t>(OscBand::ALPHA)] = {10.0f, 0.0f, 1.0f};
    bands_[static_cast<size_t>(OscBand::BETA)]  = {20.0f, 0.0f, 1.0f};
    bands_[static_cast<size_t>(OscBand::GAMMA)] = {40.0f, 0.0f, 1.0f};
}

void OscillationTracker::set_band(OscBand band, float freq_hz, float amplitude) {
    auto& osc = bands_[static_cast<size_t>(band)];
    osc.frequency = freq_hz;
    osc.amplitude = amplitude;
}

void OscillationTracker::step(float dt_ms) {
    for (size_t i = 0; i < static_cast<size_t>(OscBand::NUM_BANDS); ++i) {
        bands_[i].step(dt_ms);
    }
}

float OscillationTracker::phase(OscBand band) const {
    return bands_[static_cast<size_t>(band)].phase;
}

float OscillationTracker::value(OscBand band) const {
    return bands_[static_cast<size_t>(band)].value();
}

const Oscillator& OscillationTracker::oscillator(OscBand band) const {
    return bands_[static_cast<size_t>(band)];
}

Oscillator& OscillationTracker::oscillator(OscBand band) {
    return bands_[static_cast<size_t>(band)];
}

bool OscillationTracker::theta_gamma_coupling() const {
    return bands_[static_cast<size_t>(OscBand::THETA)].at_trough(0.8f);
}

} // namespace wuyun
