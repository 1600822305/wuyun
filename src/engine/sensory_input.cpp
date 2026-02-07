#include "engine/sensory_input.h"
#include <random>
#include <cmath>

namespace wuyun {

// =============================================================================
// VisualInput
// =============================================================================

VisualInput::VisualInput(const VisualInputConfig& config)
    : config_(config)
{
    build_receptive_fields();
}

void VisualInput::build_receptive_fields() {
    size_t n_lgn = config_.n_lgn_neurons;
    size_t w = config_.input_width;
    size_t h = config_.input_height;

    rf_center_x_.resize(n_lgn);
    rf_center_y_.resize(n_lgn);
    rf_weights_.resize(n_lgn);

    // Distribute LGN neuron receptive field centers across the image
    // Use a grid-like layout with some jitter
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> jitter(-0.3f, 0.3f);

    // Compute grid dimensions for n_lgn neurons
    size_t n_on = config_.on_off_channels ? n_lgn / 2 : n_lgn;
    size_t grid_side = static_cast<size_t>(std::ceil(std::sqrt(static_cast<float>(n_on))));

    float step_x = static_cast<float>(w) / static_cast<float>(grid_side);
    float step_y = static_cast<float>(h) / static_cast<float>(grid_side);

    // Build receptive fields for ON cells (or all if no ON/OFF split)
    for (size_t i = 0; i < n_on; ++i) {
        size_t gx = i % grid_side;
        size_t gy = i / grid_side;

        float cx = (static_cast<float>(gx) + 0.5f + jitter(rng)) * step_x;
        float cy = (static_cast<float>(gy) + 0.5f + jitter(rng)) * step_y;
        cx = std::clamp(cx, 0.0f, static_cast<float>(w) - 0.01f);
        cy = std::clamp(cy, 0.0f, static_cast<float>(h) - 0.01f);

        rf_center_x_[i] = cx;
        rf_center_y_[i] = cy;

        // Build center-surround weights for this neuron
        float r_c = config_.center_radius;
        float r_s = config_.surround_radius;

        for (size_t py = 0; py < h; ++py) {
            for (size_t px = 0; px < w; ++px) {
                float dx = static_cast<float>(px) + 0.5f - cx;
                float dy = static_cast<float>(py) + 0.5f - cy;
                float dist = std::sqrt(dx * dx + dy * dy);

                float weight = 0.0f;
                if (dist <= r_c) {
                    // Center: excitatory (ON cell: bright=excite)
                    weight = config_.center_weight * (1.0f - dist / r_c);
                } else if (dist <= r_s) {
                    // Surround: inhibitory (ON cell: bright=inhibit)
                    float norm = (dist - r_c) / (r_s - r_c);
                    weight = -config_.surround_weight * (1.0f - norm);
                }

                if (std::fabs(weight) > 0.01f) {
                    rf_weights_[i].push_back({py * w + px, weight});
                }
            }
        }
    }

    // OFF cells: same positions, inverted polarity
    if (config_.on_off_channels) {
        for (size_t i = n_on; i < n_lgn; ++i) {
            size_t on_idx = i - n_on;
            if (on_idx >= n_on) on_idx = on_idx % n_on;

            rf_center_x_[i] = rf_center_x_[on_idx];
            rf_center_y_[i] = rf_center_y_[on_idx];

            // Copy ON weights but invert sign
            for (const auto& conn : rf_weights_[on_idx]) {
                rf_weights_[i].push_back({conn.pixel_idx, -conn.weight});
            }
        }
    }
}

std::vector<float> VisualInput::encode(const std::vector<float>& pixels) const {
    size_t n_lgn = config_.n_lgn_neurons;
    std::vector<float> currents(n_lgn, config_.baseline);

    size_t n_pixels = config_.input_width * config_.input_height;
    if (pixels.size() < n_pixels) return currents;

    // Apply receptive field weights
    for (size_t i = 0; i < n_lgn; ++i) {
        float response = 0.0f;
        for (const auto& conn : rf_weights_[i]) {
            if (conn.pixel_idx < pixels.size()) {
                response += conn.weight * pixels[conn.pixel_idx];
            }
        }
        // Scale and add to baseline
        currents[i] += config_.gain * std::max(0.0f, response);
    }

    // Add noise
    if (config_.noise_amp > 0.0f) {
        static std::mt19937 noise_rng(12345);
        std::uniform_real_distribution<float> noise(0.0f, config_.noise_amp);
        for (size_t i = 0; i < n_lgn; ++i) {
            currents[i] += noise(noise_rng);
        }
    }

    return currents;
}

void VisualInput::encode_and_inject(const std::vector<float>& pixels,
                                     BrainRegion* lgn) const {
    if (!lgn) return;
    auto currents = encode(pixels);
    lgn->inject_external(currents);
}

// =============================================================================
// AuditoryInput
// =============================================================================

AuditoryInput::AuditoryInput(const AuditoryInputConfig& config)
    : config_(config)
    , prev_spectrum_(config.n_freq_bands, 0.0f)
{
}

std::vector<float> AuditoryInput::encode(const std::vector<float>& spectrum) {
    size_t n_mgn = config_.n_mgn_neurons;
    size_t n_bands = config_.n_freq_bands;
    std::vector<float> currents(n_mgn, config_.baseline);

    if (spectrum.empty()) return currents;

    // Tonotopic mapping: each MGN neuron covers a range of frequency bands
    float bands_per_neuron = static_cast<float>(n_bands) / static_cast<float>(n_mgn);

    for (size_t i = 0; i < n_mgn; ++i) {
        float band_start = static_cast<float>(i) * bands_per_neuron;
        float band_end = band_start + bands_per_neuron;

        // Average power in this neuron's frequency range
        float power = 0.0f;
        int count = 0;
        for (size_t b = static_cast<size_t>(band_start);
             b < std::min(static_cast<size_t>(std::ceil(band_end)), n_bands); ++b) {
            float val = (b < spectrum.size()) ? spectrum[b] : 0.0f;

            // Onset emphasis: difference from previous frame
            float onset = std::max(0.0f, val - prev_spectrum_[b] * config_.temporal_decay);
            power += val + onset * 0.5f;
            count++;
        }
        if (count > 0) power /= static_cast<float>(count);

        currents[i] += config_.gain * power;
    }

    // Update previous spectrum for onset detection
    for (size_t b = 0; b < n_bands; ++b) {
        prev_spectrum_[b] = (b < spectrum.size()) ? spectrum[b] : 0.0f;
    }

    // Add noise
    if (config_.noise_amp > 0.0f) {
        static std::mt19937 noise_rng(54321);
        std::uniform_real_distribution<float> noise(0.0f, config_.noise_amp);
        for (size_t i = 0; i < n_mgn; ++i) {
            currents[i] += noise(noise_rng);
        }
    }

    return currents;
}

void AuditoryInput::encode_and_inject(const std::vector<float>& spectrum,
                                       BrainRegion* mgn) {
    if (!mgn) return;
    auto currents = encode(spectrum);
    mgn->inject_external(currents);
}

} // namespace wuyun
