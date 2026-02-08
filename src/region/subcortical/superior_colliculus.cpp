#include "region/subcortical/superior_colliculus.h"
#include "core/types.h"
#include <algorithm>
#include <cmath>

namespace wuyun {

// SC superficial neurons: fast visual processing, low threshold
static NeuronParams SC_VISUAL_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -60.0f;
    p.somatic.v_threshold = -48.0f;  // Low threshold → fast response
    p.somatic.v_reset = -55.0f;
    p.somatic.tau_m = 8.0f;          // Very fast membrane (faster than cortex)
    p.somatic.r_s = 1.2f;
    p.somatic.a = 0.0f;
    p.somatic.b = 0.5f;             // Minimal adaptation
    p.somatic.tau_w = 100.0f;
    p.kappa = 0.0f;                  // No apical (not pyramidal)
    return p;
}

// SC deep neurons: multimodal integration, motor-like output
static NeuronParams SC_MOTOR_PARAMS() {
    NeuronParams p;
    p.somatic.v_rest = -58.0f;
    p.somatic.v_threshold = -45.0f;  // Slightly higher threshold (needs convergent input)
    p.somatic.v_reset = -53.0f;
    p.somatic.tau_m = 10.0f;
    p.somatic.r_s = 1.0f;
    p.somatic.a = 0.01f;
    p.somatic.b = 1.0f;
    p.somatic.tau_w = 150.0f;
    p.kappa = 0.0f;
    return p;
}

SuperiorColliculus::SuperiorColliculus(const SCConfig& config)
    : BrainRegion(config.name, config.n_superficial + config.n_deep)
    , config_(config)
    , superficial_(config.n_superficial, SC_VISUAL_PARAMS())
    , deep_(config.n_deep, SC_MOTOR_PARAMS())
    , psp_sup_(config.n_superficial, 0.0f)
    , psp_deep_(config.n_deep, 0.0f)
    , deep_preferred_dir_(config.n_deep, 0.0f)
    , fired_(config.n_superficial + config.n_deep, 0)
    , spike_type_(config.n_superficial + config.n_deep, 0)
{
    // v52: 深层运动地图 — 均匀分布偏好方向
    // 生物学: SC 深层神经元按方位角排列 (Stein & Meredith 1993)
    // 4 个神经元: RIGHT=0, UP=π/2, LEFT=π, DOWN=-π/2
    for (size_t i = 0; i < config.n_deep; ++i) {
        deep_preferred_dir_[i] = 2.0f * 3.14159265f * static_cast<float>(i)
                                / static_cast<float>(config.n_deep);
    }
}

void SuperiorColliculus::step(int32_t t, float dt) {
    // --- Superficial layer: retinotopic visual map ---
    // Receives direct retinal/LGN input, detects visual events
    float total_input = 0.0f;
    for (size_t i = 0; i < superficial_.size(); ++i) {
        superficial_.inject_basal(i, psp_sup_[i]);
        total_input += psp_sup_[i];
        psp_sup_[i] *= PSP_DECAY;
    }

    // --- Deep layer: receives from superficial + cortical feedback ---
    // Superficial → Deep feedforward
    for (size_t i = 0; i < deep_.size(); ++i) {
        // Deep gets input from superficial (broad convergence)
        float sup_drive = 0.0f;
        for (size_t j = 0; j < superficial_.size(); ++j) {
            if (superficial_.fired()[j]) sup_drive += 8.0f;
        }
        deep_.inject_basal(i, psp_deep_[i] + sup_drive);
        psp_deep_[i] *= PSP_DECAY;
    }

    superficial_.step(t, dt);
    deep_.step(t, dt);

    // --- Saliency computation ---
    // Saliency = change detection (onset/offset of visual stimuli)
    // Biology: SC responds strongly to stimulus ONSET, habituates to static scenes
    float current_input = total_input / std::max<float>(1.0f, static_cast<float>(superficial_.size()));
    float input_change = std::abs(current_input - prev_input_level_);
    prev_input_level_ = prev_input_level_ * 0.95f + current_input * 0.05f;  // Slow adaptation

    // Count deep layer firing as saliency measure
    size_t deep_fires = 0;
    for (size_t i = 0; i < deep_.size(); ++i)
        if (deep_.fired()[i]) deep_fires++;

    float firing_saliency = static_cast<float>(deep_fires) / static_cast<float>(std::max<size_t>(deep_.size(), 1));
    saliency_ = saliency_ * 0.9f + (input_change * 0.5f + firing_saliency * 0.5f) * 0.1f;

    aggregate_state();
}

void SuperiorColliculus::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        float current = is_burst(static_cast<SpikeType>(evt.spike_type)) ? 40.0f : 25.0f;
        // Route to superficial (visual input) and deep (cortical feedback)
        size_t sup_idx = evt.neuron_id % superficial_.size();
        psp_sup_[sup_idx] += current;

        // Some input also reaches deep layer (broad routing)
        size_t deep_idx = evt.neuron_id % deep_.size();
        psp_deep_[deep_idx] += current * 0.3f;  // Weaker to deep
    }
}

void SuperiorColliculus::submit_spikes(SpikeBus& bus, int32_t t) {
    bus.submit_spikes(region_id_, fired_, spike_type_, t);
}

void SuperiorColliculus::inject_external(const std::vector<float>& currents) {
    for (size_t i = 0; i < std::min(currents.size(), superficial_.size()); ++i) {
        psp_sup_[i] += currents[i];
    }
}

// =============================================================================
// v52: 视觉定向反射 — 直接视网膜→SC 快通路
//
// 生物学: 视网膜神经节细胞 → SC 浅层 (retinotopic 视觉地图)
//   SC 浅层计算刺激方位 → SC 深层 (运动地图)
//   深层编码定向运动方向 = "朝那边看/走"
//   这条通路 2-3 步就能产生运动输出, 远快于皮层 14 步
//
// 实现:
//   计算视觉 patch 的显著性质心 (center-of-mass)
//   排除中心像素 (agent 自身) 和背景 (空地=0.0)
//   food(0.9) 产生强趋近, danger(0.3) 产生弱趋近
//   danger 的弱趋近会被 PAG freeze 在学习后覆盖
//   → 生物正确: 婴儿第一次也会碰危险物, 学了才避开
// =============================================================================
void SuperiorColliculus::inject_visual_patch(
    const std::vector<float>& pixels, int width, int height, float gain)
{
    if (pixels.empty() || width <= 0 || height <= 0 || gain < 0.001f) return;

    float center_x = (width - 1) / 2.0f;
    float center_y = (height - 1) / 2.0f;
    float sum_wx = 0.0f, sum_wy = 0.0f, sum_w = 0.0f;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = static_cast<size_t>(y * width + x);
            if (idx >= pixels.size()) continue;

            float v = pixels[idx];
            if (v < 0.05f) continue;  // 忽略空地 (0.0)

            float dx = static_cast<float>(x) - center_x;
            float dy = static_cast<float>(y) - center_y;
            float dist = std::sqrt(dx * dx + dy * dy);

            if (dist < 0.5f) continue;  // 忽略中心像素 (agent 自身)

            // 权重 = 像素亮度 × 外周增益
            // 外周刺激比中心刺激更显著 (SC 外周敏感)
            float w = v * (0.5f + dist);
            sum_wx += dx * w;
            sum_wy += dy * w;
            sum_w += w;
        }
    }

    if (sum_w < 0.01f) {
        saliency_direction_ = 0.0f;
        saliency_magnitude_ = 0.0f;
        return;
    }

    // 质心方向 (GridWorld: y 向下增长, UP 动作 = y-1)
    float cx = sum_wx / sum_w;
    float cy = sum_wy / sum_w;
    saliency_direction_ = std::atan2(-cy, cx);  // -cy: 向上=正角度
    saliency_magnitude_ = std::sqrt(cx * cx + cy * cy);

    // 注入方向性电流到深层神经元
    // 偏好方向与显著性方向匹配的神经元获得更强电流
    for (size_t i = 0; i < deep_.size(); ++i) {
        float cos_sim = std::cos(deep_preferred_dir_[i] - saliency_direction_);
        if (cos_sim > 0.0f) {
            psp_deep_[i] += cos_sim * saliency_magnitude_ * gain;
        }
    }
}

void SuperiorColliculus::aggregate_state() {
    size_t offset = 0;
    auto copy_pop = [&](const NeuronPopulation& pop) {
        for (size_t i = 0; i < pop.size(); ++i) {
            fired_[offset + i] = pop.fired()[i];
            spike_type_[offset + i] = pop.spike_type()[i];
        }
        offset += pop.size();
    };
    copy_pop(superficial_);
    copy_pop(deep_);
}

} // namespace wuyun
