#pragma once
/**
 * SensoryInput — 感觉输入接口
 *
 * 将外界刺激 (图像/声音/触觉) 编码为丘脑可接受的电流模式。
 *
 * 架构:
 *   外界刺激 → SensoryEncoder → inject_external(thalamus) → SpikeBus → 皮层
 *
 * 子类:
 *   - VisualInput:   图像像素 → ON/OFF center-surround → LGN relay
 *   - AuditoryInput: 频谱功率 → tonotopic mapping → MGN relay
 *
 * 生物学基础:
 *   - 视网膜神经节细胞 ON/OFF center-surround (Kuffler 1953)
 *   - 耳蜗频率-位置映射 (von Békésy 1960)
 *   - 体感感受器→脊髓→VPL映射
 */

#include "region/brain_region.h"
#include <vector>
#include <cstdint>
#include <string>
#include <cmath>
#include <algorithm>

namespace wuyun {

// =============================================================================
// VisualInput — 视觉输入编码器
// =============================================================================

struct VisualInputConfig {
    size_t input_width   = 8;     // 输入图像宽度 (像素)
    size_t input_height  = 8;     // 输入图像高度 (像素)
    size_t n_lgn_neurons = 50;    // LGN relay 神经元数 (需匹配 ThalamicConfig)

    // Center-surround 参数
    float center_radius  = 1.0f;  // 中心半径 (像素)
    float surround_radius= 3.0f;  // 周围半径 (像素)
    float center_weight  = 1.0f;  // 中心权重
    float surround_weight= 0.5f;  // 周围权重 (抑制)

    // 输出缩放
    float gain           = 40.0f; // 电流增益 (像素强度 → nA)
    float baseline       = 5.0f;  // 基线电流 (自发活动)
    float noise_amp      = 2.0f;  // 随机噪声幅度

    // ON/OFF 通道
    bool on_off_channels = true;  // true: 前半LGN=ON, 后半=OFF
};

class VisualInput {
public:
    explicit VisualInput(const VisualInputConfig& config = {});

    /**
     * 编码图像帧为 LGN 电流向量
     *
     * @param pixels  灰度像素 [0,1], 长度 = width * height (row-major)
     * @return        LGN relay 电流向量, 长度 = n_lgn_neurons
     */
    std::vector<float> encode(const std::vector<float>& pixels) const;

    /**
     * 编码并直接注入到 LGN 区域
     */
    void encode_and_inject(const std::vector<float>& pixels, BrainRegion* lgn) const;

    // 配置查询
    size_t input_width()  const { return config_.input_width; }
    size_t input_height() const { return config_.input_height; }
    size_t n_pixels()     const { return config_.input_width * config_.input_height; }
    size_t n_lgn()        const { return config_.n_lgn_neurons; }

    const VisualInputConfig& config() const { return config_; }

private:
    VisualInputConfig config_;

    // 预计算: 像素→LGN 权重矩阵 (center-surround receptive fields)
    // rf_weights_[lgn_idx] = vector of (pixel_idx, weight) pairs
    struct RFConnection {
        size_t pixel_idx;
        float  weight;
    };
    std::vector<std::vector<RFConnection>> rf_weights_;

    void build_receptive_fields();

    // LGN 神经元的感受野中心位置 (像素空间)
    std::vector<float> rf_center_x_;
    std::vector<float> rf_center_y_;
};

// =============================================================================
// AuditoryInput — 听觉输入编码器
// =============================================================================

struct AuditoryInputConfig {
    size_t n_freq_bands   = 16;    // 频率带数 (tonotopic channels)
    size_t n_mgn_neurons  = 20;    // MGN relay 神经元数

    float gain            = 35.0f; // 功率 → 电流增益
    float baseline        = 3.0f;  // 基线电流
    float noise_amp       = 1.5f;  // 噪声
    float temporal_decay  = 0.7f;  // 时间平滑 (onset emphasis)
};

class AuditoryInput {
public:
    explicit AuditoryInput(const AuditoryInputConfig& config = {});

    /**
     * 编码频谱功率为 MGN 电流向量
     *
     * @param spectrum  频率带功率 [0,1], 长度 = n_freq_bands
     * @return          MGN relay 电流向量, 长度 = n_mgn_neurons
     */
    std::vector<float> encode(const std::vector<float>& spectrum);

    /**
     * 编码并直接注入到 MGN 区域
     */
    void encode_and_inject(const std::vector<float>& spectrum, BrainRegion* mgn);

    size_t n_freq_bands() const { return config_.n_freq_bands; }
    size_t n_mgn()        const { return config_.n_mgn_neurons; }

    const AuditoryInputConfig& config() const { return config_; }

private:
    AuditoryInputConfig config_;
    std::vector<float> prev_spectrum_;  // 上一帧 (onset 检测)
};

} // namespace wuyun
