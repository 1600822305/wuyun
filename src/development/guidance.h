#pragma once
/**
 * GuidanceField — 导向分子场
 *
 * 模拟轴突导向的化学梯度系统:
 *   - N 种导向分子, 每种在 2D 空间有浓度分布
 *   - 每个神经元表达受体 (receptor_expression), 决定对哪些分子敏感
 *   - 轴突沿梯度生长: 受体吸引的分子 → 向高浓度方向长
 *   - 连接在轴突到达目标细胞附近时形成
 *
 * 生物学:
 *   - Netrin/DCC: 吸引 (腹侧引导)
 *   - Slit/Robo: 排斥 (中线排斥)
 *   - Ephrin/Eph: 拓扑映射 (视网膜→上丘)
 *   - Semaphorin/Neuropilin: 选择性引导 (皮层层特异性)
 *
 * 实现: 每种导向分子用高斯函数描述空间浓度:
 *   concentration(x,y) = amplitude × exp(-((x-cx)²+(y-cy)²) / (2σ²))
 *   梯度 = ∂concentration/∂x, ∂concentration/∂y
 */

#include <vector>
#include <cmath>

namespace wuyun {

// 单个导向分子的空间浓度参数
struct GuidanceMolecule {
    float cx, cy;       // 浓度峰值中心
    float sigma;        // 扩散范围 (σ)
    float amplitude;    // 峰值浓度
    bool  is_attractant; // true=吸引, false=排斥

    // 在 (x,y) 处的浓度
    float concentration(float x, float y) const {
        float dx = x - cx, dy = y - cy;
        return amplitude * std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
    }

    // 在 (x,y) 处的梯度 (指向浓度增加方向)
    void gradient(float x, float y, float& gx, float& gy) const {
        float dx = x - cx, dy = y - cy;
        float c = concentration(x, y);
        float inv_s2 = 1.0f / (sigma * sigma);
        // ∂c/∂x = -c × (x-cx)/σ² → 指向 cx (浓度增加方向)
        gx = -c * dx * inv_s2;
        gy = -c * dy * inv_s2;
    }
};

// 导向分子场: N 种分子的空间浓度系统
class GuidanceField {
public:
    static constexpr int N_MOLECULES = 8;  // 8 种导向分子

    GuidanceField() : molecules_(N_MOLECULES) {}

    // 设置第 i 种分子的参数
    void set_molecule(int i, float cx, float cy, float sigma,
                      float amplitude, bool attractant) {
        if (i >= 0 && i < N_MOLECULES) {
            molecules_[i] = {cx, cy, sigma, amplitude, attractant};
        }
    }

    // 计算细胞在 (x,y) 处、受体表达为 receptors[N_MOLECULES] 时的合力方向
    // 返回: (fx, fy) 归一化方向向量
    void compute_guidance_force(float x, float y,
                                const float* receptors,
                                float& fx, float& fy) const {
        fx = 0.0f;
        fy = 0.0f;
        for (int m = 0; m < N_MOLECULES; ++m) {
            if (std::abs(receptors[m]) < 0.01f) continue;  // 不表达该受体

            float gx, gy;
            molecules_[m].gradient(x, y, gx, gy);

            // 受体强度 × 吸引/排斥方向
            float sign = molecules_[m].is_attractant ? 1.0f : -1.0f;
            fx += receptors[m] * sign * gx;
            fy += receptors[m] * sign * gy;
        }
        // 归一化
        float mag = std::sqrt(fx * fx + fy * fy);
        if (mag > 0.001f) {
            fx /= mag;
            fy /= mag;
        }
    }

    const GuidanceMolecule& molecule(int i) const { return molecules_[i]; }

private:
    std::vector<GuidanceMolecule> molecules_;
};

} // namespace wuyun
