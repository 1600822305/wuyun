#pragma once
/**
 * Neuromodulator — 神经调质系统
 *
 * 四大调质系统:
 *   DA  (多巴胺)    : VTA/SNc → 纹状体/PFC, 奖励/动机/学习信号
 *   NE  (去甲肾上腺素): 蓝斑核 → 全脑, 警觉/注意力/增益调节
 *   5HT (血清素)    : 中缝核 → 全脑, 情绪/折扣因子/风险评估
 *   ACh (乙酰胆碱)  : 基底前脑 → 皮层, 学习模式切换/注意力
 *
 * 调制效应 (02 文档 §6.2):
 *   DA  → 增益/可塑性: DA↑ → 强化学习率
 *   NE  → 增益调节:    NE↑ → 信号增益↑ (更敏锐)
 *   5HT → 折扣因子:    5HT↑ → 耐心↑ (远期奖励权重↑)
 *   ACh → 学习模式:    ACh↑ → 自下而上学习 (basal主导)
 *                       ACh↓ → 自上而下预测 (apical主导)
 *
 * 设计文档: docs/02_neuron_system_design.md §6.2
 */

#include <cstddef>
#include <cstdint>

namespace wuyun {

/** 神经调质浓度 (归一化到 0.0 ~ 1.0) */
struct NeuromodulatorLevels {
    float da  = 0.1f;   // 多巴胺 tonic baseline
    float ne  = 0.2f;   // 去甲肾上腺素 baseline
    float sht = 0.3f;   // 血清素 baseline (5-HT)
    float ach = 0.2f;   // 乙酰胆碱 baseline
};

/** 调制效应: 从调质浓度计算出的增益/学习率等 */
struct ModulationEffect {
    float gain          = 1.0f;   // 信号增益 (NE 驱动)
    float learning_rate = 1.0f;   // 学习率倍率 (DA 驱动)
    float discount      = 0.95f;  // 折扣因子 (5-HT 驱动)
    float basal_weight  = 0.5f;   // basal vs apical 权重 (ACh 驱动)
    // basal_weight=1.0: 完全自下而上; =0.0: 完全自上而下
};

/**
 * 神经调质管理器
 *
 * 管理一个脑区/系统的调质浓度和效应。
 * 支持 tonic (慢速基线) 和 phasic (快速突发) 两种模式。
 */
class NeuromodulatorSystem {
public:
    NeuromodulatorSystem();

    /** 设置 tonic 基线水平 */
    void set_tonic(const NeuromodulatorLevels& levels);

    /** 注入 phasic 突发 (例如 DA burst 信号) */
    void inject_phasic(float d_da, float d_ne, float d_sht, float d_ach);

    /** 每步更新: phasic 向 tonic 衰减 */
    void step(float dt = 1.0f);

    /** 当前总浓度 = tonic + phasic */
    NeuromodulatorLevels current() const;

    /** 计算调制效应 */
    ModulationEffect compute_effect() const;

    // 访问器
    const NeuromodulatorLevels& tonic()  const { return tonic_; }
    const NeuromodulatorLevels& phasic() const { return phasic_; }

private:
    NeuromodulatorLevels tonic_;
    NeuromodulatorLevels phasic_;   // 快速成分, 每步衰减

    // Phasic 衰减时间常数 (ms)
    float tau_da_  = 200.0f;   // DA phasic: ~200ms
    float tau_ne_  = 500.0f;   // NE phasic: ~500ms
    float tau_sht_ = 1000.0f;  // 5-HT phasic: ~1s (最慢)
    float tau_ach_ = 300.0f;   // ACh phasic: ~300ms
};

} // namespace wuyun
