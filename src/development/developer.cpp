#include "development/developer.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cstdio>

namespace wuyun {

// =============================================================================
// 发育规则 → AgentConfig 参数映射
// =============================================================================
//
// 生物学原理:
//   基因编码发育梯度 (形态发生素浓度、转录因子阈值),
//   这些梯度在空间上变化, 决定每个区域的属性。
//   我们不模拟 3D 空间, 而是用前后轴 (anterior-posterior) 梯度
//   和径向 (radial) 梯度来计算各区域参数。
//
// 前后轴位置 (归一化 [0,1], 0=最前, 1=最后):
//   FPC:     0.05  (最前)
//   dlPFC:   0.15
//   OFC:     0.20
//   ACC:     0.25
//   M1/PMC:  0.35
//   BG:      0.50  (中部)
//   Thal:    0.55
//   Hipp:    0.60
//   Amyg:    0.62
//   VTA/SNc: 0.70
//   V4/IT:   0.75
//   V2:      0.85
//   V1:      0.90  (最后)
//   LGN:     0.95

static float anterior_posterior(float position, const DevGenome& g) {
    // 增殖梯度: growth_gradient[0] = 前后轴倾斜
    // >0: 前部增殖更多 (大前额叶) — 灵长类特征
    // <0: 后部增殖更多 (大感觉区) — 啮齿类特征
    float ap_bias = g.growth_gradient[0].value;
    // 在给定位置的增殖因子: 1.0 + bias × (0.5 - position)
    // position=0(前) → 1.0 + bias×0.5 (前部增强)
    // position=1(后) → 1.0 - bias×0.5 (后部减弱)
    return 1.0f + ap_bias * (0.5f - position);
}

AgentConfig Developer::to_agent_config(const DevGenome& genome) {
    AgentConfig cfg;

    // =====================================================================
    // 1. 增殖 → 区域大小
    //    division_rounds 控制总规模, growth_gradient 控制前后比例
    // =====================================================================

    // 感觉区 (V1) 在后部 (position=0.9)
    float v1_growth = anterior_posterior(0.9f, genome);
    // 感觉区基础大小 = 2^(division_rounds[SENSORY]) / 32 (归一化到 ~1.0)
    float sensory_base = std::pow(2.0f, genome.division_rounds[0].value) / 32.0f;
    cfg.v1_size_factor = std::clamp(sensory_base * v1_growth, 0.5f, 3.0f);

    // 前额叶 (dlPFC) 在前部 (position=0.15)
    float pfc_growth = anterior_posterior(0.15f, genome);
    float pfc_base = std::pow(2.0f, genome.division_rounds[2].value) / 16.0f;
    cfg.dlpfc_size_factor = std::clamp(pfc_base * pfc_growth, 0.5f, 3.0f);

    // 皮层下 (BG) 在中部 (position=0.5)
    float bg_growth = anterior_posterior(0.5f, genome);
    float sub_base = std::pow(2.0f, genome.division_rounds[3].value) / 16.0f;
    cfg.bg_size_factor = std::clamp(sub_base * bg_growth, 0.5f, 2.0f);

    // =====================================================================
    // 2. 分化梯度 → 学习率和受体密度
    //    DA 受体密度前部高 (前额叶 DA-STDP 更强)
    //    NMDA 比后部高 (感觉区可塑性不同)
    // =====================================================================

    // DA-STDP 学习率: DA 梯度 + 基础 lr
    // da_gradient > 0 → 前部 DA 高 → 更强的 DA-STDP
    float da_factor = 1.0f + genome.da_gradient.value * 0.5f;
    cfg.da_stdp_lr = std::clamp(genome.da_stdp_lr.value * da_factor, 0.005f, 0.15f);

    // 皮层 STDP: NMDA 梯度影响
    // nmda_gradient > 0 → 后部 NMDA 高 → 感觉区 STDP 更活跃
    float nmda_factor = 1.0f + genome.nmda_gradient.value * 0.3f;
    cfg.cortical_stdp_a_plus = std::clamp(0.003f * nmda_factor, 0.001f, 0.02f);
    cfg.cortical_stdp_a_minus = std::clamp(-0.005f * nmda_factor, -0.02f, -0.001f);
    cfg.cortical_stdp_w_max = 1.5f;

    // =====================================================================
    // 3. 连接强度 → 增益参数
    //    cross_connect 概率矩阵 → 各通路增益
    // =====================================================================

    // 感觉→前额叶 连接 (Sensory→PFC cross_connect)
    float s2p_strength = genome.cross_connect[0 * 5 + 2].value;  // SENSORY→PFC
    cfg.lgn_gain = std::clamp(genome.sensory_gain.value * (1.0f + s2p_strength), 50.0f, 500.0f);
    cfg.lgn_baseline = std::clamp(genome.sensory_gain.value * 0.05f, 1.0f, 20.0f);
    cfg.lgn_noise_amp = 3.0f;

    // 前额叶→皮层下 连接 (PFC→SUB cross_connect)
    float p2b_strength = genome.cross_connect[2 * 5 + 3].value;  // PFC→SUB
    cfg.bg_to_m1_gain = std::clamp(p2b_strength * 30.0f, 2.0f, 25.0f);

    // =====================================================================
    // 4. 探索参数 → 运动噪声
    //    motor_noise 基因直接映射
    // =====================================================================

    cfg.exploration_noise = std::clamp(genome.motor_noise.value, 10.0f, 100.0f);
    // attractor/background 比例从连接矩阵的对称性推导
    float motor_recurrent = genome.cross_connect[1 * 5 + 1].value;  // MOTOR→MOTOR
    cfg.attractor_drive_ratio = std::clamp(0.3f + motor_recurrent, 0.3f, 0.9f);
    cfg.background_drive_ratio = std::clamp(0.1f - motor_recurrent * 0.2f, 0.02f, 0.3f);

    // =====================================================================
    // 5. 修剪 → 稳态可塑性
    //    pruning_threshold → homeostatic 目标发放率
    //    critical_period → 发育期长度
    // =====================================================================

    cfg.homeostatic_target_rate = std::clamp(
        genome.homeostatic_target.value / genome.pruning_threshold.value,
        1.0f, 15.0f);
    cfg.homeostatic_eta = std::clamp(genome.homeostatic_eta.value, 0.0001f, 0.01f);
    cfg.dev_period_steps = std::max<size_t>(0,
        static_cast<size_t>(genome.critical_period.value));

    // =====================================================================
    // 6. 奖赏/重放参数
    // =====================================================================

    cfg.reward_scale = std::clamp(genome.reward_gain.value / 40.0f, 0.3f, 5.0f);

    // 重放参数从连接矩阵的全局连通性推导
    float total_connectivity = 0.0f;
    for (int i = 0; i < 25; ++i) total_connectivity += genome.cross_connect[i].value;
    cfg.replay_passes = std::clamp(static_cast<int>(total_connectivity * 2.0f), 1, 15);
    cfg.replay_da_scale = 0.5f;

    // =====================================================================
    // 7. NE 探索调制
    //    从 neuromod→sensory 连接推导
    // =====================================================================

    float n2s = genome.cross_connect[4 * 5 + 0].value;  // NMOD→SENSORY
    cfg.ne_food_scale = std::clamp(n2s * 20.0f, 1.0f, 8.0f);
    cfg.ne_floor = std::clamp(0.5f + n2s, 0.4f, 1.0f);

    // =====================================================================
    // 8. 时序参数
    // =====================================================================

    cfg.brain_steps_per_action = 12;
    cfg.reward_processing_steps = 9;

    // 所有模块默认启用 (完整人脑架构)
    cfg.enable_da_stdp = true;
    cfg.enable_homeostatic = true;
    cfg.enable_cortical_stdp = true;
    cfg.enable_predictive_coding = true;
    cfg.enable_lhb = true;
    cfg.enable_amygdala = true;
    cfg.enable_synaptic_consolidation = true;
    cfg.enable_replay = true;
    cfg.enable_interleaved_replay = true;
    cfg.enable_negative_replay = true;
    cfg.enable_sleep_consolidation = true;
    cfg.enable_lc_ne = true;
    cfg.enable_nbm_ach = true;
    cfg.enable_drn_5ht = true;
    cfg.enable_nacc = true;
    cfg.enable_snc = true;
    cfg.enable_sc = true;
    cfg.enable_pag = true;
    cfg.enable_fpc = true;
    cfg.enable_ofc = true;
    cfg.enable_vmpfc = true;
    cfg.enable_acc = true;

    return cfg;
}

// =============================================================================
// 诊断: 打印发育过程
// =============================================================================

std::string Developer::development_report(const DevGenome& genome) {
    AgentConfig cfg = to_agent_config(genome);
    std::ostringstream ss;

    ss << "=== 发育报告 ===\n";

    // 增殖
    ss << "\n--- 增殖 (区域大小) ---\n";
    for (int i = 0; i < 5; ++i) {
        const char* names[] = {"感觉", "运动", "前额叶", "皮层下", "调质"};
        int n = 1 << static_cast<int>(genome.division_rounds[i].value);
        ss << "  " << names[i] << ": " << n << " 细胞 (分裂 "
           << static_cast<int>(genome.division_rounds[i].value) << " 轮)\n";
    }
    ss << "  前后轴梯度: " << genome.growth_gradient[0].value
       << (genome.growth_gradient[0].value > 0 ? " (前部大=灵长类)" : " (后部大=啮齿类)") << "\n";

    // 计算结果
    ss << "\n--- 发育 → 参数 ---\n";
    char buf[128];
    snprintf(buf, sizeof(buf), "  V1 大小: %.2f, dlPFC 大小: %.2f, BG 大小: %.2f\n",
             cfg.v1_size_factor, cfg.dlpfc_size_factor, cfg.bg_size_factor);
    ss << buf;
    snprintf(buf, sizeof(buf), "  DA-STDP lr: %.4f, 皮层 STDP a+: %.4f\n",
             cfg.da_stdp_lr, cfg.cortical_stdp_a_plus);
    ss << buf;
    snprintf(buf, sizeof(buf), "  探索噪声: %.1f, BG→M1 增益: %.1f\n",
             cfg.exploration_noise, cfg.bg_to_m1_gain);
    ss << buf;
    snprintf(buf, sizeof(buf), "  奖赏缩放: %.2f, 重放次数: %d\n",
             cfg.reward_scale, cfg.replay_passes);
    ss << buf;
    snprintf(buf, sizeof(buf), "  稳态目标: %.1f, 发育期: %zu 步\n",
             cfg.homeostatic_target_rate, cfg.dev_period_steps);
    ss << buf;

    return ss.str();
}

} // namespace wuyun
