#include "development/developer.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cstdio>

namespace wuyun {

// =============================================================================
// to_agent_config: DevGenome → AgentConfig
//
// 固定回路: 基因控制大小/增益 (内部拓扑继承 build_brain)
// 皮层: 条形码决定哪些皮层区域被激活和连接强度
//
// 当前 ClosedLoopAgent 有固定的皮层区域名称 (V1,V2,V4,IT,dlPFC,M1)
// 暂时将 5 种可进化皮层类型映射到这些固定名称:
//   ctx0 → V1 (第一层感觉处理)
//   ctx1 → V2/V4 (中间处理)
//   ctx2 → IT (高级表征)
//   ctx3 → dlPFC (决策)
//   ctx4 → FPC (规划)
//
// 未来: ClosedLoopAgent 支持动态皮层区域后, 可以按条形码创建任意数量
// =============================================================================

AgentConfig Developer::to_agent_config(const DevGenome& genome) {
    AgentConfig cfg;

    // =====================================================================
    // 固定回路参数 (继承 build_brain, 只改大小/增益)
    // =====================================================================

    cfg.bg_size_factor = std::clamp(genome.bg_size.value, 0.5f, 2.0f);
    cfg.da_stdp_lr = std::clamp(genome.da_stdp_lr.value, 0.005f, 0.15f);
    cfg.bg_to_m1_gain = std::clamp(genome.bg_gain.value, 2.0f, 20.0f);

    cfg.lgn_gain = std::clamp(genome.lgn_gain.value, 50.0f, 500.0f);
    cfg.lgn_baseline = std::clamp(genome.lgn_baseline.value, 1.0f, 20.0f);
    cfg.lgn_noise_amp = 3.0f;

    cfg.exploration_noise = std::clamp(genome.motor_noise.value, 10.0f, 100.0f);
    cfg.reward_scale = std::clamp(genome.reward_scale.value, 0.5f, 5.0f);

    cfg.homeostatic_target_rate = std::clamp(genome.homeo_target.value, 1.0f, 15.0f);
    cfg.homeostatic_eta = std::clamp(genome.homeo_eta.value, 0.0001f, 0.01f);

    cfg.ne_floor = std::clamp(genome.ne_floor.value, 0.3f, 1.0f);
    cfg.replay_passes = std::max(1, static_cast<int>(genome.replay_passes.value));
    cfg.dev_period_steps = std::max<size_t>(0, static_cast<size_t>(genome.dev_period.value));

    // =====================================================================
    // 皮层大小: 条形码 + 分裂轮数 → 区域大小因子
    // =====================================================================

    // ctx0 → V1: LGN 条形码兼容性决定 V1 大小权重
    float lgn_compat[N_CORTICAL_TYPES];
    float bg_compat[N_CORTICAL_TYPES];
    for (int t = 0; t < N_CORTICAL_TYPES; ++t) {
        float bc[BARCODE_DIM];
        for (int d = 0; d < BARCODE_DIM; ++d)
            bc[d] = genome.cortical_barcode[t][d].value;

        lgn_compat[t] = genome.barcode_compat(DevGenome::LGN_BARCODE, bc);
        float bg_bc[BARCODE_DIM];
        for (int d = 0; d < BARCODE_DIM; ++d)
            bg_bc[d] = genome.cortical_to_bg[d].value;
        bg_compat[t] = genome.barcode_compat(bc, bg_bc);
    }

    // 找到与 LGN 最兼容的皮层类型 → 映射到 V1
    // 找到与 BG 最兼容的皮层类型 → 映射到 dlPFC
    int best_sensory = 0, best_motor = 0;
    for (int t = 0; t < N_CORTICAL_TYPES; ++t) {
        if (lgn_compat[t] > lgn_compat[best_sensory]) best_sensory = t;
        if (bg_compat[t] > bg_compat[best_motor]) best_motor = t;
    }

    // V1 大小: 由最兼容感觉类型的分裂轮数决定
    float v1_div = genome.cortical_division[best_sensory].value;
    cfg.v1_size_factor = std::clamp(std::pow(2.0f, v1_div) / 16.0f, 0.5f, 3.0f);

    // dlPFC 大小: 由最兼容决策类型决定
    float pfc_div = genome.cortical_division[best_motor].value;
    cfg.dlpfc_size_factor = std::clamp(std::pow(2.0f, pfc_div) / 16.0f, 0.5f, 3.0f);

    // =====================================================================
    // 皮层 STDP 参数: 从皮层类型的兼容性强度推导
    // 高皮层间兼容性 → 强侧向连接 → 需要更保守的 STDP
    // =====================================================================

    float avg_cortical_compat = 0.0f;
    int cc = 0;
    for (int a = 0; a < N_CORTICAL_TYPES; ++a) {
        for (int b = 0; b < N_CORTICAL_TYPES; ++b) {
            if (a == b) continue;
            float bc_a[BARCODE_DIM], bc_b[BARCODE_DIM];
            for (int d = 0; d < BARCODE_DIM; ++d) {
                bc_a[d] = genome.cortical_barcode[a][d].value;
                bc_b[d] = genome.cortical_barcode[b][d].value;
            }
            avg_cortical_compat += genome.barcode_compat(bc_a, bc_b);
            cc++;
        }
    }
    avg_cortical_compat /= std::max(1, cc);

    // 高兼容性 → 更多连接 → 需要更小的 STDP 步长
    float stdp_scale = std::clamp(1.0f / (1.0f + avg_cortical_compat * 0.5f), 0.3f, 2.0f);
    cfg.cortical_stdp_a_plus = std::clamp(0.003f * stdp_scale, 0.001f, 0.02f);
    cfg.cortical_stdp_a_minus = -std::abs(cfg.cortical_stdp_a_plus) * 1.5f;
    cfg.cortical_stdp_w_max = 1.5f;

    // =====================================================================
    // 其他参数: 继承合理默认值
    // =====================================================================

    cfg.brain_steps_per_action = 12;
    cfg.reward_processing_steps = 9;
    cfg.attractor_drive_ratio = 0.5f;
    cfg.background_drive_ratio = 0.05f;
    cfg.ne_food_scale = 4.0f;
    cfg.replay_da_scale = 0.5f;

    // 全部模块启用 (49 步成果)
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
    cfg.enable_sleep_consolidation = false;  // 短评估不用睡眠
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
// 连通性检查
// =============================================================================

int Developer::check_connectivity(const DevGenome& genome) {
    // 检查: 有多少皮层类型同时兼容 LGN(输入) 和 BG(输出)
    // 至少 1 个 → 信号可以从感觉到运动
    int connected = 0;
    for (int t = 0; t < N_CORTICAL_TYPES; ++t) {
        float bc[BARCODE_DIM];
        for (int d = 0; d < BARCODE_DIM; ++d)
            bc[d] = genome.cortical_barcode[t][d].value;

        float lgn_c = genome.barcode_compat(DevGenome::LGN_BARCODE, bc);
        float lgn_p = genome.conn_prob_from_compat(lgn_c);

        float bg_bc[BARCODE_DIM];
        for (int d = 0; d < BARCODE_DIM; ++d)
            bg_bc[d] = genome.cortical_to_bg[d].value;
        float bg_c = genome.barcode_compat(bc, bg_bc);
        float bg_p = genome.conn_prob_from_compat(bg_c);

        // 如果与 LGN 和 BG 都有 >30% 连接概率 → 可连通
        if (lgn_p > 0.3f && bg_p > 0.3f) {
            connected++;
        }
    }
    return connected;
}

// =============================================================================
// 诊断报告
// =============================================================================

std::string Developer::development_report(const DevGenome& genome) {
    std::ostringstream ss;
    ss << "=== 发育报告 (v3: 骨架固定+皮层涌现) ===\n\n";

    // 固定回路
    ss << "--- 固定回路 ---\n";
    char buf[128];
    snprintf(buf, sizeof(buf), "  BG: size=%.2f, DA lr=%.4f, gain=%.1f\n",
             genome.bg_size.value, genome.da_stdp_lr.value, genome.bg_gain.value);
    ss << buf;
    snprintf(buf, sizeof(buf), "  LGN: gain=%.0f, base=%.1f\n",
             genome.lgn_gain.value, genome.lgn_baseline.value);
    ss << buf;
    snprintf(buf, sizeof(buf), "  Motor: noise=%.0f\n", genome.motor_noise.value);
    ss << buf;

    // 皮层类型
    ss << "\n--- 皮层类型 (条形码) ---\n";
    for (int t = 0; t < N_CORTICAL_TYPES; ++t) {
        int n = 1 << std::clamp(static_cast<int>(genome.cortical_division[t].value), 2, 7);
        ss << "  ctx" << t << " (" << n << "n): [";
        for (int d = 0; d < BARCODE_DIM; ++d) {
            if (d > 0) ss << ",";
            snprintf(buf, sizeof(buf), "%.2f", genome.cortical_barcode[t][d].value);
            ss << buf;
        }
        ss << "]\n";
    }

    // 兼容性
    ss << "\n--- 连接兼容性 ---\n";
    for (int a = 0; a < N_CORTICAL_TYPES; ++a) {
        for (int b = 0; b < N_CORTICAL_TYPES; ++b) {
            float bc_a[BARCODE_DIM], bc_b[BARCODE_DIM];
            for (int d = 0; d < BARCODE_DIM; ++d) {
                bc_a[d] = genome.cortical_barcode[a][d].value;
                bc_b[d] = genome.cortical_barcode[b][d].value;
            }
            float p = genome.conn_prob_from_compat(genome.barcode_compat(bc_a, bc_b));
            snprintf(buf, sizeof(buf), "%3.0f%%", p * 100.0f);
            ss << buf << " ";
        }
        ss << "  ← ctx" << a << "\n";
    }

    // LGN→皮层 兼容性
    ss << "\n--- LGN → 皮层 ---\n";
    for (int t = 0; t < N_CORTICAL_TYPES; ++t) {
        float bc[BARCODE_DIM];
        for (int d = 0; d < BARCODE_DIM; ++d)
            bc[d] = genome.cortical_barcode[t][d].value;
        float p = genome.conn_prob_from_compat(
            genome.barcode_compat(DevGenome::LGN_BARCODE, bc));
        snprintf(buf, sizeof(buf), "  LGN→ctx%d: %3.0f%%\n", t, p * 100.0f);
        ss << buf;
    }

    // 皮层→BG 兼容性
    ss << "\n--- 皮层 → BG ---\n";
    for (int t = 0; t < N_CORTICAL_TYPES; ++t) {
        float bc[BARCODE_DIM], bg_bc[BARCODE_DIM];
        for (int d = 0; d < BARCODE_DIM; ++d) {
            bc[d] = genome.cortical_barcode[t][d].value;
            bg_bc[d] = genome.cortical_to_bg[d].value;
        }
        float p = genome.conn_prob_from_compat(genome.barcode_compat(bc, bg_bc));
        snprintf(buf, sizeof(buf), "  ctx%d→BG: %3.0f%%\n", t, p * 100.0f);
        ss << buf;
    }

    // 连通性
    int conn = check_connectivity(genome);
    snprintf(buf, sizeof(buf), "\n连通皮层类型: %d/%d\n", conn, N_CORTICAL_TYPES);
    ss << buf;

    return ss.str();
}

} // namespace wuyun
