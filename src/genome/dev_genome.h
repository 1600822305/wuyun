#pragma once
/**
 * DevGenome — 发育基因组 (间接编码)
 *
 * 与直接编码 Genome 的区别:
 *   Genome (直接): 23 个浮点数 → to_agent_config() → 手工 build_brain()
 *   DevGenome (间接): ~80 个发育规则 → develop() → 大脑结构涌现
 *
 * 基因不编码"大脑长什么样"，而是编码"大脑怎么长出来":
 *   - 增殖基因: 分裂几轮 → 神经元数量涌现
 *   - 连接基因: 导向分子梯度 → 连接拓扑涌现
 *   - 分化基因: 转录因子 → 神经元类型涌现
 *
 * 生物学基础:
 *   - 基因组瓶颈 (Zador 2019): 20,000 基因 → 100 万亿突触
 *   - NDP (Nature 2023): 从单细胞通过局部通信生长功能网络
 *
 * Phase A: 增殖 + 距离连接 (~20 基因)
 * Phase B: + 导向分子 + 分化 (~50 基因)
 * Phase C: + 迁移 + 修剪 (~80 基因)
 */

#include "genome/genome.h"  // 复用 Gene 结构
#include <vector>
#include <random>

namespace wuyun {

class SimulationEngine;  // forward
struct GridWorldConfig;

// =============================================================================
// 区域类型 (发育中的细胞分类)
// =============================================================================

enum class RegionType : uint8_t {
    SENSORY   = 0,  // 感觉皮层 (V1, A1, S1...)
    MOTOR     = 1,  // 运动皮层 (M1, PMC...)
    PREFRONTAL= 2,  // 前额叶 (dlPFC, OFC, FPC...)
    SUBCORTICAL=3,  // 皮层下 (BG, Thalamus...)
    NEUROMOD  = 4,  // 调质核团 (VTA, LC, DRN...)
    N_TYPES   = 5
};

// =============================================================================
// DevGenome: 发育基因组
// =============================================================================

struct DevGenome {
    // ===================== Phase A: 增殖 + 连接 =====================

    // --- 增殖基因: 控制每种区域类型的神经元数量 ---
    // division_rounds[type] → 2^rounds 个神经元
    // 例: division_rounds[SENSORY]=5 → 2^5=32 个感觉神经元
    Gene division_rounds[5] = {
        {"div_sensory",     5.0f, 3.0f, 8.0f},   // 8~256 neurons
        {"div_motor",       4.0f, 3.0f, 7.0f},   // 8~128 neurons
        {"div_prefrontal",  4.0f, 3.0f, 7.0f},
        {"div_subcortical", 4.0f, 3.0f, 7.0f},
        {"div_neuromod",    3.0f, 2.0f, 5.0f},   // 4~32 neurons
    };

    // 抑制性神经元比例 (全局)
    Gene inhibitory_prob {"inh_prob", 0.20f, 0.10f, 0.35f};

    // --- 连接基因: 控制区域间连接 ---
    // connection_radius: 距离内的细胞有概率形成突触
    Gene connection_radius {"conn_radius", 0.3f, 0.1f, 0.6f};

    // 同类型内连接概率 (循环连接)
    Gene recurrent_prob {"recurrent_prob", 0.15f, 0.05f, 0.40f};

    // 跨类型连接概率矩阵 (5×5 = 25 个基因)
    // cross_prob[src_type * 5 + dst_type]
    Gene cross_connect[25] = {
        // →SENS  →MOT   →PFC   →SUB   →NMOD   (src↓)
        {"s2s", 0.15f, 0.02f, 0.50f}, {"s2m", 0.05f, 0.01f, 0.20f},
        {"s2p", 0.10f, 0.02f, 0.30f}, {"s2b", 0.08f, 0.01f, 0.25f},
        {"s2n", 0.03f, 0.01f, 0.15f},
        // MOT→
        {"m2s", 0.05f, 0.01f, 0.20f}, {"m2m", 0.10f, 0.02f, 0.30f},
        {"m2p", 0.08f, 0.01f, 0.25f}, {"m2b", 0.10f, 0.02f, 0.30f},
        {"m2n", 0.05f, 0.01f, 0.20f},
        // PFC→
        {"p2s", 0.08f, 0.01f, 0.25f}, {"p2m", 0.10f, 0.02f, 0.30f},
        {"p2p", 0.12f, 0.02f, 0.35f}, {"p2b", 0.15f, 0.02f, 0.40f},
        {"p2n", 0.05f, 0.01f, 0.20f},
        // SUB→
        {"b2s", 0.05f, 0.01f, 0.20f}, {"b2m", 0.10f, 0.02f, 0.30f},
        {"b2p", 0.08f, 0.01f, 0.25f}, {"b2b", 0.10f, 0.02f, 0.30f},
        {"b2n", 0.05f, 0.01f, 0.20f},
        // NMOD→
        {"n2s", 0.08f, 0.01f, 0.25f}, {"n2m", 0.05f, 0.01f, 0.20f},
        {"n2p", 0.08f, 0.01f, 0.25f}, {"n2b", 0.05f, 0.01f, 0.20f},
        {"n2n", 0.03f, 0.01f, 0.15f},
    };

    // --- 学习基因: DA-STDP 和稳态 ---
    Gene da_stdp_lr      {"da_lr",     0.05f, 0.005f, 0.15f};
    Gene homeostatic_eta {"homeo_eta", 0.005f, 0.0001f, 0.01f};
    Gene homeostatic_target {"homeo_target", 8.0f, 1.0f, 15.0f};

    // --- 感觉/运动接口基因 ---
    Gene sensory_gain    {"sens_gain",  200.0f, 50.0f, 500.0f};
    Gene motor_noise     {"mot_noise",  40.0f, 10.0f, 100.0f};
    Gene reward_gain     {"rew_gain",   100.0f, 20.0f, 300.0f};

    // ===================== Phase B: 导向分子 + 分化 =====================

    // 8 种导向分子, 每种 4 个参数 (cx, cy, sigma, amplitude) = 32 基因
    // 生物学: Netrin(吸引), Slit(排斥), Ephrin(拓扑), Semaphorin(层特异)
    Gene guidance_cx[8] = {
        {"g0_cx",0.5f,0.0f,1.0f}, {"g1_cx",0.2f,0.0f,1.0f},
        {"g2_cx",0.8f,0.0f,1.0f}, {"g3_cx",0.5f,0.0f,1.0f},
        {"g4_cx",0.3f,0.0f,1.0f}, {"g5_cx",0.7f,0.0f,1.0f},
        {"g6_cx",0.5f,0.0f,1.0f}, {"g7_cx",0.5f,0.0f,1.0f},
    };
    Gene guidance_cy[8] = {
        {"g0_cy",0.8f,0.0f,1.0f}, {"g1_cy",0.3f,0.0f,1.0f},
        {"g2_cy",0.3f,0.0f,1.0f}, {"g3_cy",0.5f,0.0f,1.0f},
        {"g4_cy",0.6f,0.0f,1.0f}, {"g5_cy",0.6f,0.0f,1.0f},
        {"g6_cy",0.1f,0.0f,1.0f}, {"g7_cy",0.9f,0.0f,1.0f},
    };
    Gene guidance_sigma[8] = {
        {"g0_s",0.2f,0.05f,0.5f}, {"g1_s",0.15f,0.05f,0.5f},
        {"g2_s",0.15f,0.05f,0.5f}, {"g3_s",0.25f,0.05f,0.5f},
        {"g4_s",0.2f,0.05f,0.5f}, {"g5_s",0.2f,0.05f,0.5f},
        {"g6_s",0.15f,0.05f,0.5f}, {"g7_s",0.15f,0.05f,0.5f},
    };
    Gene guidance_amp[8] = {
        {"g0_a",1.0f,0.1f,2.0f}, {"g1_a",1.0f,0.1f,2.0f},
        {"g2_a",0.8f,0.1f,2.0f}, {"g3_a",0.6f,0.1f,2.0f},
        {"g4_a",0.7f,0.1f,2.0f}, {"g5_a",0.7f,0.1f,2.0f},
        {"g6_a",1.0f,0.1f,2.0f}, {"g7_a",0.5f,0.1f,2.0f},
    };
    // 吸引/排斥: >0.5=吸引, <0.5=排斥
    Gene guidance_attract[8] = {
        {"g0_at",0.8f,0.0f,1.0f}, {"g1_at",0.7f,0.0f,1.0f},
        {"g2_at",0.3f,0.0f,1.0f}, {"g3_at",0.6f,0.0f,1.0f},
        {"g4_at",0.2f,0.0f,1.0f}, {"g5_at",0.5f,0.0f,1.0f},
        {"g6_at",0.9f,0.0f,1.0f}, {"g7_at",0.4f,0.0f,1.0f},
    };

    // 受体表达: 5 种区域类型 × 8 种分子 = 40 基因
    // receptor_expr[type * 8 + molecule] = 该区域类型对该分子的敏感度
    Gene receptor_expr[40] = {
        // SENSORY 对 8 种分子的受体
        {"s_r0",0.8f,0.0f,1.0f},{"s_r1",0.3f,0.0f,1.0f},{"s_r2",0.2f,0.0f,1.0f},{"s_r3",0.5f,0.0f,1.0f},
        {"s_r4",0.1f,0.0f,1.0f},{"s_r5",0.4f,0.0f,1.0f},{"s_r6",0.6f,0.0f,1.0f},{"s_r7",0.7f,0.0f,1.0f},
        // MOTOR 对 8 种分子的受体
        {"m_r0",0.3f,0.0f,1.0f},{"m_r1",0.7f,0.0f,1.0f},{"m_r2",0.5f,0.0f,1.0f},{"m_r3",0.4f,0.0f,1.0f},
        {"m_r4",0.6f,0.0f,1.0f},{"m_r5",0.2f,0.0f,1.0f},{"m_r6",0.3f,0.0f,1.0f},{"m_r7",0.1f,0.0f,1.0f},
        // PFC 对 8 种分子的受体
        {"p_r0",0.2f,0.0f,1.0f},{"p_r1",0.5f,0.0f,1.0f},{"p_r2",0.7f,0.0f,1.0f},{"p_r3",0.8f,0.0f,1.0f},
        {"p_r4",0.3f,0.0f,1.0f},{"p_r5",0.6f,0.0f,1.0f},{"p_r6",0.4f,0.0f,1.0f},{"p_r7",0.2f,0.0f,1.0f},
        // SUB 对 8 种分子的受体
        {"b_r0",0.5f,0.0f,1.0f},{"b_r1",0.4f,0.0f,1.0f},{"b_r2",0.3f,0.0f,1.0f},{"b_r3",0.6f,0.0f,1.0f},
        {"b_r4",0.8f,0.0f,1.0f},{"b_r5",0.5f,0.0f,1.0f},{"b_r6",0.2f,0.0f,1.0f},{"b_r7",0.3f,0.0f,1.0f},
        // NMOD 对 8 种分子的受体
        {"n_r0",0.4f,0.0f,1.0f},{"n_r1",0.6f,0.0f,1.0f},{"n_r2",0.4f,0.0f,1.0f},{"n_r3",0.3f,0.0f,1.0f},
        {"n_r4",0.5f,0.0f,1.0f},{"n_r5",0.3f,0.0f,1.0f},{"n_r6",0.5f,0.0f,1.0f},{"n_r7",0.6f,0.0f,1.0f},
    };

    // 分化梯度: DA/NMDA 受体密度随前后轴变化
    Gene da_gradient  {"da_grad",  0.5f, -1.0f, 1.0f};  // >0: 前部 DA 高 (前额叶样)
    Gene nmda_gradient{"nmda_grad",0.3f, -1.0f, 1.0f};  // >0: 后部 NMDA 高 (感觉样)

    // ===================== Phase C: 修剪 + 关键期 =====================
    Gene pruning_threshold {"prune_thr", 0.3f, 0.1f, 0.8f};   // 低于此活动水平的突触被修剪
    Gene critical_period   {"crit_per",  50.0f, 10.0f, 200.0f}; // 自发活动步数 (关键期长度)
    Gene spontaneous_rate  {"spont_rate", 0.1f, 0.02f, 0.3f};   // 自发活动强度

    // --- 元数据 ---
    float fitness = 0.0f;
    int   generation = 0;

    // ===================== 操作 =====================

    std::vector<Gene*> all_genes();
    std::vector<const Gene*> all_genes() const;
    size_t n_genes() const;

    void randomize(std::mt19937& rng);
    void mutate(std::mt19937& rng, float mutation_rate = 0.15f, float sigma = 0.1f);
    static DevGenome crossover(const DevGenome& a, const DevGenome& b, std::mt19937& rng);

    std::string summary() const;
    std::string to_json() const;
};

} // namespace wuyun
