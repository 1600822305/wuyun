#pragma once
/**
 * DevGenome v3 — 混合基因连接组
 *
 * 核心设计: 骨架固定, 皮层涌现
 *   固定回路 (~10 种): BG/丘脑/杏仁核/海马/VTA/LGN/M1/Hypothalamus
 *     内部拓扑写死 (3 亿年进化的产物, 不让 30 代重新发明)
 *     基因只控制: 大小/增益/学习率
 *   皮层涌现 (~5 种): 可进化皮层类型
 *     条形码由基因决定 (Barabasi 2019)
 *     皮层间连接从条形码兼容性涌现
 *     皮层↔固定回路的接口也用条形码匹配
 *
 * 效果:
 *   固定回路保证信号必达 (LGN→皮层→BG→M1, 每步 80-90% 有效)
 *   进化只需找最优皮层组合 (30 代可行)
 *   不丢弃 49 步成果 (BG 乘法增益, VTA RPE, 群体向量 等全保留)
 *
 * 生物学: 真实大脑也是这样——
 *   BG/丘脑/杏仁核/海马的基本回路在爬行动物就已固定
 *   皮层的面积分配和层间连接是哺乳类进化的主战场
 */

#include "genome/genome.h"  // Gene 结构
#include <vector>
#include <random>
#include <cmath>

namespace wuyun {

static constexpr int N_CORTICAL_TYPES = 5;   // 可进化皮层类型数
static constexpr int BARCODE_DIM = 8;         // 条形码维度

// =============================================================================
// DevGenome v3: 骨架固定 + 皮层涌现
// =============================================================================

struct DevGenome {
    // =====================================================================
    // 第一部分: 固定回路参数 (~30 基因)
    // 这些回路的"存在"和"内部拓扑"写死 (继承 build_brain)
    // 基因只控制大小/增益/学习率
    // =====================================================================

    // BG 基底节 (D1/D2/GPi/GPe/STN 内部连接写死)
    Gene bg_size       {"bg_size",     1.0f, 0.5f, 2.0f};
    Gene da_stdp_lr    {"da_lr",       0.05f, 0.005f, 0.15f};
    Gene bg_gain       {"bg_gain",     6.0f, 2.0f, 20.0f};

    // VTA DA (内部 RPE 计算写死)
    Gene vta_size      {"vta_size",    1.0f, 0.5f, 2.0f};
    Gene da_phasic_gain {"da_ph_gain", 0.5f, 0.1f, 1.5f};

    // 丘脑 (MotorThal, TRN 门控写死)
    Gene thal_size     {"thal_size",   1.0f, 0.5f, 2.0f};
    Gene thal_gate     {"thal_gate",   0.5f, 0.1f, 1.0f};

    // LGN 感觉中继
    Gene lgn_gain      {"lgn_gain",    200.0f, 50.0f, 500.0f};
    Gene lgn_baseline  {"lgn_base",    10.0f, 1.0f, 20.0f};

    // M1 运动输出
    Gene motor_noise   {"mot_noise",   40.0f, 10.0f, 100.0f};

    // Hypothalamus 奖赏感觉
    Gene reward_scale  {"rew_scale",   2.5f, 0.5f, 5.0f};

    // 杏仁核 (La→BLA→CeA 内部写死)
    Gene amyg_size     {"amyg_size",   1.0f, 0.5f, 2.0f};

    // 海马 (DG→CA3→CA1 内部写死)
    Gene hipp_size     {"hipp_size",   1.0f, 0.5f, 2.0f};

    // 稳态
    Gene homeo_target  {"homeo_tgt",   8.0f, 1.0f, 15.0f};
    Gene homeo_eta     {"homeo_eta",   0.005f, 0.0001f, 0.01f};

    // NE 探索
    Gene ne_floor      {"ne_floor",    0.6f, 0.3f, 1.0f};

    // 重放
    Gene replay_passes {"replay_n",    7.0f, 1.0f, 15.0f};

    // 发育期
    Gene dev_period    {"dev_per",     50.0f, 0.0f, 200.0f};

    // =====================================================================
    // 先验基因: 固定回路的初始连接强度
    // 先验 = 发育过程中产生的连接权重, 不是事后贴的标签
    // 生物学: TAS1R2/R3→NTS→PBN→LH→VTA 通路存在本身就是先验
    // =====================================================================
    Gene hedonic_gain  {"hedonic",  3.0f, 0.5f, 10.0f};  // Hypo→VTA 权重 (食物=好)
    Gene fear_valence  {"fear",    2.0f, 0.5f, 8.0f};    // CeA→VTA 抑制 (危险=坏)
    Gene sensory_motor {"sm_coup", 0.1f, 0.01f, 0.5f};   // 皮层→BG 初始权重
    Gene explore_drive {"expl_dr", 0.8f, 0.2f, 1.0f};    // 运动噪声倍数 (天生好奇)
    Gene approach_bias {"approach",0.05f, 0.0f, 0.2f};    // D1 初始偏置 (微弱趋近)

    // =====================================================================
    // 第二部分: 可进化皮层 (~85 基因)
    // 5 种皮层类型, 每种有条形码 + 大小 + 属性
    // =====================================================================

    // 皮层类型条形码: 5 × 8 = 40 基因
    Gene cortical_barcode[N_CORTICAL_TYPES][BARCODE_DIM];

    // 皮层类型大小: 5 基因
    Gene cortical_division[N_CORTICAL_TYPES];

    // 皮层类型抑制比: 5 基因
    Gene cortical_inh_frac[N_CORTICAL_TYPES];

    // =====================================================================
    // 第三部分: 连接兼容性 (~73 基因)
    // =====================================================================

    // 兼容性矩阵: 8×8 = 64 基因
    Gene w_connect[BARCODE_DIM][BARCODE_DIM];

    // 连接阈值
    Gene connect_threshold {"conn_thr", 0.5f, -1.0f, 2.0f};

    // 皮层→BG 接口条形码 (哪种皮层投射到 BG): 8 基因
    Gene cortical_to_bg[BARCODE_DIM];

    // LGN→皮层 接口条形码 (LGN 投射到哪种皮层): 不需要 (LGN 有固定条形码)

    // =====================================================================
    // 元数据
    // =====================================================================
    float fitness = 0.0f;
    int   generation = 0;

    // =====================================================================
    // 操作
    // =====================================================================
    DevGenome();

    std::vector<Gene*> all_genes();
    std::vector<const Gene*> all_genes() const;
    size_t n_genes() const;

    void randomize(std::mt19937& rng);
    void mutate(std::mt19937& rng, float mutation_rate = 0.15f, float sigma = 0.1f);
    static DevGenome crossover(const DevGenome& a, const DevGenome& b, std::mt19937& rng);

    std::string summary() const;
    std::string to_json() const;

    // =====================================================================
    // 条形码兼容性计算
    // =====================================================================

    // 固定类型条形码 (LGN/M1/VTA/Hypo — 不进化)
    static const float LGN_BARCODE[BARCODE_DIM];
    static const float BG_BARCODE[BARCODE_DIM];

    // 计算两个条形码之间的连接兼容性
    float barcode_compat(const float bc_a[BARCODE_DIM],
                         const float bc_b[BARCODE_DIM]) const;

    // 兼容性 → 连接概率
    float conn_prob_from_compat(float compat) const;
};

} // namespace wuyun
