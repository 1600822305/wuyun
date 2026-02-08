#include "development/developer.h"
#include "development/guidance.h"
#include "region/cortical_region.h"
#include "region/subcortical/basal_ganglia.h"
#include "region/subcortical/thalamic_relay.h"
#include "region/neuromod/vta_da.h"
#include "region/limbic/hypothalamus.h"
#include "core/types.h"
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace wuyun {

// Static storage for diagnostic access
std::vector<NeuralCell> Developer::cells_;
std::vector<DevelopedRegion> Developer::regions_;
std::vector<DevelopedConnection> Developer::connections_;

const std::vector<NeuralCell>& Developer::last_cells() { return cells_; }
const std::vector<DevelopedRegion>& Developer::last_regions() { return regions_; }
const std::vector<DevelopedConnection>& Developer::last_connections() { return connections_; }

// =============================================================================
// Phase 1: Proliferation — 祖细胞分裂产生神经元
// =============================================================================

void Developer::proliferate(const DevGenome& genome, std::mt19937& rng) {
    cells_.clear();

    // 5 种区域类型, 每种有自己的分裂轮数
    // 区域在 2D 空间中的大致位置 (归一化 [0,1]):
    //   SENSORY:    后部 (y=0.8~1.0)  — 视觉/听觉/体觉皮层
    //   MOTOR:      前上部 (y=0.2~0.4) — 运动皮层
    //   PREFRONTAL: 最前部 (y=0.0~0.2) — 前额叶
    //   SUBCORTICAL: 中下部 (y=0.4~0.6, x=0.3~0.7) — 基底节/丘脑
    //   NEUROMOD:    中心 (y=0.5~0.7, x=0.4~0.6) — VTA/LC/DRN
    struct RegionSpec {
        float cx, cy;     // 中心
        float spread;     // 扩散范围
    };
    RegionSpec specs[5] = {
        {0.5f, 0.85f, 0.15f},  // SENSORY: 后部
        {0.5f, 0.30f, 0.12f},  // MOTOR: 前上
        {0.5f, 0.10f, 0.10f},  // PREFRONTAL: 最前
        {0.5f, 0.50f, 0.10f},  // SUBCORTICAL: 中部
        {0.5f, 0.60f, 0.06f},  // NEUROMOD: 中心偏下
    };

    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    for (int type = 0; type < 5; ++type) {
        int rounds = std::clamp(static_cast<int>(genome.division_rounds[type].value), 2, 8);
        int n_cells = 1 << rounds;  // 2^rounds
        auto& spec = specs[type];

        std::normal_distribution<float> dx(spec.cx, spec.spread);
        std::normal_distribution<float> dy(spec.cy, spec.spread);

        for (int i = 0; i < n_cells; ++i) {
            NeuralCell cell;
            cell.x = std::clamp(dx(rng), 0.0f, 1.0f);
            cell.y = std::clamp(dy(rng), 0.0f, 1.0f);
            cell.region_type = static_cast<RegionType>(type);
            cell.is_inhibitory = (coin(rng) < genome.inhibitory_prob.value);
            cell.birth_order = static_cast<int>(cells_.size());
            // Phase B: 受体表达来自基因组 (区域类型决定)
            for (int m = 0; m < 8; ++m) {
                cell.receptors[m] = genome.receptor_expr[type * 8 + m].value;
            }
            cells_.push_back(cell);
        }
    }
}

// =============================================================================
// Phase 2: Region Assignment — 按空间聚类分配区域
// =============================================================================

void Developer::assign_regions(const DevGenome& genome) {
    regions_.clear();
    (void)genome;  // Phase A: 直接按 region_type 分组

    // 按 region_type 分组, 每种类型创建一个区域
    const char* names[5] = {"Sensory", "Motor", "Prefrontal", "Subcortical", "Neuromod"};

    for (int type = 0; type < 5; ++type) {
        DevelopedRegion reg;
        reg.name = names[type];
        reg.type = static_cast<RegionType>(type);
        reg.center_x = 0.0f;
        reg.center_y = 0.0f;
        int count = 0;

        for (size_t i = 0; i < cells_.size(); ++i) {
            if (static_cast<int>(cells_[i].region_type) == type) {
                reg.cell_indices.push_back(static_cast<int>(i));
                cells_[i].region_index = static_cast<int>(regions_.size());
                reg.center_x += cells_[i].x;
                reg.center_y += cells_[i].y;
                count++;
                if (cells_[i].is_inhibitory) reg.n_inhibitory++;
                else reg.n_excitatory++;
            }
        }
        if (count > 0) {
            reg.center_x /= static_cast<float>(count);
            reg.center_y /= static_cast<float>(count);
        }
        regions_.push_back(reg);
    }
}

// =============================================================================
// Phase 3: Connection Formation — 距离 + 类型概率
// =============================================================================

void Developer::form_connections(const DevGenome& genome, std::mt19937& rng) {
    connections_.clear();

    // --- Phase B: 导向分子场驱动的轴突导向 ---
    // 每个神经元伸出轴突 (growth cone), 沿导向分子梯度生长
    // 轴突到达目标细胞附近时形成突触
    // 连接拓扑从化学梯度涌现, 不从距离/概率表硬编码

    // 1. 构建导向分子场
    GuidanceField field;
    for (int m = 0; m < 8; ++m) {
        field.set_molecule(m,
            genome.guidance_cx[m].value,
            genome.guidance_cy[m].value,
            genome.guidance_sigma[m].value,
            genome.guidance_amp[m].value,
            genome.guidance_attract[m].value > 0.5f);
    }

    // 2. 模拟轴突生长
    float conn_radius = genome.connection_radius.value;
    float recurrent_p = genome.recurrent_prob.value;
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.02f);  // 轴突生长噪声

    size_t n_reg = regions_.size();
    std::vector<int> conn_count(n_reg * n_reg, 0);

    constexpr int GROWTH_STEPS = 10;   // 轴突生长步数
    constexpr float STEP_SIZE = 0.05f; // 每步移动距离
    constexpr float SYNAPSE_DIST = 0.08f; // 突触形成距离

    for (size_t i = 0; i < cells_.size(); ++i) {
        // 每个细胞伸出一根轴突, 沿梯度生长
        float ax = cells_[i].x;
        float ay = cells_[i].y;

        for (int step = 0; step < GROWTH_STEPS; ++step) {
            // 计算导向力 (基于该细胞的受体表达)
            float fx, fy;
            field.compute_guidance_force(ax, ay, cells_[i].receptors, fx, fy);

            // 加噪声 (轴突生长不完全精确)
            fx += noise(rng);
            fy += noise(rng);

            // 移动轴突
            ax += fx * STEP_SIZE;
            ay += fy * STEP_SIZE;
            ax = std::clamp(ax, 0.0f, 1.0f);
            ay = std::clamp(ay, 0.0f, 1.0f);

            // 检查附近的细胞, 形成突触
            for (size_t j = 0; j < cells_.size(); ++j) {
                if (i == j) continue;
                float dx = ax - cells_[j].x;
                float dy = ay - cells_[j].y;
                float dist = std::sqrt(dx * dx + dy * dy);

                if (dist < SYNAPSE_DIST) {
                    int src_reg = cells_[i].region_index;
                    int dst_reg = cells_[j].region_index;

                    // 同区域: 用循环连接概率
                    // 跨区域: 轴突到达 = 形成突触 (导向已经选择了方向)
                    float prob = (src_reg == dst_reg) ? recurrent_p : 0.5f;
                    if (coin(rng) < prob) {
                        conn_count[src_reg * n_reg + dst_reg]++;
                    }
                }
            }
        }
    }

    // 3. 转换为 DevelopedConnection
    for (size_t s = 0; s < n_reg; ++s) {
        for (size_t d = 0; d < n_reg; ++d) {
            int count = conn_count[s * n_reg + d];
            if (count > 0) {
                DevelopedConnection conn;
                conn.src_region = static_cast<int>(s);
                conn.dst_region = static_cast<int>(d);
                conn.n_synapses = count;
                float reg_dist = std::sqrt(
                    std::pow(regions_[s].center_x - regions_[d].center_x, 2.0f) +
                    std::pow(regions_[s].center_y - regions_[d].center_y, 2.0f));
                conn.delay = (s == d) ? 1 : (reg_dist < 0.3f ? 2 : 3);
                connections_.push_back(conn);
            }
        }
    }
}

// =============================================================================
// Phase 4: Assembly — 将发育结果转换为 SimulationEngine
// =============================================================================

SimulationEngine Developer::assemble(const DevGenome& genome, size_t vision_pixels) {
    SimulationEngine engine(10);  // max_delay=10

    // --- 创建 BrainRegion 实例 ---
    // Phase A: 每种区域类型创建一个 CorticalRegion (简化)
    // 将来 Phase B/C 会按分化结果创建不同类型的区域

    for (auto& reg : regions_) {
        if (reg.cell_indices.empty()) continue;

        size_t n_exc = static_cast<size_t>(reg.n_excitatory);
        size_t n_inh = static_cast<size_t>(reg.n_inhibitory);
        size_t n_total = n_exc + n_inh;

        if (reg.type == RegionType::SUBCORTICAL) {
            // 皮层下 → BasalGanglia (需要 D1/D2 MSN 结构)
            BasalGangliaConfig bg_cfg;
            bg_cfg.name = reg.name;
            size_t half = std::max<size_t>(2, n_exc / 2);
            bg_cfg.n_d1_msn = half;
            bg_cfg.n_d2_msn = half;
            bg_cfg.n_gpi = std::max<size_t>(2, n_inh / 2);
            bg_cfg.n_gpe = std::max<size_t>(2, n_inh / 2);
            bg_cfg.n_stn = std::max<size_t>(2, n_inh / 3);
            engine.add_region(std::make_unique<BasalGanglia>(bg_cfg));

        } else if (reg.type == RegionType::NEUROMOD) {
            // 调质核团 → VTA_DA
            VTAConfig vta_cfg;
            vta_cfg.name = reg.name;
            vta_cfg.n_da_neurons = std::max<size_t>(4, n_total);
            engine.add_region(std::make_unique<VTA_DA>(vta_cfg));

        } else {
            // 皮层区域 (感觉/运动/前额叶) → CorticalRegion
            ColumnConfig col_cfg;
            // 按 n_total 分配层大小 (大致 L4:L23:L5:L6 = 2:3:2:1)
            size_t base = std::max<size_t>(2, n_exc / 8);
            col_cfg.n_l4_stellate   = std::max<size_t>(2, base * 2);
            col_cfg.n_l23_pyramidal = std::max<size_t>(2, base * 3);
            col_cfg.n_l5_pyramidal  = std::max<size_t>(2, base * 2);
            col_cfg.n_l6_pyramidal  = std::max<size_t>(2, base);
            col_cfg.n_pv_basket     = std::max<size_t>(1, n_inh / 3);
            col_cfg.n_sst_martinotti= std::max<size_t>(1, n_inh / 3);
            col_cfg.n_vip           = std::max<size_t>(1, n_inh / 3);

            // 感觉区域: 确保 L4 能接收 vision_pixels 的输入
            if (reg.type == RegionType::SENSORY) {
                col_cfg.n_l4_stellate = std::max(col_cfg.n_l4_stellate, vision_pixels / 3);
            }

            engine.add_region(std::make_unique<CorticalRegion>(reg.name, col_cfg));
        }
    }

    // --- 感觉输入节点: LGN ---
    {
        ThalamicConfig lgn_cfg;
        lgn_cfg.name = "LGN";
        lgn_cfg.n_relay = std::max<size_t>(4, vision_pixels);
        lgn_cfg.n_trn = 2;
        engine.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));
    }

    // --- 运动输出节点: MotorThal ---
    {
        ThalamicConfig mt_cfg;
        mt_cfg.name = "MotorThal";
        mt_cfg.n_relay = 8;
        mt_cfg.n_trn = 2;
        engine.add_region(std::make_unique<ThalamicRelay>(mt_cfg));
    }

    // --- 奖赏感觉接口: Hypothalamus ---
    {
        HypothalamusConfig hypo_cfg;
        hypo_cfg.name = "Hypothalamus";
        hypo_cfg.n_lh = 4;
        hypo_cfg.n_pvn = 4;
        hypo_cfg.n_scn = 4;
        hypo_cfg.n_vlpo = 4;
        hypo_cfg.n_orexin = 4;
        hypo_cfg.n_vmh = 4;
        engine.add_region(std::make_unique<Hypothalamus>(hypo_cfg));
    }

    // --- 添加发育产生的投射 ---
    for (auto& conn : connections_) {
        if (conn.n_synapses < 1) continue;
        if (conn.src_region < 0 || conn.src_region >= (int)regions_.size()) continue;
        if (conn.dst_region < 0 || conn.dst_region >= (int)regions_.size()) continue;

        const std::string& src = regions_[conn.src_region].name;
        const std::string& dst = regions_[conn.dst_region].name;

        // 跳过自连接 (CorticalRegion 内部已有循环连接)
        if (src == dst) continue;

        engine.add_projection(src, dst, conn.delay);
    }

    // --- 固定投射: 感觉/运动/奖赏通路 ---
    // 这些是"先天布线" — 解剖学事实, 不需要从发育中涌现
    engine.add_projection("LGN", "Sensory", 2);          // 视觉输入
    engine.add_projection("Sensory", "Prefrontal", 2);    // 感觉→决策
    engine.add_projection("Prefrontal", "Subcortical", 2); // 决策→动作选择
    engine.add_projection("Subcortical", "MotorThal", 2);  // BG→丘脑
    engine.add_projection("MotorThal", "Motor", 2);        // 丘脑→运动
    engine.add_projection("Hypothalamus", "Neuromod", 1);  // 奖赏→DA

    // VTA DA 注册
    engine.register_neuromod_source("Neuromod", SimulationEngine::NeuromodType::DA);

    return engine;
}

// =============================================================================
// 主入口: develop()
// =============================================================================

SimulationEngine Developer::develop(const DevGenome& genome,
                                     size_t vision_pixels,
                                     uint32_t seed) {
    std::mt19937 rng(seed);

    // 1. 增殖: 祖细胞分裂
    proliferate(genome, rng);

    // 2. 区域分配: 按类型分组
    assign_regions(genome);

    // 3. 轴突导向: 化学梯度驱动连接
    form_connections(genome, rng);

    // 4. 组装: NeuralCell → BrainRegion → SimulationEngine
    auto engine = assemble(genome, vision_pixels);

    // 5. Phase C: 活动依赖修剪 (关键期)
    // 运行自发活动, 然后修剪低活动投射
    // 生物学: 初始过度连接 → 自发活动 → 低活动突触消退 → 精炼后的连接模式
    {
        int crit_steps = std::max(10, static_cast<int>(genome.critical_period.value));
        for (int i = 0; i < crit_steps; ++i) {
            engine.step();
        }
        // 修剪在 SimulationEngine 级别不容易实现 (投射是引用, 不能删除)
        // Phase C 简化: 关键期自发活动让 homeostatic plasticity 调整权重
        // 真正的突触修剪需要 SynapseGroup 支持 remove_synapse(), 留作未来实现
    }

    return engine;
}

} // namespace wuyun
