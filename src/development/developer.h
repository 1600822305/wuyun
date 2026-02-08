#pragma once
/**
 * Developer — 神经发育模拟器
 *
 * 从 DevGenome 发育出完整的大脑 (SimulationEngine):
 *   Phase A: 增殖 → 按距离连接 → 分配到 BrainRegion
 *   Phase B: + 导向分子轴突导向 + 转录因子分化
 *   Phase C: + 迁移 + 活动依赖修剪
 *
 * 发育过程:
 *   1. 增殖: DevGenome.division_rounds → 祖细胞分裂 → N 个 NeuralCell
 *   2. 空间分配: 细胞按位置分配到区域类型 (SENSORY/MOTOR/PFC/SUB/NMOD)
 *   3. 连接: 距离 + 类型概率 → 突触形成
 *   4. 转换: NeuralCell → NeuronPopulation + SynapseGroup → BrainRegion
 *   5. 组装: BrainRegion + Projection → SimulationEngine
 *
 * 生物学:
 *   - 增殖 = 神经干细胞分裂 (VZ/SVZ)
 *   - 空间分配 = 形态发生素梯度确定区域边界
 *   - 连接 = 轴突导向 (Phase A 用距离近似, Phase B 用化学梯度)
 */

#include "genome/dev_genome.h"
#include "engine/simulation_engine.h"
#include <vector>
#include <random>

namespace wuyun {

// =============================================================================
// NeuralCell: 发育中的神经细胞
// =============================================================================

struct NeuralCell {
    float x = 0.0f, y = 0.0f;    // 2D 位置 (归一化 [0,1])
    RegionType region_type = RegionType::SENSORY;
    bool is_inhibitory = false;
    int  birth_order = 0;          // 出生顺序 (用于层归属)
    int  region_index = 0;         // 所属区域在 regions 列表中的索引
    float receptors[8] = {};       // Phase B: 对 8 种导向分子的受体表达强度
};

// =============================================================================
// DevelopedRegion: 发育产生的区域描述
// =============================================================================

struct DevelopedRegion {
    std::string name;
    RegionType type;
    float center_x, center_y;     // 区域中心位置
    std::vector<int> cell_indices; // 属于该区域的细胞索引
    int n_excitatory = 0;
    int n_inhibitory = 0;
};

// =============================================================================
// DevelopedConnection: 发育产生的跨区域连接
// =============================================================================

struct DevelopedConnection {
    int src_region;
    int dst_region;
    int n_synapses;               // 突触数量
    int delay;                    // 传导延迟
};

// =============================================================================
// Developer: 发育模拟器
// =============================================================================

class Developer {
public:
    /**
     * 从 DevGenome 发育出完整大脑
     * @param genome 发育基因组
     * @param vision_pixels LGN 输入像素数 (用于感觉接口)
     * @return 可直接用于 ClosedLoopAgent 的 SimulationEngine
     */
    static SimulationEngine develop(const DevGenome& genome,
                                     size_t vision_pixels = 25,
                                     uint32_t seed = 42);

    // --- 发育中间结果 (用于诊断) ---

    /** 获取上次发育产生的细胞 */
    static const std::vector<NeuralCell>& last_cells();

    /** 获取上次发育产生的区域 */
    static const std::vector<DevelopedRegion>& last_regions();

    /** 获取上次发育产生的连接 */
    static const std::vector<DevelopedConnection>& last_connections();

private:
    // 发育阶段
    static void proliferate(const DevGenome& genome, std::mt19937& rng);
    static void assign_regions(const DevGenome& genome);
    static void form_connections(const DevGenome& genome, std::mt19937& rng);
    static SimulationEngine assemble(const DevGenome& genome,
                                      size_t vision_pixels);

    // 发育中间状态 (static for diagnostic access)
    static std::vector<NeuralCell> cells_;
    static std::vector<DevelopedRegion> regions_;
    static std::vector<DevelopedConnection> connections_;
};

} // namespace wuyun
