#pragma once
/**
 * SimulationEngine — 全脑仿真引擎
 *
 * 职责:
 *   1. 管理所有 BrainRegion 实例
 *   2. 统一时钟推进
 *   3. 编排 SpikeBus 脉冲收发
 *   4. 每步循环: 收脉冲 → 区域计算 → 发脉冲 → 推进总线
 *
 * 时钟层级 (02 文档 §7.1):
 *   脉冲时钟: 1 ms  (每步)
 *   振荡时钟: 10 ms (由各 Region 内部 OscillationTracker 处理)
 *   调制时钟: 100 ms (由 NeuromodulatorSystem 处理)
 *
 * 设计文档: docs/02_neuron_system_design.md §7.2
 */

#include "core/spike_bus.h"
#include "core/neuromodulator.h"
#include "region/brain_region.h"
#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace wuyun {

/** 仿真统计 */
struct SimStats {
    int32_t timestep       = 0;
    size_t  total_spikes   = 0;
    size_t  total_neurons  = 0;
    size_t  total_regions  = 0;
};

/**
 * 每步回调 (可选)
 * @param t      当前时间步
 * @param engine 引擎引用
 */
class SimulationEngine;
using StepCallback = std::function<void(int32_t t, SimulationEngine& engine)>;

class SimulationEngine {
public:
    /**
     * @param max_delay  SpikeBus 最大传导延迟 (步)
     */
    explicit SimulationEngine(int32_t max_delay = 10);

    // Non-copyable (contains unique_ptrs)
    SimulationEngine(const SimulationEngine&) = delete;
    SimulationEngine& operator=(const SimulationEngine&) = delete;
    SimulationEngine(SimulationEngine&&) = default;
    SimulationEngine& operator=(SimulationEngine&&) = default;

    // --- 区域管理 ---

    /** 添加脑区 (自动注册到 SpikeBus) */
    void add_region(std::unique_ptr<BrainRegion> region);

    /** 按名称查找脑区 */
    BrainRegion* find_region(const std::string& name);

    /** 添加跨区域投射 */
    void add_projection(const std::string& src, const std::string& dst,
                        int32_t delay, const std::string& proj_name = "");

    // --- 仿真控制 ---

    /** 运行 n 步 */
    void run(int32_t n_steps, float dt = 1.0f);

    /** 运行单步 */
    void step(float dt = 1.0f);

    /** 设置每步回调 */
    void set_callback(StepCallback cb) { callback_ = std::move(cb); }

    // --- 神经调质广播 ---

    /** 注册神经调质源区域 (DA=VTA, NE=LC, 5-HT=DRN, ACh=NBM) */
    enum class NeuromodType { DA, NE, SHT, ACh };
    void register_neuromod_source(const std::string& region_name, NeuromodType type);

    /** 获取全局神经调质水平 */
    const NeuromodulatorLevels& global_neuromod() const { return global_neuromod_; }

    // --- 访问器 ---
    int32_t    current_time() const { return t_; }
    SimStats   stats()        const;
    SpikeBus&  bus()                { return bus_; }
    const SpikeBus& bus() const    { return bus_; }

    size_t num_regions() const { return regions_.size(); }
    BrainRegion& region(size_t i) { return *regions_[i]; }
    const BrainRegion& region(size_t i) const { return *regions_[i]; }

    // --- v54: 拓扑导出 ---

    /** 导出 Graphviz DOT 格式 (脑区分组, 节点大小反映神经元数) */
    std::string export_dot() const;

    /** 导出文本拓扑摘要 (区域列表 + 投射列表) */
    std::string export_topology_summary() const;

private:
    SpikeBus bus_;
    std::vector<std::unique_ptr<BrainRegion>> regions_;
    int32_t t_ = 0;
    StepCallback callback_;

    // 神经调质广播系统
    NeuromodulatorLevels global_neuromod_;
    struct NeuromodSource {
        BrainRegion* region = nullptr;
        NeuromodType type;
    };
    std::vector<NeuromodSource> neuromod_sources_;

    void collect_and_broadcast_neuromod();
};

} // namespace wuyun
