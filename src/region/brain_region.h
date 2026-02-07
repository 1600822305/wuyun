#pragma once
/**
 * BrainRegion — 脑区基类 (Layer 4: Region)
 *
 * 统一接口:
 *   - 内部神经元群体 + 突触
 *   - SpikeBus 脉冲收发
 *   - 振荡相位追踪
 *   - 神经调质效应
 *
 * 子类: CorticalRegion, ThalamicRelay, BasalGanglia, VTA_DA, ...
 *
 * 设计文档: docs/02_neuron_system_design.md §5, §6
 */

#include "core/spike_bus.h"
#include "core/oscillation.h"
#include "core/neuromodulator.h"
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace wuyun {

/**
 * 脑区基类
 *
 * 每个脑区:
 *   1. 注册到 SpikeBus (获得 region_id)
 *   2. 每步: 接收到达脉冲 → 内部计算 → 提交输出脉冲
 *   3. 维护自身振荡和调质状态
 */
class BrainRegion {
public:
    BrainRegion(const std::string& name, size_t n_neurons);
    virtual ~BrainRegion() = default;

    // --- 生命周期 ---

    /** 注册到 SpikeBus (由 SimulationEngine 调用) */
    void register_to_bus(SpikeBus& bus);

    /**
     * 主步进函数 (每个时间步调用)
     * @param t   当前时间步
     * @param dt  步长 (ms)
     */
    virtual void step(int32_t t, float dt = 1.0f) = 0;

    /** 接收从 SpikeBus 到达的脉冲 */
    virtual void receive_spikes(const std::vector<SpikeEvent>& events) = 0;

    /** 提交输出脉冲到 SpikeBus */
    virtual void submit_spikes(SpikeBus& bus, int32_t t) = 0;

    // --- 外部输入 ---

    /** 注入外部输入电流 (感觉输入等) */
    virtual void inject_external(const std::vector<float>& currents) = 0;

    // --- 访问器 ---
    const std::string& name()       const { return name_; }
    uint32_t           region_id()  const { return region_id_; }
    size_t             n_neurons()  const { return n_neurons_; }

    OscillationTracker&       oscillation()       { return oscillation_; }
    const OscillationTracker& oscillation() const { return oscillation_; }

    NeuromodulatorSystem&       neuromod()       { return neuromod_; }
    const NeuromodulatorSystem& neuromod() const { return neuromod_; }

    /** 获取发放状态 (子类负责填充) */
    virtual const std::vector<uint8_t>& fired()      const = 0;
    virtual const std::vector<int8_t>&  spike_type()  const = 0;

protected:
    std::string name_;
    uint32_t    region_id_ = 0;
    size_t      n_neurons_;

    OscillationTracker   oscillation_;
    NeuromodulatorSystem neuromod_;
};

} // namespace wuyun
