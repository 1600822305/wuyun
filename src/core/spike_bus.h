#pragma once
/**
 * SpikeBus — 全局脉冲路由系统
 *
 * 负责跨脑区的脉冲分发:
 *   1. 每步收集所有区域的脉冲事件
 *   2. 根据连接表将脉冲路由到目标区域
 *   3. 支持轴突传导延迟 (跨区域 2-5 步)
 *
 * 延迟方案 (02 文档 §2.3):
 *   柱内: 1 步 (已在 SynapseGroup 内处理)
 *   相邻柱间: 1-2 步
 *   跨区域(皮层-皮层): 2-5 步
 *   皮层-皮层下: 1-3 步
 *   调质效应: 10-50 步 (通过 NeuromodulatorSystem 处理)
 *
 * 设计文档: docs/02_neuron_system_design.md §4, §7.2
 */

#include "types.h"
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace wuyun {

/** 单个脉冲事件 */
struct SpikeEvent {
    uint32_t region_id;    // 源区域 ID
    uint32_t dst_region;   // 目标区域 ID
    uint32_t neuron_id;    // 源神经元在区域内的 ID
    int8_t   spike_type;   // SpikeType
    int32_t  timestamp;    // 到达时间步
};

/** 跨区域投射 (一条长程连接) */
struct Projection {
    uint32_t src_region;
    uint32_t dst_region;
    int32_t  delay;        // 传导延迟 (步)
    std::string name;      // 投射名称 (如 "V1_L23→V2_L4")
};

/**
 * SpikeBus: 全局脉冲收集与路由
 *
 * 使用环形缓冲实现延迟投递。
 */
class SpikeBus {
public:
    /**
     * @param max_delay  最大传导延迟 (步)
     */
    explicit SpikeBus(int32_t max_delay = 10);

    /** 注册一个脑区 (返回 region_id) */
    uint32_t register_region(const std::string& name, size_t n_neurons);

    /** 添加跨区域投射 */
    void add_projection(uint32_t src_region, uint32_t dst_region,
                        int32_t delay, const std::string& name = "");

    /**
     * 提交脉冲事件 (每步由各区域调用)
     *
     * @param region_id   源区域 ID
     * @param fired       发放标志数组 (size = region neurons)
     * @param spike_type  脉冲类型数组 (size = region neurons)
     * @param t           当前时间步
     */
    void submit_spikes(uint32_t region_id,
                       const std::vector<uint8_t>& fired,
                       const std::vector<int8_t>& spike_type,
                       int32_t t);

    /**
     * 获取当前步应该到达目标区域的脉冲 (零拷贝: 返回内部缓冲引用)
     * 注意: 引用在 advance() 之前有效
     */
    const std::vector<SpikeEvent>& get_arriving_spikes(uint32_t dst_region, int32_t t);

    /** 推进时钟 (清理过期缓冲) */
    void advance(int32_t t);

    // 访问器
    size_t num_regions() const { return region_names_.size(); }
    size_t num_projections() const { return projections_.size(); }
    const std::string& region_name(uint32_t id) const { return region_names_[id]; }

private:
    int32_t max_delay_;

    // 区域注册表
    std::vector<std::string> region_names_;
    std::vector<size_t>      region_sizes_;

    // 投射列表
    std::vector<Projection>  projections_;

    // 延迟缓冲: delay_buffer_[slot] = vector of SpikeEvents
    // slot = t % (max_delay + 1)
    std::vector<std::vector<SpikeEvent>> delay_buffer_;

    // 查询结果缓冲 (零拷贝: 避免每次 get_arriving_spikes 分配新 vector)
    std::vector<SpikeEvent> query_result_;
};

} // namespace wuyun
