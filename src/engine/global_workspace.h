#pragma once
/**
 * GlobalWorkspace — 全局工作空间理论 (Baars 1988 / Dehaene 2001)
 *
 * 意识访问的计算模型:
 *   1. 竞争: 多个皮层区域的L5输出竞争工作空间访问权
 *   2. 点火: 最强信号超过阈值 → 全局点火 (ignition)
 *   3. 广播: 点火信号通过ILN/CeM丘脑核广播到全皮层L2/3
 *   4. 意识访问: 多个皮层区同时活跃 = "意识内容"
 *
 * 生物学基础:
 *   - L5 pyramidal: 长程皮层输出 (主要竞争者)
 *   - ILN (板内核群): 全脑广播枢纽 (CL/CM/Pf)
 *   - CeM (中央内侧核): 觉醒/意识维持
 *   - PFC L2/3: 工作空间维持 + 反馈放大
 *
 * 关键参数:
 *   - ignition_threshold: 点火所需的最小活动水平
 *   - broadcast_gain: 广播信号的放大倍数
 *   - competition_decay: 竞争积分的衰减 (防止锁定)
 *   - min_ignition_gap: 连续点火的最小间隔
 *
 * 生物学参考:
 *   - Baars (1988) A Cognitive Theory of Consciousness
 *   - Dehaene & Changeux (2011) Experimental and theoretical approaches to conscious processing
 *   - Dehaene, Kerszberg & Changeux (1998) A neuronal model of a global workspace
 */

#include "region/brain_region.h"
#include "core/population.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>

namespace wuyun {

struct GWConfig {
    std::string name = "GW";

    // === Workspace neurons ===
    size_t n_workspace = 30;    // Central workspace integrators

    // === Competition ===
    float ignition_threshold = 15.0f;  // Min salience to ignite
    float competition_decay  = 0.85f;  // Per-step decay of accumulated salience
    int32_t min_ignition_gap = 20;     // Min steps between ignition events

    // === Broadcast ===
    float broadcast_gain    = 2.5f;    // Amplification of broadcast signal
    int32_t broadcast_duration = 8;    // Steps to sustain broadcast

    // === Attention gating ===
    bool attention_gating = true;  // Only attended regions can compete
};

class GlobalWorkspace : public BrainRegion {
public:
    explicit GlobalWorkspace(const GWConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_all_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_all_; }

    // === GNW State Queries ===

    /** Is the workspace currently in ignition state? */
    bool is_ignited() const { return is_ignited_; }

    /** Region ID of current conscious content (-1 if none) */
    int32_t conscious_content_id() const { return conscious_content_id_; }

    /** Name of the source region currently in consciousness */
    const std::string& conscious_content_name() const { return conscious_content_name_; }

    /** Total number of ignition events since start */
    size_t ignition_count() const { return ignition_count_; }

    /** Current broadcast remaining steps */
    int32_t broadcast_remaining() const { return broadcast_remaining_; }

    /** Get salience map (region_id → accumulated salience) */
    const std::unordered_map<uint32_t, float>& salience_map() const { return salience_; }

    /** Register a cortical region name for readable output */
    void register_source(uint32_t region_id, const std::string& name);

    /** Current winning salience value */
    float winning_salience() const { return winning_salience_; }

    const NeuronPopulation& workspace_pop() const { return workspace_; }

private:
    void aggregate_state();

    GWConfig config_;

    // Central workspace neurons (integrators)
    NeuronPopulation workspace_;

    // Per-source region tracking
    std::unordered_map<uint32_t, float> salience_;         // Accumulated salience
    std::unordered_map<uint32_t, size_t> step_spikes_;     // This-step spike count
    std::unordered_map<uint32_t, std::string> source_names_;

    // Ignition state
    bool is_ignited_           = false;
    int32_t conscious_content_id_ = -1;
    std::string conscious_content_name_ = "";
    float winning_salience_    = 0.0f;
    size_t ignition_count_     = 0;
    int32_t broadcast_remaining_ = 0;
    int32_t last_ignition_t_   = -100;

    // Broadcast buffer: spikes to re-submit during broadcast
    float broadcast_current_   = 0.0f;

    std::vector<uint8_t> fired_all_;
    std::vector<int8_t>  spike_type_all_;
};

} // namespace wuyun
