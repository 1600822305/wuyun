#pragma once
/**
 * GapJunction — 电突触 (缝隙连接)
 *
 * 抑制性中间神经元 (PV+ basket) 之间的电突触:
 *   I_gap = g_gap · (V_pre - V_post)
 *
 * 功能:
 *   - PV+ basket 网络同步 → gamma 振荡
 *   - 超快速信号传递 (无突触延迟)
 *   - 双向对称连接
 *
 * 设计文档: docs/02_neuron_system_design.md §2.1
 */

#include <vector>
#include <cstdint>
#include <cstddef>

namespace wuyun {

/** 单条电突触连接 */
struct GapJunctionConn {
    int32_t neuron_a;   // 连接的一端
    int32_t neuron_b;   // 连接的另一端
    float   g_gap;      // 缝隙连接电导 (nS)
};

/**
 * 电突触组 — 管理一组缝隙连接
 *
 * 对称双向: A→B 和 B→A 同时传导
 */
class GapJunctionGroup {
public:
    GapJunctionGroup(size_t n_neurons);

    /** 添加一条缝隙连接 */
    void add_connection(int32_t a, int32_t b, float g_gap);

    /**
     * 计算电突触电流
     *
     * @param v_membrane  膜电位数组 (size = n_neurons)
     * @return 每个神经元的电突触电流
     */
    std::vector<float> compute_currents(const std::vector<float>& v_membrane) const;

    size_t n_connections() const { return connections_.size(); }
    size_t n_neurons()     const { return n_; }

private:
    size_t n_;
    std::vector<GapJunctionConn> connections_;
};

} // namespace wuyun
