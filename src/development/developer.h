#pragma once
/**
 * Developer — 神经发育模拟器 (v2: 完整人脑架构)
 *
 * 从 DevGenome 的发育规则计算出 AgentConfig 参数,
 * 然后用标准 ClosedLoopAgent 构建完整的 64 区域人脑。
 *
 * 与 v1 (通用区域) 的区别:
 *   v1: DevGenome → 5 个通用区域 (Sensory/Motor/PFC/Sub/Nmod) — 玩具
 *   v2: DevGenome → AgentConfig → build_brain() → 64 区域完整人脑
 *
 * 发育规则如何决定参数:
 *   1. 增殖梯度 → 区域大小 (v1_size_factor, dlpfc_size_factor, bg_size_factor)
 *      前后轴梯度决定前额叶 vs 感觉区的相对大小
 *
 *   2. 导向分子 → 连接强度 (暂不影响 — build_brain 固定投射, 但影响 gain)
 *      bg_to_m1_gain, lgn_gain 等从发育规则计算
 *
 *   3. 分化梯度 → 受体密度 / 学习率
 *      DA 梯度 → da_stdp_lr (前部高 = 更强学习)
 *      NMDA 梯度 → cortical STDP 参数
 *
 *   4. 修剪阈值 → 稳态参数
 *      pruning_threshold → homeostatic_target_rate, homeostatic_eta
 *
 * 生物学: 真实大脑的区域类型由基因决定 (PAX6→V1, FOXP2→Broca),
 *   但区域大小、连接强度、受体密度由发育梯度决定。
 *   这正是间接编码的正确层级。
 */

#include "genome/dev_genome.h"
#include "engine/closed_loop_agent.h"
#include <vector>

namespace wuyun {

class Developer {
public:
    /**
     * 从 DevGenome 计算 AgentConfig
     * 发育规则 → 参数值, 然后 ClosedLoopAgent 用这些参数构建完整人脑
     */
    static AgentConfig to_agent_config(const DevGenome& genome);

    /**
     * 诊断: 打印发育过程 (哪些基因决定了哪些参数)
     */
    static std::string development_report(const DevGenome& genome);
};

} // namespace wuyun
