#pragma once
/**
 * Developer v3 — 骨架固定 + 皮层涌现
 *
 * 固定回路 (继承 build_brain 的 49 步成果):
 *   BG D1/D2/GPi → 写死, 基因控制大小和增益
 *   VTA 内部 RPE → 写死, 基因控制 DA 增益
 *   丘脑 TRN 门控 → 写死, 基因控制门控强度
 *   杏仁核/海马 → 写死, 基因控制大小
 *   LGN/M1/Hypothalamus → 写死
 *
 * 可进化皮层 (条形码兼容性, Barabasi 2019):
 *   5 种皮层类型, 每种有 8 维条形码
 *   皮层间连接 = sigmoid(barcode_i * W * barcode_j - threshold)
 *   皮层→BG 接口 = barcode 与 cortical_to_bg 的兼容性
 *   LGN→皮层接口 = barcode 与 LGN_BARCODE 的兼容性
 *
 * 输出: AgentConfig (直接用 ClosedLoopAgent), 不需要自定义大脑
 * 但: 皮层区域的数量和连接从条形码涌现
 */

#include "genome/dev_genome.h"
#include "engine/closed_loop_agent.h"

namespace wuyun {

class Developer {
public:
    /**
     * 从 DevGenome 构建 ClosedLoopAgent
     * 固定回路: 继承 build_brain
     * 皮层: 条形码涌现
     */
    static AgentConfig to_agent_config(const DevGenome& genome);

    /**
     * 诊断: 打印条形码兼容性矩阵和皮层连接拓扑
     */
    static std::string development_report(const DevGenome& genome);

    /**
     * 检查 LGN→皮层→BG 信号通路是否连通
     * 返回: 连通的皮层类型数 (0 = 完全断开)
     */
    static int check_connectivity(const DevGenome& genome);
};

} // namespace wuyun
