/**
 * visualize_brain — 大脑拓扑可视化工具
 *
 * 用法:
 *   visualize_brain                     # 文本拓扑摘要
 *   visualize_brain --dot               # 输出 DOT 到 stdout
 *   visualize_brain --dot brain.dot     # 输出 DOT 到文件
 *
 * 工作流:
 *   创建默认 DevGenome → Developer::to_agent_config() → 构建 ClosedLoopAgent
 *   从 agent 的 brain() (SimulationEngine) 提取拓扑
 *   输出文本摘要或 Graphviz DOT 格式
 *
 * DOT 文件可粘贴到在线渲染器:
 *   https://dreampuf.github.io/GraphvizOnline/
 */

#include "genome/dev_genome.h"
#include "development/developer.h"
#include "engine/closed_loop_agent.h"
#include "engine/grid_world_env.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    setvbuf(stdout, NULL, _IONBF, 0);

    // 解析参数
    bool output_dot = false;
    std::string dot_file;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dot") == 0) {
            output_dot = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                dot_file = argv[++i];
            }
        }
    }

    printf("=== WuYun Brain Topology Visualizer ===\n\n");

    // 构建大脑: 默认 DevGenome → AgentConfig → ClosedLoopAgent
    printf("Building brain from default DevGenome...\n");
    wuyun::DevGenome genome;  // 默认基因值
    wuyun::AgentConfig cfg = wuyun::Developer::to_agent_config(genome);

    wuyun::ClosedLoopAgent agent(std::make_unique<wuyun::GridWorldEnv>(wuyun::GridWorldConfig{}), cfg);
    printf("Brain built successfully.\n\n");

    // 文本摘要
    std::string summary = agent.brain().export_topology_summary();
    printf("%s", summary.c_str());

    // DOT 输出
    if (output_dot) {
        std::string dot = agent.brain().export_dot();

        if (dot_file.empty()) {
            // 输出到 stdout
            printf("\n=== Graphviz DOT ===\n");
            printf("%s", dot.c_str());
        } else {
            // 输出到文件
            std::ofstream ofs(dot_file);
            if (ofs.is_open()) {
                ofs << dot;
                ofs.close();
                printf("\nDOT file written to: %s\n", dot_file.c_str());
                printf("Render online: https://dreampuf.github.io/GraphvizOnline/\n");
            } else {
                fprintf(stderr, "Error: cannot write to %s\n", dot_file.c_str());
                return 1;
            }
        }
    } else {
        printf("\nTip: use --dot brain.dot to generate Graphviz visualization\n");
    }

    return 0;
}
