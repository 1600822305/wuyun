/**
 * test_dev_genome.cpp — 基因连接组模型验证 (v3: 骨架固定+皮层涌现)
 */
#include "genome/dev_genome.h"
#include "development/developer.h"
#include "engine/closed_loop_agent.h"
#include "engine/grid_world_env.h"
#include <cstdio>
#include <cmath>
#ifdef _WIN32
#include <windows.h>
#endif

static int g_pass = 0, g_fail = 0;
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { printf("  [FAIL] %s\n", msg); g_fail++; return; } \
} while(0)

using namespace wuyun;

void test_genome_structure() {
    printf("\n--- 测试 1: 基因组结构 ---\n");
    DevGenome g;
    printf("  基因数: %zu\n", g.n_genes());
    TEST_ASSERT(g.n_genes() > 100, "至少 100 个基因");
    TEST_ASSERT(g.n_genes() < 250, "不超过 250 个基因");
    printf("  [PASS]\n"); g_pass++;
}

void test_barcode_compat() {
    printf("\n--- 测试 2: 条形码兼容性 ---\n");
    DevGenome g;
    // 默认: W 对角线 0.3, 其余 0 → 同维度兼容
    // LGN barcode = [1,0.8,0.1,0,...] 高维度 0,1
    // 皮层默认 barcode = [0.5,0.5,...] 均匀
    float bc[BARCODE_DIM] = {0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f};
    float compat = g.barcode_compat(DevGenome::LGN_BARCODE, bc);
    printf("  LGN → 均匀皮层 兼容性: %.3f\n", compat);
    float prob = g.conn_prob_from_compat(compat);
    printf("  连接概率: %.1f%%\n", prob * 100.0f);
    TEST_ASSERT(compat > -5.0f && compat < 5.0f, "兼容性在合理范围");
    printf("  [PASS]\n"); g_pass++;
}

void test_variation() {
    printf("\n--- 测试 3: 不同基因组→不同参数 ---\n");
    std::mt19937 rng(123);
    DevGenome g1, g2;
    g1.randomize(rng);
    g2.randomize(rng);
    AgentConfig c1 = Developer::to_agent_config(g1);
    AgentConfig c2 = Developer::to_agent_config(g2);
    printf("  g1: V1=%.2f dlPFC=%.2f lr=%.4f noise=%.0f\n",
           c1.v1_size_factor, c1.dlpfc_size_factor, c1.da_stdp_lr, c1.exploration_noise);
    printf("  g2: V1=%.2f dlPFC=%.2f lr=%.4f noise=%.0f\n",
           c2.v1_size_factor, c2.dlpfc_size_factor, c2.da_stdp_lr, c2.exploration_noise);
    bool diff = (c1.v1_size_factor != c2.v1_size_factor) ||
                (c1.da_stdp_lr != c2.da_stdp_lr);
    TEST_ASSERT(diff, "不同基因组产生不同参数");
    printf("  [PASS]\n"); g_pass++;
}

void test_agent_run() {
    printf("\n--- 测试 4: 发育→完整人脑→运行 ---\n");
    DevGenome g;
    AgentConfig cfg = Developer::to_agent_config(g);
    printf("  发育报告:\n%s\n", Developer::development_report(g).c_str());
    ClosedLoopAgent agent(std::make_unique<GridWorldEnv>(GridWorldConfig{}), cfg);
    printf("  ClosedLoopAgent 创建成功\n");
    for (int i = 0; i < 50; ++i) agent.agent_step();
    printf("  50 步运行完成\n");
    printf("  [PASS]\n"); g_pass++;
}

void test_random_robustness() {
    printf("\n--- 测试 5: 随机基因组鲁棒性 ---\n");
    std::mt19937 rng(789);
    int ok = 0;
    for (int i = 0; i < 5; ++i) {
        DevGenome g;
        g.randomize(rng);
        AgentConfig cfg = Developer::to_agent_config(g);
        try {
            ClosedLoopAgent agent(std::make_unique<GridWorldEnv>(GridWorldConfig{}), cfg);
            for (int s = 0; s < 20; ++s) agent.agent_step();
            ok++;
        } catch (...) {
            printf("  基因组 %d 崩溃\n", i);
        }
    }
    printf("  %d/5 成功\n", ok);
    TEST_ASSERT(ok >= 3, "大多数随机基因组能运行");
    printf("  [PASS]\n"); g_pass++;
}

void test_connectivity() {
    printf("\n--- 测试 6: 连通性检查 ---\n");
    DevGenome g;
    int conn = Developer::check_connectivity(g);
    printf("  默认基因组: %d/%d 皮层类型连通\n", conn, N_CORTICAL_TYPES);

    // 随机基因组的连通性分布
    std::mt19937 rng(456);
    int total_conn = 0;
    for (int i = 0; i < 10; ++i) {
        DevGenome rg;
        rg.randomize(rng);
        total_conn += Developer::check_connectivity(rg);
    }
    printf("  10 个随机基因组平均连通: %.1f/%d\n",
           static_cast<float>(total_conn) / 10.0f, N_CORTICAL_TYPES);
    printf("  [PASS]\n"); g_pass++;
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== 悟韵 基因连接组模型验证 (v3) ===\n");

    test_genome_structure();
    test_barcode_compat();
    test_variation();
    test_agent_run();
    test_random_robustness();
    test_connectivity();

    printf("\n========================================\n");
    printf("  通过: %d / %d\n", g_pass, g_pass + g_fail);
    printf("========================================\n");
    return g_fail > 0 ? 1 : 0;
}
