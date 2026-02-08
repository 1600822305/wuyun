/**
 * test_dev_genome.cpp — 间接编码发育基因组验证 (v2: 完整人脑架构)
 *
 * 测试 DevGenome → Developer::to_agent_config() → ClosedLoopAgent 管线:
 *   1. 发育规则 → AgentConfig 参数计算
 *   2. 不同基因组 → 不同大脑参数
 *   3. 交叉正确工作
 *   4. 发育的 ClosedLoopAgent 能运行
 *   5. 进化能改善适应度
 */

#include "genome/dev_genome.h"
#include "development/developer.h"
#include "engine/closed_loop_agent.h"
#include <cstdio>
#include <cmath>
#include <cassert>

#ifdef _WIN32
#include <windows.h>
#endif

static int g_pass = 0, g_fail = 0;
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { printf("  [FAIL] %s\n", msg); g_fail++; return; } \
} while(0)

using namespace wuyun;

// =========================================================================
// 测试 1: 发育规则 → AgentConfig 参数
// =========================================================================
void test_dev_to_config() {
    printf("\n--- 测试 1: 发育规则 → AgentConfig ---\n");

    DevGenome genome;
    AgentConfig cfg = Developer::to_agent_config(genome);

    printf("  V1 大小: %.2f, dlPFC 大小: %.2f, BG 大小: %.2f\n",
           cfg.v1_size_factor, cfg.dlpfc_size_factor, cfg.bg_size_factor);
    printf("  DA-STDP lr: %.4f\n", cfg.da_stdp_lr);
    printf("  探索噪声: %.1f\n", cfg.exploration_noise);
    printf("  基因数: %zu\n", genome.n_genes());

    // 参数在合理范围
    TEST_ASSERT(cfg.v1_size_factor >= 0.5f && cfg.v1_size_factor <= 3.0f, "V1 大小合理");
    TEST_ASSERT(cfg.da_stdp_lr >= 0.005f && cfg.da_stdp_lr <= 0.15f, "DA LR 合理");
    TEST_ASSERT(cfg.exploration_noise >= 10.0f && cfg.exploration_noise <= 100.0f, "噪声合理");

    // 所有模块都启用 (完整人脑)
    TEST_ASSERT(cfg.enable_lhb && cfg.enable_amygdala && cfg.enable_nacc, "完整人脑模块");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// 测试 2: 不同基因组 → 不同参数
// =========================================================================
void test_variation() {
    printf("\n--- 测试 2: 变异 (不同基因→不同参数) ---\n");

    std::mt19937 rng(123);
    DevGenome g1, g2;
    g1.randomize(rng);
    g2.randomize(rng);

    AgentConfig c1 = Developer::to_agent_config(g1);
    AgentConfig c2 = Developer::to_agent_config(g2);

    printf("  基因组 1: V1=%.2f dlPFC=%.2f lr=%.4f noise=%.1f\n",
           c1.v1_size_factor, c1.dlpfc_size_factor, c1.da_stdp_lr, c1.exploration_noise);
    printf("  基因组 2: V1=%.2f dlPFC=%.2f lr=%.4f noise=%.1f\n",
           c2.v1_size_factor, c2.dlpfc_size_factor, c2.da_stdp_lr, c2.exploration_noise);

    // 不同基因组应产生不同参数
    bool different = (c1.v1_size_factor != c2.v1_size_factor) ||
                     (c1.da_stdp_lr != c2.da_stdp_lr) ||
                     (c1.exploration_noise != c2.exploration_noise);
    TEST_ASSERT(different, "不同基因组产生不同参数");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// 测试 3: 交叉
// =========================================================================
void test_crossover() {
    printf("\n--- 测试 3: 交叉 ---\n");

    std::mt19937 rng(456);
    DevGenome p1, p2;
    p1.division_rounds[0].value = 3.0f;  // 小感觉区
    p2.division_rounds[0].value = 7.0f;  // 大感觉区

    DevGenome child = DevGenome::crossover(p1, p2, rng);
    float cv = child.division_rounds[0].value;
    printf("  父1=%.0f 父2=%.0f 子=%.0f\n", p1.division_rounds[0].value,
           p2.division_rounds[0].value, cv);
    TEST_ASSERT(cv == 3.0f || cv == 7.0f, "子代继承父母基因");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// 测试 4: 发育 → ClosedLoopAgent 能运行
// =========================================================================
void test_agent_run() {
    printf("\n--- 测试 4: 发育 → ClosedLoopAgent 运行 ---\n");

    DevGenome genome;
    AgentConfig cfg = Developer::to_agent_config(genome);

    printf("  发育报告:\n%s\n", Developer::development_report(genome).c_str());

    ClosedLoopAgent agent(cfg);
    printf("  ClosedLoopAgent 创建成功 (完整人脑架构)\n");

    // 运行 50 步
    for (int i = 0; i < 50; ++i) {
        agent.agent_step();
    }
    printf("  50 步运行完成, 无崩溃\n");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// 测试 5: 随机基因组 → ClosedLoopAgent (鲁棒性)
// =========================================================================
void test_random_genomes() {
    printf("\n--- 测试 5: 随机基因组鲁棒性 ---\n");

    std::mt19937 rng(789);
    int n_ok = 0;

    for (int i = 0; i < 5; ++i) {
        DevGenome g;
        g.randomize(rng);
        AgentConfig cfg = Developer::to_agent_config(g);

        try {
            ClosedLoopAgent agent(cfg);
            for (int s = 0; s < 20; ++s) agent.agent_step();
            n_ok++;
        } catch (...) {
            printf("  基因组 %d 崩溃!\n", i);
        }
    }
    printf("  %d/5 随机基因组成功运行\n", n_ok);
    TEST_ASSERT(n_ok >= 3, "大多数随机基因组能运行");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// main
// =========================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== 悟韵 间接编码发育基因组验证 ===\n");

    test_dev_to_config();
    test_variation();
    test_crossover();
    test_agent_run();
    test_random_genomes();

    printf("\n========================================\n");
    printf("  通过: %d / %d\n", g_pass, g_pass + g_fail);
    printf("========================================\n");

    return g_fail > 0 ? 1 : 0;
}
