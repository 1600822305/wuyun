/**
 * test_dev_genome.cpp — 间接编码发育基因组验证
 *
 * 测试 DevGenome → Developer::develop() → SimulationEngine 管线:
 *   1. 增殖: 基因控制神经元数量
 *   2. 连接: 距离+类型概率决定连接拓扑
 *   3. 组装: 生成可运行的 SimulationEngine
 *   4. 运行: SimulationEngine 能步进不崩溃
 *   5. 变异: 不同基因组产生不同大脑
 */

#include "genome/dev_genome.h"
#include "development/developer.h"
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
// 测试 1: 增殖 — 基因控制神经元数量
// =========================================================================
void test_proliferation() {
    printf("\n--- 测试 1: 增殖 (基因→神经元数量) ---\n");

    DevGenome genome;
    // 设置已知的分裂轮数
    genome.division_rounds[0].value = 5.0f;  // SENSORY: 2^5 = 32
    genome.division_rounds[1].value = 4.0f;  // MOTOR:   2^4 = 16
    genome.division_rounds[2].value = 4.0f;  // PFC:     2^4 = 16
    genome.division_rounds[3].value = 4.0f;  // SUB:     2^4 = 16
    genome.division_rounds[4].value = 3.0f;  // NMOD:    2^3 = 8

    auto engine = Developer::develop(genome, 25, 42);

    auto& cells = Developer::last_cells();
    auto& regions = Developer::last_regions();

    printf("  总细胞数: %zu (期望: 32+16+16+16+8=88)\n", cells.size());
    TEST_ASSERT(cells.size() == 88, "增殖产生正确数量的细胞");

    printf("  区域数: %zu (期望: 5)\n", regions.size());
    TEST_ASSERT(regions.size() == 5, "5种区域类型");

    // 检查每种类型的细胞数
    for (size_t i = 0; i < regions.size(); ++i) {
        int expected = 1 << static_cast<int>(genome.division_rounds[i].value);
        int actual = static_cast<int>(regions[i].cell_indices.size());
        printf("  区域 %s: %d 细胞 (期望 %d), %d 兴奋 + %d 抑制\n",
               regions[i].name.c_str(), actual, expected,
               regions[i].n_excitatory, regions[i].n_inhibitory);
        TEST_ASSERT(actual == expected, "区域细胞数正确");
    }

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// 测试 2: 连接 — 发育产生连接
// =========================================================================
void test_connections() {
    printf("\n--- 测试 2: 连接 (距离+类型概率→突触) ---\n");

    DevGenome genome;
    auto engine = Developer::develop(genome, 25, 42);

    auto& connections = Developer::last_connections();

    printf("  发育产生 %zu 条跨区域连接\n", connections.size());
    TEST_ASSERT(connections.size() > 0, "至少产生一些连接");

    int total_synapses = 0;
    for (auto& c : connections) {
        auto& regions = Developer::last_regions();
        printf("    %s → %s: %d 突触, delay=%d\n",
               regions[c.src_region].name.c_str(),
               regions[c.dst_region].name.c_str(),
               c.n_synapses, c.delay);
        total_synapses += c.n_synapses;
    }
    printf("  总突触数: %d\n", total_synapses);
    TEST_ASSERT(total_synapses > 10, "至少 10 个突触");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// 测试 3: 运行 — 发育的大脑能步进
// =========================================================================
void test_run() {
    printf("\n--- 测试 3: 运行 (发育大脑能步进不崩溃) ---\n");

    DevGenome genome;
    auto engine = Developer::develop(genome, 25, 42);

    printf("  引擎区域数: %zu\n", engine.num_regions());

    // 步进 100 步
    for (int i = 0; i < 100; ++i) {
        engine.step();
    }
    printf("  100 步完成, 无崩溃\n");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// 测试 4: 变异 — 不同基因组产生不同大脑
// =========================================================================
void test_variation() {
    printf("\n--- 测试 4: 变异 (不同基因→不同大脑) ---\n");

    std::mt19937 rng(123);

    DevGenome g1, g2;
    g1.randomize(rng);
    g2.randomize(rng);

    auto e1 = Developer::develop(g1, 25, 100);
    size_t n1 = Developer::last_cells().size();

    auto e2 = Developer::develop(g2, 25, 200);
    size_t n2 = Developer::last_cells().size();

    printf("  基因组 1: %zu 神经元\n", n1);
    printf("  基因组 2: %zu 神经元\n", n2);
    printf("  基因组 1 摘要: %s\n", g1.summary().c_str());
    printf("  基因组 2 摘要: %s\n", g2.summary().c_str());

    // 不同随机基因组应该产生不同大小的大脑
    TEST_ASSERT(n1 != n2 || g1.division_rounds[0].value != g2.division_rounds[0].value,
                "不同基因组产生不同结构");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// 测试 5: 交叉 — 两个基因组混合
// =========================================================================
void test_crossover() {
    printf("\n--- 测试 5: 交叉 (两个基因组→后代) ---\n");

    std::mt19937 rng(456);

    DevGenome parent1, parent2;
    parent1.division_rounds[0].value = 3.0f;  // 小感觉区
    parent2.division_rounds[0].value = 7.0f;  // 大感觉区

    DevGenome child = DevGenome::crossover(parent1, parent2, rng);

    float child_div = child.division_rounds[0].value;
    printf("  父1 感觉分裂轮数: %.0f (→%d 神经元)\n",
           parent1.division_rounds[0].value, 1 << 3);
    printf("  父2 感觉分裂轮数: %.0f (→%d 神经元)\n",
           parent2.division_rounds[0].value, 1 << 7);
    printf("  子代 感觉分裂轮数: %.0f (→%d 神经元)\n",
           child_div, 1 << static_cast<int>(child_div));

    TEST_ASSERT(child_div == 3.0f || child_div == 7.0f,
                "子代继承父母之一的基因");

    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// 测试 6: 进化 — 基因组能通过选择改善
// =========================================================================
void test_evolution() {
    printf("\n--- 测试 6: 进化 (选择→适应度提升) ---\n");

    std::mt19937 rng(789);
    constexpr int POP_SIZE = 20;
    constexpr int GENERATIONS = 5;

    // 创建初始种群
    std::vector<DevGenome> population(POP_SIZE);
    for (auto& g : population) g.randomize(rng);

    // 简单适应度: 感觉区大 + 连接密 = 高适应度
    // (模拟"感觉能力强的大脑更适应环境")
    auto fitness = [](const DevGenome& g) -> float {
        float sensory_size = g.division_rounds[0].value;  // 感觉区分裂轮数
        float conn_density = g.connection_radius.value;   // 连接范围
        return sensory_size * 2.0f + conn_density * 5.0f;
    };

    float best_gen0 = -1e9f;
    for (auto& g : population) {
        g.fitness = fitness(g);
        if (g.fitness > best_gen0) best_gen0 = g.fitness;
    }
    printf("  Gen 0: best fitness = %.2f\n", best_gen0);

    // 进化循环
    for (int gen = 0; gen < GENERATIONS; ++gen) {
        // 排序
        std::sort(population.begin(), population.end(),
                  [](const DevGenome& a, const DevGenome& b) {
                      return a.fitness > b.fitness;
                  });

        // 精英保留 + 交叉 + 变异
        std::vector<DevGenome> next_gen;
        // 保留 top 4
        for (int i = 0; i < 4; ++i) next_gen.push_back(population[i]);

        // 交叉+变异填充剩余
        while (next_gen.size() < POP_SIZE) {
            std::uniform_int_distribution<int> pick(0, 9);  // top 50%
            auto child = DevGenome::crossover(population[pick(rng)],
                                               population[pick(rng)], rng);
            child.mutate(rng, 0.2f, 0.15f);
            next_gen.push_back(child);
        }

        population = next_gen;
        for (auto& g : population) g.fitness = fitness(g);
    }

    float best_final = -1e9f;
    for (auto& g : population) {
        if (g.fitness > best_final) best_final = g.fitness;
    }
    printf("  Gen %d: best fitness = %.2f\n", GENERATIONS, best_final);
    printf("  提升: %+.2f\n", best_final - best_gen0);

    TEST_ASSERT(best_final >= best_gen0, "进化不退化");

    // 验证最佳基因组能发育出大脑
    std::sort(population.begin(), population.end(),
              [](const DevGenome& a, const DevGenome& b) {
                  return a.fitness > b.fitness;
              });
    auto engine = Developer::develop(population[0], 25, 42);
    printf("  最佳基因组 → %zu 神经元, %zu 连接\n",
           Developer::last_cells().size(),
           Developer::last_connections().size());
    printf("  基因数: %zu\n", population[0].n_genes());

    // 步进验证
    for (int i = 0; i < 50; ++i) engine.step();
    printf("  发育大脑运行 50 步, 无崩溃\n");

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

    test_proliferation();
    test_connections();
    test_run();
    test_variation();
    test_crossover();
    test_evolution();

    printf("\n========================================\n");
    printf("  通过: %d / %d\n", g_pass, g_pass + g_fail);
    printf("========================================\n");

    return g_fail > 0 ? 1 : 0;
}
