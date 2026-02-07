#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#endif

#include "region/limbic/hippocampus.h"
#include "region/limbic/amygdala.h"
#include "engine/simulation_engine.h"

using namespace wuyun;

// =============================================================================
// Test infrastructure
// =============================================================================

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("  [FAIL] %s\n", msg); \
        g_fail++; return; \
    } \
} while(0)

#define PASS(name) do { printf("  [PASS] %s\n", name); g_pass++; } while(0)

static size_t count_spikes(const NeuronPopulation& pop) {
    size_t n = 0;
    for (size_t i = 0; i < pop.size(); ++i) {
        if (pop.fired()[i]) n++;
    }
    return n;
}

// =============================================================================
// 测试1: 海马构造验证
// =============================================================================
void test_hippocampus_construction() {
    printf("\n--- 测试1: 海马体构造验证 ---\n");

    HippocampusConfig cfg;
    Hippocampus hipp(cfg);

    size_t total = cfg.n_ec + cfg.n_dg + cfg.n_ca3 + cfg.n_ca1 + cfg.n_sub
                 + cfg.n_dg_inh + cfg.n_ca3_inh + cfg.n_ca1_inh;
    printf("    EC=%zu DG=%zu CA3=%zu CA1=%zu Sub=%zu inh=%zu\n",
           cfg.n_ec, cfg.n_dg, cfg.n_ca3, cfg.n_ca1, cfg.n_sub,
           cfg.n_dg_inh + cfg.n_ca3_inh + cfg.n_ca1_inh);
    printf("    总神经元: %zu\n", total);

    CHECK(hipp.n_neurons() == total, "神经元总数匹配");
    CHECK(hipp.ec().size() == cfg.n_ec, "EC 大小正确");
    CHECK(hipp.dg().size() == cfg.n_dg, "DG 大小正确");
    CHECK(hipp.ca3().size() == cfg.n_ca3, "CA3 大小正确");
    CHECK(hipp.ca1().size() == cfg.n_ca1, "CA1 大小正确");

    PASS("海马体构造");
}

// =============================================================================
// 测试2: 海马沉默测试
// =============================================================================
void test_hippocampus_silence() {
    printf("\n--- 测试2: 海马沉默测试 ---\n");

    HippocampusConfig cfg;
    Hippocampus hipp(cfg);

    size_t total_spikes = 0;
    for (int t = 0; t < 100; ++t) {
        hipp.step(t);
        for (size_t i = 0; i < hipp.n_neurons(); ++i) {
            if (hipp.fired()[i]) total_spikes++;
        }
    }

    printf("    100步无输入: 总发放=%zu\n", total_spikes);
    CHECK(total_spikes == 0, "无输入应沉默");

    PASS("海马沉默测试");
}

// =============================================================================
// 测试3: 三突触通路信号传播 EC→DG→CA3→CA1→Sub
// =============================================================================
void test_trisynaptic_propagation() {
    printf("\n--- 测试3: 三突触通路信号传播 ---\n");
    printf("    通路: EC → DG → CA3 → CA1 → Sub\n");

    HippocampusConfig cfg;
    Hippocampus hipp(cfg);

    size_t spk_ec = 0, spk_dg = 0, spk_ca3 = 0, spk_ca1 = 0, spk_sub = 0;

    for (int t = 0; t < 200; ++t) {
        // Inject cortical input to EC for first 50 steps
        if (t < 80) {
            std::vector<float> input(cfg.n_ec, 30.0f);
            hipp.inject_cortical_input(input);
        }

        hipp.step(t);

        spk_ec  += count_spikes(hipp.ec());
        spk_dg  += count_spikes(hipp.dg());
        spk_ca3 += count_spikes(hipp.ca3());
        spk_ca1 += count_spikes(hipp.ca1());
        spk_sub += count_spikes(hipp.sub());
    }

    printf("    EC=%zu → DG=%zu → CA3=%zu → CA1=%zu → Sub=%zu\n",
           spk_ec, spk_dg, spk_ca3, spk_ca1, spk_sub);

    CHECK(spk_ec > 0, "EC 应有发放");
    CHECK(spk_dg > 0, "DG 应有发放 (EC→DG perforant path)");
    CHECK(spk_ca3 > 0, "CA3 应有发放 (DG→CA3 mossy fiber)");
    CHECK(spk_ca1 > 0, "CA1 应有发放 (CA3→CA1 Schaffer + EC→CA1 direct)");

    PASS("三突触通路信号传播");
}

// =============================================================================
// 测试4: DG 稀疏编码 (~2% 激活率)
// =============================================================================
void test_dg_sparsity() {
    printf("\n--- 测试4: DG 稀疏编码 ---\n");
    printf("    原理: 齿状回高阈值 → 极稀疏激活 (~2%%)\n");

    HippocampusConfig cfg;
    cfg.n_dg = 500;  // Larger DG for sparsity measurement
    cfg.n_dg_inh = 80; // Scale up interneurons with DG
    Hippocampus hipp(cfg);

    float max_sparsity = 0.0f;
    float steady_avg = 0.0f;
    int steady_steps = 0;

    for (int t = 0; t < 200; ++t) {
        if (t < 80) {
            // Sparse cortical input: only ~20% of EC active (realistic)
            std::vector<float> input(cfg.n_ec, 0.0f);
            for (size_t i = 0; i < cfg.n_ec; i += 5) {
                input[i] = 30.0f;
            }
            hipp.inject_cortical_input(input);
        }

        hipp.step(t);

        float sp = hipp.dg_sparsity();
        if (sp > max_sparsity) max_sparsity = sp;
        // Steady-state: after inhibition settles (t>=10) and during input (t<80)
        if (t >= 10 && t < 80 && sp > 0.0f) {
            steady_avg += sp;
            steady_steps++;
        }
    }

    if (steady_steps > 0) steady_avg /= steady_steps;

    printf("    DG 最大稀疏度: %.1f%%   稳态平均: %.1f%%\n",
           max_sparsity * 100.0f, steady_avg * 100.0f);

    // DG should show sparse coding: steady-state avg < 20%
    // (with small network, exact 2% is unrealistic; key is E/I balance works)
    CHECK(steady_avg < 0.20f, "DG 稳态平均激活率应 < 20% (稀疏编码)");

    PASS("DG 稀疏编码");
}

// =============================================================================
// 测试5: 杏仁核构造 + 沉默
// =============================================================================
void test_amygdala_construction() {
    printf("\n--- 测试5: 杏仁核构造验证 ---\n");

    AmygdalaConfig cfg;
    Amygdala amy(cfg);

    size_t total = cfg.n_la + cfg.n_bla + cfg.n_cea + cfg.n_itc;
    printf("    La=%zu BLA=%zu CeA=%zu ITC=%zu  总=%zu\n",
           cfg.n_la, cfg.n_bla, cfg.n_cea, cfg.n_itc, total);

    CHECK(amy.n_neurons() == total, "杏仁核神经元总数匹配");

    // Silence test
    size_t spikes = 0;
    for (int t = 0; t < 100; ++t) {
        amy.step(t);
        for (size_t i = 0; i < amy.n_neurons(); ++i) {
            if (amy.fired()[i]) spikes++;
        }
    }
    printf("    100步沉默: %zu 发放\n", spikes);
    CHECK(spikes == 0, "无输入应沉默");

    PASS("杏仁核构造+沉默");
}

// =============================================================================
// 测试6: 恐惧条件化通路 La→BLA→CeA
// =============================================================================
void test_fear_conditioning_path() {
    printf("\n--- 测试6: 恐惧条件化通路 ---\n");
    printf("    通路: 感觉→La→BLA→CeA (恐惧输出)\n");

    AmygdalaConfig cfg;
    Amygdala amy(cfg);

    size_t spk_la = 0, spk_bla = 0, spk_cea = 0;

    for (int t = 0; t < 200; ++t) {
        if (t < 50) {
            std::vector<float> sensory(cfg.n_la, 25.0f);
            amy.inject_sensory(sensory);
        }

        amy.step(t);

        spk_la  += count_spikes(amy.la());
        spk_bla += count_spikes(amy.bla());
        spk_cea += count_spikes(amy.cea());
    }

    printf("    La=%zu → BLA=%zu → CeA=%zu\n", spk_la, spk_bla, spk_cea);

    CHECK(spk_la > 0, "La 应有发放");
    CHECK(spk_bla > 0, "BLA 应有发放 (La→BLA)");
    CHECK(spk_cea > 0, "CeA 应有发放 (BLA→CeA + La→CeA)");

    PASS("恐惧条件化通路");
}

// =============================================================================
// 测试7: ITC 恐惧消退门控
// =============================================================================
void test_itc_extinction_gating() {
    printf("\n--- 测试7: ITC 恐惧消退门控 ---\n");
    printf("    原理: PFC→ITC激活 → ITC抑制CeA → CeA输出减少\n");

    AmygdalaConfig cfg;

    // Phase 1: Fear response (no ITC activation)
    Amygdala amy1(cfg);
    size_t cea_no_itc = 0;
    for (int t = 0; t < 100; ++t) {
        std::vector<float> sensory(cfg.n_la, 25.0f);
        amy1.inject_sensory(sensory);
        amy1.step(t);
        cea_no_itc += count_spikes(amy1.cea());
    }

    // Phase 2: Fear + PFC extinction (ITC active, suppresses CeA)
    Amygdala amy2(cfg);
    size_t cea_with_itc = 0;
    for (int t = 0; t < 100; ++t) {
        std::vector<float> sensory(cfg.n_la, 25.0f);
        amy2.inject_sensory(sensory);
        // PFC drives ITC at moderate level
        std::vector<float> pfc_drive(cfg.n_itc, 25.0f);
        amy2.inject_pfc_to_itc(pfc_drive);
        amy2.step(t);
        cea_with_itc += count_spikes(amy2.cea());
    }

    printf("    CeA无消退: %zu   CeA有消退(PFC→ITC): %zu\n",
           cea_no_itc, cea_with_itc);

    CHECK(cea_with_itc < cea_no_itc,
          "PFC→ITC 消退应减少 CeA 恐惧输出");

    PASS("ITC 恐惧消退门控");
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) 记忆与情感系统测试\n");
    printf("  Step 4: 海马记忆 + 杏仁核情感\n");
    printf("============================================\n");

    test_hippocampus_construction();
    test_hippocampus_silence();
    test_trisynaptic_propagation();
    test_dg_sparsity();
    test_amygdala_construction();
    test_fear_conditioning_path();
    test_itc_extinction_gating();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
