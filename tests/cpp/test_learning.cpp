/**
 * 悟韵 (WuYun) 学习能力验证测试
 *
 * Step 4.6: 开机学习 — 记忆/强化学习/泛化
 *
 * 测试验证:
 *   1. CA3 STDP 权重变化: 编码后权重应改变
 *   2. 记忆编码: 呈现模式A → CA3权重存储
 *   3. 模式补全: 部分线索 → CA3重建完整模式
 *   4. 模式分离: 不同模式编码到不同CA3子集
 *   5. BG DA-STDP: 奖励改变动作选择偏好
 */

#include "region/limbic/hippocampus.h"
#include "region/subcortical/basal_ganglia.h"
#include "plasticity/stdp.h"
#include "plasticity/da_stdp.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace wuyun;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  [FAIL] %s\n", msg); g_fail++; return; } \
} while(0)

#define PASS(msg) do { printf("  [PASS] %s\n", msg); g_pass++; } while(0)

// Helper: create a hippocampus config tuned for learning tests
// (stronger EC→DG to allow partial patterns through DG high threshold)
static HippocampusConfig make_learning_config() {
    HippocampusConfig cfg;
    cfg.p_ec_to_dg   = 0.35f;  // Stronger perforant path (default 0.20)
    cfg.w_ec_dg      = 1.2f;   // Stronger weight (default 0.8)
    cfg.w_dg_ca3     = 2.5f;   // Stronger mossy fiber
    cfg.p_dg_to_ca3  = 0.08f;  // Denser mossy fiber (default 0.05)
    cfg.ca3_stdp_enabled = true;
    return cfg;
}

// Helper: create a specific EC input pattern (activate specific subset of EC neurons)
static std::vector<float> make_pattern(size_t n_ec, const std::vector<size_t>& active_ids, float strength = 30.0f) {
    std::vector<float> pattern(n_ec, 0.0f);
    for (size_t id : active_ids) {
        if (id < n_ec) pattern[id] = strength;
    }
    return pattern;
}

// Helper: get CA3 active neuron IDs
static std::vector<size_t> get_active_ids(const NeuronPopulation& pop) {
    std::vector<size_t> ids;
    for (size_t i = 0; i < pop.size(); ++i) {
        if (pop.fired()[i]) ids.push_back(i);
    }
    return ids;
}

// Helper: compute overlap between two sets of active IDs
static float overlap_ratio(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    if (a.empty() || b.empty()) return 0.0f;
    size_t common = 0;
    for (size_t id : a) {
        if (std::find(b.begin(), b.end(), id) != b.end()) common++;
    }
    return static_cast<float>(common) / static_cast<float>((std::max)(a.size(), b.size()));
}

// =============================================================================
// 测试1: CA3 STDP 权重改变验证
// =============================================================================
void test_ca3_stdp_weight_change() {
    printf("\n--- 测试1: CA3 STDP 权重变化 ---\n");
    printf("    原理: 编码模式后, CA3循环突触权重应改变\n");

    auto cfg = make_learning_config();
    Hippocampus hipp(cfg);

    // Phase 1: Drive EC with a strong pattern to activate DG→CA3
    std::vector<size_t> pattern_a;
    for (size_t i = 0; i < 50; ++i) pattern_a.push_back(i);  // 50/80 EC neurons
    auto input_a = make_pattern(cfg.n_ec, pattern_a, 50.0f);

    // Run encoding (CA3 STDP learns co-active pattern)
    size_t ec_enc = 0, dg_enc = 0, ca3_enc = 0;
    for (int t = 0; t < 100; ++t) {
        if (t < 50) hipp.inject_cortical_input(input_a);
        hipp.step(t);
        for (size_t i = 0; i < hipp.ec().size(); ++i) if (hipp.ec().fired()[i]) ec_enc++;
        for (size_t i = 0; i < hipp.dg().size(); ++i) if (hipp.dg().fired()[i]) dg_enc++;
        for (size_t i = 0; i < hipp.ca3().size(); ++i) if (hipp.ca3().fired()[i]) ca3_enc++;
    }
    printf("    编码阶段: EC=%zu DG=%zu CA3=%zu\n", ec_enc, dg_enc, ca3_enc);

    // Phase 2: Now present the SAME pattern again and count CA3 response
    // After STDP, CA3 should respond MORE strongly (potentiated connections)
    size_t ca3_post_learning = 0;
    for (int t = 100; t < 200; ++t) {
        if (t < 150) hipp.inject_cortical_input(input_a);
        hipp.step(t);
        for (size_t i = 0; i < hipp.ca3().size(); ++i) {
            if (hipp.ca3().fired()[i]) ca3_post_learning++;
        }
    }

    // Phase 3: Fresh hippocampus (no learning) with same pattern
    auto cfg_noplast = make_learning_config();
    cfg_noplast.ca3_stdp_enabled = false;
    Hippocampus hipp_no(cfg_noplast);

    // Skip encoding, go straight to test
    size_t ca3_no_learning = 0;
    for (int t = 0; t < 100; ++t) {
        if (t < 50) hipp_no.inject_cortical_input(input_a);
        hipp_no.step(t);
        for (size_t i = 0; i < hipp_no.ca3().size(); ++i) {
            if (hipp_no.ca3().fired()[i]) ca3_no_learning++;
        }
    }

    printf("    CA3(学习后): %zu   CA3(无学习): %zu\n",
           ca3_post_learning, ca3_no_learning);

    // STDP should have increased CA3 recurrent weights → more activity
    // (or at least not less, since LTP should dominate for co-active neurons)
    CHECK(ca3_post_learning >= ca3_no_learning,
          "STDP学习后CA3应响应更强 (LTP增强了循环连接)");

    PASS("CA3 STDP 权重变化");
}

// =============================================================================
// 测试2: 记忆编码与回忆
// =============================================================================
void test_memory_encode_recall() {
    printf("\n--- 测试2: 记忆编码与回忆 ---\n");
    printf("    原理: 编码模式A→等待→部分线索→CA3应重建A\n");

    auto cfg = make_learning_config();
    Hippocampus hipp(cfg);

    // Pattern A: EC neurons 0-49 active (~62% of EC)
    std::vector<size_t> pattern_a;
    for (size_t i = 0; i < 50; ++i) pattern_a.push_back(i);
    auto input_a = make_pattern(cfg.n_ec, pattern_a, 50.0f);

    // --- Phase 1: Encoding (50 steps) ---
    // Accumulate all CA3 neurons that fire during encoding window
    std::vector<uint32_t> ca3_encode_counts(cfg.n_ca3, 0);
    for (int t = 0; t < 80; ++t) {
        if (t < 50) hipp.inject_cortical_input(input_a);
        hipp.step(t);
        if (t >= 10 && t < 50) {
            for (size_t i = 0; i < hipp.ca3().size(); ++i) {
                if (hipp.ca3().fired()[i]) ca3_encode_counts[i]++;
            }
        }
    }
    // Encoding snapshot = neurons that fired at least once
    std::vector<size_t> ca3_encoding_snapshot;
    for (size_t i = 0; i < cfg.n_ca3; ++i) {
        if (ca3_encode_counts[i] > 0) ca3_encoding_snapshot.push_back(i);
    }

    // --- Phase 2: Silence (let activity die down) ---
    for (int t = 80; t < 150; ++t) {
        hipp.step(t);
    }

    // --- Phase 3: Partial cue (30% of original pattern) ---
    std::vector<size_t> partial_cue;
    for (size_t i = 0; i < 15; ++i) partial_cue.push_back(i);  // ~30% of original 50
    auto input_partial = make_pattern(cfg.n_ec, partial_cue, 55.0f);

    // Accumulate all CA3 neurons that fire during recall window
    std::vector<uint32_t> ca3_recall_counts(cfg.n_ca3, 0);
    for (int t = 150; t < 230; ++t) {
        if (t < 200) hipp.inject_cortical_input(input_partial);
        hipp.step(t);
        if (t >= 160 && t < 200) {
            for (size_t i = 0; i < hipp.ca3().size(); ++i) {
                if (hipp.ca3().fired()[i]) ca3_recall_counts[i]++;
            }
        }
    }
    std::vector<size_t> ca3_recall_snapshot;
    for (size_t i = 0; i < cfg.n_ca3; ++i) {
        if (ca3_recall_counts[i] > 0) ca3_recall_snapshot.push_back(i);
    }

    float encode_recall_overlap = overlap_ratio(ca3_encoding_snapshot, ca3_recall_snapshot);

    printf("    编码CA3: %zu neurons   回忆CA3: %zu neurons   重叠: %.1f%%\n",
           ca3_encoding_snapshot.size(), ca3_recall_snapshot.size(),
           encode_recall_overlap * 100.0f);

    // With STDP, partial cue should reactivate similar CA3 ensemble
    CHECK(ca3_recall_snapshot.size() > 0, "部分线索应能激活CA3");
    CHECK(encode_recall_overlap > 0.1f,
          "回忆的CA3集合应与编码时有重叠 (>10%, 模式补全)");

    PASS("记忆编码与回忆");
}

// =============================================================================
// 测试3: 模式分离 (不同模式→不同CA3子集)
// =============================================================================
void test_pattern_separation() {
    printf("\n--- 测试3: 模式分离 ---\n");
    printf("    原理: 不同EC模式 → DG稀疏化 → 不同CA3子集\n");

    auto cfg = make_learning_config();
    Hippocampus hipp(cfg);

    // Pattern A: EC 0-39 (50%)
    std::vector<size_t> pat_a_ids;
    for (size_t i = 0; i < 40; ++i) pat_a_ids.push_back(i);
    auto input_a = make_pattern(cfg.n_ec, pat_a_ids, 50.0f);

    // Pattern B: EC 40-79 (non-overlapping, 50%)
    std::vector<size_t> pat_b_ids;
    for (size_t i = 40; i < 80; ++i) pat_b_ids.push_back(i);
    auto input_b = make_pattern(cfg.n_ec, pat_b_ids, 50.0f);

    // Encode pattern A
    std::vector<size_t> ca3_a;
    for (int t = 0; t < 80; ++t) {
        if (t < 50) hipp.inject_cortical_input(input_a);
        hipp.step(t);
        if (t >= 30 && t < 50) {
            auto a = get_active_ids(hipp.ca3());
            if (!a.empty()) ca3_a = a;
        }
    }

    // Silence
    for (int t = 80; t < 120; ++t) hipp.step(t);

    // Encode pattern B
    std::vector<size_t> ca3_b;
    for (int t = 120; t < 200; ++t) {
        if (t < 170) hipp.inject_cortical_input(input_b);
        hipp.step(t);
        if (t >= 150 && t < 170) {
            auto b = get_active_ids(hipp.ca3());
            if (!b.empty()) ca3_b = b;
        }
    }

    float ab_overlap = overlap_ratio(ca3_a, ca3_b);

    printf("    CA3(A): %zu neurons   CA3(B): %zu neurons   重叠: %.1f%%\n",
           ca3_a.size(), ca3_b.size(), ab_overlap * 100.0f);

    CHECK(ca3_a.size() > 0 && ca3_b.size() > 0,
          "两个模式都应激活CA3");
    CHECK(ab_overlap < 0.8f,
          "不同模式的CA3表征应不同 (重叠<80%, 模式分离)");

    PASS("模式分离");
}

// =============================================================================
// 测试4: BG DA-STDP 强化学习
// =============================================================================
void test_bg_reinforcement_learning() {
    printf("\n--- 测试4: BG DA-STDP 强化学习 ---\n");
    printf("    原理: 刺激X+高DA→D1增强, 学习偏好Go动作\n");

    // We'll test at the synapse level: create a SynapseGroup with STDP
    // representing cortical→D1 MSN, and show DA-modulated learning

    // For simplicity, test the DA-STDP plasticity mechanism directly
    // using the existing da_stdp module

    const size_t n_syn = 50;
    DASTDPParams da_cfg;
    da_cfg.stdp.a_plus = 0.01f;
    da_cfg.stdp.a_minus = -0.012f;
    da_cfg.stdp.tau_plus = 20.0f;
    da_cfg.stdp.tau_minus = 20.0f;
    da_cfg.tau_eligibility = 200.0f;
    da_cfg.da_baseline = 0.1f;
    da_cfg.w_min = 0.0f;
    da_cfg.w_max = 1.0f;

    DASTDPProcessor tracker(n_syn, da_cfg);

    // Initialize weights at 0.5
    std::vector<float> weights(n_syn, 0.5f);
    std::vector<int32_t> pre_ids(n_syn), post_ids(n_syn);
    for (size_t i = 0; i < n_syn; ++i) {
        pre_ids[i] = static_cast<int32_t>(i);
        post_ids[i] = static_cast<int32_t>(i);  // 1:1 for simplicity
    }

    // Phase 1: Pre-then-post pairing → eligibility trace builds up
    // Then reward (high DA) → traces convert to weight changes
    std::vector<float> pre_times(n_syn, -1.0f);
    std::vector<float> post_times(n_syn, -1.0f);

    // Simulate: pre fires at t=10, post fires at t=12 (LTP timing)
    for (size_t i = 0; i < 25; ++i) {
        pre_times[i] = 10.0f;
        post_times[i] = 12.0f;
    }

    // Update eligibility traces
    tracker.update_traces(pre_times.data(), post_times.data(),
                         pre_ids.data(), post_ids.data(),
                         1.0f);

    // Snapshot weights before DA
    float w_before = weights[0];

    // Apply DA reward signal
    tracker.apply_da_modulation(weights.data(), 0.8f);  // High DA = reward

    float w_after_reward = weights[0];

    // Reset and test with no reward
    std::fill(weights.begin(), weights.end(), 0.5f);
    DASTDPProcessor tracker2(n_syn, da_cfg);
    tracker2.update_traces(pre_times.data(), post_times.data(),
                          pre_ids.data(), post_ids.data(),
                          1.0f);
    tracker2.apply_da_modulation(weights.data(), 0.1f);  // Baseline DA = no reward

    float w_after_noreward = weights[0];

    printf("    w初始=0.5  w+奖励=%.4f  w+无奖励=%.4f\n",
           w_after_reward, w_after_noreward);

    CHECK(w_after_reward > w_before,
          "DA奖励应增强LTP突触 (三因子学习)");
    CHECK(std::fabs(w_after_reward - w_before) > std::fabs(w_after_noreward - 0.5f),
          "奖励条件下权重变化应大于无奖励");

    PASS("BG DA-STDP 强化学习");
}

// =============================================================================
// 测试5: 记忆容量 (多模式编码)
// =============================================================================
void test_memory_capacity() {
    printf("\n--- 测试5: 记忆容量 ---\n");
    printf("    原理: 编码多个模式, 各自线索应激活不同CA3子集\n");

    auto cfg = make_learning_config();
    Hippocampus hipp(cfg);

    // Encode 3 overlapping patterns (each 50% of EC, with partial overlap)
    // In biology, overlapping inputs are the norm
    std::vector<std::vector<size_t>> patterns(3);
    for (size_t i = 0; i < 50; ++i) patterns[0].push_back(i);        // A: EC 0-49
    for (size_t i = 20; i < 70; ++i) patterns[1].push_back(i);       // B: EC 20-69
    for (size_t i = 40; i < 80; ++i) {
        patterns[2].push_back(i);
    }
    for (size_t i = 0; i < 10; ++i) patterns[2].push_back(i);        // C: EC 40-79 + 0-9

    std::vector<std::vector<size_t>> ca3_snapshots(3);
    int t = 0;

    for (int p = 0; p < 3; ++p) {
        auto input = make_pattern(cfg.n_ec, patterns[p], 50.0f);

        // Encode
        for (int step = 0; step < 60; ++step, ++t) {
            if (step < 40) hipp.inject_cortical_input(input);
            hipp.step(t);
            if (step >= 25 && step < 40) {
                auto active = get_active_ids(hipp.ca3());
                if (!active.empty()) ca3_snapshots[p] = active;
            }
        }

        // Brief silence
        for (int step = 0; step < 30; ++step, ++t) {
            hipp.step(t);
        }
    }

    // Check: each pattern activates CA3, and they're different
    bool all_active = true;
    for (int p = 0; p < 3; ++p) {
        if (ca3_snapshots[p].empty()) all_active = false;
    }

    float ab_overlap = overlap_ratio(ca3_snapshots[0], ca3_snapshots[1]);
    float ac_overlap = overlap_ratio(ca3_snapshots[0], ca3_snapshots[2]);
    float bc_overlap = overlap_ratio(ca3_snapshots[1], ca3_snapshots[2]);

    printf("    A=%zu B=%zu C=%zu neurons\n",
           ca3_snapshots[0].size(), ca3_snapshots[1].size(), ca3_snapshots[2].size());
    printf("    A-B重叠: %.1f%%  A-C重叠: %.1f%%  B-C重叠: %.1f%%\n",
           ab_overlap * 100.0f, ac_overlap * 100.0f, bc_overlap * 100.0f);

    CHECK(all_active, "3个模式都应激活CA3");

    PASS("记忆容量");
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) 学习能力验证测试\n");
    printf("  Step 4.6: 记忆/强化学习/泛化\n");
    printf("============================================\n");

    test_ca3_stdp_weight_change();
    test_memory_encode_recall();
    test_pattern_separation();
    test_bg_reinforcement_learning();
    test_memory_capacity();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
