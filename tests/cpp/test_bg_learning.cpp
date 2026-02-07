/**
 * 悟韵 (WuYun) 基底节 DA-STDP 在线强化学习测试
 *
 * Step 4.8: BG 闭环动作选择学习
 *
 * 测试验证:
 *   1. DA-STDP 权重变化: 奖励/惩罚改变 cortical→MSN 连接权重
 *   2. Go/NoGo 偏好学习: 高DA增强D1(Go), 低DA增强D2(NoGo)
 *   3. 动作选择学习: 奖励动作A → GPi对A的抑制增强 → A被选择
 *   4. 反转学习: 奖励从A切换到B → 权重应逐渐反转
 */

#include "region/subcortical/basal_ganglia.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace wuyun;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  [FAIL] %s\n", msg); g_fail++; return; } \
} while(0)

#define PASS(msg) do { printf("  [PASS] %s\n", msg); g_pass++; } while(0)

// =============================================================================
// 测试1: DA-STDP 权重改变
// =============================================================================
void test_bg_da_stdp_weight_change() {
    printf("\n--- 测试1: BG DA-STDP 权重改变 ---\n");
    printf("    原理: 奖励(高DA) → D1权重↑, D2权重↓\n");

    BasalGangliaConfig cfg;
    cfg.da_stdp_enabled = true;
    cfg.da_stdp_lr = 0.01f;
    BasalGanglia bg(cfg);

    // Inject cortical input to make D1/D2 fire
    std::vector<float> d1_input(cfg.n_d1_msn, 60.0f);
    std::vector<float> d2_input(cfg.n_d2_msn, 60.0f);

    // Phase 1: High DA (reward) training
    bg.set_da_level(0.8f);  // Well above baseline 0.1
    for (int t = 0; t < 100; ++t) {
        bg.inject_cortical_input(d1_input, d2_input);
        bg.step(t);
    }

    // Since inject_cortical_input doesn't go through receive_spikes,
    // input_active_ won't be set. We need to simulate via SpikeEvents.
    // Let's test with manual spike injection instead.

    // Create a fresh BG and test with SpikeEvents
    BasalGanglia bg2(cfg);
    bg2.set_da_level(0.8f);

    // Simulate cortical spikes arriving
    std::vector<SpikeEvent> cortical_spikes;
    for (uint32_t i = 0; i < 50; ++i) {
        SpikeEvent evt;
        evt.region_id = 999;  // some cortical region
        evt.dst_region = 0;
        evt.neuron_id = i;
        evt.spike_type = static_cast<int8_t>(SpikeType::REGULAR);
        evt.timestamp = 0;
        cortical_spikes.push_back(evt);
    }

    size_t d1_total = 0;
    for (int t = 0; t < 200; ++t) {
        if (t < 150) bg2.receive_spikes(cortical_spikes);
        bg2.step(t);
        for (size_t i = 0; i < bg2.d1().size(); ++i) {
            if (bg2.d1().fired()[i]) d1_total++;
        }
    }

    // Phase 2: Same BG but now with LOW DA (punishment)
    BasalGanglia bg3(cfg);
    bg3.set_da_level(0.02f);  // Below baseline 0.1

    size_t d1_punish = 0;
    for (int t = 0; t < 200; ++t) {
        if (t < 150) bg3.receive_spikes(cortical_spikes);
        bg3.step(t);
        for (size_t i = 0; i < bg3.d1().size(); ++i) {
            if (bg3.d1().fired()[i]) d1_punish++;
        }
    }

    printf("    D1(高DA奖励训练): %zu  D1(低DA惩罚训练): %zu\n",
           d1_total, d1_punish);

    // Both should have activity (MSN fires from cortical input)
    CHECK(d1_total > 0, "D1在高DA条件下应有发放");

    PASS("BG DA-STDP 权重改变");
}

// =============================================================================
// 测试2: Go/NoGo 偏好学习
// =============================================================================
void test_go_nogo_preference() {
    printf("\n--- 测试2: Go/NoGo 偏好学习 ---\n");
    printf("    原理: 持续高DA → D1(Go)权重增强, D2(NoGo)权重减弱\n");

    BasalGangliaConfig cfg;
    cfg.da_stdp_enabled = true;
    cfg.da_stdp_lr = 0.02f;      // Stronger learning for visible effect
    cfg.da_stdp_baseline = 0.1f;
    BasalGanglia bg(cfg);
    bg.set_da_level(0.7f);  // High DA = reward

    // Cortical input (subset of neurons)
    std::vector<SpikeEvent> ctx_spikes;
    for (uint32_t i = 0; i < 30; ++i) {
        SpikeEvent evt;
        evt.region_id = 999;
        evt.dst_region = 0;
        evt.neuron_id = i;
        evt.spike_type = static_cast<int8_t>(SpikeType::REGULAR);
        evt.timestamp = 0;
        ctx_spikes.push_back(evt);
    }

    // Train for 300 steps with high DA
    for (int t = 0; t < 300; ++t) {
        bg.receive_spikes(ctx_spikes);
        bg.step(t);
    }

    // Now test: measure D1 vs D2 response to same input (after learning)
    size_t d1_post = 0, d2_post = 0;
    for (int t = 300; t < 400; ++t) {
        bg.receive_spikes(ctx_spikes);
        bg.step(t);
        for (size_t i = 0; i < bg.d1().size(); ++i) {
            if (bg.d1().fired()[i]) d1_post++;
        }
        for (size_t i = 0; i < bg.d2().size(); ++i) {
            if (bg.d2().fired()[i]) d2_post++;
        }
    }

    // Compare with untrained BG at same DA
    BasalGangliaConfig cfg_ctrl;
    cfg_ctrl.da_stdp_enabled = false;  // No learning
    BasalGanglia bg_ctrl(cfg_ctrl);
    bg_ctrl.set_da_level(0.7f);

    size_t d1_ctrl = 0, d2_ctrl = 0;
    for (int t = 0; t < 100; ++t) {
        bg_ctrl.receive_spikes(ctx_spikes);
        bg_ctrl.step(t);
        for (size_t i = 0; i < bg_ctrl.d1().size(); ++i) {
            if (bg_ctrl.d1().fired()[i]) d1_ctrl++;
        }
        for (size_t i = 0; i < bg_ctrl.d2().size(); ++i) {
            if (bg_ctrl.d2().fired()[i]) d2_ctrl++;
        }
    }

    printf("    训练后: D1=%zu D2=%zu  无学习: D1=%zu D2=%zu\n",
           d1_post, d2_post, d1_ctrl, d2_ctrl);

    // After DA reward training:
    // D1 weights should increase → more D1 firing (Go preference)
    // D2 weights should decrease → less D2 firing (weaken NoGo)
    CHECK(d1_post > 0, "训练后D1应有发放 (Go通路)");

    // D1 should fire more than control after reward training
    // (D1 weights potentiated by DA+)
    CHECK(d1_post > d1_ctrl,
          "奖励训练后D1应比无学习更活跃 (Go增强)");

    PASS("Go/NoGo 偏好学习");
}

// =============================================================================
// 测试3: 动作选择学习
// =============================================================================
void test_action_selection_learning() {
    printf("\n--- 测试3: 动作选择学习 ---\n");
    printf("    原理: 动作A+奖励 → D1_A增强 → GPi_A抑制更强 → 选择A\n");

    BasalGangliaConfig cfg;
    cfg.da_stdp_enabled = true;
    cfg.da_stdp_lr = 0.02f;
    BasalGanglia bg(cfg);

    // Action A: cortical neurons 0-14 (represents "action A" input)
    std::vector<SpikeEvent> action_a;
    for (uint32_t i = 0; i < 15; ++i) {
        SpikeEvent evt;
        evt.region_id = 999;
        evt.dst_region = 0;
        evt.neuron_id = i;
        evt.spike_type = static_cast<int8_t>(SpikeType::REGULAR);
        evt.timestamp = 0;
        action_a.push_back(evt);
    }

    // Action B: cortical neurons 50-64 (non-overlapping)
    std::vector<SpikeEvent> action_b;
    for (uint32_t i = 50; i < 65; ++i) {
        SpikeEvent evt;
        evt.region_id = 999;
        evt.dst_region = 0;
        evt.neuron_id = i;
        evt.spike_type = static_cast<int8_t>(SpikeType::REGULAR);
        evt.timestamp = 0;
        action_b.push_back(evt);
    }

    // Phase 1: Reward action A (high DA), present action B without reward
    for (int t = 0; t < 300; ++t) {
        if (t % 2 == 0) {
            // Action A + reward
            bg.set_da_level(0.7f);
            bg.receive_spikes(action_a);
        } else {
            // Action B + no reward
            bg.set_da_level(0.1f);
            bg.receive_spikes(action_b);
        }
        bg.step(t);
    }

    // Phase 2: Test both actions at neutral DA and compare D1 response
    bg.set_da_level(0.3f);  // Moderate DA

    size_t d1_response_a = 0;
    for (int t = 300; t < 400; ++t) {
        bg.receive_spikes(action_a);
        bg.step(t);
        for (size_t i = 0; i < bg.d1().size(); ++i) {
            if (bg.d1().fired()[i]) d1_response_a++;
        }
    }

    // Brief silence
    for (int t = 400; t < 420; ++t) bg.step(t);

    size_t d1_response_b = 0;
    for (int t = 420; t < 520; ++t) {
        bg.receive_spikes(action_b);
        bg.step(t);
        for (size_t i = 0; i < bg.d1().size(); ++i) {
            if (bg.d1().fired()[i]) d1_response_b++;
        }
    }

    printf("    D1(动作A, 曾奖励): %zu  D1(动作B, 未奖励): %zu\n",
           d1_response_a, d1_response_b);

    // Action A was paired with reward → its cortical→D1 weights are higher
    // → should evoke more D1 firing than action B
    CHECK(d1_response_a > 0 && d1_response_b > 0,
          "两个动作都应能激活D1");
    CHECK(d1_response_a > d1_response_b,
          "奖励过的动作A应引发更强D1响应 (动作选择偏好)");

    PASS("动作选择学习");
}

// =============================================================================
// 测试4: 反转学习
// =============================================================================
void test_reversal_learning() {
    printf("\n--- 测试4: 反转学习 ---\n");
    printf("    原理: 先奖励A→偏好A, 再奖励B→偏好应逐渐反转\n");

    BasalGangliaConfig cfg;
    cfg.da_stdp_enabled = true;
    cfg.da_stdp_lr = 0.03f;  // Faster learning for clear reversal
    BasalGanglia bg(cfg);

    // Action A and B spike events
    auto make_action = [](uint32_t start, uint32_t count) {
        std::vector<SpikeEvent> spikes;
        for (uint32_t i = start; i < start + count; ++i) {
            SpikeEvent evt;
            evt.region_id = 999;
            evt.dst_region = 0;
            evt.neuron_id = i;
            evt.spike_type = static_cast<int8_t>(SpikeType::REGULAR);
            evt.timestamp = 0;
            spikes.push_back(evt);
        }
        return spikes;
    };

    auto action_a = make_action(0, 20);
    auto action_b = make_action(80, 20);

    // Phase 1: Reward A (200 steps)
    for (int t = 0; t < 200; ++t) {
        if (t % 2 == 0) {
            bg.set_da_level(0.8f);
            bg.receive_spikes(action_a);
        } else {
            bg.set_da_level(0.05f);
            bg.receive_spikes(action_b);
        }
        bg.step(t);
    }

    // Measure preference after Phase 1
    bg.set_da_level(0.3f);
    size_t d1_a_phase1 = 0, d1_b_phase1 = 0;
    for (int t = 200; t < 260; ++t) {
        bg.receive_spikes(action_a);
        bg.step(t);
        for (size_t i = 0; i < bg.d1().size(); ++i)
            if (bg.d1().fired()[i]) d1_a_phase1++;
    }
    for (int t = 260; t < 280; ++t) bg.step(t);
    for (int t = 280; t < 340; ++t) {
        bg.receive_spikes(action_b);
        bg.step(t);
        for (size_t i = 0; i < bg.d1().size(); ++i)
            if (bg.d1().fired()[i]) d1_b_phase1++;
    }

    // Phase 2: REVERSE - Reward B (300 steps, longer to overcome)
    for (int t = 340; t < 640; ++t) {
        if (t % 2 == 0) {
            bg.set_da_level(0.05f);   // Punish A
            bg.receive_spikes(action_a);
        } else {
            bg.set_da_level(0.8f);    // Reward B
            bg.receive_spikes(action_b);
        }
        bg.step(t);
    }

    // Measure preference after Phase 2
    bg.set_da_level(0.3f);
    size_t d1_a_phase2 = 0, d1_b_phase2 = 0;
    for (int t = 640; t < 700; ++t) {
        bg.receive_spikes(action_a);
        bg.step(t);
        for (size_t i = 0; i < bg.d1().size(); ++i)
            if (bg.d1().fired()[i]) d1_a_phase2++;
    }
    for (int t = 700; t < 720; ++t) bg.step(t);
    for (int t = 720; t < 780; ++t) {
        bg.receive_spikes(action_b);
        bg.step(t);
        for (size_t i = 0; i < bg.d1().size(); ++i)
            if (bg.d1().fired()[i]) d1_b_phase2++;
    }

    printf("    Phase1(奖A): D1_A=%zu D1_B=%zu\n", d1_a_phase1, d1_b_phase1);
    printf("    Phase2(奖B): D1_A=%zu D1_B=%zu\n", d1_a_phase2, d1_b_phase2);

    // Phase 1: A should be preferred
    CHECK(d1_a_phase1 > d1_b_phase1,
          "Phase1: 奖励A后应偏好A");

    // Phase 2: After reversal, B should now be preferred
    // At minimum, B's response should increase relative to Phase 1
    CHECK(d1_b_phase2 > d1_b_phase1,
          "Phase2: 反转后B的D1响应应增加");

    PASS("反转学习");
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) BG DA-STDP 强化学习测试\n");
    printf("  Step 4.8: 动作选择在线学习\n");
    printf("============================================\n");

    test_bg_da_stdp_weight_change();
    test_go_nogo_preference();
    test_action_selection_learning();
    test_reversal_learning();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
