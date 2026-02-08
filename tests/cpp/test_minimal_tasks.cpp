/**
 * test_minimal_tasks.cpp — DA-STDP 能力下限诊断
 *
 * Step 25: 用极简任务验证 DA-STDP 的真实学习能力
 * 不经过 ClosedLoopAgent 的复杂管线，直接测试 BG 裸机学习。
 *
 * 三个任务:
 *   1. 2-armed bandit: 两个皮层模式，一个有奖励。能学会选对的吗？
 *   2. Contextual bandit: 模式A→选左，模式B→选右。能学条件关联吗？
 *   3. T-maze (mini agent): 3×1 grid，食物在一端。最简单的闭环空间决策。
 *
 * 如果这些都学不会 → DA-STDP 本身有问题，需要换学习机制
 * 如果能学会 → 问题在 10×10 GridWorld 太复杂，需要更好的表征
 */

#include "region/subcortical/basal_ganglia.h"
#include "engine/closed_loop_agent.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace wuyun;

static int g_pass = 0, g_fail = 0;
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); g_fail++; return; } \
} while(0)

// =========================================================================
// Task 1: 2-Armed Bandit (纯 BG 裸机)
//
// 设定: 两个皮层模式 (Pattern A = slots 0-9, Pattern B = slots 10-19)
//       每试次随机呈现一个模式
//       Pattern A 有 80% 概率给奖励 (DA=0.8)
//       Pattern B 有 20% 概率给奖励 (DA=0.35)
//       BG D1 分两组: group 0 = "选A", group 1 = "选B"
//       看 D1 group 0 (选A) 的权重是否 > group 1 (选B)
// =========================================================================
static void test_2armed_bandit() {
    printf("\n--- Task 1: 2-Armed Bandit (DA-STDP 裸机) ---\n");

    BasalGangliaConfig cfg;
    cfg.n_d1_msn = 20;
    cfg.n_d2_msn = 20;
    cfg.n_gpi = 8;
    cfg.n_gpe = 8;
    cfg.n_stn = 6;
    cfg.da_stdp_enabled = true;
    cfg.da_stdp_lr = 0.01f;
    cfg.da_stdp_elig_decay = 0.95f;
    cfg.da_stdp_w_decay = 0.001f;
    cfg.lateral_inhibition = true;
    cfg.lateral_inh_strength = 5.0f;
    BasalGanglia bg(cfg);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    int choose_a = 0, choose_b = 0;
    int correct = 0;  // chose the pattern that was presented

    for (int trial = 0; trial < 500; ++trial) {
        bool show_a = u01(rng) < 0.5f;

        // Create spike events for the selected pattern
        std::vector<SpikeEvent> events;
        int base = show_a ? 0 : 10;
        for (int i = base; i < base + 10; ++i) {
            SpikeEvent e;
            e.region_id = 999;
            e.neuron_id = static_cast<uint32_t>(i);
            e.spike_type = static_cast<int8_t>(SpikeType::REGULAR);
            e.timestamp = 0;
            events.push_back(e);
        }

        // Run 5 brain steps with pattern injection
        for (int s = 0; s < 5; ++s) {
            bg.receive_spikes(events);
            bg.step(trial * 5 + s, 1.0f);
        }

        // Check which D1 subgroup fired more (0=A, 1=B)
        int d1_a = 0, d1_b = 0;
        size_t d1_half = bg.d1().size() / 2;
        for (size_t i = 0; i < d1_half; ++i)
            if (bg.d1().fired()[i]) d1_a++;
        for (size_t i = d1_half; i < bg.d1().size(); ++i)
            if (bg.d1().fired()[i]) d1_b++;

        bool chose_a = (d1_a >= d1_b);
        if (chose_a) choose_a++; else choose_b++;
        if ((show_a && chose_a) || (!show_a && !chose_a)) correct++;

        // Deliver reward: A=80% reward, B=20% reward
        float reward_prob = show_a ? 0.80f : 0.20f;
        bool rewarded = u01(rng) < reward_prob;
        float da = rewarded ? 0.7f : 0.15f;

        // DA modulation step
        bg.set_da_level(da);
        for (int s = 0; s < 3; ++s) {
            bg.step(trial * 5 + 5 + s, 1.0f);
        }
        bg.set_da_level(0.3f);  // reset baseline
    }

    // Check D1 weights for pattern A vs pattern B slots
    float w_a_sum = 0, w_b_sum = 0;
    int w_a_n = 0, w_b_n = 0;
    for (size_t src = 0; src < 10; ++src) {  // Pattern A slots
        for (size_t idx = 0; idx < bg.d1_weights_for(src).size(); ++idx) {
            w_a_sum += bg.d1_weights_for(src)[idx];
            w_a_n++;
        }
    }
    for (size_t src = 10; src < 20; ++src) {  // Pattern B slots
        for (size_t idx = 0; idx < bg.d1_weights_for(src).size(); ++idx) {
            w_b_sum += bg.d1_weights_for(src)[idx];
            w_b_n++;
        }
    }
    float w_a_avg = w_a_n > 0 ? w_a_sum / w_a_n : 0;
    float w_b_avg = w_b_n > 0 ? w_b_sum / w_b_n : 0;

    float late_accuracy = 0;
    // Re-test last 100 trials accuracy
    int late_correct = 0;
    for (int trial = 0; trial < 100; ++trial) {
        bool show_a = u01(rng) < 0.5f;
        std::vector<SpikeEvent> events;
        int base = show_a ? 0 : 10;
        for (int i = base; i < base + 10; ++i) {
            SpikeEvent e;
            e.region_id = 999;
            e.neuron_id = static_cast<uint32_t>(i);
            e.spike_type = static_cast<int8_t>(SpikeType::REGULAR);
            e.timestamp = 0;
            events.push_back(e);
        }
        for (int s = 0; s < 5; ++s) {
            bg.receive_spikes(events);
            bg.step(5000 + trial * 5 + s, 1.0f);
        }
        int d1_a = 0, d1_b = 0;
        size_t d1_half = bg.d1().size() / 2;
        for (size_t i = 0; i < d1_half; ++i)
            if (bg.d1().fired()[i]) d1_a++;
        for (size_t i = d1_half; i < bg.d1().size(); ++i)
            if (bg.d1().fired()[i]) d1_b++;
        bool chose_a = (d1_a >= d1_b);
        if ((show_a && chose_a) || (!show_a && !chose_a)) late_correct++;
        bg.set_da_level(0.3f);
        bg.step(5000 + trial * 5 + 5, 1.0f);
    }
    late_accuracy = late_correct / 100.0f;

    printf("  Training: chose_A=%d, chose_B=%d, accuracy=%d/500=%.1f%%\n",
           choose_a, choose_b, correct, correct * 100.0f / 500);
    printf("  D1 weights: A_avg=%.4f, B_avg=%.4f, Δ=%+.4f\n",
           w_a_avg, w_b_avg, w_a_avg - w_b_avg);
    printf("  Late test accuracy: %d/100 = %.0f%%\n", late_correct, late_accuracy * 100);

    if (w_a_avg > w_b_avg + 0.01f)
        printf("  Result: Pattern A weights > Pattern B (learned reward contingency)\n");
    else
        printf("  Result: No clear weight difference (failed to learn)\n");

    // The test passes as diagnostic — we just want the data
    TEST_ASSERT(true, "2-armed bandit diagnostic completed");
    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Task 2: Contextual Bandit (条件关联)
//
// 设定: 两个模式 (A, B), 4个动作 (UP/DOWN/LEFT/RIGHT)
//       Pattern A + LEFT → 奖励
//       Pattern A + RIGHT → 无奖励
//       Pattern B + RIGHT → 奖励
//       Pattern B + LEFT → 无奖励
//       能学会 "看到A就选LEFT, 看到B就选RIGHT" 吗?
// =========================================================================
static void test_contextual_bandit() {
    printf("\n--- Task 2: Contextual Bandit (条件关联) ---\n");

    BasalGangliaConfig cfg;
    cfg.n_d1_msn = 40;  // 4 subgroups of 10
    cfg.n_d2_msn = 40;
    cfg.n_gpi = 10;
    cfg.n_gpe = 10;
    cfg.n_stn = 8;
    cfg.da_stdp_enabled = true;
    cfg.da_stdp_lr = 0.01f;
    cfg.da_stdp_elig_decay = 0.95f;
    cfg.da_stdp_w_decay = 0.001f;
    cfg.lateral_inhibition = true;
    cfg.lateral_inh_strength = 5.0f;
    BasalGanglia bg(cfg);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // D1 subgroups: 0-9=UP, 10-19=DOWN, 20-29=LEFT, 30-39=RIGHT
    auto count_subgroup = [&](int g) {
        size_t group_sz = bg.d1().size() / 4;
        size_t start = g * group_sz;
        size_t end = (g < 3) ? (g + 1) * group_sz : bg.d1().size();
        int c = 0;
        for (size_t i = start; i < end; ++i)
            if (bg.d1().fired()[i]) c++;
        return c;
    };

    int correct_total = 0;
    int block_correct[5] = {};  // 5 blocks of 200 trials

    for (int trial = 0; trial < 1000; ++trial) {
        bool show_a = u01(rng) < 0.5f;

        // Pattern injection (distinct cortical patterns)
        std::vector<SpikeEvent> events;
        int base = show_a ? 0 : 30;  // A=slots 0-29, B=slots 30-59
        for (int i = base; i < base + 20; ++i) {
            SpikeEvent e;
            e.region_id = 999;
            e.neuron_id = static_cast<uint32_t>(i);
            e.spike_type = static_cast<int8_t>(SpikeType::REGULAR);
            e.timestamp = 0;
            events.push_back(e);
        }

        // Also mark motor efference for exploration (random action)
        int explore_action = static_cast<int>(rng() % 4);
        bg.mark_motor_efference(explore_action);

        for (int s = 0; s < 5; ++s) {
            bg.receive_spikes(events);
            bg.step(trial * 8 + s, 1.0f);
        }

        // Read D1 subgroup activity
        int d1_counts[4];
        for (int g = 0; g < 4; ++g) d1_counts[g] = count_subgroup(g);

        // Winner = action with most D1 fires
        int chosen = 0;
        for (int g = 1; g < 4; ++g)
            if (d1_counts[g] > d1_counts[chosen]) chosen = g;

        // Correct: A→LEFT(2), B→RIGHT(3)
        int correct_action = show_a ? 2 : 3;
        bool is_correct = (chosen == correct_action);
        if (is_correct) {
            correct_total++;
            block_correct[trial / 200]++;
        }

        // Reward based on actual (explored) action matching rule
        bool rewarded = (show_a && explore_action == 2) ||
                        (!show_a && explore_action == 3);
        float da = rewarded ? 0.7f : 0.15f;

        bg.set_da_level(da);
        for (int s = 0; s < 3; ++s) bg.step(trial * 8 + 5 + s, 1.0f);
        bg.set_da_level(0.3f);
    }

    printf("  Overall accuracy: %d/1000 = %.1f%% (chance=25%%)\n",
           correct_total, correct_total * 0.1f);
    printf("  By block (200 trials each):\n");
    for (int b = 0; b < 5; ++b) {
        printf("    Block %d: %d/200 = %.0f%%\n", b + 1,
               block_correct[b], block_correct[b] * 100.0f / 200);
    }

    float early = (block_correct[0] + block_correct[1]) / 400.0f;
    float late = (block_correct[3] + block_correct[4]) / 400.0f;
    printf("  Early (1-2): %.1f%%, Late (4-5): %.1f%%, Improvement: %+.1f%%\n",
           early * 100, late * 100, (late - early) * 100);

    if (late > early + 0.02f)
        printf("  Result: LEARNING DETECTED (late > early)\n");
    else if (late > 0.30f)
        printf("  Result: ABOVE CHANCE but no clear improvement\n");
    else
        printf("  Result: AT CHANCE — DA-STDP failed on this task\n");

    TEST_ASSERT(true, "Contextual bandit diagnostic completed");
    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Task 3: T-Maze (极简闭环 — 1×3 grid)
//
// 设定: 3格走廊 [FOOD] [AGENT] [EMPTY]  (或反过来)
//       Agent 只能选 LEFT 或 RIGHT
//       食物固定在左边
//       3×1 视野 = 3 像素
//       能学会一直向左走吗?
// =========================================================================
static void test_tmaze() {
    printf("\n--- Task 3: T-Maze (极简闭环 1x3) ---\n");

    AgentConfig cfg;
    cfg.world_config.width = 3;
    cfg.world_config.height = 1;
    cfg.world_config.n_food = 1;
    cfg.world_config.n_danger = 0;
    cfg.world_config.vision_radius = 1;  // 3×3, but grid is 1-tall so effective 3×1
    cfg.world_config.seed = 42;

    cfg.enable_da_stdp = true;
    cfg.enable_lhb = false;         // 极简: 无 LHb
    cfg.enable_amygdala = false;    // 极简: 无杏仁核
    cfg.enable_replay = false;     // 极简: 无重放
    cfg.enable_sleep_consolidation = false;
    cfg.enable_predictive_coding = false;
    cfg.enable_cortical_stdp = false;
    cfg.enable_homeostatic = false;
    cfg.fast_eval = true;           // 无海马
    cfg.brain_steps_per_action = 20;

    ClosedLoopAgent agent(cfg);

    printf("  Environment: %zux%zu, food=%zu, vision=%zux%zu\n",
           cfg.world_config.width, cfg.world_config.height,
           cfg.world_config.n_food,
           cfg.world_config.vision_side(), cfg.world_config.vision_side());
    printf("  Brain: V1=%zu, dlPFC=%zu, BG D1=%zu neurons\n",
           agent.v1()->n_neurons(), agent.dlpfc()->n_neurons(),
           agent.bg()->d1().size());

    int left_count = 0, right_count = 0, food_count = 0;
    int block_food[5] = {};

    for (int step = 0; step < 500; ++step) {
        auto result = agent.agent_step();
        if (result.got_food) {
            food_count++;
            block_food[step / 100]++;
        }
        if (agent.last_action() == Action::LEFT) left_count++;
        if (agent.last_action() == Action::RIGHT) right_count++;
    }

    printf("  Actions: LEFT=%d, RIGHT=%d, other=%d\n",
           left_count, right_count, 500 - left_count - right_count);
    printf("  Food: %d/500\n", food_count);
    printf("  By block:\n");
    for (int b = 0; b < 5; ++b) {
        printf("    Block %d: food=%d/100\n", b + 1, block_food[b]);
    }

    float early = (block_food[0] + block_food[1]) / 200.0f;
    float late = (block_food[3] + block_food[4]) / 200.0f;
    printf("  Early food rate: %.1f%%, Late: %.1f%%, Improvement: %+.1f%%\n",
           early * 100, late * 100, (late - early) * 100);

    if (late > early + 0.02f)
        printf("  Result: LEARNING DETECTED\n");
    else if (food_count > 50)
        printf("  Result: FINDS FOOD but no clear improvement\n");
    else
        printf("  Result: STRUGGLES — even T-maze is too hard\n");

    TEST_ASSERT(true, "T-maze diagnostic completed");
    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Task 4: IT 表征质量诊断
//
// 在闭环 agent 中，注入 "食物在左"、"食物在右"、"危险在左"、"空" 四种场景
// 测量 V1/V2/V4/IT/dlPFC 各层对这些场景的响应差异
// 好的视觉层级: IT 对 "食物"(不管方向) 的响应应该相似，且与 "危险" 不同
// 差的视觉层级: IT 对所有场景响应差不多 (没有分离)
// =========================================================================
static void test_it_representation() {
    printf("\n--- Task 4: IT 表征质量诊断 ---\n");

    AgentConfig cfg;
    cfg.world_config.width = 5;
    cfg.world_config.height = 5;
    cfg.world_config.n_food = 0;
    cfg.world_config.n_danger = 0;
    cfg.world_config.vision_radius = 2;  // 5×5 vision
    cfg.world_config.seed = 42;
    cfg.enable_da_stdp = false;
    cfg.enable_lhb = false;
    cfg.enable_amygdala = false;
    cfg.enable_replay = false;
    cfg.enable_sleep_consolidation = false;
    cfg.enable_predictive_coding = false;
    cfg.enable_cortical_stdp = false;
    cfg.enable_homeostatic = false;
    cfg.fast_eval = true;

    ClosedLoopAgent agent(cfg);

    printf("  Brain: V1=%zu, V2=%zu, V4=%zu, IT=%zu, dlPFC=%zu\n",
           agent.v1()->n_neurons(),
           agent.v2() ? agent.v2()->n_neurons() : 0,
           agent.v4() ? agent.v4()->n_neurons() : 0,
           agent.it_ctx() ? agent.it_ctx()->n_neurons() : 0,
           agent.dlpfc()->n_neurons());

    // Define 4 test scenes as 5×5 pixel arrays
    // Scene 0: food on left (pixel [2][0] = 0.9)
    // Scene 1: food on right (pixel [2][4] = 0.9)
    // Scene 2: danger on left (pixel [2][0] = 0.3)
    // Scene 3: all empty
    auto make_scene = [](float left_val, float right_val) {
        std::vector<float> pixels(25, 0.0f);
        pixels[12] = 0.6f;  // center = agent
        pixels[10] = left_val;   // left of center
        pixels[14] = right_val;  // right of center
        return pixels;
    };

    std::vector<std::vector<float>> scenes = {
        make_scene(0.9f, 0.0f),  // food left
        make_scene(0.0f, 0.9f),  // food right
        make_scene(0.3f, 0.0f),  // danger left
        make_scene(0.0f, 0.0f),  // empty
    };
    const char* scene_names[] = {"food_L", "food_R", "danger_L", "empty"};

    // For each scene, inject into LGN and run 20 steps, count fires per region
    printf("\n  Scene      | V1 fires | V2 fires | V4 fires | IT fires | dlPFC fires\n");
    printf("  -----------|----------|----------|----------|----------|----------\n");

    std::vector<int> it_fires_per_scene(4, 0);

    for (int sc = 0; sc < 4; ++sc) {
        // Reset brain state by running empty steps
        for (int s = 0; s < 10; ++s) agent.brain().step();

        // Inject scene via visual encoder → LGN
        VisualInputConfig vcfg;
        vcfg.input_width = 5;
        vcfg.input_height = 5;
        vcfg.n_lgn_neurons = agent.lgn()->n_neurons();
        vcfg.gain = 200.0f;
        vcfg.baseline = 5.0f;
        vcfg.noise_amp = 0.5f;  // low noise for clean signal
        VisualInput encoder(vcfg);

        int v1_total = 0, v2_total = 0, v4_total = 0, it_total = 0, dlpfc_total = 0;

        for (int step = 0; step < 20; ++step) {
            encoder.encode_and_inject(scenes[sc], agent.lgn());
            agent.brain().step();

            // Count fires
            for (auto f : agent.v1()->fired()) v1_total += f;
            if (agent.v2()) for (auto f : agent.v2()->fired()) v2_total += f;
            if (agent.v4()) for (auto f : agent.v4()->fired()) v4_total += f;
            if (agent.it_ctx()) for (auto f : agent.it_ctx()->fired()) it_total += f;
            for (auto f : agent.dlpfc()->fired()) dlpfc_total += f;
        }

        it_fires_per_scene[sc] = it_total;

        printf("  %-10s | %8d | %8d | %8d | %8d | %8d\n",
               scene_names[sc], v1_total, v2_total, v4_total, it_total, dlpfc_total);
    }

    // Measure representation quality
    int it_food_avg = (it_fires_per_scene[0] + it_fires_per_scene[1]) / 2;
    int it_danger = it_fires_per_scene[2];
    int it_empty = it_fires_per_scene[3];

    printf("\n  IT food avg: %d, danger: %d, empty: %d\n", it_food_avg, it_danger, it_empty);

    float food_danger_ratio = (it_danger > 0) ? (float)it_food_avg / it_danger : 99.0f;
    float food_empty_ratio = (it_empty > 0) ? (float)it_food_avg / it_empty : 99.0f;

    printf("  IT food/danger ratio: %.2f (>1.5 = good separation)\n", food_danger_ratio);
    printf("  IT food/empty ratio: %.2f (>1.5 = good separation)\n", food_empty_ratio);

    // Position invariance: food_L vs food_R should be similar
    int diff_lr = std::abs(it_fires_per_scene[0] - it_fires_per_scene[1]);
    int denom = it_food_avg > 1 ? it_food_avg : 1;
    float invariance = 1.0f - (float)diff_lr / (float)denom;
    printf("  IT position invariance: %.2f (>0.7 = good, food_L=%d vs food_R=%d)\n",
           invariance, it_fires_per_scene[0], it_fires_per_scene[1]);

    if (food_danger_ratio > 1.3f && invariance > 0.5f)
        printf("  Result: IT has useful representations (food/danger separate, position invariant)\n");
    else if (food_danger_ratio > 1.1f)
        printf("  Result: IT has WEAK separation (barely distinguishes food from danger)\n");
    else
        printf("  Result: IT has NO useful separation (all scenes look the same)\n");

    TEST_ASSERT(true, "IT representation diagnostic completed");
    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Task 5: Ablation Study — 逐个关闭学习环节, 测量贡献
// =========================================================================
static void test_ablation() {
    printf("\n--- Task 5: Ablation Study (逐个关闭) ---\n");

    struct AblationConfig {
        const char* name;
        // Lambda to modify AgentConfig
        std::function<void(AgentConfig&)> modify;
    };

    AblationConfig configs[] = {
        {"全开 (baseline)",     [](AgentConfig&) {}},
        {"关 cortical STDP",    [](AgentConfig& c) { c.enable_cortical_stdp = false; }},
        {"关 predictive coding",[](AgentConfig& c) { c.enable_predictive_coding = false; }},
        {"关 amygdala",         [](AgentConfig& c) { c.enable_amygdala = false; }},
        {"关 hippocampus",      [](AgentConfig& c) { c.fast_eval = true; }},
        {"关 cerebellum",       [](AgentConfig& c) { c.enable_cerebellum = false; }},
        {"关 LHb",              [](AgentConfig& c) { c.enable_lhb = false; }},
        {"关 SWR replay",       [](AgentConfig& c) { c.enable_replay = false; }},
        {"关 sleep consolidation",[](AgentConfig& c){ c.enable_sleep_consolidation = false; }},
        {"关 synaptic consol",  [](AgentConfig& c) { c.enable_synaptic_consolidation = false; }},
        {"关 interleaved replay",[](AgentConfig& c){ c.enable_interleaved_replay = false; }},
        {"关 LC-NE",            [](AgentConfig& c) { c.enable_lc_ne = false; }},
        {"关 NBM-ACh",          [](AgentConfig& c) { c.enable_nbm_ach = false; }},
        {"关 DRN-5HT",          [](AgentConfig& c) { c.enable_drn_5ht = false; }},
        // v43: Step 40-42 新区域消融
        {"关 NAcc",              [](AgentConfig& c) { c.enable_nacc = false; }},
        {"关 SNc",               [](AgentConfig& c) { c.enable_snc = false; }},
        {"关 SC",                [](AgentConfig& c) { c.enable_sc = false; }},
        {"关 PAG",               [](AgentConfig& c) { c.enable_pag = false; }},
        {"关 FPC",               [](AgentConfig& c) { c.enable_fpc = false; }},
        {"关 OFC",               [](AgentConfig& c) { c.enable_ofc = false; }},
        {"关 vmPFC",             [](AgentConfig& c) { c.enable_vmpfc = false; }},
        {"关 all_new (回Step39)",[](AgentConfig& c) {
            c.enable_nacc = false; c.enable_snc = false; c.enable_sc = false;
            c.enable_pag = false; c.enable_fpc = false;
            c.enable_ofc = false; c.enable_vmpfc = false;
        }},
    };

    int n_configs = sizeof(configs) / sizeof(configs[0]);
    float baseline_safety = 0;

    printf("  %-25s | food | danger | safety | Δ safety\n", "Config");
    printf("  %-25s-|------|--------|--------|----------\n", "-------------------------");

    // 3 seeds 取平均, 减少随机波动
    uint32_t seeds[] = {42, 77, 123};
    int n_seeds = 3;

    for (int i = 0; i < n_configs; ++i) {
        int food_total = 0, danger_total = 0;
        for (int si = 0; si < n_seeds; ++si) {
            AgentConfig cfg;
            cfg.enable_da_stdp = true;
            cfg.world_config.seed = seeds[si];
            configs[i].modify(cfg);

            ClosedLoopAgent agent(cfg);
            for (int s = 0; s < 500; ++s) agent.agent_step();
            for (int s = 0; s < 500; ++s) {
                auto r = agent.agent_step();
                if (r.got_food) food_total++;
                if (r.hit_danger) danger_total++;
            }
        }
        float safety = (food_total + danger_total > 0)
            ? (float)food_total / (food_total + danger_total) : 0.5f;

        if (i == 0) baseline_safety = safety;
        float delta = safety - baseline_safety;

        printf("  %-25s | %4d | %6d |  %.2f  | %+.2f %s\n",
               configs[i].name, food_total, danger_total, safety, delta,
               (i > 0 && delta > 0.03f) ? "(有害)" :
               (i > 0 && delta < -0.03f) ? "(有用)" : "(中性)");
    }

    // 结论
    printf("\n  解读: Δ > 0 = 关掉后变好(该环节有害); Δ < 0 = 关掉后变差(有用)\n");

    TEST_ASSERT(true, "Ablation study completed");
    printf("  [PASS]\n"); g_pass++;
}

// =========================================================================
// Task 6: Maze spatial navigation (v48)
// =========================================================================
void test_maze() {
    printf("\n--- Task 6: Maze Spatial Navigation ---\n");

    // --- 6A: Corridor (simplest: just go right) ---
    {
        printf("\n  6A: Corridor (10x3, go right to food)\n");
        AgentConfig cfg;
        cfg.world_config.maze_type = MazeType::CORRIDOR;
        cfg.world_config.seed = 42;
        cfg.dev_period_steps = 0;  // No dev period in maze (start learning immediately)

        ClosedLoopAgent agent(cfg);
        printf("  Layout:\n%s\n", agent.world().to_string().c_str());

        int food_count = 0;
        int wall_hits = 0;
        for (int i = 0; i < 1000; ++i) {
            auto result = agent.agent_step();
            if (result.got_food) food_count++;
            if (result.hit_wall) wall_hits++;
            if (i % 200 == 199) {
                printf("    Step %4d: pos=(%d,%d) food=%d walls=%d\n",
                       i + 1, agent.world().agent_x(), agent.world().agent_y(),
                       food_count, wall_hits);
            }
        }
        printf("  Corridor result: food=%d, wall_hits=%d\n", food_count, wall_hits);
    }

    // --- 6B: T-maze (choice point: left=food, right=empty) ---
    {
        printf("\n  6B: T-maze (5x5, left=food)\n");
        AgentConfig cfg;
        cfg.world_config.maze_type = MazeType::T_MAZE;
        cfg.world_config.seed = 42;
        cfg.dev_period_steps = 0;

        ClosedLoopAgent agent(cfg);
        printf("  Layout:\n%s\n", agent.world().to_string().c_str());

        // Track visits per cell (5x5)
        int visit_count[25] = {};
        int w = static_cast<int>(agent.world().width());

        int food_count = 0;
        int wall_hits = 0;
        for (int i = 0; i < 2000; ++i) {
            auto result = agent.agent_step();
            if (result.got_food) {
                food_count++;
                printf("    *** FOOD at step %d pos=(%d,%d) ***\n", i, result.agent_x, result.agent_y);
            }
            if (result.hit_wall) wall_hits++;
            int px = agent.world().agent_x();
            int py = agent.world().agent_y();
            if (px >= 0 && px < w && py >= 0 && py < (int)agent.world().height()) {
                visit_count[py * w + px]++;
            }
            if (i % 500 == 499) {
                printf("    Step %4d: pos=(%d,%d) food=%d walls=%d action=%d\n",
                       i + 1, px, py, food_count, wall_hits,
                       static_cast<int>(agent.last_action()));
            }
        }
        printf("  T-maze result: food=%d, wall_hits=%d\n", food_count, wall_hits);
        printf("  Visit counts per cell:\n");
        for (int y = 0; y < (int)agent.world().height(); ++y) {
            printf("    ");
            for (int x = 0; x < w; ++x) {
                printf("%4d ", visit_count[y * w + x]);
            }
            printf("\n");
        }
    }

    TEST_ASSERT(true, "Maze navigation completed");
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
    printf("=== 悟韵 学习链路诊断 ===\n");

    test_ablation();
    test_maze();

    printf("\n========================================\n");
    printf("  通过: %d / %d\n", g_pass, g_pass + g_fail);
    printf("========================================\n");

    return g_fail > 0 ? 1 : 0;
}
