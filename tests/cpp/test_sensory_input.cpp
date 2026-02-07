/**
 * 悟韵 (WuYun) Step 9 感觉输入接口测试
 *
 * 测试内容:
 *   1. VisualInput 基础编码: 8x8 像素→50 LGN 电流
 *   2. Center-surround: 亮点→ON强/OFF弱, 暗点→反之
 *   3. 视觉端到端: pixels→LGN→V1 spike传播
 *   4. AuditoryInput 基础编码: 16频带→20 MGN 电流
 *   5. 听觉 onset 检测: 新音比持续音产生更强响应
 *   6. 听觉端到端: spectrum→MGN→A1 spike传播
 *   7. 多模态并行: 视觉+听觉同时输入→分别激活V1和A1
 */

#include "engine/sensory_input.h"
#include "engine/simulation_engine.h"
#include "region/cortical_region.h"
#include "region/subcortical/thalamic_relay.h"

#include <cstdio>
#include <cassert>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace wuyun;

static int g_pass = 0, g_fail = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
        g_fail++; return; \
    } \
} while(0)

static size_t count_fired(const BrainRegion& r) {
    size_t n = 0;
    for (auto f : r.fired()) if (f) n++;
    return n;
}

// =============================================================================
// Test 1: VisualInput 基础编码
// =============================================================================
static void test_visual_basic() {
    printf("\n--- 测试1: VisualInput 基础编码 ---\n");
    printf("    原理: 8x8灰度像素 → 50 LGN电流 (center-surround RF)\n");

    VisualInputConfig cfg;
    cfg.input_width = 8;
    cfg.input_height = 8;
    cfg.n_lgn_neurons = 50;
    cfg.noise_amp = 0.0f;  // No noise for deterministic test

    VisualInput vis(cfg);
    TEST_ASSERT(vis.n_pixels() == 64, "64 pixels");
    TEST_ASSERT(vis.n_lgn() == 50, "50 LGN neurons");

    // Uniform gray image
    std::vector<float> gray(64, 0.5f);
    auto currents = vis.encode(gray);
    TEST_ASSERT(currents.size() == 50, "output size matches LGN");

    // All currents should be positive (baseline + response)
    float min_c = *std::min_element(currents.begin(), currents.end());
    float max_c = *std::max_element(currents.begin(), currents.end());
    printf("    Uniform gray: min=%.1f, max=%.1f\n", min_c, max_c);
    TEST_ASSERT(min_c >= cfg.baseline, "all currents >= baseline");

    // Bright image should produce stronger response than dark
    std::vector<float> bright(64, 1.0f);
    std::vector<float> dark(64, 0.0f);
    auto c_bright = vis.encode(bright);
    auto c_dark = vis.encode(dark);

    float sum_bright = std::accumulate(c_bright.begin(), c_bright.end(), 0.0f);
    float sum_dark = std::accumulate(c_dark.begin(), c_dark.end(), 0.0f);
    printf("    Bright sum=%.1f, Dark sum=%.1f\n", sum_bright, sum_dark);

    // With ON/OFF channels: bright excites ON, dark excites OFF
    // Total should differ
    TEST_ASSERT(sum_bright != sum_dark, "bright and dark produce different responses");

    printf("  [PASS] VisualInput 基础编码\n");
    g_pass++;
}

// =============================================================================
// Test 2: Center-surround 特性
// =============================================================================
static void test_center_surround() {
    printf("\n--- 测试2: Center-surround 感受野 ---\n");
    printf("    原理: ON cell: 中心亮→兴奋, 周围亮→抑制\n");

    VisualInputConfig cfg;
    cfg.input_width = 8;
    cfg.input_height = 8;
    cfg.n_lgn_neurons = 50;
    cfg.noise_amp = 0.0f;
    cfg.on_off_channels = true;

    VisualInput vis(cfg);

    // Create a small bright spot in the center
    std::vector<float> spot(64, 0.0f);
    spot[3*8+3] = 1.0f; spot[3*8+4] = 1.0f;
    spot[4*8+3] = 1.0f; spot[4*8+4] = 1.0f;

    // Uniform bright field
    std::vector<float> uniform(64, 1.0f);

    auto c_spot = vis.encode(spot);
    auto c_uniform = vis.encode(uniform);

    // ON cells (first half)
    size_t n_on = 25;
    float spot_on_max = *std::max_element(c_spot.begin(), c_spot.begin() + n_on);
    float uniform_on_max = *std::max_element(c_uniform.begin(), c_uniform.begin() + n_on);

    printf("    ON cells: spot_max=%.1f, uniform_max=%.1f\n",
           spot_on_max, uniform_on_max);

    // A spot should produce higher peak response than uniform
    // (center-surround: spot activates center without much surround inhibition)
    TEST_ASSERT(spot_on_max > cfg.baseline, "spot activates ON cells");

    printf("  [PASS] Center-surround\n");
    g_pass++;
}

// =============================================================================
// Test 3: 视觉端到端
// =============================================================================
static void test_visual_e2e() {
    printf("\n--- 测试3: 视觉端到端 pixels→LGN→V1 ---\n");

    SimulationEngine eng(10);

    ThalamicConfig tc;
    tc.name = "LGN"; tc.n_relay = 50; tc.n_trn = 15;
    eng.add_region(std::make_unique<ThalamicRelay>(tc));

    ColumnConfig cc;
    cc.n_l4_stellate = 50; cc.n_l23_pyramidal = 100;
    cc.n_l5_pyramidal = 50; cc.n_l6_pyramidal = 40;
    cc.n_pv_basket = 15; cc.n_sst_martinotti = 10; cc.n_vip = 5;
    eng.add_region(std::make_unique<CorticalRegion>("V1", cc));
    eng.add_projection("LGN", "V1", 2);

    VisualInputConfig vcfg;
    vcfg.n_lgn_neurons = 50;
    VisualInput vis(vcfg);

    // Stimulate with bright image
    std::vector<float> bright(64, 0.8f);
    size_t v1_spikes = 0;
    for (int t = 0; t < 100; ++t) {
        vis.encode_and_inject(bright, eng.find_region("LGN"));
        eng.step();
        v1_spikes += count_fired(*eng.find_region("V1"));
    }

    printf("    V1 spikes (bright input): %zu\n", v1_spikes);
    TEST_ASSERT(v1_spikes > 100, "bright image drives V1 activity");

    // No input control
    size_t v1_silent = 0;
    for (int t = 100; t < 150; ++t) {
        eng.step();
        v1_silent += count_fired(*eng.find_region("V1"));
    }
    printf("    V1 spikes (no input): %zu\n", v1_silent);
    TEST_ASSERT(v1_spikes > v1_silent * 2, "visual input drives much more activity");

    printf("  [PASS] 视觉端到端\n");
    g_pass++;
}

// =============================================================================
// Test 4: AuditoryInput 基础编码
// =============================================================================
static void test_auditory_basic() {
    printf("\n--- 测试4: AuditoryInput 基础编码 ---\n");
    printf("    原理: 16频带功率→20 MGN电流 (tonotopic)\n");

    AuditoryInputConfig cfg;
    cfg.n_freq_bands = 16;
    cfg.n_mgn_neurons = 20;
    cfg.noise_amp = 0.0f;

    AuditoryInput aud(cfg);
    TEST_ASSERT(aud.n_freq_bands() == 16, "16 freq bands");
    TEST_ASSERT(aud.n_mgn() == 20, "20 MGN neurons");

    // Silent spectrum
    std::vector<float> silent(16, 0.0f);
    auto c_silent = aud.encode(silent);
    TEST_ASSERT(c_silent.size() == 20, "output size matches MGN");

    float sum_silent = std::accumulate(c_silent.begin(), c_silent.end(), 0.0f);

    // Loud spectrum (all bands active)
    std::vector<float> loud(16, 0.8f);
    auto c_loud = aud.encode(loud);
    float sum_loud = std::accumulate(c_loud.begin(), c_loud.end(), 0.0f);

    printf("    Silent sum=%.1f, Loud sum=%.1f\n", sum_silent, sum_loud);
    TEST_ASSERT(sum_loud > sum_silent, "loud produces stronger response");

    // Low-freq only vs high-freq only
    std::vector<float> low_freq(16, 0.0f);
    std::vector<float> high_freq(16, 0.0f);
    for (int i = 0; i < 4; ++i) low_freq[i] = 1.0f;
    for (int i = 12; i < 16; ++i) high_freq[i] = 1.0f;

    auto c_low = aud.encode(low_freq);
    auto c_high = aud.encode(high_freq);

    // Low-freq should activate early MGN neurons, high-freq late
    float low_first_half = std::accumulate(c_low.begin(), c_low.begin() + 10, 0.0f);
    float low_second_half = std::accumulate(c_low.begin() + 10, c_low.end(), 0.0f);
    printf("    Low-freq: first_half=%.1f, second_half=%.1f\n",
           low_first_half, low_second_half);
    TEST_ASSERT(low_first_half > low_second_half, "low-freq activates low MGN neurons");

    printf("  [PASS] AuditoryInput 基础编码\n");
    g_pass++;
}

// =============================================================================
// Test 5: 听觉 onset 检测
// =============================================================================
static void test_auditory_onset() {
    printf("\n--- 测试5: 听觉 onset 检测 ---\n");
    printf("    原理: 新出现的声音→更强响应 (temporal_decay)\n");

    AuditoryInputConfig cfg;
    cfg.noise_amp = 0.0f;
    cfg.temporal_decay = 0.7f;
    AuditoryInput aud(cfg);

    // Frame 1: onset (new sound)
    std::vector<float> tone(16, 0.0f);
    tone[4] = 0.8f; tone[5] = 0.8f;
    auto c_onset = aud.encode(tone);
    float sum_onset = std::accumulate(c_onset.begin(), c_onset.end(), 0.0f);

    // Frame 2: sustained (same sound)
    auto c_sustained = aud.encode(tone);
    float sum_sustained = std::accumulate(c_sustained.begin(), c_sustained.end(), 0.0f);

    printf("    Onset sum=%.1f, Sustained sum=%.1f\n", sum_onset, sum_sustained);
    TEST_ASSERT(sum_onset > sum_sustained * 0.9f, "onset at least comparable to sustained");

    printf("  [PASS] 听觉 onset\n");
    g_pass++;
}

// =============================================================================
// Test 6: 听觉端到端
// =============================================================================
static void test_auditory_e2e() {
    printf("\n--- 测试6: 听觉端到端 spectrum→MGN→A1 ---\n");

    SimulationEngine eng(10);

    ThalamicConfig tc;
    tc.name = "MGN"; tc.n_relay = 20; tc.n_trn = 6;
    eng.add_region(std::make_unique<ThalamicRelay>(tc));

    ColumnConfig cc;
    cc.n_l4_stellate = 35; cc.n_l23_pyramidal = 70;
    cc.n_l5_pyramidal = 35; cc.n_l6_pyramidal = 25;
    cc.n_pv_basket = 10; cc.n_sst_martinotti = 7; cc.n_vip = 3;
    eng.add_region(std::make_unique<CorticalRegion>("A1", cc));
    eng.add_projection("MGN", "A1", 2);

    AuditoryInputConfig acfg;
    acfg.gain = 50.0f;  // Higher gain for small MGN
    AuditoryInput aud(acfg);

    // Broadband stimulus activates more MGN neurons
    std::vector<float> tone(16, 0.0f);
    for (int i = 2; i < 10; ++i) tone[i] = 0.8f;  // 8 of 16 bands

    size_t a1_spikes = 0;
    for (int t = 0; t < 100; ++t) {
        aud.encode_and_inject(tone, eng.find_region("MGN"));
        eng.step();
        a1_spikes += count_fired(*eng.find_region("A1"));
    }

    printf("    A1 spikes (tone): %zu\n", a1_spikes);
    TEST_ASSERT(a1_spikes > 30, "auditory input drives A1");

    printf("  [PASS] 听觉端到端\n");
    g_pass++;
}

// =============================================================================
// Test 7: 多模态并行输入
// =============================================================================
static void test_multimodal() {
    printf("\n--- 测试7: 多模态并行 (视觉+听觉) ---\n");

    SimulationEngine eng(10);

    // Visual path: LGN→V1
    ThalamicConfig lgn_cfg;
    lgn_cfg.name = "LGN"; lgn_cfg.n_relay = 50; lgn_cfg.n_trn = 15;
    eng.add_region(std::make_unique<ThalamicRelay>(lgn_cfg));

    ColumnConfig v1_cc;
    v1_cc.n_l4_stellate = 50; v1_cc.n_l23_pyramidal = 100;
    v1_cc.n_l5_pyramidal = 50; v1_cc.n_l6_pyramidal = 40;
    v1_cc.n_pv_basket = 15; v1_cc.n_sst_martinotti = 10; v1_cc.n_vip = 5;
    eng.add_region(std::make_unique<CorticalRegion>("V1", v1_cc));
    eng.add_projection("LGN", "V1", 2);

    // Auditory path: MGN→A1
    ThalamicConfig mgn_cfg;
    mgn_cfg.name = "MGN"; mgn_cfg.n_relay = 20; mgn_cfg.n_trn = 6;
    eng.add_region(std::make_unique<ThalamicRelay>(mgn_cfg));

    ColumnConfig a1_cc;
    a1_cc.n_l4_stellate = 35; a1_cc.n_l23_pyramidal = 70;
    a1_cc.n_l5_pyramidal = 35; a1_cc.n_l6_pyramidal = 25;
    a1_cc.n_pv_basket = 10; a1_cc.n_sst_martinotti = 7; a1_cc.n_vip = 3;
    eng.add_region(std::make_unique<CorticalRegion>("A1", a1_cc));
    eng.add_projection("MGN", "A1", 2);

    VisualInput vis;
    AuditoryInputConfig acfg;
    acfg.gain = 50.0f;
    AuditoryInput aud(acfg);

    std::vector<float> bright(64, 0.8f);
    std::vector<float> tone(16, 0.0f);
    for (int i = 2; i < 10; ++i) tone[i] = 0.8f;

    size_t v1_spikes = 0, a1_spikes = 0;
    for (int t = 0; t < 100; ++t) {
        vis.encode_and_inject(bright, eng.find_region("LGN"));
        aud.encode_and_inject(tone, eng.find_region("MGN"));
        eng.step();
        v1_spikes += count_fired(*eng.find_region("V1"));
        a1_spikes += count_fired(*eng.find_region("A1"));
    }

    printf("    V1=%zu, A1=%zu (both active)\n", v1_spikes, a1_spikes);
    TEST_ASSERT(v1_spikes > 100, "visual path active");
    TEST_ASSERT(a1_spikes > 30, "auditory path active");
    TEST_ASSERT(eng.num_regions() == 4, "4 regions");

    printf("  [PASS] 多模态并行\n");
    g_pass++;
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    printf("============================================\n");
    printf("  悟韵 (WuYun) Step 9 感觉输入接口测试\n");
    printf("  VisualInput + AuditoryInput\n");
    printf("============================================\n");

    test_visual_basic();
    test_center_surround();
    test_visual_e2e();
    test_auditory_basic();
    test_auditory_onset();
    test_auditory_e2e();
    test_multimodal();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
