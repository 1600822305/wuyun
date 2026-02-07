/**
 * 悟韵 (WuYun) 地基组件单元测试
 *
 * 测试项:
 *   1. NMDA Mg²⁺ 电压门控 B(V)
 *   2. STP 集成到 SynapseGroup
 *   3. SpikeBus 跨区域脉冲路由
 *   4. DA-STDP 三因子学习
 *   5. 神经调质系统
 *   6. 特化神经元参数集验证
 */

#include "core/types.h"
#include "core/population.h"
#include "core/synapse_group.h"
#include "core/spike_bus.h"
#include "core/neuromodulator.h"
#include "plasticity/stdp.h"
#include "plasticity/stp.h"
#include "plasticity/da_stdp.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

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
// 测试1: NMDA Mg²⁺ 电压门控
// =============================================================================
void test_nmda_mg_block() {
    printf("\n--- 测试1: NMDA Mg²⁺ 电压门控 B(V) ---\n");
    printf("    公式: B(V) = 1/(1 + [Mg²⁺]/3.57 · exp(-0.062·V))\n");

    // B(V) at different voltages with [Mg²⁺]=1.0 mM
    float mg = 1.0f;
    auto B = [mg](float v) {
        return 1.0f / (1.0f + (mg / 3.57f) * std::exp(-0.062f * v));
    };

    float b_neg65 = B(-65.0f);  // 静息电位: 应该几乎完全阻断
    float b_neg40 = B(-40.0f);  // 中等去极化: 部分开放
    float b_0     = B(0.0f);    // 完全去极化: 几乎完全开放

    printf("    B(-65mV) = %.4f (静息: 应近0, Mg²⁺阻断)\n", b_neg65);
    printf("    B(-40mV) = %.4f (中等去极化: 部分开放)\n", b_neg40);
    printf("    B( 0mV)  = %.4f (完全去极化: 应近1)\n", b_0);

    CHECK(b_neg65 < 0.15f, "B(-65) 应该 < 0.15 (Mg²⁺阻断)");
    CHECK(b_neg40 > 0.2f && b_neg40 < 0.8f, "B(-40) 应该在 0.2~0.8 之间");
    CHECK(b_0 > 0.7f, "B(0) 应该 > 0.7 (开放)");
    CHECK(b_neg65 < b_neg40 && b_neg40 < b_0, "B(V) 应随去极化单调递增");

    // 验证 SynapseGroup 实际使用: NMDA vs AMPA 电流差异
    size_t n = 5;
    std::vector<int32_t> pre = {0}, post = {0};
    std::vector<float> w = {0.5f};
    std::vector<int32_t> d = {1};

    SynapseGroup ampa(n, n, pre, post, w, d, AMPA_PARAMS, CompartmentType::BASAL);
    SynapseGroup nmda(n, n, pre, post, w, d, NMDA_PARAMS, CompartmentType::BASAL);

    // Simulate pre neuron 0 firing
    std::vector<uint8_t> fired(n, 0); fired[0] = 1;
    std::vector<int8_t> st(n, 0); st[0] = static_cast<int8_t>(SpikeType::REGULAR);
    ampa.deliver_spikes(fired, st);
    nmda.deliver_spikes(fired, st);

    // Compute currents at resting potential (-65 mV)
    std::vector<float> v_rest(n, -65.0f);
    auto i_ampa = ampa.step_and_compute(v_rest);
    auto i_nmda = nmda.step_and_compute(v_rest);

    printf("    AMPA I[0] = %.4f   NMDA I[0] = %.4f (at -65mV)\n", i_ampa[0], i_nmda[0]);
    CHECK(std::abs(i_nmda[0]) < std::abs(i_ampa[0]),
          "NMDA电流 at -65mV 应弱于 AMPA (Mg²⁺阻断)");

    PASS("NMDA Mg²⁺ 电压门控");
}

// =============================================================================
// 测试2: STP 集成到 SynapseGroup
// =============================================================================
void test_stp_integration() {
    printf("\n--- 测试2: STP 集成 (Tsodyks-Markram) ---\n");
    printf("    原理: 高频发放→STD→突触减弱; 低频→STF→突触增强\n");

    size_t n = 10;
    std::vector<int32_t> pre_ids, post_ids;
    std::vector<float> weights;
    std::vector<int32_t> delays;
    for (size_t i = 0; i < n; ++i) {
        pre_ids.push_back(0);
        post_ids.push_back(static_cast<int32_t>(i));
        weights.push_back(0.5f);
        delays.push_back(1);
    }

    // SynapseGroup with STD (depression)
    SynapseGroup syn_std(n, n, pre_ids, post_ids, weights, delays,
                          AMPA_PARAMS, CompartmentType::BASAL);
    syn_std.enable_stp(STP_DEPRESSION);  // U=0.5, tau_D=200, tau_F=20

    // SynapseGroup without STP
    SynapseGroup syn_plain(n, n, pre_ids, post_ids, weights, delays,
                            AMPA_PARAMS, CompartmentType::BASAL);

    std::vector<float> v(n, -65.0f);
    std::vector<uint8_t> fired(n, 0);
    std::vector<int8_t> st(n, 0);

    // First spike: STP gain = U * x = 0.5 * 1.0 = 0.5
    fired[0] = 1; st[0] = static_cast<int8_t>(SpikeType::REGULAR);
    syn_std.deliver_spikes(fired, st);
    syn_plain.deliver_spikes(fired, st);

    auto i_std_1  = syn_std.step_and_compute(v);
    auto i_plain_1 = syn_plain.step_and_compute(v);

    printf("    第1个脉冲: STP电流=%.4f  无STP=%.4f\n", i_std_1[0], i_plain_1[0]);

    // Second spike immediately: STD should reduce current
    syn_std.deliver_spikes(fired, st);
    syn_plain.deliver_spikes(fired, st);

    auto i_std_2  = syn_std.step_and_compute(v);
    auto i_plain_2 = syn_plain.step_and_compute(v);

    printf("    第2个脉冲: STP电流=%.4f  无STP=%.4f\n", i_std_2[0], i_plain_2[0]);

    CHECK(std::abs(i_std_2[0]) < std::abs(i_std_1[0]),
          "STD: 连续脉冲后电流应减弱 (资源耗竭)");
    CHECK(syn_std.has_stp(), "has_stp() 应返回 true");

    PASS("STP 集成 (STD 资源耗竭)");
}

// =============================================================================
// 测试3: SpikeBus 跨区域路由
// =============================================================================
void test_spike_bus() {
    printf("\n--- 测试3: SpikeBus 跨区域脉冲路由 ---\n");
    printf("    原理: 区域A→(delay=3)→区域B, 脉冲在3步后到达\n");

    SpikeBus bus(10);

    uint32_t region_a = bus.register_region("V1", 100);
    uint32_t region_b = bus.register_region("V2", 50);
    uint32_t region_c = bus.register_region("PFC", 30);

    CHECK(bus.num_regions() == 3, "应有3个区域");

    bus.add_projection(region_a, region_b, 3, "V1→V2");
    bus.add_projection(region_a, region_c, 5, "V1→PFC");

    CHECK(bus.num_projections() == 2, "应有2条投射");

    // V1 neuron 5 fires at t=10
    std::vector<uint8_t> fired_a(100, 0);
    std::vector<int8_t> st_a(100, 0);
    fired_a[5] = 1;
    st_a[5] = static_cast<int8_t>(SpikeType::REGULAR);

    bus.submit_spikes(region_a, fired_a, st_a, 10);

    // At t=12: nothing should arrive yet
    auto arriving_12 = bus.get_arriving_spikes(region_b, 12);
    CHECK(arriving_12.empty(), "t=12: V2不应收到脉冲 (delay=3)");

    // At t=13: spike should arrive at V2
    auto arriving_13 = bus.get_arriving_spikes(region_b, 13);
    CHECK(arriving_13.size() == 1, "t=13: V2应收到1个脉冲");
    CHECK(arriving_13[0].neuron_id == 5, "脉冲来自V1 neuron 5");

    // At t=13: PFC should not receive yet (delay=5)
    auto arriving_pfc_13 = bus.get_arriving_spikes(region_c, 13);
    CHECK(arriving_pfc_13.empty(), "t=13: PFC不应收到脉冲 (delay=5)");

    // At t=15: PFC should receive
    auto arriving_pfc_15 = bus.get_arriving_spikes(region_c, 15);
    CHECK(arriving_pfc_15.size() == 1, "t=15: PFC应收到1个脉冲");

    printf("    V1[5] fires@t=10 → V2 arrives@t=13 ✓ → PFC arrives@t=15 ✓\n");

    PASS("SpikeBus 延迟路由");
}

// =============================================================================
// 测试4: DA-STDP 三因子学习
// =============================================================================
void test_da_stdp() {
    printf("\n--- 测试4: DA-STDP 三因子学习 ---\n");
    printf("    原理: STDP→资格痕迹 → DA到达时转化为权重变化\n");

    DASTDPParams params;
    params.stdp.a_plus = 0.01f;
    params.stdp.a_minus = -0.012f;
    params.stdp.tau_plus = 20.0f;
    params.stdp.tau_minus = 20.0f;
    params.tau_eligibility = 1000.0f;
    params.da_baseline = 0.1f;
    params.w_min = 0.0f;
    params.w_max = 1.0f;

    size_t n_syn = 3;
    DASTDPProcessor processor(n_syn, params);

    // Synapse 0: pre fires at t=5, post fires at t=7 (Δt=+2 → LTP)
    // Synapse 1: pre fires at t=7, post fires at t=5 (Δt=-2 → LTD)
    // Synapse 2: no activity
    float pre_times[]  = {5.0f, 7.0f, -1.0f};
    float post_times[] = {7.0f, 5.0f, -1.0f};
    int32_t pre_ids[]  = {0, 1, 2};
    int32_t post_ids[] = {0, 1, 2};

    processor.update_traces(pre_times, post_times, pre_ids, post_ids);

    auto& traces = processor.traces();
    printf("    资格痕迹: [0]=%.6f (LTP)  [1]=%.6f (LTD)  [2]=%.6f (无)\n",
           traces[0], traces[1], traces[2]);

    CHECK(traces[0] > 0.0f, "突触0: pre→post → 正资格痕迹 (LTP候选)");
    CHECK(traces[1] < 0.0f, "突触1: post→pre → 负资格痕迹 (LTD候选)");
    CHECK(std::abs(traces[2]) < 1e-10f, "突触2: 无活动 → 零痕迹");

    // 无 DA: 权重不应变化
    float weights[] = {0.5f, 0.5f, 0.5f};
    processor.apply_da_modulation(weights, params.da_baseline);  // DA = baseline → no change
    printf("    DA=baseline: w=[%.4f, %.4f, %.4f] (应不变)\n",
           weights[0], weights[1], weights[2]);
    CHECK(std::abs(weights[0] - 0.5f) < 0.001f, "DA=baseline: 权重不变");

    // DA burst (reward signal): 应该强化 LTP, 弱化 LTD
    float weights2[] = {0.5f, 0.5f, 0.5f};
    processor.apply_da_modulation(weights2, 0.8f);  // DA = 0.8 >> baseline
    printf("    DA=0.8 (reward): w=[%.4f, %.4f, %.4f]\n",
           weights2[0], weights2[1], weights2[2]);
    CHECK(weights2[0] > 0.5f, "DA reward: 突触0 (LTP) 应增强");
    CHECK(weights2[1] < 0.5f, "DA reward: 突触1 (LTD) 应减弱");

    PASS("DA-STDP 三因子学习");
}

// =============================================================================
// 测试5: 神经调质系统
// =============================================================================
void test_neuromodulator() {
    printf("\n--- 测试5: 神经调质系统 (DA/NE/5-HT/ACh) ---\n");

    NeuromodulatorSystem nm;

    // Set tonic baseline
    nm.set_tonic({0.1f, 0.2f, 0.3f, 0.2f});
    auto cur = nm.current();
    printf("    Tonic: DA=%.2f NE=%.2f 5HT=%.2f ACh=%.2f\n",
           cur.da, cur.ne, cur.sht, cur.ach);
    CHECK(std::abs(cur.da - 0.1f) < 0.01f, "Tonic DA = 0.1");

    // Inject phasic DA burst (reward)
    nm.inject_phasic(0.5f, 0.0f, 0.0f, 0.0f);
    cur = nm.current();
    printf("    DA burst: DA=%.2f (应为~0.6)\n", cur.da);
    CHECK(cur.da > 0.5f, "DA burst 后浓度应 > 0.5");

    // Compute modulation effect
    auto eff = nm.compute_effect();
    printf("    调制效应: gain=%.2f lr=%.2f discount=%.2f basal_w=%.2f\n",
           eff.gain, eff.learning_rate, eff.discount, eff.basal_weight);
    CHECK(eff.learning_rate > 1.0f, "高DA → 学习率 > 1.0");

    // Step: phasic decays
    for (int i = 0; i < 500; ++i) nm.step(1.0f);
    cur = nm.current();
    printf("    500步衰减后: DA=%.4f (应接近tonic 0.1)\n", cur.da);
    CHECK(cur.da < 0.15f, "Phasic DA 应已衰减接近 tonic");

    PASS("神经调质系统");
}

// =============================================================================
// 测试6: 特化神经元参数集
// =============================================================================
void test_specialized_params() {
    printf("\n--- 测试6: 特化神经元参数集验证 ---\n");

    // 丘脑 Tonic 模式
    auto th_tonic = THALAMIC_RELAY_TONIC_PARAMS();
    CHECK(th_tonic.kappa > 0.0f, "丘脑 Tonic: κ > 0 (有apical)");
    CHECK(th_tonic.burst_spike_count == 1, "丘脑 Tonic: 单脉冲");
    printf("    丘脑 Tonic: κ=%.1f ✓\n", th_tonic.kappa);

    // 丘脑 Burst 模式
    auto th_burst = THALAMIC_RELAY_BURST_PARAMS();
    CHECK(th_burst.kappa > th_tonic.kappa, "丘脑 Burst: κ > Tonic κ");
    CHECK(th_burst.burst_spike_count >= 3, "丘脑 Burst: 多脉冲 burst");
    printf("    丘脑 Burst: κ=%.1f, burst=%d ✓\n", th_burst.kappa, th_burst.burst_spike_count);

    // TRN
    auto trn = TRN_PARAMS();
    CHECK(trn.kappa == 0.0f, "TRN: κ=0 (单区室, 纯抑制)");
    printf("    TRN: κ=0 (纯抑制门控) ✓\n");

    // MSN D1/D2
    auto d1 = MSN_D1_PARAMS();
    auto d2 = MSN_D2_PARAMS();
    CHECK(d1.somatic.v_rest < -75.0f, "MSN D1: 超极化静息 (down state)");
    CHECK(d2.somatic.v_rest < -75.0f, "MSN D2: 超极化静息 (down state)");
    printf("    MSN D1: v_rest=%.0f (超极化) ✓\n", d1.somatic.v_rest);
    printf("    MSN D2: v_rest=%.0f (超极化) ✓\n", d2.somatic.v_rest);

    // 颗粒细胞
    auto gc = GRANULE_CELL_PARAMS();
    CHECK(gc.somatic.v_threshold > -50.0f, "颗粒细胞: 高阈值 (稀疏编码)");
    printf("    颗粒细胞: threshold=%.0f (稀疏) ✓\n", gc.somatic.v_threshold);

    // 浦肯野
    auto pk = PURKINJE_PARAMS();
    CHECK(pk.somatic.tau_m <= 10.0f, "浦肯野: 快速膜时间常数");
    printf("    浦肯野: tau_m=%.0f (高频) ✓\n", pk.somatic.tau_m);

    // DA 神经元
    auto da = DOPAMINE_NEURON_PARAMS();
    CHECK(da.somatic.tau_w >= 400.0f, "DA神经元: 非常慢适应");
    CHECK(da.burst_spike_count >= 3, "DA神经元: phasic burst 能力");
    printf("    DA神经元: tau_w=%.0f, burst=%d ✓\n", da.somatic.tau_w, da.burst_spike_count);

    // 验证: 丘脑 burst 可实际产生 burst (功能测试)
    NeuronPopulation thal(10, th_burst);
    size_t burst_count = 0;
    for (int t = 0; t < 100; ++t) {
        // 持续注入 basal + apical 直到 t=30
        if (t < 30) {
            for (size_t i = 0; i < 10; ++i) {
                thal.inject_basal(i, 30.0f);
                thal.inject_apical(i, 40.0f);
            }
        }
        thal.step(t);
        for (size_t i = 0; i < 10; ++i) {
            auto spike = static_cast<SpikeType>(thal.spike_type()[i]);
            if (is_burst(spike)) burst_count++;
        }
    }
    printf("    丘脑 burst 功能: burst=%zu (50步内)\n", burst_count);
    CHECK(burst_count > 0, "丘脑 Burst 模式应能产生 burst");

    PASS("特化神经元参数集");
}

// =============================================================================
// Main
// =============================================================================
int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    printf("============================================\n");
    printf("  悟韵 (WuYun) 地基组件单元测试\n");
    printf("  Layer 0-1: Synapse/Signal/Plasticity/Bus\n");
    printf("============================================\n");

    test_nmda_mg_block();
    test_stp_integration();
    test_spike_bus();
    test_da_stdp();
    test_neuromodulator();
    test_specialized_params();

    printf("\n============================================\n");
    printf("  结果: %d 通过, %d 失败, 共 %d 测试\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("============================================\n");

    return g_fail > 0 ? 1 : 0;
}
