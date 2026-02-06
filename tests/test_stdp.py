"""
STDP 可塑性规则验证测试

Case 1: 经典 STDP — pre→post (LTP)
Case 2: 经典 STDP — post→pre (LTD)
Case 3: 时间窗口外 → 无变化
Case 4: 三因子 STDP — 无 DA → 权重不变
Case 5: 三因子 STDP — DA 到达 → 权重变化
Case 6: 抑制性 STDP — 相关活动 → 增强抑制
Case 7: 软边界 — 权重不超出 [w_min, w_max]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from wuyun.synapse.plasticity.classical_stdp import ClassicalSTDP, ClassicalSTDPParams
from wuyun.synapse.plasticity.da_modulated_stdp import DAModulatedSTDP, DAModulatedSTDPParams
from wuyun.synapse.plasticity.inhibitory_stdp import InhibitorySTDP, InhibitorySTDPParams
from wuyun.synapse.synapse_base import SynapseBase
from wuyun.spike.signal_types import SynapseType, CompartmentType, PlasticityType


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# =============================================================================
# Case 1: 经典 STDP — LTP
# =============================================================================

def test_case_1_classical_ltp():
    """pre 先于 post → 权重增加 (LTP)"""
    print_header("Case 1: 经典 STDP — pre→post = LTP")

    rule = ClassicalSTDP()
    w_init = 0.5

    # pre at t=100, post at t=110 → Δt = +10ms → LTP
    dw = rule.compute_weight_update(
        pre_spike_times=[100],
        post_spike_times=[110],
        current_weight=w_init,
        w_min=0.0, w_max=1.0,
    )

    # 理论值: A+ * exp(-10/20) * soft_bound = 0.005 * 0.6065 * 0.5 ≈ 0.001516
    expected_raw = 0.005 * np.exp(-10.0 / 20.0)
    expected_dw = expected_raw * (1.0 - w_init) / 1.0  # 软边界

    print(f"  pre=100ms, post=110ms → Δt=+10ms")
    print(f"  Δw = {dw:.6f} (理论 ≈ {expected_dw:.6f})")
    print(f"  新权重 = {w_init + dw:.6f}")

    assert dw > 0, f"LTP: Δw 应 > 0, 得到 {dw}"
    assert abs(dw - expected_dw) < 1e-6, f"Δw 偏差过大: {dw} vs {expected_dw}"

    print(f"  ✅ PASS: pre→post → LTP, Δw > 0")
    return True


# =============================================================================
# Case 2: 经典 STDP — LTD
# =============================================================================

def test_case_2_classical_ltd():
    """post 先于 pre → 权重减少 (LTD)"""
    print_header("Case 2: 经典 STDP — post→pre = LTD")

    rule = ClassicalSTDP()
    w_init = 0.5

    # post at t=100, pre at t=110 → Δt = 100-110 = -10ms → LTD
    dw = rule.compute_weight_update(
        pre_spike_times=[110],
        post_spike_times=[100],
        current_weight=w_init,
        w_min=0.0, w_max=1.0,
    )

    expected_raw = -0.00525 * np.exp(-10.0 / 20.0)
    expected_dw = expected_raw * (w_init - 0.0) / 1.0  # 软边界

    print(f"  pre=110ms, post=100ms → Δt=-10ms")
    print(f"  Δw = {dw:.6f} (理论 ≈ {expected_dw:.6f})")
    print(f"  新权重 = {w_init + dw:.6f}")

    assert dw < 0, f"LTD: Δw 应 < 0, 得到 {dw}"
    assert abs(dw - expected_dw) < 1e-6, f"Δw 偏差过大: {dw} vs {expected_dw}"

    print(f"  ✅ PASS: post→pre → LTD, Δw < 0")
    return True


# =============================================================================
# Case 3: 时间窗口外 → 无变化
# =============================================================================

def test_case_3_outside_window():
    """Δt = 200ms >> τ=20ms → Δw ≈ 0"""
    print_header("Case 3: 时间窗口外 → Δw ≈ 0")

    rule = ClassicalSTDP()

    dw = rule.compute_weight_update(
        pre_spike_times=[100],
        post_spike_times=[300],  # Δt = +200ms
        current_weight=0.5,
        w_min=0.0, w_max=1.0,
    )

    # exp(-200/20) = exp(-10) ≈ 4.5e-5, 乘以 A+=0.005 ≈ 2.3e-7
    print(f"  Δt = +200ms, Δw = {dw:.10f}")
    assert abs(dw) < 1e-4, f"窗口外 Δw 应接近 0, 得到 {dw}"

    # 无脉冲
    dw_empty = rule.compute_weight_update(
        pre_spike_times=[], post_spike_times=[],
        current_weight=0.5, w_min=0.0, w_max=1.0,
    )
    assert dw_empty == 0.0, f"无脉冲 Δw 应为 0, 得到 {dw_empty}"

    print(f"  ✅ PASS: 时间窗口外 Δw ≈ 0")
    return True


# =============================================================================
# Case 4: 三因子 STDP — 无 DA → 权重不变
# =============================================================================

def test_case_4_three_factor_no_da():
    """有 pre/post 配对 → eligibility > 0, 但 DA=0 → Δw=0"""
    print_header("Case 4: 三因子 STDP — 无 DA → 权重不变")

    rule = DAModulatedSTDP()

    # 1. compute_weight_update 始终返回 0 (三因子不直接改权重)
    dw_direct = rule.compute_weight_update(
        pre_spike_times=[100], post_spike_times=[110],
        current_weight=0.5, w_min=0.0, w_max=1.0,
    )
    assert dw_direct == 0.0, f"三因子 compute_weight_update 应返回 0, 得到 {dw_direct}"

    # 2. 更新资格痕迹 (pre→post → 正向 STDP)
    eligibility = 0.0
    eligibility = rule.update_eligibility(
        pre_spike_times=[100], post_spike_times=[110],
        current_eligibility=eligibility, dt=1.0,
    )
    print(f"  pre=100, post=110 → eligibility = {eligibility:.6f}")
    assert eligibility > 0, f"LTP 配对后 eligibility 应 > 0, 得到 {eligibility}"

    # 3. DA=0 → 权重不变
    dw = rule.apply_modulated_update(
        eligibility=eligibility, modulation=0.0,
        current_weight=0.5, w_min=0.0, w_max=1.0,
    )
    print(f"  DA=0 → Δw = {dw:.6f}")
    assert dw == 0.0, f"DA=0 时 Δw 应为 0, 得到 {dw}"

    print(f"  ✅ PASS: 三因子 STDP — 无 DA → 资格痕迹存在但权重不变")
    return True


# =============================================================================
# Case 5: 三因子 STDP — DA 到达 → 权重变化
# =============================================================================

def test_case_5_three_factor_with_da():
    """DA 到达 → eligibility × DA → 权重变化"""
    print_header("Case 5: 三因子 STDP — DA 到达 → 权重变化")

    rule = DAModulatedSTDP()

    # 建立资格痕迹
    eligibility = 0.0
    eligibility = rule.update_eligibility(
        pre_spike_times=[100], post_spike_times=[110],
        current_eligibility=eligibility, dt=1.0,
    )
    print(f"  资格痕迹: {eligibility:.6f}")

    # DA=1.0 → 完整学习
    dw_full = rule.apply_modulated_update(
        eligibility=eligibility, modulation=1.0,
        current_weight=0.5, w_min=0.0, w_max=1.0,
    )
    print(f"  DA=1.0 → Δw = {dw_full:.6f}")
    assert dw_full > 0, f"DA=1.0 + 正向 eligibility → Δw 应 > 0, 得到 {dw_full}"

    # DA=0.5 → 弱化学习
    dw_half = rule.apply_modulated_update(
        eligibility=eligibility, modulation=0.5,
        current_weight=0.5, w_min=0.0, w_max=1.0,
    )
    print(f"  DA=0.5 → Δw = {dw_half:.6f}")
    assert 0 < dw_half < dw_full, \
        f"DA=0.5 的 Δw ({dw_half}) 应在 0 和 DA=1.0 的 Δw ({dw_full}) 之间"

    # 资格痕迹衰减验证
    elig_fresh = eligibility
    # 模拟 500ms 无脉冲衰减
    for _ in range(500):
        elig_fresh = rule.update_eligibility([], [], elig_fresh, dt=1.0)
    print(f"  500ms 衰减后: eligibility = {elig_fresh:.6f} (初始 {eligibility:.6f})")
    assert elig_fresh < eligibility * 0.7, \
        f"500ms 后资格痕迹应明显衰减 (τ_e=1000ms, 理论 ~60.6%)"

    print(f"  ✅ PASS: 三因子 STDP — DA 调制权重变化正确")
    return True


# =============================================================================
# Case 6: 抑制性 STDP — 相关活动 → 增强抑制
# =============================================================================

def test_case_6_inhibitory_stdp():
    """同步 pre/post → 增强抑制; 不相关 → 减弱抑制"""
    print_header("Case 6: 抑制性 STDP — 对称窗口")

    rule = InhibitorySTDP()

    # 同步活动: |Δt| = 5ms → 增强抑制
    dw_sync = rule.compute_weight_update(
        pre_spike_times=[100],
        post_spike_times=[105],
        current_weight=0.5,
        w_min=0.0, w_max=1.0,
    )
    print(f"  同步 |Δt|=5ms → Δw = {dw_sync:.6f}")
    assert dw_sync > 0, f"同步活动应增强抑制 (Δw > 0), 得到 {dw_sync}"

    # 对称验证: pre-post 和 post-pre 应相同
    dw_reverse = rule.compute_weight_update(
        pre_spike_times=[105],
        post_spike_times=[100],
        current_weight=0.5,
        w_min=0.0, w_max=1.0,
    )
    print(f"  反向 |Δt|=5ms → Δw = {dw_reverse:.6f}")
    assert abs(dw_sync - dw_reverse) < 1e-10, "对称窗口: 正反向 Δw 应相同"

    # 不相关: 只有 pre, 无 post → 减弱抑制
    dw_uncorr = rule.compute_weight_update(
        pre_spike_times=[100],
        post_spike_times=[],
        current_weight=0.5,
        w_min=0.0, w_max=1.0,
    )
    print(f"  不相关 (pre only) → Δw = {dw_uncorr:.6f}")
    assert dw_uncorr < 0, f"不相关应减弱抑制 (Δw < 0), 得到 {dw_uncorr}"

    print(f"  ✅ PASS: 抑制性 STDP — 对称窗口 + E/I 平衡调节")
    return True


# =============================================================================
# Case 7: 软边界 — 权重不超出 [w_min, w_max]
# =============================================================================

def test_case_7_soft_boundary():
    """反复 LTP → 权重趋近但不超过 w_max"""
    print_header("Case 7: 软边界 — 权重收敛")

    rule = ClassicalSTDP()

    w = 0.5
    w_max = 1.0
    w_min = 0.0

    print(f"  初始权重: {w:.4f}")

    # 反复 LTP (Δt=+5ms)
    for i in range(200):
        dw = rule.compute_weight_update(
            pre_spike_times=[i * 10],
            post_spike_times=[i * 10 + 5],
            current_weight=w,
            w_min=w_min, w_max=w_max,
        )
        w = np.clip(w + dw, w_min, w_max)

        if i in [0, 9, 49, 99, 199]:
            print(f"    迭代 {i+1:3d}: w={w:.6f}, dw={dw:.8f}")

    assert w < w_max, f"权重应不超过 w_max={w_max}, 得到 {w}"
    assert w > 0.7, f"200 次 LTP 后权重应明显增加, 得到 {w}"

    # 验证接近上限时 dw 趋近 0
    dw_final = rule.compute_weight_update(
        pre_spike_times=[2000],
        post_spike_times=[2005],
        current_weight=w,
        w_min=w_min, w_max=w_max,
    )
    dw_mid = rule.compute_weight_update(
        pre_spike_times=[2000],
        post_spike_times=[2005],
        current_weight=0.5,
        w_min=w_min, w_max=w_max,
    )
    print(f"  接近上限 (w={w:.4f}): dw={dw_final:.8f}")
    print(f"  中间位置 (w=0.5):    dw={dw_mid:.8f}")
    assert abs(dw_final) < abs(dw_mid), \
        "软边界: 接近上限时 Δw 应比中间位置小"

    # 反复 LTD → 权重趋近但不低于 w_min
    w = 0.5
    for i in range(200):
        dw = rule.compute_weight_update(
            pre_spike_times=[i * 10 + 5],
            post_spike_times=[i * 10],  # post 先于 pre → LTD
            current_weight=w,
            w_min=w_min, w_max=w_max,
        )
        w = np.clip(w + dw, w_min, w_max)

    print(f"  200次 LTD 后: w={w:.6f}")
    assert w > w_min, f"权重应不低于 w_min={w_min}, 得到 {w}"
    assert w < 0.3, f"200 次 LTD 后权重应明显减少, 得到 {w}"

    print(f"  ✅ PASS: 软边界 — 权重收敛, 不超出 [w_min, w_max]")
    return True


# =============================================================================
# Case 8: SynapseBase 集成验证
# =============================================================================

def test_case_8_synapse_integration():
    """验证 SynapseBase 正确委托给 PlasticityRule"""
    print_header("Case 8: SynapseBase 集成")

    rule = ClassicalSTDP()

    syn = SynapseBase(
        pre_id=0, post_id=1,
        weight=0.5,
        synapse_type=SynapseType.AMPA,
        target_compartment=CompartmentType.BASAL,
        plasticity_rule=rule,
    )

    print(f"  初始: {syn}")
    w_before = syn.weight

    # 通过 SynapseBase 接口更新权重
    dw = syn.update_weight_stdp(
        pre_spike_times=[100],
        post_spike_times=[110],
    )

    print(f"  update_weight_stdp(pre=100, post=110)")
    print(f"  Δw = {dw:.6f}, 新权重 = {syn.weight:.6f}")

    assert dw > 0, f"LTP: Δw 应 > 0"
    assert syn.weight > w_before, f"权重应增加"

    # 无规则时应返回 0
    syn_no_rule = SynapseBase(pre_id=0, post_id=1, weight=0.5)
    dw_none = syn_no_rule.update_weight_stdp([100], [110])
    assert dw_none == 0.0, f"无规则时 Δw 应为 0"

    print(f"  ✅ PASS: SynapseBase 正确委托给 PlasticityRule")
    return True


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  悟韵 (WuYun) STDP 可塑性规则验证测试                  ║")
    print("║  测试突触学习规则: 经典STDP / 三因子STDP / 抑制性STDP  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    results = {}
    tests = [
        ("Case 1: 经典 STDP — LTP", test_case_1_classical_ltp),
        ("Case 2: 经典 STDP — LTD", test_case_2_classical_ltd),
        ("Case 3: 时间窗口外", test_case_3_outside_window),
        ("Case 4: 三因子 — 无DA", test_case_4_three_factor_no_da),
        ("Case 5: 三因子 — DA调制", test_case_5_three_factor_with_da),
        ("Case 6: 抑制性 STDP", test_case_6_inhibitory_stdp),
        ("Case 7: 软边界", test_case_7_soft_boundary),
        ("Case 8: SynapseBase集成", test_case_8_synapse_integration),
    ]

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "PASS" if passed else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()

    # 总结
    print_header("总结")
    all_pass = True
    for name, result in results.items():
        icon = "✅" if result == "PASS" else "❌"
        if result != "PASS":
            all_pass = False
        print(f"  {icon} {result}: {name}")

    print()
    if all_pass:
        print("🎉 所有测试通过! STDP 可塑性规则验证完毕。")
        print("   三种学习规则 + SynapseBase 集成均工作正常。")
    else:
        print("❌ 存在失败的测试，请检查。")
        sys.exit(1)