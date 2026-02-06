"""
Phase 2 测试 — 多柱 + 丘脑路由

7 个测试用例，不用 pytest，直接 python tests/test_phase2_thalamus.py 运行。
用 print() 输出所有关键数值，assert 做断言。

审查修复 (2026-02-06):
- 工厂函数使用 ff_connection_strength=1.5 确保 L6 能发放
- 测试增益使用调优后的默认 GainParams (不再手动传)
- 加强断言: L6 必须有活动, Col1 必须有活动, TC 不应无限增长
"""

import sys
import os
import math
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 添加项目根目录到 sys.path，支持直接运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# 测试 1: TC 中继基础
# ============================================================================

def test_tc_relay_basic():
    """TC 中继基础: 注入感觉电流，TC 应该发放"""
    print("\n" + "=" * 60)
    print("测试 1: TC 中继基础")
    print("=" * 60)

    from wuyun.thalamus import create_thalamic_nucleus

    nucleus = create_thalamic_nucleus(nucleus_id=0, n_tc=10, n_trn=5, seed=42)
    print(f"  创建核团: {nucleus}")

    total_tc_spikes = 0
    total_trn_spikes = 0

    for t in range(100):
        nucleus.inject_sensory_current(50.0)
        nucleus.step(t)

        tc_out = nucleus.get_tc_output()
        trn_out = nucleus.get_trn_output()
        total_tc_spikes += len(tc_out)
        total_trn_spikes += len(trn_out)

        if (t + 1) % 20 == 0:
            tc_rate = nucleus.get_tc_firing_rate()
            trn_rate = nucleus.get_trn_firing_rate()
            tc_burst = nucleus.get_tc_burst_ratio()
            print(f"  t={t+1:3d}: TC 发放数={total_tc_spikes}, TRN 发放数={total_trn_spikes}, "
                  f"TC rate={tc_rate:.1f}Hz, TRN rate={trn_rate:.1f}Hz, "
                  f"TC burst ratio={tc_burst:.3f}")

    assert total_tc_spikes > 0, f"TC 应该有发放, 实际 {total_tc_spikes}"
    print(f"  ✓ TC 总发放数: {total_tc_spikes} > 0")
    print("  PASSED")


# ============================================================================
# 测试 2: TRN 门控效应
# ============================================================================

def test_trn_gating():
    """TRN 门控: 强 TRN 驱动应抑制 TC 发放"""
    print("\n" + "=" * 60)
    print("测试 2: TRN 门控效应")
    print("=" * 60)

    from wuyun.thalamus import create_thalamic_nucleus

    # 场景 A: 只注入感觉电流
    nucleus_a = create_thalamic_nucleus(nucleus_id=0, n_tc=10, n_trn=5, seed=42)
    tc_spikes_a = 0
    for t in range(200):
        nucleus_a.inject_sensory_current(50.0)
        nucleus_a.step(t)
        tc_spikes_a += len(nucleus_a.get_tc_output())

    tc_rate_a = nucleus_a.get_tc_firing_rate()

    # 场景 B: 感觉电流 + 强 TRN 驱动
    nucleus_b = create_thalamic_nucleus(nucleus_id=1, n_tc=10, n_trn=5, seed=42)
    tc_spikes_b = 0
    for t in range(200):
        nucleus_b.inject_sensory_current(50.0)
        nucleus_b.inject_trn_drive_current(80.0)
        nucleus_b.step(t)
        tc_spikes_b += len(nucleus_b.get_tc_output())

    tc_rate_b = nucleus_b.get_tc_firing_rate()

    print(f"  场景 A (只感觉): TC 总发放={tc_spikes_a}, rate={tc_rate_a:.1f}Hz")
    print(f"  场景 B (感觉+TRN): TC 总发放={tc_spikes_b}, rate={tc_rate_b:.1f}Hz")
    ratio = tc_spikes_b / max(tc_spikes_a, 1)
    print(f"  B/A 比率: {ratio:.3f}")

    assert tc_spikes_b < tc_spikes_a, (
        f"TRN 驱动应抑制 TC: B({tc_spikes_b}) < A({tc_spikes_a})"
    )
    print(f"  ✓ TRN 门控有效: B({tc_spikes_b}) < A({tc_spikes_a})")
    print("  PASSED")


# ============================================================================
# 测试 3: Tonic/Burst 双模式
# ============================================================================

def test_tonic_burst_modes():
    """TC 的 tonic/burst 双模式"""
    print("\n" + "=" * 60)
    print("测试 3: Tonic/Burst 双模式")
    print("=" * 60)

    from wuyun.thalamus import create_thalamic_nucleus

    # 场景 A: 有皮层反馈 → TC 应倾向 burst (apical 被激活 → Ca²⁺ spike)
    # TC 的 apical Ca²⁺ 阈值 = -35mV, 静息 = -70mV, τ_a=25ms
    # 需要足够强的电流让 V_a 从 -70 升到 -35 (差 35mV)
    nucleus_a = create_thalamic_nucleus(nucleus_id=0, n_tc=10, n_trn=5, seed=42)
    for t in range(500):
        nucleus_a.inject_sensory_current(80.0)
        nucleus_a.inject_cortical_feedback_current(200.0)  # 强反馈驱动 apical
        nucleus_a.step(t)
    burst_a = nucleus_a.get_tc_burst_ratio()

    # 场景 B: 无反馈 → TC 只有 regular (无 apical 输入 → 无 Ca²⁺)
    nucleus_b = create_thalamic_nucleus(nucleus_id=1, n_tc=10, n_trn=5, seed=42)
    for t in range(500):
        nucleus_b.inject_sensory_current(80.0)
        nucleus_b.step(t)
    burst_b = nucleus_b.get_tc_burst_ratio()

    # 场景 C: 先有反馈后撤去
    nucleus_c = create_thalamic_nucleus(nucleus_id=2, n_tc=10, n_trn=5, seed=42)
    for t in range(300):
        nucleus_c.inject_sensory_current(80.0)
        nucleus_c.inject_cortical_feedback_current(200.0)
        nucleus_c.step(t)
    burst_c_early = nucleus_c.get_tc_burst_ratio()
    for t in range(300, 600):
        nucleus_c.inject_sensory_current(80.0)
        nucleus_c.step(t)
    burst_c_late = nucleus_c.get_tc_burst_ratio()

    print(f"  场景 A (有反馈): burst ratio = {burst_a:.4f}")
    print(f"  场景 B (无反馈): burst ratio = {burst_b:.4f}")
    print(f"  场景 C (先有后撤): early={burst_c_early:.4f}, late={burst_c_late:.4f}")

    assert burst_a > burst_b, (
        f"有反馈时 burst 应更高: A({burst_a:.4f}) > B({burst_b:.4f})"
    )
    print(f"  ✓ 有反馈 burst({burst_a:.4f}) > 无反馈 burst({burst_b:.4f})")
    print("  PASSED")


# ============================================================================
# 测试 4: 丘脑-皮层环路 (★闭环验证)
# ============================================================================

def test_thalamocortical_loop():
    """丘脑-皮层环路: 手动连线 TC ↔ Column, 验证 L6 闭环"""
    print("\n" + "=" * 60)
    print("测试 4: 丘脑-皮层环路 (★闭环)")
    print("=" * 60)

    from wuyun.thalamus import create_thalamic_nucleus
    from wuyun.circuit import create_sensory_column

    nucleus = create_thalamic_nucleus(nucleus_id=0, n_tc=10, n_trn=5, seed=42)
    # ★ 关键修复: 使用 ff_connection_strength=1.5 确保 L5/L6 能被激活
    column = create_sensory_column(
        column_id=0, n_per_layer=20, seed=42, ff_connection_strength=1.5
    )

    print(f"  核团: {nucleus}")
    print(f"  皮层柱: {column}")

    has_nan = False
    l6_ever_fired = False
    tc_rates_history = []

    for t in range(500):
        # 1. 注入感觉电流
        nucleus.inject_sensory_current(80.0)
        # 2. 核团步进
        nucleus.step(t)
        # 3. TC 输出 → Column L4
        tc_count = len(nucleus.get_tc_output())
        column.inject_feedforward_current(tc_count * 30.0)
        # 4. Column 步进
        column.step(t)
        # 5. Column L6 → TC 反馈 + TRN 驱动 (★闭环关键)
        summary = column.get_output_summary()
        l6_rate = summary['l6_firing_rate']
        if l6_rate > 0:
            l6_ever_fired = True
            nucleus.inject_cortical_feedback_current(l6_rate * 10.0)
            nucleus.inject_trn_drive_current(l6_rate * 5.0)  # ★ L6→TRN 负反馈

        # 检查 NaN
        tc_rate = nucleus.get_tc_firing_rate()
        if math.isnan(tc_rate) or math.isnan(l6_rate):
            has_nan = True

        if (t + 1) % 100 == 0:
            rates = column.get_layer_firing_rates()
            burst_ratios = column.get_layer_burst_ratios()
            tc_rates_history.append(tc_rate)
            print(f"  t={t+1:3d}: TC rate={tc_rate:.1f}Hz, "
                  f"L4={rates.get(4,0):.1f}, L23={rates.get(23,0):.1f}, "
                  f"L5={rates.get(5,0):.1f}, L6={rates.get(6,0):.1f}Hz, "
                  f"L23 burst={burst_ratios.get(23,0):.3f}")

    # 最终检查
    final_tc = nucleus.get_tc_firing_rate()
    final_rates = column.get_layer_firing_rates()

    assert not has_nan, "系统中出现 NaN!"
    print(f"  ✓ 无 NaN")

    # ★ 加强断言: L6 必须有活动 (闭环验证)
    assert l6_ever_fired, "L6 从未发放 — 丘脑-皮层环路是开环的!"
    print(f"  ✓ L6 有发放 (闭环确认)")

    # ★ 加强断言: TC 不应无限增长
    assert final_tc < 200, f"TC 发放率异常高: {final_tc:.1f}Hz (可能缺少负反馈)"
    print(f"  ✓ TC 发放率稳定: {final_tc:.1f}Hz < 200Hz")

    # ★ 加强断言: L4 必须有活动
    assert final_rates.get(4, 0) > 0, "L4 无活动 — 丘脑输入未到达皮层!"
    print(f"  ✓ L4 有活动: {final_rates.get(4,0):.1f}Hz")

    print("  PASSED")


# ============================================================================
# 测试 5: 双柱层级预测编码 (★核心测试)
# ============================================================================

def test_hierarchical_prediction():
    """双柱层级预测编码 — 验证闭环 + Col1 活跃"""
    print("\n" + "=" * 60)
    print("测试 5: 双柱层级预测编码 (★核心)")
    print("=" * 60)

    from wuyun.circuit.multi_column import create_hierarchical_network

    # 使用默认 GainParams (已调优) + ff_connection_strength=1.5 (默认)
    network = create_hierarchical_network(
        n_columns=2, n_per_layer=20, n_tc=8, n_trn=4, seed=42
    )
    print(f"  网络: {network}")

    # 给 nucleus_0 注入恒定感觉输入
    early_regular = 0
    late_regular = 0
    col0_l6_ever_fired = False
    col1_ever_active = False

    for t in range(1000):
        # 注入感觉输入到核团 0
        network.router.nuclei[0].inject_sensory_current(80.0)
        network.step(t)

        # 收集预测误差 (L2/3 regular)
        col0_summary = network.columns[0].get_output_summary()
        regular_count = len(col0_summary.get('prediction_error', []))
        if t < 500:
            early_regular += regular_count
        else:
            late_regular += regular_count

        # 检查 L6 闭环
        if col0_summary['l6_firing_rate'] > 0:
            col0_l6_ever_fired = True

        # 检查 Col1 活跃
        col1_summary = network.columns[1].get_output_summary()
        if col1_summary['l23_firing_rate'] > 0:
            col1_ever_active = True

        if (t + 1) % 200 == 0:
            state = network.get_network_state()
            for col_id in sorted(state['columns'].keys()):
                cs = state['columns'][col_id]
                print(f"  t={t+1:4d} Col{col_id}: "
                      f"L23 rate={cs['l23_firing_rate']:.1f}Hz, "
                      f"L6 rate={cs['firing_rates'].get(6,0):.1f}Hz, "
                      f"L23 burst={cs['burst_ratios'].get(23,0):.3f}, "
                      f"errors={cs['prediction_error_count']}")
            for nid in sorted(state['nuclei'].keys()):
                ns = state['nuclei'][nid]
                print(f"  t={t+1:4d} Nuc{nid}: "
                      f"TC rate={ns['tc_firing_rate']:.1f}Hz, "
                      f"TC burst={ns['tc_burst_ratio']:.3f}")

    print(f"\n  前 500 步 regular 总数: {early_regular}")
    print(f"  后 500 步 regular 总数: {late_regular}")

    # 检查权重在合理范围
    all_weights_ok = True
    for col_id, column in network.columns.items():
        for sg in column.synapse_groups:
            if np.any(sg.weights < 0) or np.any(sg.weights > 1):
                all_weights_ok = False
                break

    assert all_weights_ok, "存在权重超出 [0, 1] 范围!"
    print(f"  ✓ 所有权重在 [0, 1] 范围内")

    # ★ 加强断言: L6 必须有活动 (闭环)
    assert col0_l6_ever_fired, "Col0 L6 从未发放 — 环路是开环的!"
    print(f"  ✓ Col0 L6 有发放 (闭环确认)")

    # ★ 加强断言: Col1 必须有活动 (层级传递)
    assert col1_ever_active, "Col1 从未活跃 — 层级误差传递失败!"
    print(f"  ✓ Col1 有活动 (层级传递确认)")

    # ★ 加强断言: TC 不应无限增长
    final_tc_rate = network.router.nuclei[0].get_tc_firing_rate()
    assert final_tc_rate < 200, f"Nuc0 TC 发放率过高: {final_tc_rate:.1f}Hz"
    print(f"  ✓ TC 发放率稳定: {final_tc_rate:.1f}Hz < 200Hz")

    print("  PASSED")


# ============================================================================
# 测试 6: 注意力切换 (TRN 竞争)
# ============================================================================

def test_attention_switching():
    """注意力切换: TRN 竞争实现 winner-take-all"""
    print("\n" + "=" * 60)
    print("测试 6: 注意力切换 (TRN 竞争)")
    print("=" * 60)

    from wuyun.circuit.multi_column import create_hierarchical_network

    # 使用默认 GainParams (已调优)
    network = create_hierarchical_network(
        n_columns=2, n_per_layer=20, n_tc=8, n_trn=4, seed=42
    )

    # 阶段 1 (0-300): nucleus_0 强(80), nucleus_1 弱(20)
    phase1_col0_rate = 0.0
    phase1_col1_rate = 0.0
    phase1_count = 0

    for t in range(300):
        network.router.nuclei[0].inject_sensory_current(80.0)
        network.router.nuclei[1].inject_sensory_current(20.0)
        network.step(t)

        if t >= 100:  # 跳过初始瞬态
            s0 = network.columns[0].get_output_summary()
            s1 = network.columns[1].get_output_summary()
            phase1_col0_rate += s0['l23_firing_rate']
            phase1_col1_rate += s1['l23_firing_rate']
            phase1_count += 1

    avg_p1_c0 = phase1_col0_rate / max(phase1_count, 1)
    avg_p1_c1 = phase1_col1_rate / max(phase1_count, 1)

    # 阶段 2 (300-1500): 反转 nucleus_0 弱(5), nucleus_1 强(120)
    # 延长到 1200 步，让 firing_rate 窗口 (1000ms) 完全滑过
    phase2_col0_rate = 0.0
    phase2_col1_rate = 0.0
    phase2_count = 0

    for t in range(300, 1500):
        network.router.nuclei[0].inject_sensory_current(5.0)
        network.router.nuclei[1].inject_sensory_current(120.0)
        network.step(t)

        if t >= 1300:  # 只统计最后 200 步 (窗口已完全滑过)
            s0 = network.columns[0].get_output_summary()
            s1 = network.columns[1].get_output_summary()
            phase2_col0_rate += s0['l23_firing_rate']
            phase2_col1_rate += s1['l23_firing_rate']
            phase2_count += 1

    avg_p2_c0 = phase2_col0_rate / max(phase2_count, 1)
    avg_p2_c1 = phase2_col1_rate / max(phase2_count, 1)

    print(f"  阶段 1 (nuc0=80, nuc1=20):")
    print(f"    Col0 avg L23 rate: {avg_p1_c0:.2f}Hz")
    print(f"    Col1 avg L23 rate: {avg_p1_c1:.2f}Hz")
    print(f"  阶段 2 (nuc0=5, nuc1=120):")
    print(f"    Col0 avg L23 rate: {avg_p2_c0:.2f}Hz")
    print(f"    Col1 avg L23 rate: {avg_p2_c1:.2f}Hz")

    # 阶段 1: 柱 0 应更活跃
    assert avg_p1_c0 > avg_p1_c1, (
        f"阶段 1 柱 0 应更活跃: {avg_p1_c0:.2f} > {avg_p1_c1:.2f}"
    )
    print(f"  ✓ 阶段 1: Col0({avg_p1_c0:.2f}) > Col1({avg_p1_c1:.2f})")

    # 阶段 2: 柱 1 应更活跃
    assert avg_p2_c1 > avg_p2_c0, (
        f"阶段 2 柱 1 应更活跃: {avg_p2_c1:.2f} > {avg_p2_c0:.2f}"
    )
    print(f"  ✓ 阶段 2: Col1({avg_p2_c1:.2f}) > Col0({avg_p2_c0:.2f})")
    print("  PASSED")


# ============================================================================
# 测试 7: 长期稳定性 + 稳态可塑性
# ============================================================================

def test_long_term_stability():
    """长期稳定性 + 稳态可塑性 — 验证 L6 闭环 + 权重稳定"""
    print("\n" + "=" * 60)
    print("测试 7: 长期稳定性 + 稳态可塑性")
    print("=" * 60)

    from wuyun.circuit.multi_column import create_hierarchical_network
    from wuyun.synapse.plasticity.homeostatic import HomeostaticPlasticity

    # 使用默认 GainParams (已调优)
    network = create_hierarchical_network(
        n_columns=2, n_per_layer=20, n_tc=8, n_trn=4, seed=42
    )
    homeostatic = HomeostaticPlasticity()

    all_weights_valid = True
    col0_l6_ever_fired = False

    for t in range(2000):
        network.router.nuclei[0].inject_sensory_current(80.0)
        network.step(t)

        # 检查 L6 闭环
        s0 = network.columns[0].get_output_summary()
        if s0['l6_firing_rate'] > 0:
            col0_l6_ever_fired = True

        # 每 500 步应用稳态可塑性
        if (t + 1) % 500 == 0:
            for col in network.columns.values():
                col.apply_homeostatic_scaling(homeostatic)

            # 收集权重统计
            for col_id, col in network.columns.items():
                weights = np.concatenate([sg.weights for sg in col.synapse_groups]) if col.synapse_groups else np.array([])
                if len(weights) > 0:
                    w_min = min(weights)
                    w_max = max(weights)
                    w_mean = sum(weights) / len(weights)

                    if w_min < 0 or w_max > 1:
                        all_weights_valid = False

                    rates = col.get_layer_firing_rates()
                    print(f"  t={t+1:4d} Col{col_id}: "
                          f"w_min={w_min:.4f}, w_max={w_max:.4f}, "
                          f"w_mean={w_mean:.4f}, "
                          f"L23={rates.get(23,0):.1f}Hz, "
                          f"L5={rates.get(5,0):.1f}Hz, "
                          f"L6={rates.get(6,0):.1f}Hz")

    assert all_weights_valid, "权重超出 [0, 1] 范围!"
    print(f"  ✓ 权重始终在 [0, 1]")

    # 检查发放率不超过 200Hz
    final_rates = network.columns[0].get_layer_firing_rates()
    for layer_id, rate in final_rates.items():
        assert rate < 200, f"Col0 L{layer_id} 发放率过高: {rate:.1f}Hz"
    print(f"  ✓ 发放率在合理范围")

    # ★ 加强断言: L6 必须有活动
    assert col0_l6_ever_fired, "Col0 L6 从未发放 — 2000 步后环路仍是开环的!"
    print(f"  ✓ Col0 L6 有发放 (闭环确认)")

    # ★ 加强断言: TC 不应无限增长
    final_tc = network.router.nuclei[0].get_tc_firing_rate()
    assert final_tc < 200, f"TC 发放率过高: {final_tc:.1f}Hz"
    print(f"  ✓ TC 发放率稳定: {final_tc:.1f}Hz < 200Hz")

    print("  PASSED")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("Phase 2 测试: 多柱 + 丘脑路由")
    print("=" * 60)

    tests = [
        ("测试 1: TC 中继基础", test_tc_relay_basic),
        ("测试 2: TRN 门控效应", test_trn_gating),
        ("测试 3: Tonic/Burst 双模式", test_tonic_burst_modes),
        ("测试 4: 丘脑-皮层环路", test_thalamocortical_loop),
        ("测试 5: 双柱层级预测编码", test_hierarchical_prediction),
        ("测试 6: 注意力切换", test_attention_switching),
        ("测试 7: 长期稳定性", test_long_term_stability),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append((name, f"ERROR: {type(e).__name__}: {e}"))
            print(f"  ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"结果: {passed}/{passed + failed} 通过")
    if errors:
        print("失败:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()