"""Phase 1.8: P2 前置补充测试

测试 3 项 Phase 2 前置依赖:
1. 稳态可塑性 (HomeostaticPlasticity)
2. 丘脑中继/TRN 神经元参数 (THALAMIC_RELAY_PARAMS, TRN_PARAMS)
3. CorticalColumn 标准接口扩展 (侧向/输出汇总/稳态集成)
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wuyun.synapse.plasticity.homeostatic import HomeostaticPlasticity, HomeostaticParams
from wuyun.neuron.neuron_base import THALAMIC_RELAY_PARAMS, TRN_PARAMS, NeuronBase
from wuyun.circuit.column_factory import create_sensory_column
from wuyun.spike.signal_types import SpikeType, NeuronType


def test_homeostatic_scaling_factor():
    """稳态可塑性: 高发放率→缩小, 低发放率→放大"""
    h = HomeostaticPlasticity(HomeostaticParams(target_rate=5.0))

    # 发放率 = 目标 → factor ≈ 1.0
    f_normal = h.compute_scaling_factor(5.0)
    assert 0.99 < f_normal < 1.01

    # 发放率太高 → factor < 1.0
    f_high = h.compute_scaling_factor(15.0)
    assert f_high < 1.0

    # 发放率太低 → factor > 1.0
    f_low = h.compute_scaling_factor(1.0)
    assert f_low > 1.0

    print(f"  rate=5Hz → factor={f_normal:.4f}")
    print(f"  rate=15Hz → factor={f_high:.4f}")
    print(f"  rate=1Hz → factor={f_low:.4f}")
    print("  ✅ PASS: 稳态缩放方向正确")


def test_homeostatic_weight_scaling():
    """稳态可塑性: 实际权重缩放在 [w_min, w_max] 内"""
    h = HomeostaticPlasticity()

    # 高发放率 → 权重缩小
    new_w = h.scale_weight(0.5, 15.0, 0.0, 1.0)
    assert new_w < 0.5

    # 低发放率 → 权重放大
    new_w = h.scale_weight(0.5, 1.0, 0.0, 1.0)
    assert new_w > 0.5

    # 不超出边界
    new_w = h.scale_weight(0.99, 0.1, 0.0, 1.0)
    assert new_w <= 1.0
    new_w = h.scale_weight(0.01, 50.0, 0.0, 1.0)
    assert new_w >= 0.0

    print("  ✅ PASS: 权重缩放边界正确")


def test_thalamic_relay_params():
    """丘脑中继神经元: 参数正确, 能发放 burst"""
    tc = NeuronBase(neuron_id=1, params=THALAMIC_RELAY_PARAMS)

    assert tc.params.neuron_type == NeuronType.THALAMIC_RELAY
    assert tc.params.kappa == 0.3
    assert tc.has_apical  # TC 有 apical (接收皮层反馈)
    assert tc.params.apical.ca_duration == 40  # TC burst 持续更长
    assert tc.params.burst_spike_count == 4

    # 验证 TC 能发放
    fired = False
    for t in range(100):
        tc.inject_basal_current(80.0)  # 强感觉输入
        st = tc.step(t)
        if st.is_active:
            fired = True

    assert fired, "TC 应该能发放"
    print(f"  TC neuron: κ={tc.params.kappa}, ca_dur={tc.params.apical.ca_duration}")
    print("  ✅ PASS: 丘脑中继参数正确")


def test_trn_params():
    """TRN 神经元: 抑制性, 无 apical, 低阈值"""
    trn = NeuronBase(neuron_id=2, params=TRN_PARAMS)

    assert trn.params.neuron_type == NeuronType.TRN
    assert trn.params.kappa == 0.0  # 抑制性, 单区室
    assert not trn.has_apical
    assert trn.params.somatic.v_threshold == -45.0  # 低阈值

    # 验证 TRN 容易被激活
    fired = False
    for t in range(50):
        trn.inject_basal_current(60.0)
        st = trn.step(t)
        if st.is_active:
            fired = True
            break

    assert fired, "TRN 应该容易被激活 (低阈值)"
    print("  ✅ PASS: TRN 参数正确")


def test_column_lateral_interface():
    """皮层柱: 侧向接口和输出汇总"""
    col = create_sensory_column(
        column_id=0, n_per_layer=10, seed=42, ff_connection_strength=1.5,
    )

    # 注入前馈激活
    for t in range(100):
        col.inject_feedforward_current(50.0)
        col.step(t)

    # 测试 get_output_summary
    summary = col.get_output_summary()
    assert 'prediction_error' in summary
    assert 'match_signal' in summary
    assert 'prediction' in summary
    assert 'drive' in summary
    assert 'l23_firing_rate' in summary

    # 测试 get_neuron_ids
    l4_ids = col.get_neuron_ids(4, excitatory_only=True)
    assert len(l4_ids) > 0
    l23_all = col.get_neuron_ids(23, excitatory_only=False)
    l23_exc = col.get_neuron_ids(23, excitatory_only=True)
    assert len(l23_all) >= len(l23_exc)

    # 测试侧向注入 (不报错即可)
    col.inject_lateral_current(10.0)
    col.step(100)

    print(f"  output_summary keys: {list(summary.keys())}")
    print(f"  L4 exc IDs: {len(l4_ids)}, L23 all: {len(l23_all)}, L23 exc: {len(l23_exc)}")
    print("  ✅ PASS: 侧向接口和输出汇总正确")


def test_homeostatic_integration():
    """稳态可塑性集成: CorticalColumn.apply_homeostatic_scaling()"""
    col = create_sensory_column(
        column_id=0, n_per_layer=30, seed=42, ff_connection_strength=1.5,
    )
    h = HomeostaticPlasticity()

    # 记录初始权重
    initial_weights = []
    for sg in col.synapse_groups:
        if sg.is_excitatory:
            initial_weights.extend(sg.weights.tolist())

    # 跑 500 步让网络活跃
    for t in range(500):
        col.inject_feedforward_current(50.0)
        col.step(t)

    # 应用稳态缩放
    col.apply_homeostatic_scaling(h)

    # 权重应该有变化 (因为发放率不可能刚好 = target_rate)
    new_weights = []
    for sg in col.synapse_groups:
        if sg.is_excitatory:
            new_weights.extend(sg.weights.tolist())
    changed = sum(1 for a, b in zip(initial_weights, new_weights) if abs(a - b) > 1e-6)

    # 至少有一些权重发生了变化
    assert changed > 0, f"应有权重变化, 但 {changed}/{len(initial_weights)} 变了"

    # 所有权重仍在合法范围内
    for sg in col.synapse_groups:
        for w in sg.weights:
            assert sg.w_min <= w <= sg.w_max, f"权重越界: {w}"

    print(f"  {changed}/{len(initial_weights)} 个兴奋性权重被缩放")
    print("  ✅ PASS: 稳态可塑性集成正确")


if __name__ == '__main__':
    tests = [
        test_homeostatic_scaling_factor,
        test_homeostatic_weight_scaling,
        test_thalamic_relay_params,
        test_trn_params,
        test_column_lateral_interface,
        test_homeostatic_integration,
    ]

    print("=" * 60)
    print("  Phase 1.8: P2 前置补充测试")
    print("=" * 60)

    passed = 0
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAIL: {e}")

    print(f"\n{'=' * 60}")
    print(f"  结果: {passed}/{len(tests)} 通过")
    print(f"{'=' * 60}")