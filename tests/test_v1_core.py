"""Phase V1: 向量化核心引擎数值等价性验证

验证 NeuronPopulation + SynapseGroup 的输出与
原始 NeuronBase + SynapseBase 在相同输入下完全一致。

测试策略:
1. 用相同参数创建 NeuronBase 和 NeuronPopulation
2. 注入相同电流
3. 逐步比较 V_soma, V_apical, 发放时间, 脉冲类型
4. 误差 < 1e-10 视为数值等价
"""

import sys
import os
import time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wuyun.neuron.neuron_base import (
    NeuronBase, NeuronParams,
    L23_PYRAMIDAL_PARAMS, L5_PYRAMIDAL_PARAMS,
    BASKET_PV_PARAMS, GRANULE_PARAMS, PLACE_CELL_PARAMS,
)
from wuyun.synapse.synapse_base import SynapseBase, AMPA_PARAMS, GABA_A_PARAMS, NMDA_PARAMS
from wuyun.spike.signal_types import SpikeType, SynapseType, CompartmentType
from wuyun.spike.spike_bus import SpikeBus
from wuyun.spike.spike import Spike

from wuyun.core.population import NeuronPopulation
from wuyun.core.synapse_group import SynapseGroup


TOL = 1e-10  # 数值等价容差


# =========================================================================
# 测试 1: 单神经元 regular spike 等价性
# =========================================================================

def test_1_regular_spike_equivalence():
    """L2/3 锥体 — 只注入 basal 电流 → regular spike"""
    print("\n" + "=" * 70)
    print("测试 1: Regular spike 数值等价性 (L2/3 pyramidal)")
    print("=" * 70)

    params = L23_PYRAMIDAL_PARAMS

    # 旧: NeuronBase
    old = NeuronBase(neuron_id=0, params=params)

    # 新: NeuronPopulation (n=1)
    pop = NeuronPopulation(1, params)

    max_v_err = 0.0
    max_va_err = 0.0
    spike_match = True
    current = 8.0

    for t in range(200):
        # 注入相同电流
        old.inject_basal_current(current)
        pop.i_basal[0] = current

        # 推进
        old_type = old.step(t)
        pop.step(t)

        # 比较 V_soma
        v_err = abs(old.soma.v - pop.v_soma[0])
        va_err = abs(old.apical.v - pop.v_apical[0])
        max_v_err = max(max_v_err, v_err)
        max_va_err = max(max_va_err, va_err)

        # 比较脉冲类型
        new_type_val = pop.spike_type[0]
        old_type_val = old_type.value
        if old_type_val != new_type_val:
            print(f"  ✗ t={t}: old={old_type.name}({old_type_val}) vs new={new_type_val}")
            spike_match = False

    print(f"  V_soma max error:   {max_v_err:.2e}")
    print(f"  V_apical max error: {max_va_err:.2e}")
    print(f"  Spike types match:  {spike_match}")
    assert max_v_err < TOL, f"V_soma 误差 {max_v_err} > {TOL}"
    assert max_va_err < TOL, f"V_apical 误差 {max_va_err} > {TOL}"
    assert spike_match, "脉冲类型不匹配"
    print("  ✓ Regular spike 数值等价验证通过")


# =========================================================================
# 测试 2: burst spike 等价性
# =========================================================================

def test_2_burst_spike_equivalence():
    """L5 锥体 — basal + apical 同时输入 → burst"""
    print("\n" + "=" * 70)
    print("测试 2: Burst spike 数值等价性 (L5 pyramidal, κ=0.6)")
    print("=" * 70)

    params = L5_PYRAMIDAL_PARAMS

    old = NeuronBase(neuron_id=0, params=params)
    pop = NeuronPopulation(1, params)

    max_v_err = 0.0
    max_va_err = 0.0
    spike_mismatch = 0
    old_spikes = []
    new_spikes = []

    for t in range(300):
        # basal + apical 同时注入 (apical 需足够强以触发 Ca²⁺ → burst)
        old.inject_basal_current(25.0)
        old.inject_apical_current(50.0)
        pop.i_basal[0] = 25.0
        pop.i_apical[0] = 50.0

        old_type = old.step(t)
        pop.step(t)

        v_err = abs(old.soma.v - pop.v_soma[0])
        va_err = abs(old.apical.v - pop.v_apical[0])
        max_v_err = max(max_v_err, v_err)
        max_va_err = max(max_va_err, va_err)

        if old_type.is_active:
            old_spikes.append((t, old_type.name))
        if pop.spike_type[0] != 0:
            new_spikes.append((t, SpikeType(int(pop.spike_type[0])).name))

        if old_type.value != pop.spike_type[0]:
            spike_mismatch += 1

    print(f"  V_soma max error:   {max_v_err:.2e}")
    print(f"  V_apical max error: {max_va_err:.2e}")
    print(f"  Old spikes: {len(old_spikes)}, New spikes: {len(new_spikes)}")
    print(f"  Spike mismatches: {spike_mismatch}")

    if old_spikes[:5]:
        print(f"  Old first 5: {old_spikes[:5]}")
        print(f"  New first 5: {new_spikes[:5]}")

    # 检查有 burst
    has_burst_old = any(s[1].startswith('BURST') for s in old_spikes)
    has_burst_new = any(s[1].startswith('BURST') for s in new_spikes)
    print(f"  Has burst: old={has_burst_old}, new={has_burst_new}")

    assert max_v_err < TOL, f"V_soma 误差 {max_v_err} > {TOL}"
    assert max_va_err < TOL, f"V_apical 误差 {max_va_err} > {TOL}"
    assert spike_mismatch == 0, f"{spike_mismatch} 步脉冲类型不匹配"
    assert has_burst_old and has_burst_new, "应该产生 burst"
    print("  ✓ Burst spike 数值等价验证通过")


# =========================================================================
# 测试 3: 单区室 (PV, κ=0) 等价性
# =========================================================================

def test_3_single_compartment_equivalence():
    """PV 篮状细胞 — κ=0, 无顶端树突"""
    print("\n" + "=" * 70)
    print("测试 3: 单区室等价性 (PV, κ=0)")
    print("=" * 70)

    params = BASKET_PV_PARAMS

    old = NeuronBase(neuron_id=0, params=params)
    pop = NeuronPopulation(1, params)

    max_v_err = 0.0
    spike_mismatch = 0

    for t in range(200):
        old.inject_basal_current(6.0)
        pop.i_basal[0] = 6.0

        old_type = old.step(t)
        pop.step(t)

        v_err = abs(old.soma.v - pop.v_soma[0])
        max_v_err = max(max_v_err, v_err)

        if old_type.value != pop.spike_type[0]:
            spike_mismatch += 1

    print(f"  V_soma max error: {max_v_err:.2e}")
    print(f"  Spike mismatches: {spike_mismatch}")
    assert max_v_err < TOL, f"V_soma 误差 {max_v_err} > {TOL}"
    assert spike_mismatch == 0, f"{spike_mismatch} 步不匹配"
    print("  ✓ 单区室数值等价验证通过")


# =========================================================================
# 测试 4: 多神经元群体批量 vs 逐个
# =========================================================================

def test_4_population_batch_vs_individual():
    """N=20 个 L2/3 锥体, 不同电流, 验证批量结果 = 逐个结果"""
    print("\n" + "=" * 70)
    print("测试 4: 群体批量 vs 逐个 (N=20, 不同电流)")
    print("=" * 70)

    params = L23_PYRAMIDAL_PARAMS
    N = 20

    # 旧: N 个独立 NeuronBase
    old_neurons = [NeuronBase(neuron_id=i, params=params) for i in range(N)]

    # 新: 一个 NeuronPopulation(N)
    pop = NeuronPopulation(N, params)

    rng = np.random.RandomState(42)
    currents = rng.uniform(0, 15, size=N)

    max_v_err = 0.0
    total_mismatches = 0

    for t in range(300):
        for i in range(N):
            old_neurons[i].inject_basal_current(currents[i])
        pop.i_basal[:] = currents

        old_types = [n.step(t) for n in old_neurons]
        pop.step(t)

        for i in range(N):
            v_err = abs(old_neurons[i].soma.v - pop.v_soma[i])
            max_v_err = max(max_v_err, v_err)
            if old_types[i].value != pop.spike_type[i]:
                total_mismatches += 1

    print(f"  V_soma max error: {max_v_err:.2e}")
    print(f"  Total mismatches: {total_mismatches} / {300 * N}")
    assert max_v_err < TOL, f"V_soma 误差 {max_v_err} > {TOL}"
    assert total_mismatches == 0, f"有 {total_mismatches} 步不匹配"
    print("  ✓ 群体批量等价验证通过")


# =========================================================================
# 测试 5: SynapseGroup 电流计算等价性
# =========================================================================

def test_5_synapse_group_equivalence():
    """AMPA 突触组电流计算 vs 逐个 SynapseBase"""
    print("\n" + "=" * 70)
    print("测试 5: SynapseGroup 电流计算等价性")
    print("=" * 70)

    N_pre = 5
    N_post = 3

    # 构建连接: pre[i] → post[j], 概率 0.6
    rng = np.random.RandomState(123)
    pre_ids = []
    post_ids = []
    weights = []
    for i in range(N_pre):
        for j in range(N_post):
            if rng.random() < 0.6:
                pre_ids.append(i)
                post_ids.append(j)
                weights.append(rng.uniform(0.2, 0.8))

    K = len(pre_ids)
    pre_ids = np.array(pre_ids, dtype=np.int32)
    post_ids = np.array(post_ids, dtype=np.int32)
    weights_arr = np.array(weights)
    delays = np.ones(K, dtype=np.int32)

    print(f"  突触数: {K}")

    # 旧: SpikeBus + SynapseBase
    bus = SpikeBus()
    old_synapses = []
    old_post_neurons = [NeuronBase(neuron_id=100 + j, params=L23_PYRAMIDAL_PARAMS) for j in range(N_post)]

    for k in range(K):
        syn = SynapseBase(
            pre_id=pre_ids[k],
            post_id=100 + post_ids[k],
            weight=weights_arr[k],
            delay=1,
            synapse_type=SynapseType.AMPA,
            target_compartment=CompartmentType.BASAL,
            params=AMPA_PARAMS,
        )
        old_post_neurons[post_ids[k]].add_synapse(syn)
        bus.register_synapse(syn)
        old_synapses.append(syn)

    # 新: SynapseGroup
    group = SynapseGroup(
        pre_ids=pre_ids,
        post_ids=post_ids,
        weights=weights_arr,
        delays=delays,
        synapse_type=SynapseType.AMPA,
        target=CompartmentType.BASAL,
        tau_decay=AMPA_PARAMS.tau_decay,
        e_rev=AMPA_PARAMS.e_rev,
        g_max=AMPA_PARAMS.g_max,
        n_post=N_post,
    )
    pop_post = NeuronPopulation(N_post, L23_PYRAMIDAL_PARAMS)

    max_current_err = 0.0

    for t in range(100):
        # 模拟突触前发放: 每 10 步某些 pre 发放
        pre_fired = np.zeros(N_pre, dtype=bool)
        pre_spike_type = np.zeros(N_pre, dtype=np.int8)
        if t % 10 < 3:
            active = t % N_pre
            pre_fired[active] = True
            pre_spike_type[active] = SpikeType.REGULAR.value

            # 旧: emit spike
            spike = Spike(source_id=active, timestamp=t, spike_type=SpikeType.REGULAR)
            bus.emit(spike)

        # 旧: deliver + step synapses + collect currents
        bus.step(t)
        old_currents = np.zeros(N_post)
        for j in range(N_post):
            n = old_post_neurons[j]
            v_soma = n.soma.v
            for syn in n._synapses_basal:
                syn.step(t)
                old_currents[j] += syn.compute_current(v_soma)

        # 新: deliver + step_and_compute
        group.deliver_spikes(pre_fired, pre_spike_type)
        new_currents = group.step_and_compute(pop_post.v_soma)

        err = np.abs(old_currents - new_currents).max()
        max_current_err = max(max_current_err, err)

    print(f"  Max current error: {max_current_err:.2e}")
    assert max_current_err < 1e-8, f"电流误差 {max_current_err} > 1e-8"
    print("  ✓ SynapseGroup 电流计算等价验证通过")


# =========================================================================
# 测试 6: 性能对比
# =========================================================================

def test_6_performance_comparison():
    """NeuronPopulation vs NeuronBase 性能对比"""
    print("\n" + "=" * 70)
    print("测试 6: 性能对比")
    print("=" * 70)

    N = 200
    T = 1000
    params = L23_PYRAMIDAL_PARAMS

    # 旧: N 个 NeuronBase
    old_neurons = [NeuronBase(neuron_id=i, params=params) for i in range(N)]

    rng = np.random.RandomState(42)
    currents = rng.uniform(3, 10, size=N)

    t0 = time.time()
    for t in range(T):
        for i, n in enumerate(old_neurons):
            n.inject_basal_current(currents[i])
            n.step(t)
    old_time = time.time() - t0

    # 新: NeuronPopulation(N)
    pop = NeuronPopulation(N, params)

    t0 = time.time()
    for t in range(T):
        pop.i_basal[:] = currents
        pop.step(t)
    new_time = time.time() - t0

    speedup = old_time / new_time if new_time > 0 else float('inf')
    print(f"  NeuronBase (N={N}, T={T}): {old_time:.3f}s")
    print(f"  NeuronPopulation:           {new_time:.3f}s")
    print(f"  加速比: {speedup:.1f}×")
    assert new_time < old_time, "NeuronPopulation 应该更快"
    print("  ✓ 性能验证通过")


# =========================================================================
# 主函数
# =========================================================================

def main():
    tests = [
        ("测试 1: Regular spike 等价", test_1_regular_spike_equivalence),
        ("测试 2: Burst spike 等价", test_2_burst_spike_equivalence),
        ("测试 3: 单区室等价", test_3_single_compartment_equivalence),
        ("测试 4: 群体批量等价", test_4_population_batch_vs_individual),
        ("测试 5: SynapseGroup 等价", test_5_synapse_group_equivalence),
        ("测试 6: 性能对比", test_6_performance_comparison),
    ]

    passed = 0
    failed = 0
    errors = []

    total_start = time.time()
    for name, fn in tests:
        try:
            start = time.time()
            fn()
            elapsed = time.time() - start
            print(f"  ⏱ {elapsed:.2f}s")
            passed += 1
        except AssertionError as e:
            elapsed = time.time() - start
            print(f"  ✗ 断言失败: {e}")
            print(f"  ⏱ {elapsed:.2f}s")
            failed += 1
            errors.append((name, str(e)))
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ✗ 异常: {type(e).__name__}: {e}")
            print(f"  ⏱ {elapsed:.2f}s")
            failed += 1
            errors.append((name, f"{type(e).__name__}: {e}"))

    total = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"Phase V1 测试结果: {passed}/{passed + failed} 通过")
    print(f"总耗时: {total:.2f}s")
    if errors:
        print("\n失败:")
        for name, err in errors:
            print(f"  ✗ {name}: {err}")
    print("=" * 70)
    assert failed == 0, f"{failed} 个测试失败"


if __name__ == "__main__":
    main()
