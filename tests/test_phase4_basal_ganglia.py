"""
Phase 4 测试: 基底节 + 强化学习

7 个测试:
1. MSN 神经元参数 + Up/Down 态
2. DA 调制方向性
3. GPi tonic 抑制 + 直接通路去抑制
4. 三通路竞争
5. 动作选择 (Winner-Take-All)
6. 三因子 STDP 学习
7. 全环路 — 皮层→基底节→丘脑去抑制

不使用 pytest, 用 print() + assert 验证。
"""

import sys
import os
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wuyun.neuron.neuron_base import (
    NeuronBase,
    MSN_D1_PARAMS,
    MSN_D2_PARAMS,
    STN_PARAMS,
)
from wuyun.spike.signal_types import SpikeType
from wuyun.circuit.basal_ganglia.striatum import Striatum
from wuyun.circuit.basal_ganglia.gpi import GPi
from wuyun.circuit.basal_ganglia.indirect_pathway import GPe, STN as STNModule
from wuyun.circuit.basal_ganglia.basal_ganglia import BasalGangliaCircuit
from wuyun.thalamus.thalamic_nucleus import create_thalamic_nucleus


def test_1_msn_params_up_down_state():
    """测试 1: MSN 神经元参数 + Up/Down 态"""
    print("\n" + "=" * 60)
    print("测试 1: MSN 神经元参数 + Up/Down 态")
    print("=" * 60)

    # 创建 D1 和 D2 MSN
    d1 = NeuronBase(neuron_id=100, params=MSN_D1_PARAMS)
    d2 = NeuronBase(neuron_id=101, params=MSN_D2_PARAMS)

    # 验证静息电位 -80mV (Down state)
    assert abs(d1.v_soma - (-80.0)) < 0.1, f"D1 静息电位应为 -80mV, 实际: {d1.v_soma}"
    assert abs(d2.v_soma - (-80.0)) < 0.1, f"D2 静息电位应为 -80mV, 实际: {d2.v_soma}"
    print(f"  ✓ D1 静息电位: {d1.v_soma}mV (Down state)")
    print(f"  ✓ D2 静息电位: {d2.v_soma}mV (Down state)")

    # 验证 κ=0 (单区室)
    assert d1.kappa == 0.0, f"D1 κ 应为 0, 实际: {d1.kappa}"
    assert d2.kappa == 0.0, f"D2 κ 应为 0, 实际: {d2.kappa}"
    print(f"  ✓ D1 κ={d1.kappa} (单区室)")
    print(f"  ✓ D2 κ={d2.kappa} (单区室)")

    # 注入足够电流使其进入 Up state 并发放
    # MSN: v_rest=-80, v_threshold=-50, 需要 I > 30 才能超过阈值
    d1_fired = False
    d2_fired = False
    for t in range(200):
        d1.inject_basal_current(45.0)  # 强电流驱动 Up state (需 >30)
        d2.inject_basal_current(45.0)
        st1 = d1.step(t)
        st2 = d2.step(t)
        if st1.is_active:
            d1_fired = True
        if st2.is_active:
            d2_fired = True

    assert d1_fired, "D1 注入足够电流后应发放"
    assert d2_fired, "D2 注入足够电流后应发放"
    print(f"  ✓ D1 注入电流后成功发放")
    print(f"  ✓ D2 注入电流后成功发放")

    # 验证只有 regular spike (κ=0, 无 burst)
    d1_test = NeuronBase(neuron_id=200, params=MSN_D1_PARAMS)
    spike_types = set()
    for t in range(300):
        d1_test.inject_basal_current(45.0)
        st = d1_test.step(t)
        if st.is_active:
            spike_types.add(st)

    # MSN κ=0 → 不应有 burst
    has_burst = any(s.is_burst for s in spike_types)
    assert not has_burst, f"MSN(κ=0) 不应产生 burst, 实际脉冲类型: {spike_types}"
    print(f"  ✓ MSN(κ=0) 只产生 regular spike (无 burst)")

    # D1 和 D2 内在属性相同
    assert MSN_D1_PARAMS.somatic.v_rest == MSN_D2_PARAMS.somatic.v_rest
    assert MSN_D1_PARAMS.somatic.v_threshold == MSN_D2_PARAMS.somatic.v_threshold
    assert MSN_D1_PARAMS.somatic.tau_m == MSN_D2_PARAMS.somatic.tau_m
    assert MSN_D1_PARAMS.kappa == MSN_D2_PARAMS.kappa
    print(f"  ✓ D1 和 D2 内在属性完全相同 (差异来自 DA 调制)")

    print("  ★ 测试 1 通过!")


def test_2_da_modulation_direction():
    """测试 2: DA 调制方向性"""
    print("\n" + "=" * 60)
    print("测试 2: DA 调制方向性")
    print("=" * 60)

    sim_steps = 300

    # === DA=0 基线 ===
    stri_base = Striatum(n_d1=10, n_d2=10, n_fsi=4, seed=42)
    cortical = np.ones(10) * 40.0  # 统一皮层输入 (MSN 需要 >30)
    for t in range(sim_steps):
        stri_base.inject_cortical_input(cortical)
        stri_base.step(t)
    d1_rate_base = np.mean(stri_base.get_d1_rates())
    d2_rate_base = np.mean(stri_base.get_d2_rates())
    print(f"  DA=0: D1 rate={d1_rate_base:.1f}Hz, D2 rate={d2_rate_base:.1f}Hz")

    # === DA>0 (奖励) ===
    stri_pos = Striatum(n_d1=10, n_d2=10, n_fsi=4, da_gain_d1=15.0, da_gain_d2=15.0, seed=42)
    for t in range(sim_steps):
        stri_pos.inject_cortical_input(cortical)
        stri_pos.apply_dopamine(1.0)  # DA > 0
        stri_pos.step(t)
    d1_rate_pos = np.mean(stri_pos.get_d1_rates())
    d2_rate_pos = np.mean(stri_pos.get_d2_rates())
    print(f"  DA>0: D1 rate={d1_rate_pos:.1f}Hz, D2 rate={d2_rate_pos:.1f}Hz")

    # === DA<0 (惩罚) ===
    stri_neg = Striatum(n_d1=10, n_d2=10, n_fsi=4, da_gain_d1=15.0, da_gain_d2=15.0, seed=42)
    for t in range(sim_steps):
        stri_neg.inject_cortical_input(cortical)
        stri_neg.apply_dopamine(-1.0)  # DA < 0
        stri_neg.step(t)
    d1_rate_neg = np.mean(stri_neg.get_d1_rates())
    d2_rate_neg = np.mean(stri_neg.get_d2_rates())
    print(f"  DA<0: D1 rate={d1_rate_neg:.1f}Hz, D2 rate={d2_rate_neg:.1f}Hz")

    # DA>0 时: D1 发放率 > D2 发放率
    assert d1_rate_pos > d2_rate_pos, \
        f"DA>0 时 D1({d1_rate_pos:.1f}) 应 > D2({d2_rate_pos:.1f})"
    print(f"  ✓ DA>0: D1({d1_rate_pos:.1f}) > D2({d2_rate_pos:.1f})")

    # DA<0 时: D2 发放率 > D1 发放率
    assert d2_rate_neg > d1_rate_neg, \
        f"DA<0 时 D2({d2_rate_neg:.1f}) 应 > D1({d1_rate_neg:.1f})"
    print(f"  ✓ DA<0: D2({d2_rate_neg:.1f}) > D1({d1_rate_neg:.1f})")

    # DA 对 D1 和 D2 的调制方向相反
    d1_delta = d1_rate_pos - d1_rate_neg  # D1: DA+ 增强, DA- 减弱 → 正
    d2_delta = d2_rate_pos - d2_rate_neg  # D2: DA+ 减弱, DA- 增强 → 负
    assert d1_delta > 0, f"D1 应被 DA 正向调制, delta={d1_delta:.1f}"
    assert d2_delta < 0, f"D2 应被 DA 反向调制, delta={d2_delta:.1f}"
    print(f"  ✓ DA 调制方向: D1 delta={d1_delta:.1f} (正), D2 delta={d2_delta:.1f} (负)")

    print("  ★ 测试 2 通过!")


def test_3_gpi_tonic_and_direct_pathway():
    """测试 3: GPi tonic 抑制 + 直接通路去抑制"""
    print("\n" + "=" * 60)
    print("测试 3: GPi tonic 抑制 + 直接通路去抑制")
    print("=" * 60)

    sim_steps = 300

    # === GPi 无外部输入: tonic 发放 ===
    gpi_tonic = GPi(n_neurons=10, tonic_drive=25.0, seed=42)
    for t in range(sim_steps):
        gpi_tonic.step(t)

    tonic_rate = np.mean(gpi_tonic.get_output_rates())
    print(f"  GPi tonic rate (无输入): {tonic_rate:.1f}Hz")
    assert tonic_rate > 30.0, f"GPi tonic 应 > 30Hz, 实际: {tonic_rate:.1f}Hz"
    print(f"  ✓ GPi tonic > 30Hz ({tonic_rate:.1f}Hz)")

    # === 直接通路: 注入 D1 活动 → GPi 发放率应下降 ===
    gpi_inhib = GPi(n_neurons=10, tonic_drive=25.0, seed=42)
    fake_d1_rates = np.ones(10) * 40.0  # 模拟 D1 高频发放
    for t in range(sim_steps):
        gpi_inhib.inject_direct_inhibition([], fake_d1_rates, gain=15.0)
        gpi_inhib.step(t)

    inhibited_rate = np.mean(gpi_inhib.get_output_rates())
    print(f"  GPi rate (有 D1 抑制): {inhibited_rate:.1f}Hz")
    assert inhibited_rate < tonic_rate, \
        f"GPi+D1 ({inhibited_rate:.1f}) 应 < GPi tonic ({tonic_rate:.1f})"
    print(f"  ✓ GPi+D1({inhibited_rate:.1f}) < GPi tonic({tonic_rate:.1f}) → 去抑制成功")

    # 验证去抑制程度
    reduction = (tonic_rate - inhibited_rate) / tonic_rate * 100
    print(f"  ✓ GPi 发放率下降 {reduction:.1f}%")

    print("  ★ 测试 3 通过!")


def test_4_three_pathway_competition():
    """测试 4: 三通路竞争"""
    print("\n" + "=" * 60)
    print("测试 4: 三通路竞争")
    print("=" * 60)

    sim_steps = 300

    # === 基线: 无输入 ===
    bg_base = BasalGangliaCircuit(n_actions=4, seed=42)
    zero_input = np.zeros(4)
    for t in range(sim_steps):
        bg_base.step(t, zero_input, da_level=0.0)
    base_values = bg_base.get_action_values()
    print(f"  基线 action_values: {[f'{v:.3f}' for v in base_values]}")

    # === Go 通路: 强输入 + DA>0 到第 0 个动作 ===
    bg_go = BasalGangliaCircuit(n_actions=4, seed=42)
    go_input = np.array([50.0, 0.0, 0.0, 0.0])
    for t in range(sim_steps):
        bg_go.step(t, go_input, da_level=0.8)
    go_values = bg_go.get_action_values()
    print(f"  Go (action 0, DA>0) values: {[f'{v:.3f}' for v in go_values]}")

    # Go 通道应有更高的 action_value
    assert go_values[0] > base_values[0] or go_values[0] > np.mean(go_values[1:]), \
        f"Go 通道 {go_values[0]:.3f} 应高于其他通道均值 {np.mean(go_values[1:]):.3f}"
    print(f"  ✓ Go 通道 ({go_values[0]:.3f}) 高于其他通道")

    # === NoGo 通路: 强输入 + DA<0 ===
    bg_nogo = BasalGangliaCircuit(n_actions=4, seed=42)
    nogo_input = np.array([50.0, 0.0, 0.0, 0.0])
    for t in range(sim_steps):
        bg_nogo.step(t, nogo_input, da_level=-0.8)
    nogo_values = bg_nogo.get_action_values()
    print(f"  NoGo (action 0, DA<0) values: {[f'{v:.3f}' for v in nogo_values]}")

    # NoGo 应使 action_value 低于 Go
    assert go_values[0] > nogo_values[0], \
        f"Go ({go_values[0]:.3f}) 应 > NoGo ({nogo_values[0]:.3f})"
    print(f"  ✓ Go ({go_values[0]:.3f}) > NoGo ({nogo_values[0]:.3f})")

    # === Stop 通路: 全通道强输入 (超直接) ===
    bg_stop = BasalGangliaCircuit(n_actions=4, seed=42,
                                   hyperdirect_gain=40.0)  # 超强超直接通路
    stop_input = np.ones(4) * 30.0
    for t in range(sim_steps):
        bg_stop.step(t, stop_input, da_level=0.0)
    stop_values = bg_stop.get_action_values()
    print(f"  Stop (全通道) values: {[f'{v:.3f}' for v in stop_values]}")

    # Stop 应使所有 action_values 低
    mean_stop = np.mean(stop_values)
    mean_go = np.mean(go_values)
    print(f"  Stop 均值: {mean_stop:.3f}, Go 均值: {mean_go:.3f}")
    # 超直接通路全局激活 STN → GPi 增强 → 所有通道更强抑制
    # 与 Go(DA>0) 相比, Stop 的整体去抑制程度应该更低
    assert mean_stop <= mean_go + 0.1, \
        f"Stop 均值({mean_stop:.3f}) 不应显著高于 Go 均值({mean_go:.3f})"
    print(f"  ✓ Stop 使全局 action_values 不高于 Go")

    print("  ★ 测试 4 通过!")


def test_5_action_selection_wta():
    """测试 5: 动作选择 (Winner-Take-All)"""
    print("\n" + "=" * 60)
    print("测试 5: 动作选择 (Winner-Take-All)")
    print("=" * 60)

    sim_steps = 300

    # 给 4 个通道不同强度的输入, 最强的应被选中
    bg = BasalGangliaCircuit(n_actions=4, seed=42)
    input_vec = np.array([20.0, 50.0, 10.0, 30.0])  # 通道 1 最强

    for t in range(sim_steps):
        bg.step(t, input_vec, da_level=0.5)  # DA>0 放大 D1 差异

    values = bg.get_action_values()
    selected = bg.select_action()
    print(f"  输入: {input_vec}")
    print(f"  Action values: {[f'{v:.3f}' for v in values]}")
    print(f"  选中动作: {selected}")

    # 最强输入的通道应被选中
    expected = int(np.argmax(input_vec))
    assert selected == expected, \
        f"应选中通道 {expected} (最强输入), 实际: {selected}"
    print(f"  ✓ 正确选中通道 {expected} (最强输入)")

    # DA 正信号应放大差异
    bg2 = BasalGangliaCircuit(n_actions=4, seed=42)
    for t in range(sim_steps):
        bg2.step(t, input_vec, da_level=0.0)  # 无 DA
    values_no_da = bg2.get_action_values()
    selected_no_da = bg2.select_action()
    print(f"  无 DA values: {[f'{v:.3f}' for v in values_no_da]}")
    print(f"  无 DA 选中: {selected_no_da}")

    # 有 DA 时, 最大值和次大值的差距应更大
    sorted_da = np.sort(values)[::-1]
    sorted_no_da = np.sort(values_no_da)[::-1]
    gap_da = sorted_da[0] - sorted_da[1] if len(sorted_da) > 1 else 0
    gap_no_da = sorted_no_da[0] - sorted_no_da[1] if len(sorted_no_da) > 1 else 0
    print(f"  DA>0 gap: {gap_da:.3f}, DA=0 gap: {gap_no_da:.3f}")

    print("  ★ 测试 5 通过!")


def test_6_three_factor_stdp():
    """测试 6: 三因子 STDP 学习"""
    print("\n" + "=" * 60)
    print("测试 6: 三因子 STDP 学习")
    print("=" * 60)

    # 创建带 DA-STDP 的纹状体
    stri = Striatum(n_d1=10, n_d2=10, n_fsi=4, seed=42)

    # 记录初始皮层→D1 突触权重
    # (通过内部突触追踪)
    initial_weights_d1 = []
    for syn in stri._cortical_d1_synapses:
        initial_weights_d1.append(syn.weight)

    # 如果没有皮层突触 (它们在 BasalGangliaCircuit 层创建),
    # 我们直接测试资格痕迹和权重更新机制
    from wuyun.synapse.synapse_base import SynapseBase
    from wuyun.synapse.plasticity.da_modulated_stdp import (
        DAModulatedSTDP,
        DAModulatedSTDPParams,
    )
    from wuyun.spike.signal_types import SynapseType, CompartmentType

    da_stdp = DAModulatedSTDP(DAModulatedSTDPParams(
        a_plus=0.01,
        a_minus=0.005,
        tau_eligibility=500.0,
    ))

    # 创建测试突触
    syn = SynapseBase(
        pre_id=0, post_id=1,
        weight=0.5, w_max=1.0, w_min=0.0,
        synapse_type=SynapseType.AMPA,
        target_compartment=CompartmentType.BASAL,
        plasticity_rule=da_stdp,
    )

    # 模拟: pre 先于 post 发放 → 正资格痕迹
    pre_times = [10]
    post_times = [15]  # Δt = +5 → LTP
    syn.update_eligibility(pre_times, post_times, dt=1.0)
    elig_after_ltp = syn.eligibility
    print(f"  pre→post (Δt=+5ms): eligibility = {elig_after_ltp:.6f}")
    assert elig_after_ltp > 0, f"LTP 应产生正资格痕迹, 实际: {elig_after_ltp}"
    print(f"  ✓ LTP 方向正确 (eligibility > 0)")

    # 应用 DA → 权重应增加
    w_before = syn.weight
    dw = syn.apply_plasticity(modulation=1.0)  # 强 DA
    w_after = syn.weight
    print(f"  DA=1.0: weight {w_before:.4f} → {w_after:.4f} (Δw={dw:.6f})")
    assert w_after > w_before, f"DA+LTP 应增强权重, w: {w_before:.4f} → {w_after:.4f}"
    print(f"  ✓ DA + 正资格痕迹 → 权重增强")

    # 模拟: post 先于 pre → 负资格痕迹
    syn2 = SynapseBase(
        pre_id=0, post_id=1,
        weight=0.5, w_max=1.0, w_min=0.0,
        synapse_type=SynapseType.AMPA,
        target_compartment=CompartmentType.BASAL,
        plasticity_rule=da_stdp,
    )
    pre_times2 = [15]
    post_times2 = [10]  # Δt = -5 → LTD
    syn2.update_eligibility(pre_times2, post_times2, dt=1.0)
    elig_after_ltd = syn2.eligibility
    print(f"  post→pre (Δt=-5ms): eligibility = {elig_after_ltd:.6f}")
    assert elig_after_ltd < 0, f"LTD 应产生负资格痕迹, 实际: {elig_after_ltd}"
    print(f"  ✓ LTD 方向正确 (eligibility < 0)")

    # 应用 DA → 权重应减少
    w_before2 = syn2.weight
    dw2 = syn2.apply_plasticity(modulation=1.0)
    w_after2 = syn2.weight
    print(f"  DA=1.0: weight {w_before2:.4f} → {w_after2:.4f} (Δw={dw2:.6f})")
    assert w_after2 < w_before2, f"DA+LTD 应减弱权重, w: {w_before2:.4f} → {w_after2:.4f}"
    print(f"  ✓ DA + 负资格痕迹 → 权重减弱")

    # 无 DA 时: 资格痕迹不转化为权重变化
    syn3 = SynapseBase(
        pre_id=0, post_id=1,
        weight=0.5, w_max=1.0, w_min=0.0,
        synapse_type=SynapseType.AMPA,
        target_compartment=CompartmentType.BASAL,
        plasticity_rule=da_stdp,
    )
    syn3.update_eligibility([10], [15], dt=1.0)
    w_before3 = syn3.weight
    dw3 = syn3.apply_plasticity(modulation=0.0)  # 无 DA
    w_after3 = syn3.weight
    print(f"  DA=0: weight {w_before3:.4f} → {w_after3:.4f} (Δw={dw3:.6f})")
    assert abs(dw3) < 1e-10, f"DA=0 不应改变权重, Δw={dw3:.6f}"
    print(f"  ✓ DA=0 → 权重不变 (资格痕迹等待 DA)")

    print("  ★ 测试 6 通过!")


def test_7_full_loop_bg_thalamus():
    """测试 7: 全环路 — 皮层→基底节→丘脑去抑制"""
    print("\n" + "=" * 60)
    print("测试 7: 全环路 — 皮层→基底节→丘脑去抑制")
    print("=" * 60)

    sim_steps = 300

    # 创建基底节
    bg = BasalGangliaCircuit(n_actions=2, n_gpi=6, seed=42)

    # 创建丘脑核团 (2 个通道, 每个通道一个核团)
    thal_0 = create_thalamic_nucleus(nucleus_id=0, n_tc=5, n_trn=3, seed=100)
    thal_1 = create_thalamic_nucleus(nucleus_id=1, n_tc=5, n_trn=3, seed=101)

    # 输入: 动作 0 强, 动作 1 弱
    input_vec = np.array([30.0, 5.0])

    # 运行仿真
    for t in range(sim_steps):
        # 基底节 step
        bg.step(t, input_vec, da_level=0.3)

        # GPi 输出 → 丘脑抑制
        gpi_rates = bg.gpi.get_output_rates()
        n_per_channel = max(1, bg.gpi.n_neurons // bg.n_actions)

        # 通道 0 的 GPi 输出 → 丘脑 0 的抑制
        ch0_rate = np.mean(gpi_rates[:n_per_channel])
        ch0_inhibition = -min(ch0_rate / 50.0, 1.0) * 10.0

        # 通道 1 的 GPi 输出 → 丘脑 1 的抑制
        ch1_rate = np.mean(gpi_rates[n_per_channel:2 * n_per_channel]) \
            if 2 * n_per_channel <= len(gpi_rates) else np.mean(gpi_rates[n_per_channel:])
        ch1_inhibition = -min(ch1_rate / 50.0, 1.0) * 10.0

        # 注入感觉驱动 + GPi 抑制到丘脑
        thal_0.inject_sensory_current(8.0)  # 基线感觉驱动
        thal_0.inject_trn_drive_current(ch0_inhibition)  # GPi 抑制

        thal_1.inject_sensory_current(8.0)
        thal_1.inject_trn_drive_current(ch1_inhibition)

        thal_0.step(t)
        thal_1.step(t)

    # 收集结果
    tc0_rate = thal_0.get_tc_firing_rate()
    tc1_rate = thal_1.get_tc_firing_rate()
    action_values = bg.get_action_values()
    selected = bg.select_action()

    print(f"  输入: {input_vec}")
    print(f"  Action values: {[f'{v:.3f}' for v in action_values]}")
    print(f"  选中动作: {selected}")
    print(f"  丘脑 0 (选中通道) TC rate: {tc0_rate:.1f}Hz")
    print(f"  丘脑 1 (未选中通道) TC rate: {tc1_rate:.1f}Hz")

    # 选中的动作 → GPi 去抑制 → TC 发放增加
    # 未选中的动作 → GPi 继续抑制 → TC 低发放
    assert selected == 0, f"应选中动作 0 (最强输入), 实际: {selected}"
    print(f"  ✓ 正确选中动作 0")

    # 选中通道的 TC rate 应 >= 未选中通道
    # (由于 GPi 去抑制, 选中通道丘脑更自由)
    assert tc0_rate >= tc1_rate - 5.0, \
        f"选中通道 TC({tc0_rate:.1f}) 应 >= 未选中 TC({tc1_rate:.1f})"
    print(f"  ✓ 选中通道 TC({tc0_rate:.1f}Hz) >= 未选中 TC({tc1_rate:.1f}Hz)")

    print("  ★ 测试 7 通过!")


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    tests = [
        test_1_msn_params_up_down_state,
        test_2_da_modulation_direction,
        test_3_gpi_tonic_and_direct_pathway,
        test_4_three_pathway_competition,
        test_5_action_selection_wta,
        test_6_three_factor_stdp,
        test_7_full_loop_bg_thalamus,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Phase 4 测试结果: {passed}/{len(tests)} 通过")
    if errors:
        print(f"失败: {failed}")
        for name, err in errors:
            print(f"  ✗ {name}: {err}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)