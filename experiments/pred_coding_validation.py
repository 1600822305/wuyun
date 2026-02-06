"""
Phase 1-D: 预测编码验证实验

4 个实验验证皮层柱能自发形成预测编码:
  实验 1: 全层级联激活 (L4→L23→L5→L6)
  实验 2: L6 预测反馈环路 → burst 比率随时间上升
  实验 3: 新奇刺激 → 预测误差飙升 (burst 骤降)
  实验 4: STDP 权重演化观察

关键约束:
  - 不修改神经元动力学参数
  - 不修改可塑性规则
  - 只调连接拓扑/权重/网络规模
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List, Tuple

from wuyun.spike.signal_types import SpikeType
from wuyun.circuit.column_factory import create_sensory_column
from wuyun.circuit.cortical_column import CorticalColumn
from experiments.utils import (
    collect_window_stats,
    snapshot_weights,
    print_header,
    print_window_table,
    print_weight_table,
)


# =============================================================================
# 实验 1: 全层级联激活
# =============================================================================

def experiment_1_full_cascade(
    n_per_layer: int = 50,
    duration: int = 500,
    ff_current: float = 30.0,
    seed: int = 42,
    ff_strength: float = 1.0,
) -> Dict:
    """验证 L4→L23→L5→L6 全链路连通

    创建 n=50 的皮层柱, 给 L4 持续前馈输入, 运行 500ms。

    预期结果:
    - L4:  高发放率 (直接接收输入)
    - L23: 中等发放率 (L4→L23 前馈)
    - L5:  有发放 (L23→L5 前馈) ← 关键!
    - L6:  有发放 (L5→L6 前馈) ← 关键!

    Returns:
        {
            'layer_rates': {layer_id: float},       各层发放率 (Hz)
            'layer_spike_counts': {layer_id: int},  各层总脉冲数
            'layer_active_neurons': {layer_id: int}, 各层活跃神经元数
            'windows': list,                         每 50ms 窗口统计
            'passed': bool,
        }
    """
    print_header("实验 1: 全层级联激活 (L4→L23→L5→L6)")
    print(f"  参数: n={n_per_layer}, duration={duration}ms, I_ff={ff_current}, "
          f"ff_strength={ff_strength}, seed={seed}")

    col = create_sensory_column(
        column_id=0, n_per_layer=n_per_layer,
        seed=seed, ff_connection_strength=ff_strength,
    )
    print(f"  柱: {col}")

    # 运行仿真，按 50ms 窗口统计
    windows = collect_window_stats(
        col, duration, ff_current,
        window_size=50, start_time=0,
    )

    # 打印窗口统计
    print_window_table(windows)

    # 各层最终发放率
    rates = col.get_layer_firing_rates()
    print(f"\n  最终各层平均发放率:")
    for lid in [4, 23, 5, 6]:
        print(f"    L{lid}: {rates.get(lid, 0):.1f} Hz")

    # 各层活跃神经元数
    active = {}
    spike_counts = {4: 0, 23: 0, 5: 0, 6: 0}
    for lid, layer in col.layers.items():
        count = 0
        for n in layer.neurons:
            if n.spike_train.count > 0:
                count += 1
        active[lid] = count

    # 从窗口累计总脉冲
    for w in windows:
        for lid in [4, 23, 5, 6]:
            spike_counts[lid] += w['layer_spikes'][lid]

    print(f"\n  各层活跃神经元数:")
    for lid in [4, 23, 5, 6]:
        print(f"    L{lid}: {active[lid]}/{col.layers[lid].n_total} neurons, "
              f"{spike_counts[lid]} total spikes")

    # 通过条件
    l5_rate = rates.get(5, 0)
    l6_rate = rates.get(6, 0)
    passed = l5_rate > 0 and l6_rate > 0

    if passed:
        print(f"\n  ✅ PASS: L5 ({l5_rate:.1f} Hz) 和 L6 ({l6_rate:.1f} Hz) 均有发放")
    else:
        print(f"\n  ❌ FAIL: L5={l5_rate:.1f} Hz, L6={l6_rate:.1f} Hz (需 > 0)")

    return {
        'layer_rates': rates,
        'layer_spike_counts': spike_counts,
        'layer_active_neurons': active,
        'windows': windows,
        'passed': passed,
    }


# =============================================================================
# 实验 2: L6 预测反馈环路
# =============================================================================

def experiment_2_l6_prediction_loop(
    n_per_layer: int = 50,
    duration: int = 1000,
    ff_current: float = 30.0,
    seed: int = 42,
    ff_strength: float = 1.0,
) -> Dict:
    """验证 L6 反馈使 L23 burst 比率随时间上升

    步骤:
    1. 创建 n=50 皮层柱
    2. 给 L4 持续前馈输入, 运行 1000ms
    3. 统计每 100ms 窗口的 L23 burst 比率

    预期行为:
    - 前期 (0-200ms): L6 刚开始激活, 反馈弱 → burst 比率低
    - 后期 (500-1000ms): L6 反馈积累 → burst 比率上升

    Returns:
        {
            'windows': list,
            'early_burst_ratio': float,  前 200ms 平均 burst 比率
            'late_burst_ratio': float,   后 200ms 平均 burst 比率
            'passed': bool,
        }
    """
    print_header("实验 2: L6 预测反馈环路 → burst 比率变化")
    print(f"  参数: n={n_per_layer}, duration={duration}ms, I_ff={ff_current}, "
          f"ff_strength={ff_strength}, seed={seed}")

    col = create_sensory_column(
        column_id=1, n_per_layer=n_per_layer,
        seed=seed, ff_connection_strength=ff_strength,
    )
    print(f"  柱: {col}")

    # 运行仿真，按 100ms 窗口统计
    windows = collect_window_stats(
        col, duration, ff_current,
        window_size=100, start_time=0,
    )

    print_window_table(windows)

    # 前期 (前 2 个窗口 = 0-200ms) vs 后期 (后 2 个窗口)
    n_windows = len(windows)
    early_windows = windows[:2]  # 0-200ms
    late_windows = windows[-2:]  # 800-1000ms (or last 2)

    def avg_burst_ratio(ws):
        if not ws:
            return 0.0
        total_r = sum(w['l23_regular'] for w in ws)
        total_b = sum(w['l23_burst'] for w in ws)
        total = total_r + total_b
        return total_b / total if total > 0 else 0.0

    early_ratio = avg_burst_ratio(early_windows)
    late_ratio = avg_burst_ratio(late_windows)

    print(f"\n  前期 (0-200ms) L23 burst 比率: {early_ratio:.3f}")
    print(f"  后期 ({(n_windows-2)*100}-{n_windows*100}ms) L23 burst 比率: {late_ratio:.3f}")
    print(f"  变化: {late_ratio - early_ratio:+.3f}")

    # L6 发放趋势
    print(f"\n  L6 发放趋势:")
    for w in windows:
        bar = '█' * min(w['l6_spikes'], 50)
        print(f"    {w['window_start']:>4}-{w['window_end']:<4}ms: "
              f"{w['l6_spikes']:>4} spikes  {bar}")

    # 通过条件: 后期 burst 比率 > 前期
    passed = late_ratio > early_ratio
    if passed:
        print(f"\n  ✅ PASS: 后期 burst 比率 ({late_ratio:.3f}) > "
              f"前期 ({early_ratio:.3f})")
    else:
        print(f"\n  ❌ FAIL: 后期 burst 比率 ({late_ratio:.3f}) <= "
              f"前期 ({early_ratio:.3f})")

    return {
        'windows': windows,
        'early_burst_ratio': early_ratio,
        'late_burst_ratio': late_ratio,
        'passed': passed,
    }


# =============================================================================
# 实验 3: 新奇刺激 → 预测误差飙升
# =============================================================================

def experiment_3_novelty_detection(
    n_per_layer: int = 50,
    seed: int = 42,
    ff_strength: float = 1.0,
) -> Dict:
    """改变刺激模式后, burst 比率骤降 (预测失败)

    步骤:
    1. Phase A (适应期, 0-500ms): I_ff=30 → 系统适应 → burst 比率上升
    2. Phase B (新奇期, 500-700ms): I_ff=60 → 预测失效 → burst 骤降
    3. Phase C (再适应期, 700-1000ms): I_ff=60 → 重新适应 → burst 回升

    Returns:
        {
            'windows': list,
            'phase_a_burst_ratio': float,   适应期末尾
            'phase_b_burst_ratio': float,   新奇期
            'phase_c_burst_ratio': float,   再适应期末尾
            'passed': bool,
        }
    """
    print_header("实验 3: 新奇检测 → 预测误差飙升")
    print(f"  参数: n={n_per_layer}, Phase A: I=30 (0-500ms), "
          f"Phase B: I=60 (500-700ms), Phase C: I=60 (700-1000ms)")

    col = create_sensory_column(
        column_id=2, n_per_layer=n_per_layer,
        seed=seed, ff_connection_strength=ff_strength,
    )
    print(f"  柱: {col}")

    # 手动分阶段运行
    all_windows = []
    window_size = 100

    # Phase A: 适应期 0-500ms, I=30
    current_window = _new_manual_window(0, window_size)
    for t in range(0, 500):
        col.inject_feedforward_current(30.0)
        col.step(t)
        _accumulate_manual(col, current_window)
        if (t + 1) % window_size == 0 and t > 0:
            _finalize_manual(current_window, 'A')
            all_windows.append(current_window)
            current_window = _new_manual_window(t + 1, window_size)

    # Phase B: 新奇期 500-700ms, I=60
    for t in range(500, 700):
        col.inject_feedforward_current(60.0)
        col.step(t)
        _accumulate_manual(col, current_window)
        if (t + 1) % window_size == 0:
            _finalize_manual(current_window, 'B')
            all_windows.append(current_window)
            current_window = _new_manual_window(t + 1, window_size)

    # Phase C: 再适应期 700-1000ms, I=60
    for t in range(700, 1000):
        col.inject_feedforward_current(60.0)
        col.step(t)
        _accumulate_manual(col, current_window)
        if (t + 1) % window_size == 0:
            _finalize_manual(current_window, 'C')
            all_windows.append(current_window)
            current_window = _new_manual_window(t + 1, window_size)

    # 打印结果
    print(f"\n  {'窗口(ms)':<15} {'Phase':>6} {'L23 reg':>8} {'L23 bst':>8} "
          f"{'burst%':>8} {'L6 spk':>8}")
    print(f"  {'-' * 60}")
    for w in all_windows:
        print(f"  {w['window_start']:>4}-{w['window_end']:<8} "
              f"{w['phase']:>6} "
              f"{w['l23_regular']:>8} {w['l23_burst']:>8} "
              f"{w['l23_burst_ratio']:>7.1%} "
              f"{w['l6_spikes']:>8}")

    # 计算各阶段 burst 比率
    def phase_ratio(phase_label):
        ws = [w for w in all_windows if w['phase'] == phase_label]
        if not ws:
            return 0.0
        total_r = sum(w['l23_regular'] for w in ws)
        total_b = sum(w['l23_burst'] for w in ws)
        total = total_r + total_b
        return total_b / total if total > 0 else 0.0

    # 适应期末尾: Phase A 后半段 (300-500ms 的窗口)
    phase_a_late = [w for w in all_windows
                    if w['phase'] == 'A' and w['window_start'] >= 300]
    phase_a_ratio = _windows_burst_ratio(phase_a_late)

    # 新奇期: Phase B (500-700ms)
    phase_b_ratio = phase_ratio('B')

    # 再适应期: Phase C 后半段 (800-1000ms)
    phase_c_late = [w for w in all_windows
                    if w['phase'] == 'C' and w['window_start'] >= 800]
    phase_c_ratio = _windows_burst_ratio(phase_c_late)

    print(f"\n  Phase A 末尾 (300-500ms) burst 比率: {phase_a_ratio:.3f}")
    print(f"  Phase B 新奇 (500-700ms) burst 比率: {phase_b_ratio:.3f}")
    print(f"  Phase C 末尾 (800-1000ms) burst 比率: {phase_c_ratio:.3f}")

    # 通过条件: Phase B burst 比率 < Phase A 末尾
    passed = phase_b_ratio < phase_a_ratio
    if passed:
        print(f"\n  ✅ PASS: 新奇期 burst ({phase_b_ratio:.3f}) < "
              f"适应期 ({phase_a_ratio:.3f}) → 预测误差增加")
    else:
        print(f"\n  ❌ FAIL: 新奇期 burst ({phase_b_ratio:.3f}) >= "
              f"适应期 ({phase_a_ratio:.3f})")

    return {
        'windows': all_windows,
        'phase_a_burst_ratio': phase_a_ratio,
        'phase_b_burst_ratio': phase_b_ratio,
        'phase_c_burst_ratio': phase_c_ratio,
        'passed': passed,
    }


def _new_manual_window(start, size):
    return {
        'window_start': start,
        'window_end': start + size,
        'l23_regular': 0,
        'l23_burst': 0,
        'l23_burst_ratio': 0.0,
        'l5_regular': 0,
        'l5_burst': 0,
        'l6_spikes': 0,
        'layer_spikes': {4: 0, 23: 0, 5: 0, 6: 0},
        'phase': '',
    }


def _accumulate_manual(column, window):
    for neuron in column.layers[23].excitatory:
        st = neuron.current_spike_type
        if st == SpikeType.REGULAR:
            window['l23_regular'] += 1
        elif st.is_burst:
            window['l23_burst'] += 1

    if 5 in column.layers:
        for neuron in column.layers[5].excitatory:
            st = neuron.current_spike_type
            if st == SpikeType.REGULAR:
                window['l5_regular'] += 1
            elif st.is_burst:
                window['l5_burst'] += 1

    if 6 in column.layers:
        for neuron in column.layers[6].excitatory:
            st = neuron.current_spike_type
            if st.is_active:
                window['l6_spikes'] += 1

    for lid, layer in column.layers.items():
        spikes = layer.get_last_spikes()
        window['layer_spikes'][lid] += len(spikes)


def _finalize_manual(window, phase):
    window['phase'] = phase
    total = window['l23_regular'] + window['l23_burst']
    window['l23_burst_ratio'] = window['l23_burst'] / total if total > 0 else 0.0


def _windows_burst_ratio(ws):
    if not ws:
        return 0.0
    total_r = sum(w['l23_regular'] for w in ws)
    total_b = sum(w['l23_burst'] for w in ws)
    total = total_r + total_b
    return total_b / total if total > 0 else 0.0


# =============================================================================
# 实验 4: STDP 权重演化
# =============================================================================

def experiment_4_stdp_weight_evolution(
    n_per_layer: int = 50,
    duration: int = 2000,
    ff_current: float = 30.0,
    seed: int = 42,
    ff_strength: float = 1.0,
) -> Dict:
    """观察预测编码过程中权重的变化方向

    步骤:
    1. 创建 n=50 皮层柱
    2. 记录初始权重快照
    3. 运行 2000ms 持续刺激
    4. 每 500ms 记录权重快照
    5. 比较权重变化

    预期:
    - L4→L23 basal 权重: 活跃连接增强 (classical STDP LTP)
    - L6→L23 apical 权重: 预测匹配的连接增强
    - PV+→soma 权重: 随活动调整
    - 所有权重在 [w_min, w_max] 内
    - 无 NaN/Inf

    Returns:
        {
            'snapshots': [(time, snapshot), ...],
            'has_nan': bool,
            'has_overflow': bool,
            'weight_stable': bool,
            'passed': bool,
        }
    """
    print_header("实验 4: STDP 权重演化")
    print(f"  参数: n={n_per_layer}, duration={duration}ms, I_ff={ff_current}, "
          f"ff_strength={ff_strength}, seed={seed}")

    col = create_sensory_column(
        column_id=3, n_per_layer=n_per_layer,
        seed=seed, ff_connection_strength=ff_strength,
    )
    print(f"  柱: {col}")

    # 初始快照
    snapshots = []
    snap0 = snapshot_weights(col)
    snapshots.append((0, snap0))

    # 运行仿真，每 500ms 拍快照
    snapshot_interval = 500
    for t in range(duration):
        col.inject_feedforward_current(ff_current)
        col.step(t)

        # STDP 权重更新: 每个时间步对所有突触做 STDP
        # (经典 STDP 通过 pre/post spike times 更新)
        for syn in col.synapses:
            # 获取突触前后神经元
            pre_neuron = col.get_neuron(syn.pre_id)
            post_neuron = col.get_neuron(syn.post_id)
            if pre_neuron and post_neuron:
                pre_times = pre_neuron.spike_train.get_recent_times(window_ms=50)
                post_times = post_neuron.spike_train.get_recent_times(window_ms=50)
                if pre_times and post_times:
                    syn.update_weight_stdp(pre_times, post_times)

        # 快照
        if (t + 1) % snapshot_interval == 0:
            snap = snapshot_weights(col)
            snapshots.append((t + 1, snap))

    # 打印权重表
    print_weight_table(snapshots)

    # 检查权重稳定性
    has_nan = False
    has_overflow = False
    for syn in col.synapses:
        w = syn.weight
        if np.isnan(w) or np.isinf(w):
            has_nan = True
        if w < syn.w_min - 0.001 or w > syn.w_max + 0.001:
            has_overflow = True

    # 检查权重变化方向
    initial = snapshots[0][1]
    final = snapshots[-1][1]

    print(f"\n  权重变化摘要:")
    for cat in ['ff_l4_l23', 'ff_l23_l5', 'fb_l6_l23', 'inh_pv']:
        i_mean = initial.get(cat, {}).get('mean', 0)
        f_mean = final.get(cat, {}).get('mean', 0)
        delta = f_mean - i_mean
        direction = '↑' if delta > 0.001 else ('↓' if delta < -0.001 else '→')
        print(f"    {cat:<15}: {i_mean:.4f} → {f_mean:.4f} ({delta:+.4f}) {direction}")

    weight_stable = not has_nan and not has_overflow

    print(f"\n  NaN/Inf: {'有 ❌' if has_nan else '无 ✅'}")
    print(f"  溢出 [w_min,w_max]: {'有 ❌' if has_overflow else '无 ✅'}")

    passed = weight_stable
    if passed:
        print(f"\n  ✅ PASS: 权重稳定, 无 NaN/Inf, 无溢出")
    else:
        print(f"\n  ❌ FAIL: 权重不稳定")

    return {
        'snapshots': snapshots,
        'has_nan': has_nan,
        'has_overflow': has_overflow,
        'weight_stable': weight_stable,
        'passed': passed,
    }


# =============================================================================
# 主程序: 运行所有实验
# =============================================================================

def run_all_experiments(
    n_per_layer: int = 50,
    ff_strength: float = 1.0,
    seed: int = 42,
):
    """运行全部 4 个实验"""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  悟韵 (WuYun) Phase 1-D: 预测编码验证实验                     ║")
    print("║  验证皮层柱能自发形成预测编码                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    results = {}

    # 实验 1
    r1 = experiment_1_full_cascade(
        n_per_layer=n_per_layer, seed=seed, ff_strength=ff_strength,
    )
    results['实验1: 全层级联'] = r1['passed']

    # 实验 2
    r2 = experiment_2_l6_prediction_loop(
        n_per_layer=n_per_layer, seed=seed, ff_strength=ff_strength,
    )
    results['实验2: L6预测环路'] = r2['passed']

    # 实验 3
    r3 = experiment_3_novelty_detection(
        n_per_layer=n_per_layer, seed=seed, ff_strength=ff_strength,
    )
    results['实验3: 新奇检测'] = r3['passed']

    # 实验 4
    r4 = experiment_4_stdp_weight_evolution(
        n_per_layer=n_per_layer, seed=seed, ff_strength=ff_strength,
    )
    results['实验4: STDP权重'] = r4['passed']

    # 总结
    print_header("总结")
    all_pass = True
    for name, passed in results.items():
        icon = "✅" if passed else "❌"
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {icon} {status}: {name}")

    print()
    if all_pass:
        print("🎉 所有实验通过! 预测编码验证完毕。")
    else:
        print("⚠️  部分实验未通过，可能需要调整 ff_strength 参数。")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='预测编码验证实验')
    parser.add_argument('--n', type=int, default=50, help='每层神经元数')
    parser.add_argument('--ff-strength', type=float, default=1.0,
                        help='前馈连接强度倍率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    run_all_experiments(
        n_per_layer=args.n,
        ff_strength=args.ff_strength,
        seed=args.seed,
    )