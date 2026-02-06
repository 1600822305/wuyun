"""
Phase 1-D: 预测编码自动化验证测试

Case 1: 全层级联 — L5 和 L6 发放率 > 0
Case 2: L6 预测反馈 → 后期 burst 比率 > 前期
Case 3: 新奇检测 → 刺激变化时 burst 比率变化
Case 4: 权重稳定性 — 无 NaN/Inf, 无溢出
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from wuyun.spike.signal_types import SpikeType
from wuyun.circuit.column_factory import create_sensory_column
from experiments.utils import snapshot_weights


# =============================================================================
# 通用参数
# =============================================================================

N_PER_LAYER = 30
FF_STRENGTH = 1.5
SEED = 42


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# =============================================================================
# Case 1: 全层级联激活
# =============================================================================

def test_case_1_full_cascade():
    """n=30, 300ms, I=50 → L5 和 L6 发放率 > 0"""
    print_header("Case 1: 全层级联激活")

    col = create_sensory_column(
        column_id=0, n_per_layer=N_PER_LAYER,
        seed=SEED, ff_connection_strength=FF_STRENGTH,
    )
    print(f"  柱: {col}")

    duration = 300
    ff_current = 50.0  # 足够强驱动全链路

    for t in range(duration):
        col.inject_feedforward_current(ff_current)
        col.step(t)

    rates = col.get_layer_firing_rates()
    for lid in [4, 23, 5, 6]:
        print(f"  L{lid}: {rates.get(lid, 0):.1f} Hz")

    l5_rate = rates.get(5, 0)
    l6_rate = rates.get(6, 0)

    assert l5_rate > 0, f"L5 发放率应 > 0, 得到 {l5_rate:.1f} Hz"
    assert l6_rate > 0, f"L6 发放率应 > 0, 得到 {l6_rate:.1f} Hz"

    print(f"\n  ✅ PASS: L5={l5_rate:.1f} Hz, L6={l6_rate:.1f} Hz 均 > 0")
    return True


# =============================================================================
# Case 2: L6 预测反馈 → burst 增加
# =============================================================================

def test_case_2_l6_prediction_feedback():
    """n=30, 600ms, I=50 → 后期 burst 比率 > 前期

    用较强输入 (I=50) 确保 L6 有足够活动产生 NMDA 反馈,
    使 apical 能积累到 Ca²⁺ 阈值。
    """
    print_header("Case 2: L6 预测反馈 → burst 比率增加")

    col = create_sensory_column(
        column_id=1, n_per_layer=N_PER_LAYER,
        seed=SEED, ff_connection_strength=FF_STRENGTH,
    )

    duration = 600
    ff_current = 50.0

    window_size = 100
    windows = []
    win_regular = 0
    win_burst = 0

    for t in range(duration):
        col.inject_feedforward_current(ff_current)
        col.step(t)

        for neuron in col.layers[23].excitatory:
            st = neuron.current_spike_type
            if st == SpikeType.REGULAR:
                win_regular += 1
            elif st.is_burst:
                win_burst += 1

        if (t + 1) % window_size == 0:
            total = win_regular + win_burst
            ratio = win_burst / total if total > 0 else 0.0
            windows.append({
                'start': t + 1 - window_size,
                'end': t + 1,
                'regular': win_regular,
                'burst': win_burst,
                'ratio': ratio,
            })
            win_regular = 0
            win_burst = 0

    for w in windows:
        print(f"  {w['start']:>4}-{w['end']:<4}ms: "
              f"reg={w['regular']:>4} bst={w['burst']:>4} "
              f"ratio={w['ratio']:.3f}")

    early = windows[:2]  # 0-200ms
    late = windows[-2:]   # 400-600ms

    def avg_ratio(ws):
        total_r = sum(w['regular'] for w in ws)
        total_b = sum(w['burst'] for w in ws)
        total = total_r + total_b
        return total_b / total if total > 0 else 0.0

    early_ratio = avg_ratio(early)
    late_ratio = avg_ratio(late)

    print(f"\n  前期 (0-200ms) burst 比率: {early_ratio:.4f}")
    print(f"  后期 (400-600ms) burst 比率: {late_ratio:.4f}")

    assert late_ratio > early_ratio, \
        f"后期 burst 比率 ({late_ratio:.4f}) 应 > 前期 ({early_ratio:.4f})"

    print(f"\n  ✅ PASS: 后期 ({late_ratio:.4f}) > 前期 ({early_ratio:.4f})")
    return True


# =============================================================================
# Case 3: 新奇检测
# =============================================================================

def test_case_3_novelty_detection():
    """n=30, 600ms, I=50→100 → 输入突然增强 → burst 比率骤降

    预测编码原理:
      适应期: L6 学会预测 I=50 的活动模式 → apical 反馈匹配 → burst 比率高
      新奇期: I 突然翻倍至 100, 大量新神经元被强 basal 驱动发放,
              但 L6 预测还是旧模式 → 无匹配 apical → regular spike 飙升
              → burst 比率骤降 = "预测误差洪流" = 新奇检测

    这是正确的生物学行为: 意外强刺激 = 惊讶 = 预测误差。
    """
    print_header("Case 3: 新奇检测 (输入增强)")

    col = create_sensory_column(
        column_id=2, n_per_layer=N_PER_LAYER,
        seed=SEED, ff_connection_strength=FF_STRENGTH,
    )

    window_size = 100
    windows = []
    win_regular = 0
    win_burst = 0

    def record_window(start, end, phase):
        nonlocal win_regular, win_burst
        total = win_regular + win_burst
        ratio = win_burst / total if total > 0 else 0.0
        windows.append({
            'start': start,
            'end': end,
            'regular': win_regular,
            'burst': win_burst,
            'ratio': ratio,
            'phase': phase,
        })
        win_regular = 0
        win_burst = 0

    def collect_l23():
        nonlocal win_regular, win_burst
        for neuron in col.layers[23].excitatory:
            st = neuron.current_spike_type
            if st == SpikeType.REGULAR:
                win_regular += 1
            elif st.is_burst:
                win_burst += 1

    # Phase A: 0-300ms, I=50 (建立预测, L6 反馈逐渐生效)
    for t in range(0, 300):
        col.inject_feedforward_current(50.0)
        col.step(t)
        collect_l23()
        if (t + 1) % window_size == 0:
            record_window(t + 1 - window_size, t + 1, 'A')

    # Phase B: 300-500ms, I=100 (新奇强刺激, 大量新 regular spike)
    for t in range(300, 500):
        col.inject_feedforward_current(100.0)
        col.step(t)
        collect_l23()
        if (t + 1) % window_size == 0:
            record_window(t + 1 - window_size, t + 1, 'B')

    # Phase C: 500-600ms, I=100 (持续, 系统开始重新适应)
    for t in range(500, 600):
        col.inject_feedforward_current(100.0)
        col.step(t)
        collect_l23()
        if (t + 1) % window_size == 0:
            record_window(t + 1 - window_size, t + 1, 'C')

    for w in windows:
        print(f"  {w['start']:>4}-{w['end']:<4}ms [{w['phase']}]: "
              f"reg={w['regular']:>4} bst={w['burst']:>4} "
              f"ratio={w['ratio']:.3f}")

    # 适应期末尾 (Phase A 最后一个窗口, L6 反馈已建立)
    phase_a_stable = [w for w in windows
                      if w['phase'] == 'A' and w['start'] >= 100]
    # 新奇期首窗 (Phase B 第一个窗口, L6 预测最失配)
    phase_b_first = [w for w in windows if w['phase'] == 'B'][:1]

    adapt_ratio = (sum(w['burst'] for w in phase_a_stable)
                   / max(sum(w['regular'] + w['burst'] for w in phase_a_stable), 1))
    novel_ratio = (sum(w['burst'] for w in phase_b_first)
                   / max(sum(w['regular'] + w['burst'] for w in phase_b_first), 1))

    print(f"\n  适应期 (100-300ms, I=50) burst 比率: {adapt_ratio:.4f}")
    print(f"  新奇期首窗 (300-400ms, I=100) burst 比率: {novel_ratio:.4f}")

    # 通过条件: 新奇期 burst 比率 < 适应期 (强输入驱动大量 regular = 预测误差)
    assert novel_ratio < adapt_ratio, \
        f"新奇期 burst 比率 ({novel_ratio:.4f}) 应 < 适应期 ({adapt_ratio:.4f})"

    print(f"\n  ✅ PASS: 新奇期 ({novel_ratio:.4f}) < 适应期 ({adapt_ratio:.4f})")
    return True


# =============================================================================
# Case 4: 权重稳定性
# =============================================================================

def test_case_4_weight_stability():
    """n=30, 500ms → 所有权重 ∈ [w_min, w_max], 无 NaN/Inf"""
    print_header("Case 4: STDP 权重稳定性")

    col = create_sensory_column(
        column_id=3, n_per_layer=N_PER_LAYER,
        seed=SEED, ff_connection_strength=FF_STRENGTH,
    )
    print(f"  柱: {col}")

    duration = 500
    ff_current = 50.0

    snap_initial = snapshot_weights(col)

    for t in range(duration):
        col.inject_feedforward_current(ff_current)
        col.step(t)

        # STDP 每 10ms 更新一次 (减少计算量)
        if t % 10 == 0:
            for syn in col.synapses:
                pre_neuron = col.get_neuron(syn.pre_id)
                post_neuron = col.get_neuron(syn.post_id)
                if pre_neuron and post_neuron:
                    pre_times = pre_neuron.spike_train.get_recent_times(window_ms=50)
                    post_times = post_neuron.spike_train.get_recent_times(window_ms=50)
                    if pre_times and post_times:
                        syn.update_weight_stdp(pre_times, post_times)

    has_nan = False
    has_overflow = False
    for syn in col.synapses:
        w = syn.weight
        if np.isnan(w) or np.isinf(w):
            has_nan = True
        if w < syn.w_min - 0.001 or w > syn.w_max + 0.001:
            has_overflow = True

    snap_final = snapshot_weights(col)

    print(f"\n  权重变化:")
    for cat in ['ff_l4_l23', 'ff_l23_l5', 'ff_l5_l6',
                'fb_l6_l23', 'fb_l6_l5', 'inh_pv']:
        i_mean = snap_initial.get(cat, {}).get('mean', 0)
        f_mean = snap_final.get(cat, {}).get('mean', 0)
        count = snap_final.get(cat, {}).get('count', 0)
        delta = f_mean - i_mean
        direction = '↑' if delta > 0.001 else ('↓' if delta < -0.001 else '→')
        print(f"    {cat:<15}: {i_mean:.4f} → {f_mean:.4f} "
              f"({delta:+.4f}) {direction} [{count} synapses]")

    assert not has_nan, "不应有 NaN/Inf 权重"
    assert not has_overflow, "权重不应溢出 [w_min, w_max]"

    print(f"\n  ✅ PASS: 所有 {col.n_synapses} 个权重稳定, "
          f"无 NaN/Inf, 无溢出")
    return True


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  悟韵 (WuYun) Phase 1-D: 预测编码验证测试                     ║")
    print("║  证明皮层柱能自发形成预测编码                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    results = {}
    tests = [
        ("Case 1: 全层级联", test_case_1_full_cascade),
        ("Case 2: L6预测反馈", test_case_2_l6_prediction_feedback),
        ("Case 3: 新奇检测", test_case_3_novelty_detection),
        ("Case 4: 权重稳定性", test_case_4_weight_stability),
    ]

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "PASS" if passed else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()

    print_header("总结")
    all_pass = True
    for name, result in results.items():
        icon = "✅" if result == "PASS" else "❌"
        if result != "PASS":
            all_pass = False
        print(f"  {icon} {result}: {name}")

    print()
    if all_pass:
        print("🎉 所有测试通过! Phase 1-D 预测编码验证完毕。")
    else:
        print("❌ 存在失败的测试，请检查。")
        sys.exit(1)