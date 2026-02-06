"""
Phase 1-D: 预测编码自动化验证测试

Case 1: 全层级联 — L5 和 L6 发放率 > 0
Case 2: L6 预测反馈 → 后期 burst 比率 > 前期
Case 3: 新奇检测 → 刺激变化时 burst 比率变化
Case 4: 权重稳定性 — 无 NaN/Inf, 无溢出
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from wuyun.spike.signal_types import SpikeType
from wuyun.circuit.column_factory import create_sensory_column


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
    """反馈电流逐渐增加 → burst 比率应增加

    验证双区室核心机制: basal(前馈) + apical(反馈) 同时存在 → burst
    前期只有前馈, 后期加入反馈 → burst 比率增加
    """
    print_header("Case 2: 反馈电流 → burst 比率增加")

    col = create_sensory_column(
        column_id=1, n_per_layer=N_PER_LAYER,
        seed=SEED, ff_connection_strength=FF_STRENGTH,
    )

    ff_current = 50.0

    # 前期: 只有前馈 (0-200ms)
    early_regular = 0
    early_burst = 0
    for t in range(200):
        col.inject_feedforward_current(ff_current)
        col.step(t)
        pop = col.layers[23].exc_pop
        for i in np.nonzero(pop.fired)[0]:
            st = SpikeType(int(pop.spike_type[i]))
            if st == SpikeType.REGULAR:
                early_regular += 1
            elif st.is_burst:
                early_burst += 1

    # 后期: 前馈 + 反馈 (200-400ms)
    late_regular = 0
    late_burst = 0
    for t in range(200, 400):
        col.inject_feedforward_current(ff_current)
        col.inject_feedback_current(40.0)  # 模拟高层预测反馈
        col.step(t)
        pop = col.layers[23].exc_pop
        for i in np.nonzero(pop.fired)[0]:
            st = SpikeType(int(pop.spike_type[i]))
            if st == SpikeType.REGULAR:
                late_regular += 1
            elif st.is_burst:
                late_burst += 1

    early_total = early_regular + early_burst
    late_total = late_regular + late_burst
    early_ratio = early_burst / early_total if early_total > 0 else 0.0
    late_ratio = late_burst / late_total if late_total > 0 else 0.0

    print(f"  前期 (0-200ms, 纯前馈): reg={early_regular} bst={early_burst} ratio={early_ratio:.3f}")
    print(f"  后期 (200-400ms, +反馈): reg={late_regular} bst={late_burst} ratio={late_ratio:.3f}")

    assert late_ratio > early_ratio, \
        f"后期 burst 比率 ({late_ratio:.4f}) 应 > 前期 ({early_ratio:.4f})"

    print(f"\n  ✅ PASS: 后期 ({late_ratio:.4f}) > 前期 ({early_ratio:.4f})")
    return True


# =============================================================================
# Case 3: 新奇检测
# =============================================================================

def test_case_3_novelty_detection():
    """新奇检测: 有反馈(预测匹配)→burst多; 撤除反馈(新奇)→regular多

    预测编码原理:
      匹配期: basal(前馈) + apical(反馈) → burst = 预测匹配
      新奇期: 只有 basal, apical 撤除 → regular = 预测误差
    """
    print_header("Case 3: 新奇检测 (反馈撤除)")

    col = create_sensory_column(
        column_id=2, n_per_layer=N_PER_LAYER,
        seed=SEED, ff_connection_strength=FF_STRENGTH,
    )

    ff_current = 50.0
    fb_current = 40.0

    def count_l23_spikes(col):
        reg, bst = 0, 0
        pop = col.layers[23].exc_pop
        for i in np.nonzero(pop.fired)[0]:
            st = SpikeType(int(pop.spike_type[i]))
            if st == SpikeType.REGULAR:
                reg += 1
            elif st.is_burst:
                bst += 1
        return reg, bst

    # Phase A: 0-300ms, 前馈+反馈 (预测匹配 → burst 多)
    match_regular, match_burst = 0, 0
    for t in range(300):
        col.inject_feedforward_current(ff_current)
        col.inject_feedback_current(fb_current)
        col.step(t)
        if t >= 100:  # 跳过初始瞬态
            r, b = count_l23_spikes(col)
            match_regular += r
            match_burst += b

    # Phase B: 300-600ms, 只有前馈 (新奇 → regular 多)
    novel_regular, novel_burst = 0, 0
    for t in range(300, 600):
        col.inject_feedforward_current(ff_current)
        col.step(t)
        if t >= 400:  # 跳过过渡期
            r, b = count_l23_spikes(col)
            novel_regular += r
            novel_burst += b

    match_total = match_regular + match_burst
    novel_total = novel_regular + novel_burst
    match_ratio = match_burst / match_total if match_total > 0 else 0.0
    novel_ratio = novel_burst / novel_total if novel_total > 0 else 0.0

    print(f"  匹配期 (100-300ms, ff+fb): reg={match_regular} bst={match_burst} ratio={match_ratio:.3f}")
    print(f"  新奇期 (400-600ms, ff only): reg={novel_regular} bst={novel_burst} ratio={novel_ratio:.3f}")

    assert match_ratio > novel_ratio, \
        f"匹配期 burst 比率 ({match_ratio:.4f}) 应 > 新奇期 ({novel_ratio:.4f})"

    print(f"\n  ✅ PASS: 匹配期 ({match_ratio:.4f}) > 新奇期 ({novel_ratio:.4f})")
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

    # 记录初始权重
    initial_weights = np.concatenate([sg.weights.copy() for sg in col.synapse_groups])

    for t in range(duration):
        col.inject_feedforward_current(ff_current)
        col.step(t)

    has_nan = False
    has_overflow = False
    for sg in col.synapse_groups:
        w = sg.weights
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            has_nan = True
        if np.any(w < sg.w_min - 0.001) or np.any(w > sg.w_max + 0.001):
            has_overflow = True

    final_weights = np.concatenate([sg.weights for sg in col.synapse_groups])
    delta = final_weights - initial_weights
    changed = int(np.sum(np.abs(delta) > 1e-6))
    print(f"\n  权重变化: {changed}/{len(final_weights)} 个权重有变化")
    print(f"  权重范围: [{final_weights.min():.4f}, {final_weights.max():.4f}]")

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