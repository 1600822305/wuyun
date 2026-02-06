"""
Phase 2.8 前置补充测试

测试 3 项 P3 前置依赖:
  1. 海马神经元参数预设 (GRANULE/PLACE_CELL/GRID_CELL)
  2. 短时程可塑性 STP (Tsodyks-Markram)
  3. Theta 振荡时钟 (OscillationClock)

不使用 pytest, 用 print 输出详细数值。
"""

import sys
import os
import math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ============================================================================
# 测试 1: 海马神经元参数预设
# ============================================================================

def test_hippocampal_neuron_params():
    """验证 GRANULE/PLACE_CELL/GRID_CELL 参数预设可用且属性正确"""
    print("\n" + "=" * 60)
    print("测试 1: 海马神经元参数预设")
    print("=" * 60)

    from wuyun.neuron.neuron_base import (
        NeuronBase, GRANULE_PARAMS, PLACE_CELL_PARAMS, GRID_CELL_PARAMS,
    )
    from wuyun.spike.signal_types import NeuronType

    # --- GRANULE ---
    g = NeuronBase(neuron_id=1000, params=GRANULE_PARAMS)
    print(f"  GRANULE: type={g.params.neuron_type.name}, "
          f"κ={g.params.kappa}, threshold={g.params.somatic.v_threshold}mV")
    assert g.params.neuron_type == NeuronType.GRANULE
    assert g.params.kappa == 0.0, "颗粒细胞应为单区室 (κ=0)"
    assert g.apical is None, "颗粒细胞不应有顶端树突"
    assert g.params.somatic.v_threshold == -40.0, "颗粒细胞应有高阈值 (-40mV)"
    print(f"  ✓ GRANULE: 单区室, 高阈值 (-40mV)")

    # --- PLACE_CELL ---
    p = NeuronBase(neuron_id=1001, params=PLACE_CELL_PARAMS)
    print(f"  PLACE_CELL: type={p.params.neuron_type.name}, "
          f"κ={p.params.kappa}, burst_count={p.params.burst_spike_count}")
    assert p.params.neuron_type == NeuronType.PLACE_CELL
    assert p.params.kappa == 0.3, "位置细胞应有中等耦合 (κ=0.3)"
    assert p.apical is not None, "位置细胞应有顶端树突"
    assert p.params.burst_spike_count == 3
    print(f"  ✓ PLACE_CELL: 双区室 (κ=0.3), burst=3")

    # --- GRID_CELL ---
    gc = NeuronBase(neuron_id=1002, params=GRID_CELL_PARAMS)
    print(f"  GRID_CELL: type={gc.params.neuron_type.name}, "
          f"κ={gc.params.kappa}, a={gc.params.somatic.a}")
    assert gc.params.neuron_type == NeuronType.GRID_CELL
    assert gc.params.kappa == 0.2
    assert gc.apical is not None, "网格细胞应有顶端树突"
    assert gc.params.somatic.a == 0.03, "网格细胞需要较强亚阈适应 (振荡)"
    print(f"  ✓ GRID_CELL: 双区室 (κ=0.2), a=0.03 (振荡)")

    # --- 功能测试: GRANULE 稀疏激活 ---
    granule = NeuronBase(neuron_id=2000, params=GRANULE_PARAMS)
    place = NeuronBase(neuron_id=2001, params=PLACE_CELL_PARAMS)

    spikes_granule = 0
    spikes_place = 0
    current = 25.0  # 较强输入 (确保两种神经元都能发放)

    for t in range(500):
        granule.inject_basal_current(current)
        place.inject_basal_current(current)
        st_g = granule.step(t)
        st_p = place.step(t)
        if st_g.is_active:
            spikes_granule += 1
        if st_p.is_active:
            spikes_place += 1

    rate_g = spikes_granule / 0.5  # 500ms → Hz
    rate_p = spikes_place / 0.5
    print(f"\n  功能验证 (I=25.0, 500ms):")
    print(f"    GRANULE 发放率: {rate_g:.1f} Hz")
    print(f"    PLACE_CELL 发放率: {rate_p:.1f} Hz")

    # GRANULE 高阈值(-40mV) 应比 PLACE_CELL(-50mV) 更稀疏
    assert rate_g <= rate_p, (
        f"GRANULE ({rate_g:.1f}Hz) 应不高于 PLACE_CELL ({rate_p:.1f}Hz)"
    )
    print(f"  ✓ GRANULE ≤ PLACE_CELL: {rate_g:.1f} ≤ {rate_p:.1f}")

    print("  PASSED")


# ============================================================================
# 测试 2: 短时程可塑性 STP — 基础动力学
# ============================================================================

def test_stp_basic():
    """验证 STP 囊泡耗竭和易化的基本动力学"""
    print("\n" + "=" * 60)
    print("测试 2: STP 基础动力学")
    print("=" * 60)

    from wuyun.synapse.short_term_plasticity import (
        ShortTermPlasticity, STPParams,
        MOSSY_FIBER_STP, SCHAFFER_COLLATERAL_STP,
    )

    # --- 苔藓纤维 (强易化) ---
    stp_mf = ShortTermPlasticity(MOSSY_FIBER_STP)
    print(f"  苔藓纤维: {stp_mf}")

    assert stp_mf.n == 1.0, "初始囊泡池应满"
    assert stp_mf.p == MOSSY_FIBER_STP.p0
    initial_eff = stp_mf.get_efficacy()
    print(f"  初始效能: {initial_eff:.4f} (p0={MOSSY_FIBER_STP.p0} × n=1.0)")

    # 连续 10 个脉冲 (间隔 20ms)
    efficacies_mf = []
    for i in range(10):
        for _ in range(20):  # 20ms 间隔
            stp_mf.step()
        eff = stp_mf.on_spike()
        efficacies_mf.append(eff)
        print(f"  脉冲 {i+1}: eff={eff:.4f}, n={stp_mf.n:.4f}, p={stp_mf.p:.4f}")

    # ★ 苔藓纤维的核心特征: 配对脉冲易化 (PPF)
    # 第 2 个脉冲的效能应 >> 第 1 个 (释放概率已被易化)
    ppf_ratio = efficacies_mf[1] / efficacies_mf[0]
    peak_eff = max(efficacies_mf)
    peak_idx = efficacies_mf.index(peak_eff)
    print(f"\n  配对脉冲比 (PPF): {ppf_ratio:.1f}x")
    print(f"  峰值效能: {peak_eff:.4f} (脉冲 {peak_idx+1})")

    assert ppf_ratio > 1.5, (
        f"苔藓纤维 PPF 应 > 1.5x: 实际 {ppf_ratio:.2f}x"
    )
    assert peak_eff > efficacies_mf[0] * 2, (
        f"峰值效能应至少为首次的 2 倍: {peak_eff:.4f} vs {efficacies_mf[0]:.4f}"
    )
    print(f"  ✓ 苔藓纤维 PPF: {ppf_ratio:.1f}x (去极化器效应确认)")

    # --- Schaffer collateral (抑制主导) ---
    stp_sc = ShortTermPlasticity(SCHAFFER_COLLATERAL_STP)
    efficacies_sc = []
    for i in range(10):
        for _ in range(20):
            stp_sc.step()
        eff = stp_sc.on_spike()
        efficacies_sc.append(eff)

    print(f"\n  Schaffer collateral:")
    print(f"    首次效能: {efficacies_sc[0]:.4f}")
    print(f"    末次效能: {efficacies_sc[-1]:.4f}")

    assert efficacies_sc[-1] < efficacies_sc[0], (
        f"Schaffer 应抑制: 末次 {efficacies_sc[-1]:.4f} < 首次 {efficacies_sc[0]:.4f}"
    )
    print(f"  ✓ Schaffer 抑制: {efficacies_sc[0]:.4f} → {efficacies_sc[-1]:.4f}")

    print("  PASSED")


# ============================================================================
# 测试 3: STP — 恢复动力学
# ============================================================================

def test_stp_recovery():
    """验证 STP 在脉冲停止后能恢复到基线"""
    print("\n" + "=" * 60)
    print("测试 3: STP 恢复动力学")
    print("=" * 60)

    from wuyun.synapse.short_term_plasticity import (
        ShortTermPlasticity, MOSSY_FIBER_STP,
    )

    stp = ShortTermPlasticity(MOSSY_FIBER_STP)

    # 先发 5 个高频脉冲耗竭囊泡池
    for i in range(5):
        for _ in range(10):
            stp.step()
        stp.on_spike()

    n_after_burst = stp.n
    p_after_burst = stp.p
    print(f"  高频脉冲后: n={n_after_burst:.4f}, p={p_after_burst:.4f}")
    assert n_after_burst < 0.9, f"囊泡池应被显著耗竭: n={n_after_burst:.4f}"

    # 等待 2000ms 恢复
    for _ in range(2000):
        stp.step()

    n_recovered = stp.n
    p_recovered = stp.p
    print(f"  恢复 2000ms 后: n={n_recovered:.4f}, p={p_recovered:.4f}")

    assert n_recovered > 0.85, f"囊泡池应基本恢复: n={n_recovered:.4f}"
    assert abs(p_recovered - MOSSY_FIBER_STP.p0) < 0.01, (
        f"释放概率应恢复到 p0: p={p_recovered:.4f} vs p0={MOSSY_FIBER_STP.p0}"
    )
    print(f"  ✓ 囊泡池恢复: {n_after_burst:.4f} → {n_recovered:.4f}")
    print(f"  ✓ 释放概率恢复: {p_after_burst:.4f} → {p_recovered:.4f} ≈ p0={MOSSY_FIBER_STP.p0}")

    # reset 测试
    stp.reset()
    assert stp.n == 1.0 and stp.p == MOSSY_FIBER_STP.p0
    print(f"  ✓ reset() 正常")

    print("  PASSED")


# ============================================================================
# 测试 4: Theta 振荡时钟 — 相位推进
# ============================================================================

def test_oscillation_clock_phase():
    """验证振荡时钟的相位推进和周期"""
    print("\n" + "=" * 60)
    print("测试 4: Theta 振荡时钟 — 相位推进")
    print("=" * 60)

    from wuyun.spike.oscillation_clock import (
        OscillationClock, THETA_PARAMS,
    )
    from wuyun.spike.signal_types import OscillationBand

    clock = OscillationClock()
    clock.add_band(OscillationBand.THETA, THETA_PARAMS)

    # Theta = 6Hz → 周期 = 1000/6 ≈ 166.7ms
    expected_period = 1000.0 / THETA_PARAMS.frequency
    print(f"  Theta 频率: {THETA_PARAMS.frequency}Hz, 周期: {expected_period:.1f}ms")

    initial_phase = clock.get_phase(OscillationBand.THETA)
    print(f"  初始相位: {initial_phase:.4f} rad")

    # 推进一个完整周期
    steps = int(expected_period)
    for _ in range(steps):
        clock.step()

    final_phase = clock.get_phase(OscillationBand.THETA)
    # 应该接近 2π (一个完整周期)
    expected_final = 2.0 * math.pi * THETA_PARAMS.frequency * steps / 1000.0
    expected_final_mod = expected_final % (2.0 * math.pi)
    print(f"  {steps}步后相位: {final_phase:.4f} rad (期望: {expected_final_mod:.4f})")

    phase_error = abs(final_phase - expected_final_mod)
    assert phase_error < 0.1, f"相位误差过大: {phase_error:.4f}"
    print(f"  ✓ 相位推进正确, 误差: {phase_error:.6f} rad")

    # 验证 OscillationState 输出
    state = clock.get_state()
    assert state.theta_power == 1.0
    assert state.theta_phase == final_phase
    print(f"  ✓ get_state() 正常: theta_power={state.theta_power}, theta_phase={state.theta_phase:.4f}")

    print("  PASSED")


# ============================================================================
# 测试 5: Theta 编码/检索相位门控
# ============================================================================

def test_theta_phase_gating():
    """验证 Theta 编码/检索相位交替"""
    print("\n" + "=" * 60)
    print("测试 5: Theta 编码/检索相位门控")
    print("=" * 60)

    from wuyun.spike.oscillation_clock import (
        OscillationClock, THETA_PARAMS,
    )
    from wuyun.spike.signal_types import OscillationBand

    clock = OscillationClock()
    clock.add_band(OscillationBand.THETA, THETA_PARAMS)

    encoding_count = 0
    retrieval_count = 0
    transitions = 0
    prev_encoding = None
    total_steps = 1000  # 1 秒, 应有 ~6 个 theta 周期

    for t in range(total_steps):
        clock.step()
        is_enc = clock.is_encoding_phase()

        if is_enc:
            encoding_count += 1
        else:
            retrieval_count += 1

        if prev_encoding is not None and is_enc != prev_encoding:
            transitions += 1
        prev_encoding = is_enc

    print(f"  1000ms 内:")
    print(f"    编码相位步数: {encoding_count}")
    print(f"    检索相位步数: {retrieval_count}")
    print(f"    相位转换次数: {transitions}")

    # 编码和检索应该各占约一半
    ratio = encoding_count / total_steps
    print(f"    编码占比: {ratio:.3f} (期望 ≈0.5)")
    assert 0.35 < ratio < 0.65, f"编码占比异常: {ratio:.3f}"
    print(f"  ✓ 编码/检索各约 50%")

    # 应该有 ~12 次转换 (每个周期 2 次转换 × 6 个周期)
    expected_transitions = 2 * THETA_PARAMS.frequency  # 每秒
    print(f"    期望转换: ~{expected_transitions:.0f}, 实际: {transitions}")
    assert transitions >= 8, f"转换次数过少: {transitions}"
    print(f"  ✓ 相位转换正常: {transitions} 次")

    # 编码/检索强度测试
    clock.reset()
    max_enc = 0.0
    max_ret = 0.0
    for t in range(200):
        clock.step()
        enc_str = clock.get_encoding_strength()
        ret_str = clock.get_retrieval_strength()
        max_enc = max(max_enc, enc_str)
        max_ret = max(max_ret, ret_str)

        # 编码和检索强度不应同时为最大
        assert not (enc_str > 0.9 and ret_str > 0.9), (
            f"t={t}: 编码和检索不应同时最大 (enc={enc_str:.3f}, ret={ret_str:.3f})"
        )

    print(f"  ✓ 编码最大强度: {max_enc:.3f}")
    print(f"  ✓ 检索最大强度: {max_ret:.3f}")
    assert max_enc > 0.8 and max_ret > 0.8, "编码和检索应各自达到高强度"
    print(f"  ✓ 编码和检索互斥且各达到高强度")

    print("  PASSED")


# ============================================================================
# 测试 6: 多频段振荡 + 调制
# ============================================================================

def test_multi_band_modulation():
    """验证多频段振荡和 CTC 调制因子"""
    print("\n" + "=" * 60)
    print("测试 6: 多频段振荡 + 调制")
    print("=" * 60)

    from wuyun.spike.oscillation_clock import (
        OscillationClock, THETA_PARAMS, GAMMA_PARAMS,
    )
    from wuyun.spike.signal_types import OscillationBand

    clock = OscillationClock()
    clock.add_band(OscillationBand.THETA, THETA_PARAMS)
    clock.add_band(OscillationBand.GAMMA, GAMMA_PARAMS)

    print(f"  Theta: {THETA_PARAMS.frequency}Hz, Gamma: {GAMMA_PARAMS.frequency}Hz")

    mod_min_theta = 1.0
    mod_max_theta = 0.0
    mod_min_gamma = 1.0
    mod_max_gamma = 0.0

    for t in range(500):
        clock.step()
        mod_t = clock.get_modulation(OscillationBand.THETA)
        mod_g = clock.get_modulation(OscillationBand.GAMMA)

        mod_min_theta = min(mod_min_theta, mod_t)
        mod_max_theta = max(mod_max_theta, mod_t)
        mod_min_gamma = min(mod_min_gamma, mod_g)
        mod_max_gamma = max(mod_max_gamma, mod_g)

    print(f"  Theta 调制范围: [{mod_min_theta:.3f}, {mod_max_theta:.3f}]")
    print(f"  Gamma 调制范围: [{mod_min_gamma:.3f}, {mod_max_gamma:.3f}]")

    # Theta power=1.0 → 调制范围应为 [0, 1]
    assert mod_min_theta < 0.1, f"Theta 最小调制应接近 0: {mod_min_theta:.3f}"
    assert mod_max_theta > 0.9, f"Theta 最大调制应接近 1: {mod_max_theta:.3f}"
    print(f"  ✓ Theta 调制: 全范围 [0, 1]")

    # Gamma power=0.5 → 调制范围应为 [0.25, 0.75]
    assert mod_min_gamma > 0.15, f"Gamma 最小调制异常: {mod_min_gamma:.3f}"
    assert mod_max_gamma < 0.85, f"Gamma 最大调制异常: {mod_max_gamma:.3f}"
    print(f"  ✓ Gamma 调制: 受限范围 (power=0.5)")

    # 未添加的频段应返回 0.5 (无调制)
    mod_alpha = clock.get_modulation(OscillationBand.ALPHA)
    assert mod_alpha == 0.5, f"未添加频段应返回 0.5: {mod_alpha}"
    print(f"  ✓ 未注册频段返回 0.5 (无调制)")

    # reset 测试
    clock.reset()
    assert clock.time == 0
    assert clock.get_phase(OscillationBand.THETA) == 0.0
    print(f"  ✓ reset() 正常")

    print("  PASSED")


# ============================================================================
# 测试 7: STP + 频率依赖 (苔藓纤维去极化器效应)
# ============================================================================

def test_stp_frequency_dependence():
    """验证苔藓纤维的频率依赖特性: 短脉冲串的峰值效能随频率变化"""
    print("\n" + "=" * 60)
    print("测试 7: STP 频率依赖 (苔藓纤维去极化器)")
    print("=" * 60)

    from wuyun.synapse.short_term_plasticity import (
        ShortTermPlasticity, MOSSY_FIBER_STP, DEPRESSING_STP,
    )

    # --- 苔藓纤维: 短脉冲串 (5 spikes) 的峰值效能 ---
    freqs = [5, 10, 20, 50]  # Hz
    peak_effs_mf = []
    burst_size = 5

    for freq in freqs:
        stp = ShortTermPlasticity(MOSSY_FIBER_STP)
        isi = int(1000 / freq)

        effs = []
        for i in range(burst_size):
            for _ in range(isi):
                stp.step()
            eff = stp.on_spike()
            effs.append(eff)

        peak = max(effs)
        peak_effs_mf.append(peak)
        print(f"  苔藓 {freq}Hz: 峰值效能 = {peak:.4f} (序列: {[f'{e:.4f}' for e in effs]})")

    # 苔藓纤维关键特征: 所有频率的峰值效能都 > 基线 p0
    baseline = MOSSY_FIBER_STP.p0
    for i, freq in enumerate(freqs):
        assert peak_effs_mf[i] > baseline * 1.5, (
            f"{freq}Hz 峰值 ({peak_effs_mf[i]:.4f}) 应显著高于基线 ({baseline:.4f})"
        )
    print(f"  ✓ 所有频率峰值效能 >> 基线 p0={baseline}")

    # --- 对比: 纯抑制型 STP (应该单调下降) ---
    peak_effs_dep = []
    for freq in freqs:
        stp = ShortTermPlasticity(DEPRESSING_STP)
        isi = int(1000 / freq)
        effs = []
        for i in range(burst_size):
            for _ in range(isi):
                stp.step()
            eff = stp.on_spike()
            effs.append(eff)
        peak_effs_dep.append(effs[0])  # 抑制型: 首个脉冲最强

    print(f"\n  抑制型对比:")
    for i, freq in enumerate(freqs):
        print(f"    {freq}Hz: 首脉冲效能 = {peak_effs_dep[i]:.4f}")

    # 苔藓纤维的 PPF 比率应远高于抑制型
    # 苔藓: peak/first >> 1, 抑制: peak/first ≈ 1 (首个就是最大)
    stp_mf_test = ShortTermPlasticity(MOSSY_FIBER_STP)
    stp_dep_test = ShortTermPlasticity(DEPRESSING_STP)
    for _ in range(20):
        stp_mf_test.step()
        stp_dep_test.step()
    eff1_mf = stp_mf_test.on_spike()
    eff1_dep = stp_dep_test.on_spike()
    for _ in range(20):
        stp_mf_test.step()
        stp_dep_test.step()
    eff2_mf = stp_mf_test.on_spike()
    eff2_dep = stp_dep_test.on_spike()

    ppf_mf = eff2_mf / eff1_mf if eff1_mf > 0 else 0
    ppf_dep = eff2_dep / eff1_dep if eff1_dep > 0 else 0
    print(f"\n  PPF 对比 (20ms ISI):")
    print(f"    苔藓纤维: {ppf_mf:.2f}x")
    print(f"    抑制型:   {ppf_dep:.2f}x")
    assert ppf_mf > ppf_dep, (
        f"苔藓纤维 PPF ({ppf_mf:.2f}) 应高于抑制型 ({ppf_dep:.2f})"
    )
    print(f"  ✓ 苔藓纤维易化 > 抑制型: {ppf_mf:.2f}x > {ppf_dep:.2f}x")

    print("  PASSED")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("Phase 2.8 测试: P3 前置补充")
    print("=" * 60)

    tests = [
        ("测试 1: 海马神经元参数预设", test_hippocampal_neuron_params),
        ("测试 2: STP 基础动力学", test_stp_basic),
        ("测试 3: STP 恢复动力学", test_stp_recovery),
        ("测试 4: Theta 振荡相位推进", test_oscillation_clock_phase),
        ("测试 5: Theta 编码/检索门控", test_theta_phase_gating),
        ("测试 6: 多频段振荡 + 调制", test_multi_band_modulation),
        ("测试 7: STP 频率依赖去极化器", test_stp_frequency_dependence),
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
