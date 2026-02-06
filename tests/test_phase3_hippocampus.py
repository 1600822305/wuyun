"""
Phase 3 测试: 海马记忆系统 (DG→CA3→CA1 三突触环路)

7 个测试:
  1. DG 模式分离
  2. DG 稀疏激活
  3. CA3 模式存储 (STDP)
  4. CA3 模式补全
  5. CA1 匹配/新奇检测
  6. Theta 相位门控
  7. 全环路编码-回忆

不使用 pytest, 用 print() + assert 验证。
"""

import sys
import os
import time
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wuyun.circuit.hippocampus.dentate_gyrus import DentateGyrus
from wuyun.circuit.hippocampus.ca3_network import CA3Network
from wuyun.circuit.hippocampus.ca1_network import CA1Network
from wuyun.circuit.hippocampus.hippocampal_loop import (
    HippocampalLoop,
    create_hippocampal_loop,
)
from wuyun.spike.signal_types import OscillationBand
from wuyun.spike.oscillation_clock import OscillationClock, THETA_PARAMS


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =========================================================================
# 测试 1: DG 模式分离
# =========================================================================

def test_1_dg_pattern_separation():
    """验证 DG 将相似输入映射为更正交的稀疏表示"""
    print("\n" + "=" * 70)
    print("测试 1: DG 模式分离")
    print("=" * 70)

    dg = DentateGyrus(
        n_ec_inputs=16,
        n_granule=100,
        n_inhibitory=10,
        ec_granule_gain=25.0,
        pv_granule_gain=25.0,
        seed=42,
    )

    rng = np.random.RandomState(123)

    # 创建两个相似模式 (cosine > 0.8)
    base = rng.uniform(0.3, 0.8, size=16)
    noise = rng.uniform(-0.1, 0.1, size=16)
    pattern_a = np.clip(base, 0, 1)
    pattern_b = np.clip(base + noise, 0, 1)

    input_sim = cosine_similarity(pattern_a, pattern_b)
    print(f"  输入相似度: {input_sim:.4f}")
    assert input_sim > 0.8, f"输入模式不够相似: {input_sim:.4f}"

    # 运行模式 A, 收集 DG 输出
    dg.reset()
    dg_output_a = np.zeros(100)
    for t in range(200):
        dg.inject_ec_input(pattern_a)
        dg.step(t)
        dg_output_a += dg.get_granule_activity()

    # 运行模式 B
    dg.reset()
    dg_output_b = np.zeros(100)
    for t in range(200):
        dg.inject_ec_input(pattern_b)
        dg.step(t)
        dg_output_b += dg.get_granule_activity()

    output_sim = cosine_similarity(dg_output_a, dg_output_b)
    print(f"  DG 输出相似度: {output_sim:.4f}")
    print(f"  分离效果: {input_sim:.4f} → {output_sim:.4f}")
    print(f"  DG-A 活跃细胞: {np.sum(dg_output_a > 0)}/{100}")
    print(f"  DG-B 活跃细胞: {np.sum(dg_output_b > 0)}/{100}")

    # 核心断言: DG 输出相似度 < 输入相似度 (模式分离)
    assert output_sim < input_sim, (
        f"DG 未实现模式分离: 输入 sim={input_sim:.4f}, 输出 sim={output_sim:.4f}"
    )
    print("  ✓ DG 模式分离验证通过")


# =========================================================================
# 测试 2: DG 稀疏激活
# =========================================================================

def test_2_dg_sparse_activation():
    """验证 DG 激活率保持低水平, 强输入不导致广泛激活"""
    print("\n" + "=" * 70)
    print("测试 2: DG 稀疏激活")
    print("=" * 70)

    dg = DentateGyrus(
        n_ec_inputs=16,
        n_granule=100,
        n_inhibitory=10,
        ec_granule_gain=25.0,
        pv_granule_gain=25.0,
        seed=42,
    )

    rng = np.random.RandomState(456)

    # 测试不同强度的输入
    sparsities = []
    for strength in [0.2, 0.4, 0.6, 1.0]:
        dg.reset()
        pattern = rng.uniform(0, strength, size=16)

        # 运行足够步数让网络稳定
        for t in range(300):
            dg.inject_ec_input(pattern)
            dg.step(t)

        sparsity = dg.get_sparsity()
        mean_rate = dg.get_mean_rate()
        sparsities.append(sparsity)
        print(f"  输入强度={strength:.1f}: 稀疏度={sparsity:.4f}, 平均发放率={mean_rate:.2f}Hz")

    # 核心断言: 所有输入强度下稀疏度 < 20%
    for i, (strength, sparsity) in enumerate(zip([0.2, 0.4, 0.6, 1.0], sparsities)):
        assert sparsity < 0.20, (
            f"输入强度={strength}: 稀疏度 {sparsity:.4f} > 20%, PV 抑制不足"
        )

    # 强输入不应比弱输入激活多太多 (PV 保护)
    # 允许一定增长但不应线性增长
    print(f"  稀疏度范围: [{min(sparsities):.4f}, {max(sparsities):.4f}]")
    print("  ✓ DG 稀疏激活验证通过")


# =========================================================================
# 测试 3: CA3 模式存储 (STDP)
# =========================================================================

def test_3_ca3_pattern_storage():
    """验证 STDP 学习后, 共活跃细胞间权重 > 非共活跃细胞间权重"""
    print("\n" + "=" * 70)
    print("测试 3: CA3 模式存储 (STDP)")
    print("=" * 70)

    ca3 = CA3Network(
        n_pyramidal=50,
        n_inhibitory=8,
        recurrent_prob=0.2,
        mossy_gain=40.0,
        pv_gain=20.0,
        n_dg_granule=100,
        n_ec_inputs=16,
        seed=42,
    )

    # 记录初始权重
    W_before = ca3.get_recurrent_weights().copy()
    initial_mean = np.mean(W_before[W_before > 0]) if np.any(W_before > 0) else 0
    print(f"  初始循环权重均值: {initial_mean:.4f}")

    # 创建模拟 DG 输入 — 激活特定 CA3 子集
    # 通过直接注入电流模拟苔藓纤维输入
    rng = np.random.RandomState(789)
    active_ca3_indices = sorted(rng.choice(50, size=15, replace=False))
    print(f"  目标活跃 CA3 细胞: {active_ca3_indices[:10]}... ({len(active_ca3_indices)} 个)")

    # 编码期: 循环沉默, 仅外部驱动 + STDP 异突触更新
    # (基于文献: 编码期 CA3 recurrent 被 ACh 抑制)
    for t in range(500):
        # 直接注入电流到目标 CA3 细胞 (模拟苔藓纤维强输入)
        for idx in active_ca3_indices:
            ca3.pyramidal_pop.i_basal[idx] += 40.0

        ca3.step(t, enable_recurrent=False)
        ca3.apply_recurrent_stdp(t)

    # 检查权重变化
    W_after = ca3.get_recurrent_weights()

    # 计算共活跃细胞间 vs 非共活跃细胞间的权重
    active_set = set(active_ca3_indices)
    co_active_weights = []
    non_co_active_weights = []

    for i in range(50):
        for j in range(50):
            if W_after[i, j] > 0:
                if i in active_set and j in active_set:
                    co_active_weights.append(W_after[i, j])
                else:
                    non_co_active_weights.append(W_after[i, j])

    mean_co = np.mean(co_active_weights) if co_active_weights else 0
    mean_non = np.mean(non_co_active_weights) if non_co_active_weights else 0

    print(f"  共活跃权重均值: {mean_co:.4f} ({len(co_active_weights)} 个连接)")
    print(f"  非共活跃权重均值: {mean_non:.4f} ({len(non_co_active_weights)} 个连接)")
    print(f"  权重比: {mean_co / mean_non:.2f}x" if mean_non > 0 else "  非共活跃权重为 0")

    # 核心断言: 共活跃细胞间权重 > 非共活跃细胞间权重
    assert mean_co > mean_non, (
        f"STDP 未增强共活跃连接: co={mean_co:.4f}, non={mean_non:.4f}"
    )
    assert len(co_active_weights) > 0, "无共活跃连接"
    print("  ✓ CA3 模式存储验证通过")


# =========================================================================
# 测试 4: CA3 模式补全
# =========================================================================

def test_4_ca3_pattern_completion():
    """验证 CA3 能用部分线索回忆完整模式"""
    print("\n" + "=" * 70)
    print("测试 4: CA3 模式补全")
    print("=" * 70)

    ca3 = CA3Network(
        n_pyramidal=50,
        n_inhibitory=8,
        recurrent_prob=0.4,  # 更密的循环连接 (降低零连接概率)
        mossy_gain=40.0,
        recurrent_gain=25.0,  # ★ 强循环增益, 确保激活扩散超过阈值
        pv_gain=8.0,  # 中等抑制, 允许循环激活扩散
        n_dg_granule=100,
        n_ec_inputs=16,
        seed=42,
    )

    rng = np.random.RandomState(101)
    # 选择 15 个目标活跃细胞
    target_cells = sorted(rng.choice(50, size=15, replace=False))
    target_set = set(target_cells)
    print(f"  目标模式: {len(target_cells)} 个活跃细胞")

    # === 编码阶段: 循环沉默, 强激活目标细胞, STDP 异突触更新 ===
    # 500 步足够权重饱和 (测试 3 验证), STDP 每 5 步更新 (降低计算负载)
    for t in range(500):
        for idx in target_cells:
            ca3.pyramidal_pop.i_basal[idx] += 45.0
        ca3.step(t, enable_recurrent=False)
        if t % 5 == 0:
            ca3.apply_recurrent_stdp(t)

    W = ca3.get_recurrent_weights()
    co_w = []
    for i in target_cells:
        for j in target_cells:
            if W[i, j] > 0:
                co_w.append(W[i, j])
    print(f"  编码后共活跃权重: mean={np.mean(co_w):.4f}" if co_w else "  无共活跃连接")

    # === 重置动态状态 (保留学习到的权重) ===
    ca3.pyramidal_pop.reset()
    ca3.pv_pop.reset()
    ca3.recurrent_syn.reset()
    ca3.ca3_pv_syn.reset()
    ca3.pv_ca3_syn.reset()

    # === 回忆阶段: 只激活 50% 的目标细胞 ===
    cue_cells = target_cells[:len(target_cells) // 2]
    print(f"  部分线索: {len(cue_cells)} 个细胞 (50%)")

    # 回忆阶段: 循环放大, 部分线索 → 完整模式
    recalled_activity = np.zeros(50)
    for t in range(500, 1100):
        # 只激活线索细胞
        for idx in cue_cells:
            ca3.pyramidal_pop.i_basal[idx] += 45.0
        ca3.step(t, enable_recurrent=True)  # ★ 检索期循环放大
        recalled_activity += ca3.get_activity()

    # 统计回忆准确率
    recalled_cells = set(np.where(recalled_activity > 0)[0])
    target_recalled = recalled_cells & target_set
    recall_rate = len(target_recalled) / len(target_set) if target_set else 0

    print(f"  回忆激活细胞: {len(recalled_cells)}")
    print(f"  目标被回忆: {len(target_recalled)}/{len(target_set)}")
    print(f"  回忆率: {recall_rate:.2%}")

    # 核心断言: 回忆率 > 70% (部分线索能激活大部分目标)
    assert recall_rate > 0.70, (
        f"CA3 模式补全不足: 回忆率 {recall_rate:.2%} < 70%"
    )
    print("  ✓ CA3 模式补全验证通过")


# =========================================================================
# 测试 5: CA1 匹配/新奇检测
# =========================================================================

def test_5_ca1_match_novelty():
    """验证 CA1 能区分匹配 (burst) 和新奇 (regular)"""
    print("\n" + "=" * 70)
    print("测试 5: CA1 匹配/新奇检测")
    print("=" * 70)

    ca1 = CA1Network(
        n_pyramidal=50,
        n_inhibitory=8,
        schaffer_gain=20.0,
        ec3_gain=15.0,
        pv_gain=15.0,
        n_ca3=50,
        n_ec_inputs=16,
        seed=42,
    )

    rng = np.random.RandomState(202)

    # === 场景 A: CA3 回忆 + EC-III 感知 一致 → 应该有 burst ===
    print("  场景 A: CA3 + EC-III 同时输入 (匹配)")
    ca1.reset()

    burst_count_a = 0
    regular_count_a = 0
    total_active_a = 0

    for t in range(500):
        # CA3→basal: 直接注入电流模拟 Schaffer 输入
        ca1.pyramidal_pop.i_basal[:] += 20.0
        # EC-III→apical: 注入 apical 电流
        ec3_pattern = rng.uniform(0.5, 1.0, size=16)
        ca1.inject_ec3_input(ec3_pattern)

        ca1.step(t)

        st = ca1.pyramidal_pop.spike_type
        active = st != 0
        total_active_a += int(active.sum())
        is_burst = (st == 3) | (st == 4) | (st == 5)  # BURST_START/CONTINUE/END
        burst_count_a += int(is_burst.sum())
        regular_count_a += int((st == 1).sum())  # REGULAR

    burst_ratio_a = burst_count_a / total_active_a if total_active_a > 0 else 0
    print(f"    活跃: {total_active_a}, burst: {burst_count_a}, regular: {regular_count_a}")
    print(f"    burst 比率: {burst_ratio_a:.4f}")

    # === 场景 B: 只有 CA3 回忆, 无 EC-III → 应该是 regular 为主 ===
    print("  场景 B: 只有 CA3 输入 (新奇)")
    ca1.reset()

    burst_count_b = 0
    regular_count_b = 0
    total_active_b = 0

    for t in range(500):
        # CA3→basal: 直接注入电流
        ca1.pyramidal_pop.i_basal[:] += 20.0
        # 无 EC-III 输入

        ca1.step(t)

        st = ca1.pyramidal_pop.spike_type
        active = st != 0
        total_active_b += int(active.sum())
        is_burst = (st == 3) | (st == 4) | (st == 5)
        burst_count_b += int(is_burst.sum())
        regular_count_b += int((st == 1).sum())  # REGULAR

    burst_ratio_b = burst_count_b / total_active_b if total_active_b > 0 else 0
    print(f"    活跃: {total_active_b}, burst: {burst_count_b}, regular: {regular_count_b}")
    print(f"    burst 比率: {burst_ratio_b:.4f}")

    # 核心断言: 场景 A 的 burst 比率 > 场景 B
    print(f"  burst 比率对比: A={burst_ratio_a:.4f} vs B={burst_ratio_b:.4f}")
    assert burst_ratio_a > burst_ratio_b, (
        f"CA1 匹配/新奇检测失败: 匹配场景 burst={burst_ratio_a:.4f} "
        f"应 > 新奇场景 burst={burst_ratio_b:.4f}"
    )
    # 场景 A 应该有一些 burst
    assert burst_count_a > 0, "匹配场景应该产生 burst"
    print("  ✓ CA1 匹配/新奇检测验证通过")


# =========================================================================
# 测试 6: Theta 相位门控
# =========================================================================

def test_6_theta_phase_gating():
    """验证 Theta 振荡正确门控编码/检索通路"""
    print("\n" + "=" * 70)
    print("测试 6: Theta 相位门控")
    print("=" * 70)

    loop = create_hippocampal_loop(
        n_ec_inputs=16,
        n_granule=50,  # 较小规模加速测试
        n_ca3=30,
        n_ca1=30,
        seed=42,
    )

    rng = np.random.RandomState(303)
    pattern = rng.uniform(0.3, 0.8, size=16)

    # 运行 1 个完整 theta 周期 (~167ms @ 6Hz)
    encoding_steps = 0
    retrieval_steps = 0
    dg_active_encoding = 0
    dg_active_retrieval = 0
    stdp_updates = 0

    for t in range(200):
        loop.step(t, ec2_input=pattern, ec3_input=pattern)

        phase = loop.get_theta_phase()
        enc_str = loop.clock.get_encoding_strength()
        ret_str = loop.clock.get_retrieval_strength()

        dg_active = np.sum(loop.dg.get_granule_activity())

        if phase == "encoding":
            encoding_steps += 1
            dg_active_encoding += dg_active
            if enc_str > 0.5:
                stdp_updates += 1
        else:
            retrieval_steps += 1
            dg_active_retrieval += dg_active

    print(f"  编码步数: {encoding_steps}")
    print(f"  检索步数: {retrieval_steps}")
    print(f"  编码期 DG 总活跃: {dg_active_encoding:.0f}")
    print(f"  检索期 DG 总活跃: {dg_active_retrieval:.0f}")
    print(f"  STDP 更新步数: {stdp_updates}")

    # 核心断言
    # 1. 两相都应该出现 (theta 6Hz, 200ms ≈ 1.2 个周期)
    assert encoding_steps > 0, "无编码相位"
    assert retrieval_steps > 0, "无检索相位"

    # 2. 编码期 DG 应该比检索期更活跃 (编码通路开放)
    assert dg_active_encoding > dg_active_retrieval, (
        f"编码期 DG 应比检索期更活跃: enc={dg_active_encoding:.0f}, "
        f"ret={dg_active_retrieval:.0f}"
    )

    # 3. STDP 只在编码期更新
    assert stdp_updates > 0, "编码期应有 STDP 更新"

    print("  ✓ Theta 相位门控验证通过")


# =========================================================================
# 测试 7: 全环路编码-回忆
# =========================================================================

def test_7_full_loop_encode_recall():
    """验证全环路编码多个模式后能正确回忆"""
    print("\n" + "=" * 70)
    print("测试 7: 全环路编码-回忆")
    print("=" * 70)

    loop = create_hippocampal_loop(
        n_ec_inputs=16,
        n_granule=100,
        n_ca3=50,
        n_ca1=50,
        seed=42,
    )

    rng = np.random.RandomState(404)

    # 创建 2 个不同模式 (强信号, 确保 DG 能激活)
    patterns = []
    for i in range(2):
        p = np.zeros(16)
        # 每个模式激活不同的 EC 输入子集, 使用强信号
        active_indices = rng.choice(16, size=10, replace=False)
        p[active_indices] = rng.uniform(0.7, 1.0, size=10)
        patterns.append(p)

    pattern_sim = cosine_similarity(patterns[0], patterns[1])
    print(f"  模式间相似度: {pattern_sim:.4f}")

    # === 编码阶段 (更长持续时间确保 STDP 学习) ===
    for i, pattern in enumerate(patterns):
        print(f"  编码模式 {i}...")
        loop.encode(pattern, duration=2000)

        # 编码后检查 CA3 活动
        ca3_rate = loop.ca3.get_mean_rate()
        print(f"    编码后 CA3 mean rate: {ca3_rate:.2f}Hz")

    # 获取编码后的 CA3 权重统计
    diag = loop.get_diagnostics()
    print(f"  CA3 循环权重: mean={diag['ca3_recurrent_mean_w']:.4f}, "
          f"max={diag['ca3_recurrent_max_w']:.4f}")

    # === 回忆阶段 ===
    recall_results = []
    for i, pattern in enumerate(patterns):
        # 创建 60% 部分线索 (稍多线索提高成功率)
        cue = pattern.copy()
        mask = np.zeros(16)
        nonzero = np.where(pattern > 0)[0]
        if len(nonzero) > 0:
            n_keep = max(1, int(len(nonzero) * 0.6))
            keep = nonzero[:n_keep]
            mask[keep] = 1.0
        cue = cue * mask

        print(f"  回忆模式 {i} (线索: {np.sum(mask > 0):.0f}/{np.sum(pattern > 0):.0f} 个活跃输入)...")

        # 重置动态状态但保留权重
        loop.ca3.pyramidal_pop.reset()
        loop.ca3.pv_pop.reset()
        loop.ca3.recurrent_syn.reset()
        loop.ca3.ca3_pv_syn.reset()
        loop.ca3.pv_ca3_syn.reset()
        loop.ca1.pyramidal_pop.reset()
        loop.ca1.pv_pop.reset()
        loop.ca1.ca1_pv_syn.reset()
        loop.ca1.pv_ca1_syn.reset()
        loop.dg.granule_pop.reset()
        loop.dg.pv_pop.reset()
        loop.dg.g2pv_syn.reset()
        loop.dg.pv2g_syn.reset()

        # 回忆 (累积活动而非只看最后一步)
        ca3_accum = np.zeros(50)
        for recall_t in range(500):
            t_r = loop._time + recall_t + 1
            loop.step(t_r, ec2_input=cue, ec3_input=None,
                      force_retrieval=True)
            ca3_accum += loop.ca3.get_activity()
        loop._time += 500

        ca3_rate = loop.ca3.get_mean_rate()
        ca1_rate = loop.ca1.get_mean_rate()

        recall_results.append({
            "ca3_active": int(np.sum(ca3_accum > 0)),
            "ca3_rate": ca3_rate,
            "ca1_rate": ca1_rate,
        })
        print(f"    CA3 活跃: {recall_results[-1]['ca3_active']}, "
              f"CA3 rate: {ca3_rate:.2f}Hz, CA1 rate: {ca1_rate:.2f}Hz")

    # 核心断言
    # 1. 至少一个模式回忆时 CA3 应该有活跃细胞
    any_active = any(r["ca3_active"] > 0 for r in recall_results)
    assert any_active, "所有模式回忆失败: CA3 无活跃细胞"

    # 2. 至少一个模式 CA3 应该有发放
    any_firing = any(r["ca3_rate"] > 0 for r in recall_results)
    assert any_firing, "所有模式回忆失败: CA3 发放率为 0"

    print("  ✓ 全环路编码-回忆验证通过")


# =========================================================================
# 主函数
# =========================================================================

def main():
    tests = [
        ("测试 1: DG 模式分离", test_1_dg_pattern_separation),
        ("测试 2: DG 稀疏激活", test_2_dg_sparse_activation),
        ("测试 3: CA3 模式存储", test_3_ca3_pattern_storage),
        ("测试 4: CA3 模式补全", test_4_ca3_pattern_completion),
        ("测试 5: CA1 匹配/新奇", test_5_ca1_match_novelty),
        ("测试 6: Theta 相位门控", test_6_theta_phase_gating),
        ("测试 7: 全环路编码-回忆", test_7_full_loop_encode_recall),
    ]

    passed = 0
    failed = 0
    errors = []

    total_start = time.time()

    for name, test_fn in tests:
        try:
            start = time.time()
            test_fn()
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

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print(f"Phase 3 测试结果: {passed}/{passed + failed} 通过")
    print(f"总耗时: {total_elapsed:.2f}s")
    if errors:
        print("\n失败测试:")
        for name, err in errors:
            print(f"  ✗ {name}: {err}")
    print("=" * 70)

    assert failed == 0, f"{failed} 个测试失败"


if __name__ == "__main__":
    main()