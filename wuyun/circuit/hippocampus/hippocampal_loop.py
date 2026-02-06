"""
3-D: HippocampalLoop — 海马全环路 + Theta 门控

整合 DG + CA3 + CA1 + OscillationClock:
  编码相 (theta谷): EC→DG→CA3 通路开放, CA3→CA1 抑制
  检索相 (theta峰): CA3→CA1 通路开放, EC→DG 抑制

step() 内部流程:
  1. clock.step()                        # Theta 相位推进
  2. enc_str = clock.get_encoding_strength()
  3. ret_str = clock.get_retrieval_strength()
  # === 编码通路 (调制 by enc_str) ===
  4. dg.inject_ec_input(ec2_input * enc_str)
  5. dg.step(t)
  6. ca3.inject_mossy_input(dg spikes, dg rates * enc_str)
  7. ca3.inject_ec_input(ec2_input * enc_str * 0.3)
  8. ca3.step(t)
  9. if enc_str > 0.5: ca3.apply_recurrent_stdp(t)
  # === 检索通路 (调制 by ret_str) ===
  10. ca1.inject_schaffer_input(ca3 spikes, ca3 rates * ret_str)
  11. ca1.inject_ec3_input(ec3_input * ret_str)
  12. ca1.step(t)

依赖: spike/ ← synapse/ ← neuron/ ← circuit/hippocampus/
"""

from typing import Optional, Dict
import numpy as np

from wuyun.spike.signal_types import OscillationBand
from wuyun.spike.oscillation_clock import OscillationClock, THETA_PARAMS

from wuyun.circuit.hippocampus.dentate_gyrus import DentateGyrus
from wuyun.circuit.hippocampus.ca3_network import CA3Network
from wuyun.circuit.hippocampus.ca1_network import CA1Network


class HippocampalLoop:
    """海马全环路 — Theta 门控的编码/检索系统

    整合 DG→CA3→CA1 三突触环路与 Theta 振荡门控。
    Theta 振荡 (6Hz, ~167ms/周期) 自动交替编码和检索相位:
    - 编码相 (theta 谷): EC→DG→CA3 通路活跃, STDP 更新循环权重
    - 检索相 (theta 峰): CA3→CA1 通路活跃, 回忆与感知比较

    使用模式:
        loop = create_hippocampal_loop()
        # 编码
        loop.encode(pattern_a, duration=500)
        # 回忆
        loop.recall(partial_cue, duration=300)
        match = loop.get_match_signal()
    """

    def __init__(
        self,
        dg: DentateGyrus,
        ca3: CA3Network,
        ca1: CA1Network,
        clock: OscillationClock,
        n_ec_inputs: int = 16,
    ):
        self.dg = dg
        self.ca3 = ca3
        self.ca1 = ca1
        self.clock = clock
        self.n_ec_inputs = n_ec_inputs
        self._time = 0

    def step(
        self,
        t: int,
        ec2_input: Optional[np.ndarray] = None,
        ec3_input: Optional[np.ndarray] = None,
        force_retrieval: bool = False,
    ) -> None:
        """单步推进 (theta 自动切换编码/检索)

        基于文献 (Cutsuridis 2010; PLOS CB 2025) 的相位依赖路由:
        - 编码相: EC→DG→CA3 活跃, CA3 循环沉默, STDP 异突触更新
        - 检索相: EC→CA3 直接通路 + CA3 循环放大, DG 通路抑制

        Args:
            t: 当前仿真时间步
            ec2_input: EC-II 输入 (→ DG + CA3), shape=(n_ec_inputs,)
            ec3_input: EC-III 输入 (→ CA1 apical), shape=(n_ec_inputs,)
            force_retrieval: 强制检索模式 (recall 时使用)
        """
        self._time = t

        # 默认零输入
        if ec2_input is None:
            ec2_input = np.zeros(self.n_ec_inputs)
        if ec3_input is None:
            ec3_input = np.zeros(self.n_ec_inputs)

        # === 1. Theta 相位推进 ===
        self.clock.step()
        enc_str = self.clock.get_encoding_strength()
        ret_str = self.clock.get_retrieval_strength()
        is_encoding = self.clock.is_encoding_phase() and not force_retrieval

        # === 2. 编码通路 ===
        # 编码相: DG→CA3 活跃 (强), EC→CA3 直接通路也活跃
        # 检索相: DG 通路弱化, EC→CA3 直接通路增强 (用于线索输入)
        if is_encoding:
            eff_enc = max(enc_str, 0.6)
            eff_direct = eff_enc  # 编码期 EC→CA3 强驱动 (确保 CA3 可靠激活)
        else:
            eff_enc = 0.1  # 检索期 DG 通路几乎沉默
            eff_direct = max(ret_str, 0.3)  # 检索期 EC→CA3 增强

        # EC-II → DG
        self.dg.inject_ec_input(ec2_input * eff_enc)
        self.dg.step(t)

        # DG → CA3 (mossy fiber)
        dg_spikes = self.dg.get_output_spikes()
        dg_rates = self.dg.get_granule_rates()
        self.ca3.inject_mossy_input(dg_spikes, dg_rates * eff_enc)

        # EC-II → CA3 (穿通纤维直接通路)
        self.ca3.inject_ec_input(ec2_input * eff_direct)

        # CA3 step: 编码期循环沉默, 检索期循环放大
        self.ca3.step(t, enable_recurrent=not is_encoding)

        # 编码期 STDP 异突触更新
        # 每 5 步更新一次 (降低计算负载, 50ms STDP 窗口仍能捕获脉冲对)
        if is_encoding and enc_str > 0.3 and t % 5 == 0:
            self.ca3.apply_recurrent_stdp(t)

        # === 3. 检索通路 ===
        # CA3 → CA1 (Schaffer collateral)
        ca3_spikes = self.ca3.get_output_spikes()
        ca3_rates = self.ca3.get_firing_rates()
        eff_ret = ret_str if not is_encoding else 0.1
        self.ca1.inject_schaffer_input(ca3_spikes, ca3_rates * eff_ret)

        # EC-III → CA1 (穿通纤维, → apical)
        self.ca1.inject_ec3_input(ec3_input * eff_ret)
        self.ca1.step(t)

    def encode(self, ec_pattern: np.ndarray, duration: int = 500) -> None:
        """编码一个模式

        运行 duration 步, EC-II 和 EC-III 都接收同一模式。
        Theta 门控确保编码相时 DG→CA3 通路活跃, STDP 更新。

        Args:
            ec_pattern: EC 输入模式, shape=(n_ec_inputs,)
            duration: 编码持续时间 (ms/时间步)
        """
        for i in range(duration):
            t = self._time + i + 1
            self.step(t, ec2_input=ec_pattern, ec3_input=ec_pattern)
        self._time += duration

    def recall(self, partial_cue: np.ndarray, duration: int = 300) -> None:
        """用部分线索回忆

        强制检索模式: EC→CA3 直接通路 + CA3 循环放大。
        DG 通路沉默, CA1 只依赖 CA3 回忆。
        基于文献: 回忆时循环连接被 ACh 撤退放大 (Cutsuridis 2010)。

        Args:
            partial_cue: 部分线索, shape=(n_ec_inputs,)
            duration: 回忆持续时间 (ms/时间步)
        """
        for i in range(duration):
            t = self._time + i + 1
            # 强制检索: EC→CA3 直接 + 循环放大, DG 沉默
            self.step(t, ec2_input=partial_cue, ec3_input=None,
                      force_retrieval=True)
        self._time += duration

    # =========================================================================
    # 状态查询
    # =========================================================================

    def get_ca1_output(self) -> np.ndarray:
        """获取 CA1 输出 (发放率向量)"""
        return self.ca1.get_output()

    def get_match_signal(self) -> float:
        """获取匹配度 (CA1 burst 比率)"""
        return self.ca1.get_match_signal()

    def get_novelty_signal(self) -> float:
        """获取新奇度 (CA1 regular 比率)"""
        return self.ca1.get_novelty_signal()

    def get_theta_phase(self) -> str:
        """获取当前 theta 相位

        Returns:
            "encoding" 或 "retrieval"
        """
        if self.clock.is_encoding_phase():
            return "encoding"
        return "retrieval"

    def get_diagnostics(self) -> Dict:
        """获取诊断信息

        Returns:
            包含 DG 稀疏度, CA3 活跃度, CA1 匹配/新奇, 权重统计
        """
        ca3_weights = self.ca3.get_recurrent_weights()
        nonzero_weights = ca3_weights[ca3_weights > 0]

        return {
            "dg_sparsity": self.dg.get_sparsity(),
            "dg_mean_rate": self.dg.get_mean_rate(),
            "ca3_mean_rate": self.ca3.get_mean_rate(),
            "ca3_active_frac": float(np.mean(self.ca3.get_activity())),
            "ca1_mean_rate": self.ca1.get_mean_rate(),
            "ca1_match": self.ca1.get_match_signal(),
            "ca1_novelty": self.ca1.get_novelty_signal(),
            "ca1_burst_ratio": self.ca1.get_burst_ratio(),
            "ca3_recurrent_mean_w": float(np.mean(nonzero_weights)) if len(nonzero_weights) > 0 else 0.0,
            "ca3_recurrent_max_w": float(np.max(nonzero_weights)) if len(nonzero_weights) > 0 else 0.0,
            "ca3_recurrent_n_synapses": len(nonzero_weights),
            "theta_phase": self.get_theta_phase(),
            "encoding_strength": self.clock.get_encoding_strength(),
            "retrieval_strength": self.clock.get_retrieval_strength(),
        }

    def reset(self) -> None:
        """重置所有状态 (保留连接结构和学习到的权重)"""
        self.dg.reset()
        self.ca3.reset()
        self.ca1.reset()
        self.clock.reset()
        self._time = 0

    def __repr__(self) -> str:
        return (
            f"HippocampalLoop(dg={self.dg}, ca3={self.ca3}, ca1={self.ca1}, "
            f"theta={self.get_theta_phase()})"
        )


# =============================================================================
# 工厂函数
# =============================================================================

def create_hippocampal_loop(
    n_ec_inputs: int = 16,
    n_granule: int = 100,
    n_ca3: int = 50,
    n_ca1: int = 50,
    seed: int = 42,
) -> HippocampalLoop:
    """创建完整的海马环路

    Args:
        n_ec_inputs: EC 输入维度
        n_granule: DG 颗粒细胞数
        n_ca3: CA3 锥体细胞数
        n_ca1: CA1 锥体细胞数
        seed: 随机种子

    Returns:
        配置好的 HippocampalLoop 实例
    """
    # DG
    dg = DentateGyrus(
        n_ec_inputs=n_ec_inputs,
        n_granule=n_granule,
        n_inhibitory=max(2, n_granule // 10),
        seed=seed,
    )

    # CA3
    ca3 = CA3Network(
        n_pyramidal=n_ca3,
        n_inhibitory=max(2, n_ca3 // 6),
        n_dg_granule=n_granule,
        n_ec_inputs=n_ec_inputs,
        seed=seed + 1,
    )

    # CA1
    ca1 = CA1Network(
        n_pyramidal=n_ca1,
        n_inhibitory=max(2, n_ca1 // 6),
        n_ca3=n_ca3,
        n_ec_inputs=n_ec_inputs,
        seed=seed + 2,
    )

    # Theta clock
    clock = OscillationClock()
    clock.add_band(OscillationBand.THETA, THETA_PARAMS)

    return HippocampalLoop(
        dg=dg,
        ca3=ca3,
        ca1=ca1,
        clock=clock,
        n_ec_inputs=n_ec_inputs,
    )