"""
3-C: CA1Network — 比较器与输出层 (向量化版本)

核心功能: 比较 CA3 回忆与 EC-III 直接感知，检测新奇/匹配
生物学: CA1 同时接收 CA3 (via Schaffer) 和 EC-III (via 穿通纤维)

预测编码对应 (★核心创新):
  - CA3→basal (前馈回忆) + EC-III→apical (反馈感知) 同时 → BURST = 匹配
  - 只有 CA3→basal                                        → REGULAR = 新奇
  - 只有 EC-III→apical                                    → 沉默 (感知但无回忆)

组成 (向量化):
  - pyramidal_pop: NeuronPopulation (n_pyramidal, PLACE_CELL)
  - pv_pop: NeuronPopulation (n_inhibitory, BASKET_PV)
  - ca1_pv_syn: SynapseGroup (CA1→PV, AMPA)
  - pv_ca1_syn: SynapseGroup (PV→CA1, GABA_A)
  - CA3→CA1: Schaffer collateral (STP 电流注入) → BASAL
  - EC-III→CA1: 矩阵乘法电流注入 → APICAL

依赖: core/ (NeuronPopulation, SynapseGroup)
"""

from typing import List, Dict, Optional
import numpy as np

from wuyun.spike.spike import Spike
from wuyun.spike.signal_types import (
    SpikeType,
    SynapseType,
    CompartmentType,
)
from wuyun.synapse.synapse_base import AMPA_PARAMS, GABA_A_PARAMS
from wuyun.synapse.short_term_plasticity import (
    ShortTermPlasticity,
    SCHAFFER_COLLATERAL_STP,
)
from wuyun.neuron.neuron_base import (
    PLACE_CELL_PARAMS,
    BASKET_PV_PARAMS,
)
from wuyun.core.population import NeuronPopulation
from wuyun.core.synapse_group import SynapseGroup

# SpikeType 整数编码
_NONE = 0
_REGULAR = SpikeType.REGULAR.value
_BURST_START = SpikeType.BURST_START.value
_BURST_CONTINUE = SpikeType.BURST_CONTINUE.value
_BURST_END = SpikeType.BURST_END.value


class CA1Network:
    """CA1 — 比较器与输出层 (向量化版本)

    利用双区室神经元的 regular/burst 机制天然实现匹配/新奇检测:
    - CA3→basal: 前馈回忆信号
    - EC-III→apical: 反馈感知信号
    - 两者同时 → burst (匹配), 只有 basal → regular (新奇)

    这是预测编码在海马中的直接体现:
    - burst 比率高 → 回忆与感知一致 → 熟悉
    - regular 比率高 → 回忆与感知不一致 → 新奇
    """

    def __init__(
        self,
        n_pyramidal: int = 50,
        n_inhibitory: int = 8,
        schaffer_gain: float = 20.0,
        ec3_gain: float = 15.0,
        pv_gain: float = 20.0,
        n_ca3: int = 50,
        n_ec_inputs: int = 16,
        seed: int = 42,
    ):
        self.n_pyramidal = n_pyramidal
        self.n_inhibitory = n_inhibitory
        self.schaffer_gain = schaffer_gain
        self.ec3_gain = ec3_gain
        self.pv_gain = pv_gain
        self.n_ca3 = n_ca3
        self.n_ec_inputs = n_ec_inputs
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        # === 向量化神经元群体 ===
        self.pyramidal_pop = NeuronPopulation(n_pyramidal, PLACE_CELL_PARAMS)
        self.pv_pop = NeuronPopulation(n_inhibitory, BASKET_PV_PARAMS)

        # === STP 实例 (CA3→CA1 Schaffer collateral) ===
        self._schaffer_stp: Dict[int, ShortTermPlasticity] = {}
        self._schaffer_conn: Dict[int, List[int]] = {}

        # === EC-III→CA1 直接通路连接矩阵 ===
        self._ec3_conn = np.zeros((n_ec_inputs, n_pyramidal))

        # === 构建连接 ===
        self._build_schaffer_connections()
        self._build_ec3_connections()
        self._build_synapse_groups()

        # === 追踪 ===
        self._step_count = 0

    def _build_schaffer_connections(self) -> None:
        """构建 CA3→CA1 Schaffer collateral 连接 (p=0.3)

        附加 SCHAFFER_COLLATERAL_STP (轻度抑制型)
        Target: BASAL (前馈回忆信号)
        """
        for ca3_idx in range(self.n_ca3):
            targets = []
            for ca1_idx in range(self.n_pyramidal):
                if self._rng.random() < 0.3:
                    targets.append(ca1_idx)
                    key = ca3_idx * 10000 + ca1_idx
                    self._schaffer_stp[key] = ShortTermPlasticity(
                        SCHAFFER_COLLATERAL_STP
                    )
            self._schaffer_conn[ca3_idx] = targets

    def _build_ec3_connections(self) -> None:
        """构建 EC-III→CA1 穿通纤维连接 (p=0.25)

        Target: APICAL (★反馈通路 — 感知信号)
        """
        for i in range(self.n_ec_inputs):
            for j in range(self.n_pyramidal):
                if self._rng.random() < 0.25:
                    self._ec3_conn[i, j] = 0.3 + 0.4 * self._rng.random()

    def _build_synapse_groups(self) -> None:
        """构建内部突触组"""
        # CA1 → PV (全连接, AMPA)
        n_ca1pv = self.n_pyramidal * self.n_inhibitory
        ca1pv_pre = np.repeat(np.arange(self.n_pyramidal), self.n_inhibitory)
        ca1pv_post = np.tile(np.arange(self.n_inhibitory), self.n_pyramidal)
        self.ca1_pv_syn = SynapseGroup(
            pre_ids=ca1pv_pre,
            post_ids=ca1pv_post,
            weights=np.full(n_ca1pv, 0.5),
            delays=np.ones(n_ca1pv, dtype=np.int32),
            synapse_type=SynapseType.AMPA,
            target=CompartmentType.SOMA,
            tau_decay=AMPA_PARAMS.tau_decay,
            e_rev=AMPA_PARAMS.e_rev,
            g_max=AMPA_PARAMS.g_max,
            n_post=self.n_inhibitory,
        )

        # PV → CA1 (全连接, GABA_A)
        n_pvca1 = self.n_inhibitory * self.n_pyramidal
        pvca1_pre = np.repeat(np.arange(self.n_inhibitory), self.n_pyramidal)
        pvca1_post = np.tile(np.arange(self.n_pyramidal), self.n_inhibitory)
        self.pv_ca1_syn = SynapseGroup(
            pre_ids=pvca1_pre,
            post_ids=pvca1_post,
            weights=np.full(n_pvca1, 0.6),
            delays=np.ones(n_pvca1, dtype=np.int32),
            synapse_type=SynapseType.GABA_A,
            target=CompartmentType.SOMA,
            tau_decay=GABA_A_PARAMS.tau_decay,
            e_rev=GABA_A_PARAMS.e_rev,
            g_max=GABA_A_PARAMS.g_max,
            n_post=self.n_pyramidal,
        )

    # =========================================================================
    # 外部接口
    # =========================================================================

    def inject_schaffer_input(
        self,
        ca3_spikes: List[Spike],
        ca3_rates: np.ndarray,
    ) -> None:
        """注入 CA3→CA1 Schaffer collateral 输入

        使用 STP 调制: Schaffer 是轻度抑制型 (高频时效能下降)
        Target: BASAL (前馈回忆)

        Args:
            ca3_spikes: CA3 锥体细胞的输出脉冲列表
            ca3_rates: CA3 锥体细胞发放率向量, shape=(n_ca3,)
        """
        active_ca3 = set()
        for spike in ca3_spikes:
            ca3_idx = spike.source_id - 3000
            if 0 <= ca3_idx < self.n_ca3:
                active_ca3.add(ca3_idx)

        schaffer_gain = self.schaffer_gain
        for ca3_idx in range(self.n_ca3):
            targets = self._schaffer_conn.get(ca3_idx, [])
            for ca1_idx in targets:
                key = ca3_idx * 10000 + ca1_idx
                stp = self._schaffer_stp.get(key)
                if stp is None:
                    continue
                stp.step()
                if ca3_idx in active_ca3:
                    efficacy = stp.on_spike()
                    self.pyramidal_pop.i_basal[ca1_idx] += schaffer_gain * efficacy

    def inject_ec3_input(self, pattern: np.ndarray) -> None:
        """注入 EC-III 直接感知输入

        Target: APICAL (★反馈通路)
        这是匹配/新奇检测的关键: EC-III→apical + CA3→basal → burst

        Args:
            pattern: EC-III 输入向量, shape=(n_ec_inputs,)
        """
        currents = pattern @ self._ec3_conn * self.ec3_gain
        mask = currents > 0
        self.pyramidal_pop.i_apical[mask] += currents[mask]

    def step(self, t: int) -> None:
        """推进一个时间步

        Args:
            t: 当前仿真时间步
        """
        self._step_count += 1

        # --- 1. 更新 CA1 锥体细胞 ---
        self.pyramidal_pop.step(t)

        # --- 2. CA1→PV 传递 ---
        self.ca1_pv_syn.deliver_spikes(
            self.pyramidal_pop.fired, self.pyramidal_pop.spike_type)
        i_ca1pv = self.ca1_pv_syn.step_and_compute(self.pv_pop.v_soma)
        self.pv_pop.i_soma += i_ca1pv

        # --- 3. 更新 PV ---
        self.pv_pop.step(t)

        # PV 发放 → 全局抑制电流
        pv_firing = self.pv_pop.fired.sum()
        if pv_firing > 0:
            self.pyramidal_pop.i_soma -= pv_firing * self.pv_gain

        # --- 4. PV→CA1 传递 ---
        self.pv_ca1_syn.deliver_spikes(
            self.pv_pop.fired, self.pv_pop.spike_type)
        i_pvca1 = self.pv_ca1_syn.step_and_compute(self.pyramidal_pop.v_soma)
        self.pyramidal_pop.i_soma += i_pvca1

    # =========================================================================
    # 匹配/新奇检测
    # =========================================================================

    def get_match_signal(self) -> float:
        """获取匹配信号 (burst 比率)

        burst = CA3 回忆 + EC-III 感知 同时激活 → 预测匹配

        Returns:
            匹配度 [0, 1], burst/(burst+regular)
        """
        st = self.pyramidal_pop.spike_type
        active = st != _NONE
        total_active = int(active.sum())
        if total_active == 0:
            return 0.0
        is_burst = (st == _BURST_START) | (st == _BURST_CONTINUE) | (st == _BURST_END)
        return int(is_burst.sum()) / total_active

    def get_novelty_signal(self) -> float:
        """获取新奇信号 (regular 比率)

        regular = 只有 CA3 回忆, 无 EC-III 感知 → 预测误差/新奇

        Returns:
            新奇度 [0, 1], regular/(burst+regular)
        """
        st = self.pyramidal_pop.spike_type
        active = st != _NONE
        total_active = int(active.sum())
        if total_active == 0:
            return 0.0
        return int((st == _REGULAR).sum()) / total_active

    def get_activity(self) -> np.ndarray:
        """获取 CA1 锥体细胞活跃状态 (binary)

        Returns:
            binary 向量, shape=(n_pyramidal,)
        """
        return self.pyramidal_pop.fired.astype(np.float64)

    def get_output(self) -> np.ndarray:
        """获取 CA1 输出 (发放率向量, → EC-V 记忆巩固)

        Returns:
            发放率向量, shape=(n_pyramidal,), 单位 Hz
        """
        return self.pyramidal_pop.get_firing_rates(
            window_ms=1000, current_time=self._step_count)

    def get_mean_rate(self) -> float:
        """获取 CA1 平均发放率 (Hz)"""
        return float(np.mean(self.get_output()))

    def get_burst_ratio(self) -> float:
        """获取整体 burst 比率 (窗口统计)

        Returns:
            burst 比率 [0, 1]
        """
        st = self.pyramidal_pop.spike_type
        active = st != _NONE
        total_active = int(active.sum())
        if total_active == 0:
            return 0.0
        is_burst = (st == _BURST_START) | (st == _BURST_CONTINUE) | (st == _BURST_END)
        return int(is_burst.sum()) / total_active

    def reset(self) -> None:
        """重置所有状态"""
        self.pyramidal_pop.reset()
        self.pv_pop.reset()
        self.ca1_pv_syn.reset()
        self.pv_ca1_syn.reset()
        for stp in self._schaffer_stp.values():
            stp.reset()
        self._step_count = 0

    def __repr__(self) -> str:
        return (
            f"CA1Network(pyramidal={self.n_pyramidal}, pv={self.n_inhibitory}, "
            f"match={self.get_match_signal():.2f}, "
            f"novelty={self.get_novelty_signal():.2f})"
        )