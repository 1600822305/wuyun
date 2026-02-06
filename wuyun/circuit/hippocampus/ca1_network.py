"""
3-C: CA1Network — 比较器与输出层

核心功能: 比较 CA3 回忆与 EC-III 直接感知，检测新奇/匹配
生物学: CA1 同时接收 CA3 (via Schaffer) 和 EC-III (via 穿通纤维)

预测编码对应 (★核心创新):
  - CA3→basal (前馈回忆) + EC-III→apical (反馈感知) 同时 → BURST = 匹配
  - 只有 CA3→basal                                        → REGULAR = 新奇
  - 只有 EC-III→apical                                    → 沉默 (感知但无回忆)

组成:
  - n_pyramidal 个 PLACE_CELL 神经元 (兴奋性, 双区室 κ=0.3)
  - n_inhibitory 个 BASKET_PV 神经元 (抑制性)
  - 内部 SpikeBus
  - CA3→CA1: Schaffer collateral (SCHAFFER_COLLATERAL_STP) → BASAL
  - EC-III→CA1: 穿通纤维 (直接感知) → APICAL
  - CA1→PV / PV→CA1: 反馈抑制

依赖: spike/ ← synapse/ ← neuron/ ← 本模块
"""

from typing import List, Dict, Optional
import numpy as np

from wuyun.spike.spike import Spike
from wuyun.spike.spike_bus import SpikeBus
from wuyun.spike.signal_types import (
    SpikeType,
    SynapseType,
    CompartmentType,
)
from wuyun.synapse.synapse_base import SynapseBase, AMPA_PARAMS, GABA_A_PARAMS
from wuyun.synapse.plasticity.inhibitory_stdp import InhibitorySTDP
from wuyun.synapse.short_term_plasticity import (
    ShortTermPlasticity,
    SCHAFFER_COLLATERAL_STP,
)
from wuyun.neuron.neuron_base import (
    NeuronBase,
    PLACE_CELL_PARAMS,
    BASKET_PV_PARAMS,
)


class CA1Network:
    """CA1 — 比较器与输出层

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

        # === 创建神经元 ===
        # CA1 锥体细胞: ID 范围 [4000, 4000+n_pyramidal)
        self.pyramidal_neurons: List[NeuronBase] = []
        for i in range(n_pyramidal):
            n = NeuronBase(neuron_id=4000 + i, params=PLACE_CELL_PARAMS)
            self.pyramidal_neurons.append(n)

        # PV 抑制性神经元: ID 范围 [4500, 4500+n_inhibitory)
        self.pv_neurons: List[NeuronBase] = []
        for i in range(n_inhibitory):
            n = NeuronBase(neuron_id=4500 + i, params=BASKET_PV_PARAMS)
            self.pv_neurons.append(n)

        # === 内部 SpikeBus ===
        self._bus = SpikeBus()

        # === 突触存储 ===
        self._synapses: List[SynapseBase] = []

        # === STP 实例 (CA3→CA1 Schaffer collateral) ===
        self._schaffer_stp: Dict[int, ShortTermPlasticity] = {}

        # === 连接矩阵 ===
        # CA3→CA1 Schaffer 连接: schaffer_conn[ca3_idx] = [ca1_idx_list]
        self._schaffer_conn: Dict[int, List[int]] = {}
        # EC-III→CA1 直接通路连接矩阵
        self._ec3_conn = np.zeros((n_ec_inputs, n_pyramidal))

        # === 构建连接 ===
        self._build_schaffer_connections()
        self._build_ec3_connections()
        self._build_inhibitory_connections()

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

    def _build_inhibitory_connections(self) -> None:
        """构建 CA1→PV 和 PV→CA1 连接"""
        inh_stdp = InhibitorySTDP()

        # CA1 → PV (全连接, AMPA)
        for ca1 in self.pyramidal_neurons:
            for pv in self.pv_neurons:
                syn = SynapseBase(
                    pre_id=ca1.id,
                    post_id=pv.id,
                    weight=0.5,
                    delay=1,
                    synapse_type=SynapseType.AMPA,
                    target_compartment=CompartmentType.SOMA,
                    params=AMPA_PARAMS,
                )
                pv.add_synapse(syn)
                self._bus.register_synapse(syn)
                self._synapses.append(syn)

        # PV → CA1 (全连接, GABA_A + InhibitorySTDP)
        for pv in self.pv_neurons:
            for ca1 in self.pyramidal_neurons:
                syn = SynapseBase(
                    pre_id=pv.id,
                    post_id=ca1.id,
                    weight=0.6,
                    delay=1,
                    synapse_type=SynapseType.GABA_A,
                    target_compartment=CompartmentType.SOMA,
                    plasticity_rule=inh_stdp,
                    params=GABA_A_PARAMS,
                )
                ca1.add_synapse(syn)
                self._bus.register_synapse(syn)
                self._synapses.append(syn)

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
                    current = self.schaffer_gain * efficacy
                    self.pyramidal_neurons[ca1_idx].inject_basal_current(current)

    def inject_ec3_input(self, pattern: np.ndarray) -> None:
        """注入 EC-III 直接感知输入

        Target: APICAL (★反馈通路)
        这是匹配/新奇检测的关键: EC-III→apical + CA3→basal → burst

        Args:
            pattern: EC-III 输入向量, shape=(n_ec_inputs,)
        """
        currents = pattern @ self._ec3_conn
        for i, neuron in enumerate(self.pyramidal_neurons):
            if currents[i] > 0:
                neuron.inject_apical_current(currents[i] * self.ec3_gain)

    def step(self, t: int) -> None:
        """推进一个时间步

        Args:
            t: 当前仿真时间步
        """
        self._step_count += 1

        # 更新 CA1 锥体细胞
        for neuron in self.pyramidal_neurons:
            spike_type = neuron.step(t)
            if spike_type.is_active:
                self._bus.emit(Spike(neuron.id, t, spike_type))

        # 更新 PV 细胞
        for neuron in self.pv_neurons:
            spike_type = neuron.step(t)
            if spike_type.is_active:
                self._bus.emit(Spike(neuron.id, t, spike_type))
                for ca1 in self.pyramidal_neurons:
                    ca1.inject_somatic_current(-self.pv_gain)

        # SpikeBus 分发
        self._bus.step(t)

    # =========================================================================
    # 匹配/新奇检测
    # =========================================================================

    def get_match_signal(self) -> float:
        """获取匹配信号 (burst 比率)

        burst = CA3 回忆 + EC-III 感知 同时激活 → 预测匹配

        Returns:
            匹配度 [0, 1], burst/(burst+regular)
        """
        total_active = 0
        total_burst = 0
        for n in self.pyramidal_neurons:
            st = n.current_spike_type
            if st.is_active:
                total_active += 1
                if st.is_burst:
                    total_burst += 1
        return total_burst / total_active if total_active > 0 else 0.0

    def get_novelty_signal(self) -> float:
        """获取新奇信号 (regular 比率)

        regular = 只有 CA3 回忆, 无 EC-III 感知 → 预测误差/新奇

        Returns:
            新奇度 [0, 1], regular/(burst+regular)
        """
        total_active = 0
        total_regular = 0
        for n in self.pyramidal_neurons:
            st = n.current_spike_type
            if st.is_active:
                total_active += 1
                if st == SpikeType.REGULAR:
                    total_regular += 1
        return total_regular / total_active if total_active > 0 else 0.0

    def get_activity(self) -> np.ndarray:
        """获取 CA1 锥体细胞活跃状态 (binary)

        Returns:
            binary 向量, shape=(n_pyramidal,)
        """
        return np.array([
            1.0 if n.current_spike_type.is_active else 0.0
            for n in self.pyramidal_neurons
        ])

    def get_output(self) -> np.ndarray:
        """获取 CA1 输出 (发放率向量, → EC-V 记忆巩固)

        Returns:
            发放率向量, shape=(n_pyramidal,), 单位 Hz
        """
        return np.array([n.firing_rate for n in self.pyramidal_neurons])

    def get_mean_rate(self) -> float:
        """获取 CA1 平均发放率 (Hz)"""
        return float(np.mean(self.get_output()))

    def get_burst_ratio(self) -> float:
        """获取整体 burst 比率 (窗口统计)

        Returns:
            burst 比率 [0, 1]
        """
        ratios = [n.burst_ratio for n in self.pyramidal_neurons]
        active_ratios = [r for r in ratios if r > 0]
        return float(np.mean(active_ratios)) if active_ratios else 0.0

    def reset(self) -> None:
        """重置所有状态"""
        for n in self.pyramidal_neurons:
            n.reset()
        for n in self.pv_neurons:
            n.reset()
        for s in self._synapses:
            s.reset()
        for stp in self._schaffer_stp.values():
            stp.reset()
        self._bus.reset()
        self._step_count = 0

    def __repr__(self) -> str:
        return (
            f"CA1Network(pyramidal={self.n_pyramidal}, pv={self.n_inhibitory}, "
            f"match={self.get_match_signal():.2f}, "
            f"novelty={self.get_novelty_signal():.2f})"
        )