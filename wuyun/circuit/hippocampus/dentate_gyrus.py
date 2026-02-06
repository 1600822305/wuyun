"""
3-A: DentateGyrus — 齿状回模式分离器

核心功能: 将 EC 输入空间的相似模式映射为正交化稀疏表示
生物学: DG 颗粒细胞层是人脑中最稀疏的编码之一 (~2% 激活率)

组成:
  - n_granule 个 GRANULE 神经元 (兴奋性, 高阈值 -40mV)
  - n_inhibitory 个 BASKET_PV 神经元 (抑制性, 强全局抑制)
  - 内部 SpikeBus
  - EC→Granule: 发散连接 (每个 EC 输入连接多个 granule, 随机权重)
  - Granule→PV: 兴奋性连接
  - PV→Granule: 全局抑制 (→ 竞争性稀疏)

依赖: spike/ ← synapse/ ← neuron/ ← 本模块
"""

from typing import List, Optional
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
from wuyun.neuron.neuron_base import (
    NeuronBase,
    GRANULE_PARAMS,
    BASKET_PV_PARAMS,
)


class DentateGyrus:
    """齿状回 — 模式分离器

    将高维 EC 输入通过展开编码 (n_granule >> n_ec) + 竞争抑制
    映射为极稀疏的正交化表示。

    关键机制:
    1. 展开编码: n_ec=16 → n_granule=100 (6倍展开)
    2. 随机发散连接: 每个 EC 输入随机连接 ~30% 的颗粒细胞
    3. 高阈值: 颗粒细胞 v_threshold=-40mV (比标准 -50mV 高 10mV)
    4. 强全局抑制: PV→Granule 强抑制确保只有最强输入的细胞存活

    效果: 两个相似输入 (cosine > 0.8) → DG 输出相似度大幅降低
    """

    def __init__(
        self,
        n_ec_inputs: int = 16,
        n_granule: int = 100,
        n_inhibitory: int = 10,
        ec_granule_prob: float = 0.3,
        ec_granule_gain: float = 25.0,
        pv_granule_gain: float = 25.0,
        seed: int = 42,
    ):
        self.n_ec_inputs = n_ec_inputs
        self.n_granule = n_granule
        self.n_inhibitory = n_inhibitory
        self.ec_granule_prob = ec_granule_prob
        self.ec_granule_gain = ec_granule_gain
        self.pv_granule_gain = pv_granule_gain
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        # === 创建神经元 ===
        # Granule 细胞: ID 范围 [1000, 1000+n_granule)
        self.granule_neurons: List[NeuronBase] = []
        for i in range(n_granule):
            n = NeuronBase(neuron_id=1000 + i, params=GRANULE_PARAMS)
            self.granule_neurons.append(n)

        # PV 抑制性神经元: ID 范围 [2000, 2000+n_inhibitory)
        self.pv_neurons: List[NeuronBase] = []
        for i in range(n_inhibitory):
            n = NeuronBase(neuron_id=2000 + i, params=BASKET_PV_PARAMS)
            self.pv_neurons.append(n)

        # === 内部 SpikeBus ===
        self._bus = SpikeBus()

        # === 创建内部连接 ===
        self._synapses: List[SynapseBase] = []
        self._build_internal_connections()

        # === EC→Granule 连接矩阵 (随机, 用于电流注入) ===
        # ec_conn[i][j] = weight if EC_i → Granule_j connected, else 0
        self._ec_conn = np.zeros((n_ec_inputs, n_granule))
        self._build_ec_connections()

        # === 发放率追踪 ===
        self._granule_spike_counts = np.zeros(n_granule)
        self._step_count = 0

    def _build_ec_connections(self) -> None:
        """构建 EC→Granule 随机发散连接矩阵"""
        for i in range(self.n_ec_inputs):
            for j in range(self.n_granule):
                if self._rng.random() < self.ec_granule_prob:
                    # 随机权重 [0.3, 0.7]
                    self._ec_conn[i, j] = 0.3 + 0.4 * self._rng.random()

    def _build_internal_connections(self) -> None:
        """构建内部突触连接

        Granule→PV: 全连接, AMPA, target=SOMA
        PV→Granule: 全连接, GABA_A, target=SOMA + InhibitorySTDP
        """
        inh_stdp = InhibitorySTDP()

        # Granule → PV (兴奋性, 全连接)
        for g in self.granule_neurons:
            for pv in self.pv_neurons:
                syn = SynapseBase(
                    pre_id=g.id,
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

        # PV → Granule (抑制性, 全连接 + InhibitorySTDP)
        for pv in self.pv_neurons:
            for g in self.granule_neurons:
                syn = SynapseBase(
                    pre_id=pv.id,
                    post_id=g.id,
                    weight=0.6,
                    delay=1,
                    synapse_type=SynapseType.GABA_A,
                    target_compartment=CompartmentType.SOMA,
                    plasticity_rule=inh_stdp,
                    params=GABA_A_PARAMS,
                )
                g.add_synapse(syn)
                self._bus.register_synapse(syn)
                self._synapses.append(syn)

    # =========================================================================
    # 外部接口
    # =========================================================================

    def inject_ec_input(self, pattern: np.ndarray) -> None:
        """注入 EC-II 输入模式

        将 EC 输入向量通过随机连接矩阵转换为颗粒细胞的电流注入。

        Args:
            pattern: EC 输入向量, shape=(n_ec_inputs,), 值范围 [0, 1]
        """
        # pattern @ ec_conn → granule currents, shape=(n_granule,)
        currents = pattern @ self._ec_conn  # (n_ec,) @ (n_ec, n_granule) → (n_granule,)
        for i, neuron in enumerate(self.granule_neurons):
            if currents[i] > 0:
                neuron.inject_basal_current(currents[i] * self.ec_granule_gain)

    def step(self, t: int) -> None:
        """推进一个时间步

        1. 更新所有颗粒细胞
        2. 更新所有 PV 细胞
        3. SpikeBus 分发脉冲
        4. 记录发放

        Args:
            t: 当前仿真时间步
        """
        self._step_count += 1

        # 更新颗粒细胞
        for i, neuron in enumerate(self.granule_neurons):
            spike_type = neuron.step(t)
            if spike_type.is_active:
                self._bus.emit(Spike(neuron.id, t, spike_type))
                self._granule_spike_counts[i] += 1

        # 更新 PV 细胞
        for neuron in self.pv_neurons:
            spike_type = neuron.step(t)
            if spike_type.is_active:
                self._bus.emit(Spike(neuron.id, t, spike_type))
                # PV 发放 → 注入全局抑制电流到所有颗粒细胞
                for g in self.granule_neurons:
                    g.inject_somatic_current(-self.pv_granule_gain)

        # SpikeBus 分发
        self._bus.step(t)

    def get_granule_activity(self) -> np.ndarray:
        """获取颗粒细胞当前活跃状态 (binary)

        Returns:
            binary 向量, shape=(n_granule,), 1=最近发放, 0=沉默
        """
        return np.array([
            1.0 if n.current_spike_type.is_active else 0.0
            for n in self.granule_neurons
        ])

    def get_granule_rates(self) -> np.ndarray:
        """获取颗粒细胞发放率向量

        Returns:
            发放率向量, shape=(n_granule,), 单位 Hz
        """
        return np.array([n.firing_rate for n in self.granule_neurons])

    def get_output_spikes(self) -> List[Spike]:
        """获取当前时间步的颗粒细胞输出脉冲 (→ CA3 mossy fiber)

        Returns:
            活跃颗粒细胞的 Spike 列表
        """
        spikes = []
        for n in self.granule_neurons:
            if n.current_spike_type.is_active:
                spikes.append(Spike(
                    source_id=n.id,
                    timestamp=self._step_count,
                    spike_type=n.current_spike_type,
                ))
        return spikes

    def get_sparsity(self) -> float:
        """获取当前激活率 (目标 ~2-5%)

        Returns:
            当前活跃颗粒细胞比例 [0, 1]
        """
        active = sum(
            1 for n in self.granule_neurons
            if n.current_spike_type.is_active
        )
        return active / self.n_granule if self.n_granule > 0 else 0.0

    def get_mean_rate(self) -> float:
        """获取颗粒细胞平均发放率 (Hz)"""
        rates = self.get_granule_rates()
        return float(np.mean(rates))

    def reset(self) -> None:
        """重置所有状态"""
        for n in self.granule_neurons:
            n.reset()
        for n in self.pv_neurons:
            n.reset()
        for s in self._synapses:
            s.reset()
        self._bus.reset()
        self._granule_spike_counts[:] = 0
        self._step_count = 0

    def __repr__(self) -> str:
        return (
            f"DentateGyrus(ec={self.n_ec_inputs}, granule={self.n_granule}, "
            f"pv={self.n_inhibitory}, sparsity={self.get_sparsity():.3f})"
        )