"""
3-A: DentateGyrus — 齿状回模式分离器 (向量化版本)

核心功能: 将 EC 输入空间的相似模式映射为正交化稀疏表示
生物学: DG 颗粒细胞层是人脑中最稀疏的编码之一 (~2% 激活率)

组成 (向量化):
  - granule_pop: NeuronPopulation (n_granule, GRANULE)
  - pv_pop: NeuronPopulation (n_inhibitory, BASKET_PV)
  - g2pv_syn: SynapseGroup (Granule→PV, AMPA)
  - pv2g_syn: SynapseGroup (PV→Granule, GABA_A)
  - EC→Granule: 矩阵乘法电流注入

依赖: core/ (NeuronPopulation, SynapseGroup)
"""

from typing import List
import numpy as np

from wuyun.spike.spike import Spike
from wuyun.spike.signal_types import (
    SpikeType,
    SynapseType,
    CompartmentType,
)
from wuyun.synapse.synapse_base import AMPA_PARAMS, GABA_A_PARAMS
from wuyun.neuron.neuron_base import GRANULE_PARAMS, BASKET_PV_PARAMS
from wuyun.core.population import NeuronPopulation
from wuyun.core.synapse_group import SynapseGroup


class DentateGyrus:
    """齿状回 — 模式分离器 (向量化版本)

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

        # === 向量化神经元群体 ===
        self.granule_pop = NeuronPopulation(n_granule, GRANULE_PARAMS)
        self.pv_pop = NeuronPopulation(n_inhibitory, BASKET_PV_PARAMS)

        # === EC→Granule 连接矩阵 ===
        self._ec_conn = np.zeros((n_ec_inputs, n_granule))
        self._build_ec_connections()

        # === 内部突触组 ===
        self._build_synapse_groups()

        # === 发放率追踪 ===
        self._granule_spike_counts = np.zeros(n_granule)
        self._step_count = 0

    def _build_ec_connections(self) -> None:
        """构建 EC→Granule 随机发散连接矩阵"""
        for i in range(self.n_ec_inputs):
            for j in range(self.n_granule):
                if self._rng.random() < self.ec_granule_prob:
                    self._ec_conn[i, j] = 0.3 + 0.4 * self._rng.random()

    def _build_synapse_groups(self) -> None:
        """构建内部突触组

        Granule→PV: 全连接, AMPA, target=SOMA
        PV→Granule: 全连接, GABA_A, target=SOMA
        """
        # Granule → PV: n_granule × n_inhibitory 全连接
        n_g2pv = self.n_granule * self.n_inhibitory
        g2pv_pre = np.repeat(np.arange(self.n_granule), self.n_inhibitory)
        g2pv_post = np.tile(np.arange(self.n_inhibitory), self.n_granule)
        self.g2pv_syn = SynapseGroup(
            pre_ids=g2pv_pre,
            post_ids=g2pv_post,
            weights=np.full(n_g2pv, 0.5),
            delays=np.ones(n_g2pv, dtype=np.int32),
            synapse_type=SynapseType.AMPA,
            target=CompartmentType.SOMA,
            tau_decay=AMPA_PARAMS.tau_decay,
            e_rev=AMPA_PARAMS.e_rev,
            g_max=AMPA_PARAMS.g_max,
            n_post=self.n_inhibitory,
        )

        # PV → Granule: n_inhibitory × n_granule 全连接
        n_pv2g = self.n_inhibitory * self.n_granule
        pv2g_pre = np.repeat(np.arange(self.n_inhibitory), self.n_granule)
        pv2g_post = np.tile(np.arange(self.n_granule), self.n_inhibitory)
        self.pv2g_syn = SynapseGroup(
            pre_ids=pv2g_pre,
            post_ids=pv2g_post,
            weights=np.full(n_pv2g, 0.6),
            delays=np.ones(n_pv2g, dtype=np.int32),
            synapse_type=SynapseType.GABA_A,
            target=CompartmentType.SOMA,
            tau_decay=GABA_A_PARAMS.tau_decay,
            e_rev=GABA_A_PARAMS.e_rev,
            g_max=GABA_A_PARAMS.g_max,
            n_post=self.n_granule,
        )

    # =========================================================================
    # 外部接口
    # =========================================================================

    def inject_ec_input(self, pattern: np.ndarray) -> None:
        """注入 EC-II 输入模式

        将 EC 输入向量通过随机连接矩阵转换为颗粒细胞的电流注入。

        Args:
            pattern: EC 输入向量, shape=(n_ec_inputs,), 值范围 [0, 1]
        """
        currents = pattern @ self._ec_conn * self.ec_granule_gain
        self.granule_pop.i_basal += currents

    def step(self, t: int) -> None:
        """推进一个时间步

        1. 更新颗粒细胞
        2. Granule→PV 传递
        3. 更新 PV 细胞
        4. PV→Granule 传递 + 全局抑制
        5. 记录发放

        Args:
            t: 当前仿真时间步
        """
        self._step_count += 1

        # --- 1. 更新颗粒细胞 ---
        self.granule_pop.step(t)

        # --- 2. Granule→PV 传递 ---
        self.g2pv_syn.deliver_spikes(self.granule_pop.fired, self.granule_pop.spike_type)
        i_g2pv = self.g2pv_syn.step_and_compute(self.pv_pop.v_soma)
        self.pv_pop.i_soma += i_g2pv

        # --- 3. 更新 PV 细胞 ---
        self.pv_pop.step(t)

        # --- 4. PV→Granule 传递 + 全局抑制 ---
        self.pv2g_syn.deliver_spikes(self.pv_pop.fired, self.pv_pop.spike_type)
        i_pv2g = self.pv2g_syn.step_and_compute(self.granule_pop.v_soma)
        self.granule_pop.i_soma += i_pv2g

        # PV 发放 → 额外全局抑制电流
        pv_firing = self.pv_pop.fired.sum()
        if pv_firing > 0:
            self.granule_pop.i_soma -= pv_firing * self.pv_granule_gain

        # --- 5. 记录发放 ---
        self._granule_spike_counts += self.granule_pop.fired.astype(np.float64)

    def get_granule_activity(self) -> np.ndarray:
        """获取颗粒细胞当前活跃状态 (binary)

        Returns:
            binary 向量, shape=(n_granule,), 1=最近发放, 0=沉默
        """
        return self.granule_pop.fired.astype(np.float64)

    def get_granule_rates(self) -> np.ndarray:
        """获取颗粒细胞发放率向量

        Returns:
            发放率向量, shape=(n_granule,), 单位 Hz
        """
        return self.granule_pop.get_firing_rates(
            window_ms=1000, current_time=self._step_count)

    def get_output_spikes(self) -> List[Spike]:
        """获取当前时间步的颗粒细胞输出脉冲 (→ CA3 mossy fiber)

        Returns:
            活跃颗粒细胞的 Spike 列表
        """
        spikes = []
        fired_idx = np.nonzero(self.granule_pop.fired)[0]
        for i in fired_idx:
            spikes.append(Spike(
                source_id=1000 + i,
                timestamp=self._step_count,
                spike_type=SpikeType(int(self.granule_pop.spike_type[i])),
            ))
        return spikes

    def get_sparsity(self) -> float:
        """获取当前激活率 (目标 ~2-5%)

        Returns:
            当前活跃颗粒细胞比例 [0, 1]
        """
        return float(self.granule_pop.fired.sum()) / self.n_granule

    def get_mean_rate(self) -> float:
        """获取颗粒细胞平均发放率 (Hz)"""
        return float(self.get_granule_rates().mean())

    def reset(self) -> None:
        """重置所有状态"""
        self.granule_pop.reset()
        self.pv_pop.reset()
        self.g2pv_syn.reset()
        self.pv2g_syn.reset()
        self._granule_spike_counts[:] = 0
        self._step_count = 0

    def __repr__(self) -> str:
        return (
            f"DentateGyrus(ec={self.n_ec_inputs}, granule={self.n_granule}, "
            f"pv={self.n_inhibitory}, sparsity={self.get_sparsity():.3f})"
        )