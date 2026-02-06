"""
4-A: Striatum — 纹状体 (基底节输入核, 向量化版本)

包含 D1-MSN (直接通路) 和 D2-MSN (间接通路) 两个群体 + FSI 抑制。

DA 调制效应 (物理属性, 非硬编码决策):
- D1 受体: DA 增强 MSN 兴奋性 (升高 Up-state 概率)
- D2 受体: DA 降低 MSN 兴奋性 (降低 Up-state 概率)

内部连接 (向量化 SynapseGroup):
- FSI→D1 (GABA_A, 全连接, weight=0.5)
- FSI→D2 (GABA_A, 全连接, weight=0.5)
- D1→D1 侧向抑制 (GABA_A, p=0.15, weight=0.1)
- D2→D2 侧向抑制 (GABA_A, p=0.15, weight=0.1)

依赖: core/ (NeuronPopulation, SynapseGroup)
"""

from typing import List, Optional
import numpy as np

from wuyun.spike.spike import Spike
from wuyun.spike.signal_types import (
    SpikeType,
    SynapseType,
    CompartmentType,
)
from wuyun.synapse.synapse_base import AMPA_PARAMS, GABA_A_PARAMS
from wuyun.synapse.plasticity.da_modulated_stdp import (
    DAModulatedSTDP,
    DAModulatedSTDPParams,
)
from wuyun.neuron.neuron_base import (
    MSN_D1_PARAMS,
    MSN_D2_PARAMS,
    BASKET_PV_PARAMS,
)
from wuyun.core.population import NeuronPopulation
from wuyun.core.synapse_group import SynapseGroup


class Striatum:
    """纹状体 — 基底节输入核 (向量化版本)

    包含 D1-MSN (直接通路) 和 D2-MSN (间接通路) 两个群体。
    接收皮层谷氨酸兴奋性输入 + DA 调制。

    Attributes:
        d1_pop: NeuronPopulation (D1-MSN, 直接通路)
        d2_pop: NeuronPopulation (D2-MSN, 间接通路)
        fsi_pop: NeuronPopulation (FSI, PV+ 抑制性)
    """

    def __init__(
        self,
        n_d1: int = 20,
        n_d2: int = 20,
        n_fsi: int = 8,
        da_gain_d1: float = 10.0,
        da_gain_d2: float = 10.0,
        seed: int = None,
    ):
        self.n_d1 = n_d1
        self.n_d2 = n_d2
        self.n_fsi = n_fsi
        self.da_gain_d1 = da_gain_d1
        self.da_gain_d2 = da_gain_d2
        self._rng = np.random.RandomState(seed)

        # === 向量化神经元群体 ===
        self.d1_pop = NeuronPopulation(n_d1, MSN_D1_PARAMS)
        self.d2_pop = NeuronPopulation(n_d2, MSN_D2_PARAMS)
        self.fsi_pop = NeuronPopulation(n_fsi, BASKET_PV_PARAMS)

        # === 皮层→MSN 突触 (带 DA-STDP, 分别追踪) ===
        # 这些在外部构建时添加，暂保持空列表兼容
        self._cortical_d1_synapses = []
        self._cortical_d2_synapses = []

        # === DA-STDP 规则 ===
        self._da_stdp = DAModulatedSTDP(DAModulatedSTDPParams(
            a_plus=0.005,
            a_minus=0.00525,
            tau_plus=20.0,
            tau_minus=20.0,
            tau_eligibility=1000.0,
        ))

        # === 构建内部连接 (SynapseGroup) ===
        self._build_synapse_groups()

        # === 发放率追踪 ===
        self._d1_spike_counts = np.zeros(n_d1)
        self._d2_spike_counts = np.zeros(n_d2)
        self._rate_window = 100
        self._step_count = 0
        self._d1_recent_spikes: List[List[Spike]] = []
        self._d2_recent_spikes: List[List[Spike]] = []

    def _build_synapse_groups(self) -> None:
        """构建纹状体内部连接 (向量化 SynapseGroup)"""
        # --- FSI → D1 (GABA_A, 全连接, weight=0.5) ---
        n_fsi_d1 = self.n_fsi * self.n_d1
        fsi_d1_pre = np.repeat(np.arange(self.n_fsi), self.n_d1)
        fsi_d1_post = np.tile(np.arange(self.n_d1), self.n_fsi)
        self.fsi_d1_syn = SynapseGroup(
            pre_ids=fsi_d1_pre, post_ids=fsi_d1_post,
            weights=np.full(n_fsi_d1, 0.5),
            delays=np.ones(n_fsi_d1, dtype=np.int32),
            synapse_type=SynapseType.GABA_A,
            target=CompartmentType.SOMA,
            tau_decay=GABA_A_PARAMS.tau_decay,
            e_rev=GABA_A_PARAMS.e_rev,
            g_max=GABA_A_PARAMS.g_max,
            n_post=self.n_d1,
        )

        # --- FSI → D2 (GABA_A, 全连接, weight=0.5) ---
        n_fsi_d2 = self.n_fsi * self.n_d2
        fsi_d2_pre = np.repeat(np.arange(self.n_fsi), self.n_d2)
        fsi_d2_post = np.tile(np.arange(self.n_d2), self.n_fsi)
        self.fsi_d2_syn = SynapseGroup(
            pre_ids=fsi_d2_pre, post_ids=fsi_d2_post,
            weights=np.full(n_fsi_d2, 0.5),
            delays=np.ones(n_fsi_d2, dtype=np.int32),
            synapse_type=SynapseType.GABA_A,
            target=CompartmentType.SOMA,
            tau_decay=GABA_A_PARAMS.tau_decay,
            e_rev=GABA_A_PARAMS.e_rev,
            g_max=GABA_A_PARAMS.g_max,
            n_post=self.n_d2,
        )

        # --- D1 → D1 侧向抑制 (GABA_A, p=0.15, weight=0.1) ---
        d1d1_pre_list, d1d1_post_list = [], []
        for i in range(self.n_d1):
            for j in range(self.n_d1):
                if i != j and self._rng.random() < 0.15:
                    d1d1_pre_list.append(i)
                    d1d1_post_list.append(j)
        if d1d1_pre_list:
            d1d1_pre = np.array(d1d1_pre_list, dtype=np.int32)
            d1d1_post = np.array(d1d1_post_list, dtype=np.int32)
            n_d1d1 = len(d1d1_pre_list)
        else:
            d1d1_pre = np.zeros(0, dtype=np.int32)
            d1d1_post = np.zeros(0, dtype=np.int32)
            n_d1d1 = 0
        self.d1_lateral_syn = SynapseGroup(
            pre_ids=d1d1_pre, post_ids=d1d1_post,
            weights=np.full(n_d1d1, 0.1),
            delays=np.ones(n_d1d1, dtype=np.int32),
            synapse_type=SynapseType.GABA_A,
            target=CompartmentType.SOMA,
            tau_decay=GABA_A_PARAMS.tau_decay,
            e_rev=GABA_A_PARAMS.e_rev,
            g_max=GABA_A_PARAMS.g_max,
            n_post=self.n_d1,
        )

        # --- D2 → D2 侧向抑制 (GABA_A, p=0.15, weight=0.1) ---
        d2d2_pre_list, d2d2_post_list = [], []
        for i in range(self.n_d2):
            for j in range(self.n_d2):
                if i != j and self._rng.random() < 0.15:
                    d2d2_pre_list.append(i)
                    d2d2_post_list.append(j)
        if d2d2_pre_list:
            d2d2_pre = np.array(d2d2_pre_list, dtype=np.int32)
            d2d2_post = np.array(d2d2_post_list, dtype=np.int32)
            n_d2d2 = len(d2d2_pre_list)
        else:
            d2d2_pre = np.zeros(0, dtype=np.int32)
            d2d2_post = np.zeros(0, dtype=np.int32)
            n_d2d2 = 0
        self.d2_lateral_syn = SynapseGroup(
            pre_ids=d2d2_pre, post_ids=d2d2_post,
            weights=np.full(n_d2d2, 0.1),
            delays=np.ones(n_d2d2, dtype=np.int32),
            synapse_type=SynapseType.GABA_A,
            target=CompartmentType.SOMA,
            tau_decay=GABA_A_PARAMS.tau_decay,
            e_rev=GABA_A_PARAMS.e_rev,
            g_max=GABA_A_PARAMS.g_max,
            n_post=self.n_d2,
        )

    # =========================================================================
    # 外部接口
    # =========================================================================

    def inject_cortical_input(self, input_vector: np.ndarray) -> None:
        """注入皮层 → 纹状体输入 (谷氨酸兴奋性)

        将 input_vector 映射为电流注入到 D1 和 D2 的 basal。
        D1 和 D2 接收相同的皮层输入。

        Args:
            input_vector: 皮层输入向量, 长度自动映射到 D1/D2 数量
        """
        n_in = len(input_vector)
        # 映射输入到 D1
        d1_idx = np.arange(self.n_d1) % n_in
        d1_current = input_vector[d1_idx]
        self.d1_pop.i_basal += np.maximum(d1_current, 0.0)

        # 映射输入到 D2 (相同输入)
        d2_idx = np.arange(self.n_d2) % n_in
        d2_current = input_vector[d2_idx]
        self.d2_pop.i_basal += np.maximum(d2_current, 0.0)

    def apply_dopamine(self, da_level: float) -> None:
        """DA 调制

        D1: 注入 +da_gain_d1 * da_level 电流 (兴奋)
        D2: 注入 -da_gain_d2 * da_level 电流 (抑制)

        Args:
            da_level: 多巴胺水平 (>0 = 奖励, <0 = 惩罚, 0 = 基线)
        """
        self.d1_pop.i_soma += self.da_gain_d1 * da_level
        self.d2_pop.i_soma += -self.da_gain_d2 * da_level

    def step(self, t: int) -> None:
        """推进一个时间步

        Args:
            t: 当前仿真时间步
        """
        self._step_count += 1

        # --- 1. 更新 D1 ---
        self.d1_pop.step(t)
        self._d1_spike_counts += self.d1_pop.fired.astype(np.float64)

        # --- 2. 更新 D2 ---
        self.d2_pop.step(t)
        self._d2_spike_counts += self.d2_pop.fired.astype(np.float64)

        # --- 3. D1→D1 侧向抑制 ---
        self.d1_lateral_syn.deliver_spikes(self.d1_pop.fired, self.d1_pop.spike_type)
        i_d1lat = self.d1_lateral_syn.step_and_compute(self.d1_pop.v_soma)
        self.d1_pop.i_soma += i_d1lat

        # --- 4. D2→D2 侧向抑制 ---
        self.d2_lateral_syn.deliver_spikes(self.d2_pop.fired, self.d2_pop.spike_type)
        i_d2lat = self.d2_lateral_syn.step_and_compute(self.d2_pop.v_soma)
        self.d2_pop.i_soma += i_d2lat

        # --- 5. 更新 FSI ---
        self.fsi_pop.step(t)

        # --- 6. FSI→D1 传递 ---
        self.fsi_d1_syn.deliver_spikes(self.fsi_pop.fired, self.fsi_pop.spike_type)
        i_fsi_d1 = self.fsi_d1_syn.step_and_compute(self.d1_pop.v_soma)
        self.d1_pop.i_soma += i_fsi_d1

        # --- 7. FSI→D2 传递 ---
        self.fsi_d2_syn.deliver_spikes(self.fsi_pop.fired, self.fsi_pop.spike_type)
        i_fsi_d2 = self.fsi_d2_syn.step_and_compute(self.d2_pop.v_soma)
        self.d2_pop.i_soma += i_fsi_d2

        # --- 8. 记录最近脉冲 (滑动窗口) ---
        d1_spikes = self._collect_spikes(self.d1_pop, 5000)
        d2_spikes = self._collect_spikes(self.d2_pop, 5200)
        self._d1_recent_spikes.append(d1_spikes)
        self._d2_recent_spikes.append(d2_spikes)
        if len(self._d1_recent_spikes) > self._rate_window:
            self._d1_recent_spikes.pop(0)
        if len(self._d2_recent_spikes) > self._rate_window:
            self._d2_recent_spikes.pop(0)

    @staticmethod
    def _collect_spikes(pop: NeuronPopulation, id_offset: int) -> List[Spike]:
        """从 Population 收集 Spike 列表"""
        spikes = []
        fired_idx = np.nonzero(pop.fired)[0]
        for i in fired_idx:
            spikes.append(Spike(
                id_offset + i, 0,
                SpikeType(int(pop.spike_type[i]))))
        return spikes

    def get_d1_rates(self) -> np.ndarray:
        """D1 群体发放率 (Hz)"""
        if self._step_count == 0:
            return np.zeros(self.n_d1)
        return self._d1_spike_counts * (1000.0 / self._step_count)

    def get_d2_rates(self) -> np.ndarray:
        """D2 群体发放率 (Hz)"""
        if self._step_count == 0:
            return np.zeros(self.n_d2)
        return self._d2_spike_counts * (1000.0 / self._step_count)

    def get_d1_spikes(self) -> List[Spike]:
        """获取最近一步的 D1 输出脉冲"""
        if self._d1_recent_spikes:
            return self._d1_recent_spikes[-1]
        return []

    def get_d2_spikes(self) -> List[Spike]:
        """获取最近一步的 D2 输出脉冲"""
        if self._d2_recent_spikes:
            return self._d2_recent_spikes[-1]
        return []

    def apply_da_modulated_plasticity(self, da_level: float) -> None:
        """用 DA 信号触发三因子权重更新

        遍历所有皮层→MSN 突触:
        1. 更新资格痕迹 (STDP 增量 + 衰减)
        2. DA 信号 × 资格痕迹 → 权重变化

        Args:
            da_level: DA 调制信号水平
        """
        # 皮层→MSN 突触仍用旧 SynapseBase (DA-STDP 逐突触)
        cortical_synapses = self._cortical_d1_synapses + self._cortical_d2_synapses
        for syn in cortical_synapses:
            # DA 到达时: 资格痕迹 → 权重变化
            if abs(da_level) > 0.01:
                syn.apply_plasticity(modulation=da_level)

    def reset_spike_counts(self) -> None:
        """重置发放计数器"""
        self._d1_spike_counts = np.zeros(self.n_d1)
        self._d2_spike_counts = np.zeros(self.n_d2)
        self._step_count = 0
        self._d1_recent_spikes = []
        self._d2_recent_spikes = []

    def get_state(self) -> dict:
        """返回当前状态供调试"""
        return {
            "d1_rates": self.get_d1_rates().tolist(),
            "d2_rates": self.get_d2_rates().tolist(),
            "d1_mean_v": float(np.mean(self.d1_pop.v_soma)),
            "d2_mean_v": float(np.mean(self.d2_pop.v_soma)),
            "step_count": self._step_count,
        }