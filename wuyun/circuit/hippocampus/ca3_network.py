"""
3-B: CA3Network — 自联想记忆网络 (向量化版本)

核心功能: 通过循环连接存储模式，部分线索触发完整回忆
生物学: CA3 是海马中唯一具有大量循环兴奋连接的区域

组成 (向量化):
  - pyramidal_pop: NeuronPopulation (n_pyramidal, PLACE_CELL)
  - pv_pop: NeuronPopulation (n_inhibitory, BASKET_PV)
  - recurrent_syn: SynapseGroup (CA3→CA3, AMPA, STDP)
  - ca3_pv_syn: SynapseGroup (CA3→PV, AMPA)
  - pv_ca3_syn: SynapseGroup (PV→CA3, GABA_A)
  - DG→CA3: 苔藓纤维 (STP 电流注入)
  - EC→CA3: 矩阵乘法电流注入

依赖: core/ (NeuronPopulation, SynapseGroup)
"""

from typing import List, Optional, Dict
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
    MOSSY_FIBER_STP,
)
from wuyun.neuron.neuron_base import (
    PLACE_CELL_PARAMS,
    BASKET_PV_PARAMS,
)
from wuyun.core.population import NeuronPopulation
from wuyun.core.synapse_group import SynapseGroup


class CA3Network:
    """CA3 — 自联想记忆网络 (向量化版本)

    通过 Hebbian 学习 (STDP) 在循环连接中存储模式。
    编码期: 活跃细胞间循环权重增强 → 形成吸引子
    检索期: 部分线索激活子集 → 循环激活扩散 → 完整模式回忆

    关键机制:
    1. 苔藓纤维 (DG→CA3): 强去极化器, 附加 MOSSY_FIBER_STP
    2. 循环连接 (CA3→CA3): STDP 学习
    3. PV 反馈抑制: 防止过度兴奋
    """

    def __init__(
        self,
        n_pyramidal: int = 50,
        n_inhibitory: int = 8,
        recurrent_prob: float = 0.2,
        mossy_gain: float = 40.0,
        ec_direct_gain: float = 30.0,
        recurrent_gain: float = 12.0,
        pv_gain: float = 10.0,
        n_dg_granule: int = 100,
        n_ec_inputs: int = 16,
        seed: int = 42,
    ):
        self.n_pyramidal = n_pyramidal
        self.n_inhibitory = n_inhibitory
        self.recurrent_prob = recurrent_prob
        self.mossy_gain = mossy_gain
        self.ec_direct_gain = ec_direct_gain
        self.recurrent_gain = recurrent_gain
        self.pv_gain = pv_gain
        self.n_dg_granule = n_dg_granule
        self.n_ec_inputs = n_ec_inputs
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        # === 向量化神经元群体 ===
        self.pyramidal_pop = NeuronPopulation(n_pyramidal, PLACE_CELL_PARAMS)
        self.pv_pop = NeuronPopulation(n_inhibitory, BASKET_PV_PARAMS)

        # === STP 实例 (DG→CA3 苔藓纤维) ===
        self._mossy_stp: Dict[int, ShortTermPlasticity] = {}
        self._mossy_conn: Dict[int, List[int]] = {}

        # === EC→CA3 直接通路连接矩阵 ===
        self._ec_conn = np.zeros((n_ec_inputs, n_pyramidal))

        # === 构建连接 ===
        self._build_mossy_connections()
        self._build_ec_connections()
        self._build_synapse_groups()

        # === 追踪 ===
        self._step_count = 0

        # === 兼容旧 API: 暴露 pyramidal_neurons 属性 ===
        # 某些测试/外部代码通过 ca3.pyramidal_neurons[i] 访问
        self._pyramidal_neuron_ids = list(range(3000, 3000 + n_pyramidal))

    def _build_mossy_connections(self) -> None:
        """构建 DG→CA3 苔藓纤维连接"""
        for g_idx in range(self.n_dg_granule):
            n_targets = self._rng.randint(3, 6)
            targets = self._rng.choice(
                self.n_pyramidal, size=min(n_targets, self.n_pyramidal),
                replace=False,
            )
            self._mossy_conn[g_idx] = list(targets)
            for ca3_idx in targets:
                key = g_idx * 10000 + ca3_idx
                self._mossy_stp[key] = ShortTermPlasticity(MOSSY_FIBER_STP)

    def _build_ec_connections(self) -> None:
        """构建 EC→CA3 直接通路 (p=0.25, 中等强度)"""
        for i in range(self.n_ec_inputs):
            for j in range(self.n_pyramidal):
                if self._rng.random() < 0.25:
                    self._ec_conn[i, j] = 0.3 + 0.4 * self._rng.random()

    def _build_synapse_groups(self) -> None:
        """构建所有内部 SynapseGroup"""
        # --- CA3→CA3 循环连接 (STDP) ---
        pre_list, post_list = [], []
        for i in range(self.n_pyramidal):
            for j in range(self.n_pyramidal):
                if i == j:
                    continue
                if self._rng.random() < self.recurrent_prob:
                    pre_list.append(i)
                    post_list.append(j)

        n_rec = len(pre_list)
        self.recurrent_syn = SynapseGroup(
            pre_ids=np.array(pre_list, dtype=np.int32),
            post_ids=np.array(post_list, dtype=np.int32),
            weights=np.full(n_rec, 0.1),
            delays=np.ones(n_rec, dtype=np.int32),
            synapse_type=SynapseType.AMPA,
            target=CompartmentType.BASAL,
            tau_decay=AMPA_PARAMS.tau_decay,
            e_rev=AMPA_PARAMS.e_rev,
            g_max=AMPA_PARAMS.g_max,
            n_post=self.n_pyramidal,
            w_max=1.0,
            w_min=0.0,
        )

        # --- CA3 → PV (全连接, AMPA) ---
        n_ca3pv = self.n_pyramidal * self.n_inhibitory
        ca3pv_pre = np.repeat(np.arange(self.n_pyramidal), self.n_inhibitory)
        ca3pv_post = np.tile(np.arange(self.n_inhibitory), self.n_pyramidal)
        self.ca3_pv_syn = SynapseGroup(
            pre_ids=ca3pv_pre,
            post_ids=ca3pv_post,
            weights=np.full(n_ca3pv, 0.5),
            delays=np.ones(n_ca3pv, dtype=np.int32),
            synapse_type=SynapseType.AMPA,
            target=CompartmentType.SOMA,
            tau_decay=AMPA_PARAMS.tau_decay,
            e_rev=AMPA_PARAMS.e_rev,
            g_max=AMPA_PARAMS.g_max,
            n_post=self.n_inhibitory,
        )

        # --- PV → CA3 (全连接, GABA_A) ---
        n_pvca3 = self.n_inhibitory * self.n_pyramidal
        pvca3_pre = np.repeat(np.arange(self.n_inhibitory), self.n_pyramidal)
        pvca3_post = np.tile(np.arange(self.n_pyramidal), self.n_inhibitory)
        self.pv_ca3_syn = SynapseGroup(
            pre_ids=pvca3_pre,
            post_ids=pvca3_post,
            weights=np.full(n_pvca3, 0.3),
            delays=np.ones(n_pvca3, dtype=np.int32),
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

    def inject_mossy_input(
        self,
        dg_spikes: List[Spike],
        dg_rates: np.ndarray,
    ) -> None:
        """注入 DG→CA3 苔藓纤维输入

        双通道注入:
        1. 脉冲通道: STP 调制的强去极化 (spike-based)
        2. 速率通道: 基于 DG 发放率的持续弱电流 (rate-based)

        Args:
            dg_spikes: DG 颗粒细胞的输出脉冲列表
            dg_rates: DG 颗粒细胞发放率向量, shape=(n_dg_granule,)
        """
        active_granules = set()
        for spike in dg_spikes:
            g_idx = spike.source_id - 1000
            if 0 <= g_idx < self.n_dg_granule:
                active_granules.add(g_idx)

        nonzero_rate_granules = set(np.nonzero(dg_rates[:self.n_dg_granule])[0])
        relevant_granules = active_granules | nonzero_rate_granules
        if not relevant_granules:
            return

        mossy_gain = self.mossy_gain
        for g_idx in relevant_granules:
            targets = self._mossy_conn.get(g_idx)
            if not targets:
                continue
            rate = float(dg_rates[g_idx])
            is_active = g_idx in active_granules

            for ca3_idx in targets:
                key = g_idx * 10000 + ca3_idx
                stp = self._mossy_stp.get(key)
                if stp is None:
                    continue
                stp.step()

                if is_active:
                    efficacy = stp.on_spike()
                    self.pyramidal_pop.i_basal[ca3_idx] += mossy_gain * efficacy

                if rate > 0:
                    rate_current = mossy_gain * 0.3 * min(rate / 10.0, 1.0)
                    self.pyramidal_pop.i_basal[ca3_idx] += rate_current

    def inject_ec_input(self, pattern: np.ndarray) -> None:
        """注入 EC-II 直接通路输入 (弱)

        Args:
            pattern: EC 输入向量, shape=(n_ec_inputs,)
        """
        currents = pattern @ self._ec_conn * self.ec_direct_gain
        mask = currents > 0
        self.pyramidal_pop.i_basal[mask] += currents[mask]

    def step(self, t: int, enable_recurrent: bool = True) -> None:
        """推进一个时间步

        基于文献的相位依赖路由 (Cutsuridis 2010; PLOS CB 2025):
        - 编码期 (enable_recurrent=False): 循环沉默, 仅外部驱动
        - 检索期 (enable_recurrent=True): 循环放大, 部分线索→完整模式

        Args:
            t: 当前仿真时间步
            enable_recurrent: 是否启用循环连接放大
        """
        self._step_count += 1

        # --- 1. 更新 CA3 锥体细胞 ---
        self.pyramidal_pop.step(t)

        # --- 2. 循环连接电流放大 (仅检索期) ---
        if enable_recurrent and self.pyramidal_pop.fired.any():
            fired_idx = np.nonzero(self.pyramidal_pop.fired)[0]
            # 找出 pre 在 fired_idx 中的循环突触
            pre_fired = self.pyramidal_pop.fired[self.recurrent_syn.pre_ids]
            active_syns = np.nonzero(pre_fired)[0]
            if len(active_syns) > 0:
                post_targets = self.recurrent_syn.post_ids[active_syns]
                syn_weights = self.recurrent_syn.weights[active_syns]
                currents = self.recurrent_gain * syn_weights
                np.add.at(self.pyramidal_pop.i_basal, post_targets, currents)

        # --- 3. CA3→PV 传递 + PV 更新 (仅检索期) ---
        if enable_recurrent:
            self.ca3_pv_syn.deliver_spikes(
                self.pyramidal_pop.fired, self.pyramidal_pop.spike_type)
            i_ca3pv = self.ca3_pv_syn.step_and_compute(self.pv_pop.v_soma)
            self.pv_pop.i_soma += i_ca3pv

            self.pv_pop.step(t)

            # PV→CA3 传递
            self.pv_ca3_syn.deliver_spikes(
                self.pv_pop.fired, self.pv_pop.spike_type)
            i_pvca3 = self.pv_ca3_syn.step_and_compute(self.pyramidal_pop.v_soma)
            self.pyramidal_pop.i_soma += i_pvca3

        # --- 4. 循环突触门控 (传递脉冲到延迟缓冲) ---
        self.recurrent_syn.deliver_spikes(
            self.pyramidal_pop.fired, self.pyramidal_pop.spike_type)
        # 循环突触电流在下一步通过 step_and_compute 生效
        i_rec = self.recurrent_syn.step_and_compute(self.pyramidal_pop.v_soma)
        if enable_recurrent:
            self.pyramidal_pop.i_basal += i_rec

    def apply_recurrent_stdp(self, t: int) -> None:
        """对循环连接应用 STDP 更新

        仅在编码期调用。使用 SynapseGroup 的批量 STDP 更新。

        Args:
            t: 当前时间步
        """
        pre_cache = self.pyramidal_pop.get_all_recent_spike_times(window_ms=50)
        post_cache = pre_cache  # CA3→CA3 循环, pre 和 post 是同一个 Population
        self.recurrent_syn.update_weights_stdp(
            pre_spike_cache=pre_cache,
            post_spike_cache=post_cache,
            a_plus=0.02,
            a_minus=0.01,
            tau_plus=20.0,
            tau_minus=20.0,
        )

    # =========================================================================
    # 状态查询
    # =========================================================================

    def get_activity(self) -> np.ndarray:
        """获取 CA3 锥体细胞活跃状态 (binary)

        Returns:
            binary 向量, shape=(n_pyramidal,)
        """
        return self.pyramidal_pop.fired.astype(np.float64)

    def get_output_spikes(self) -> List[Spike]:
        """获取当前时间步的 CA3 输出脉冲 (→ CA1 Schaffer collateral)

        Returns:
            活跃 CA3 锥体细胞的 Spike 列表
        """
        spikes = []
        fired_idx = np.nonzero(self.pyramidal_pop.fired)[0]
        for i in fired_idx:
            spikes.append(Spike(
                source_id=3000 + i,
                timestamp=self._step_count,
                spike_type=SpikeType(int(self.pyramidal_pop.spike_type[i])),
            ))
        return spikes

    def get_firing_rates(self) -> np.ndarray:
        """获取 CA3 锥体细胞发放率向量

        Returns:
            发放率向量, shape=(n_pyramidal,), 单位 Hz
        """
        return self.pyramidal_pop.get_firing_rates(
            window_ms=1000, current_time=self._step_count)

    def get_mean_rate(self) -> float:
        """获取 CA3 平均发放率 (Hz)"""
        return float(np.mean(self.get_firing_rates()))

    def get_recurrent_weights(self) -> np.ndarray:
        """获取循环连接权重矩阵

        Returns:
            权重矩阵, shape=(n_pyramidal, n_pyramidal)
            w[i][j] = CA3_i → CA3_j 的权重, 无连接为 0
        """
        W = np.zeros((self.n_pyramidal, self.n_pyramidal))
        for k in range(self.recurrent_syn.K):
            i = self.recurrent_syn.pre_ids[k]
            j = self.recurrent_syn.post_ids[k]
            W[i, j] = self.recurrent_syn.weights[k]
        return W

    def reset(self) -> None:
        """重置所有状态"""
        self.pyramidal_pop.reset()
        self.pv_pop.reset()
        self.recurrent_syn.reset()
        self.ca3_pv_syn.reset()
        self.pv_ca3_syn.reset()
        for stp in self._mossy_stp.values():
            stp.reset()
        self._step_count = 0

    def __repr__(self) -> str:
        return (
            f"CA3Network(pyramidal={self.n_pyramidal}, pv={self.n_inhibitory}, "
            f"recurrent_syns={self.recurrent_syn.K}, "
            f"mean_rate={self.get_mean_rate():.1f}Hz)"
        )