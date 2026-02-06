"""
3-B: CA3Network — 自联想记忆网络

核心功能: 通过循环连接存储模式，部分线索触发完整回忆
生物学: CA3 是海马中唯一具有大量循环兴奋连接的区域

组成:
  - n_pyramidal 个 PLACE_CELL 神经元 (兴奋性, 双区室)
  - n_inhibitory 个 BASKET_PV 神经元 (抑制性)
  - 内部 SpikeBus
  - DG→CA3: 苔藓纤维 (MOSSY_FIBER_STP, 少量强连接, "去极化器")
  - EC→CA3: 直接通路 (弱, 穿通纤维)
  - CA3→CA3: 循环连接 (★核心, ClassicalSTDP 学习)
  - CA3→PV / PV→CA3: 反馈抑制

依赖: spike/ ← synapse/ ← neuron/ ← 本模块
"""

from typing import List, Optional, Dict
import numpy as np

from wuyun.spike.spike import Spike
from wuyun.spike.spike_bus import SpikeBus
from wuyun.spike.signal_types import (
    SpikeType,
    SynapseType,
    CompartmentType,
)
from wuyun.synapse.synapse_base import SynapseBase, AMPA_PARAMS, GABA_A_PARAMS
from wuyun.synapse.plasticity.classical_stdp import ClassicalSTDP, ClassicalSTDPParams
from wuyun.synapse.plasticity.inhibitory_stdp import InhibitorySTDP
from wuyun.synapse.short_term_plasticity import (
    ShortTermPlasticity,
    MOSSY_FIBER_STP,
)
from wuyun.neuron.neuron_base import (
    NeuronBase,
    PLACE_CELL_PARAMS,
    BASKET_PV_PARAMS,
)


class CA3Network:
    """CA3 — 自联想记忆网络

    通过 Hebbian 学习 (ClassicalSTDP) 在循环连接中存储模式。
    编码期: 活跃细胞间循环权重增强 → 形成吸引子
    检索期: 部分线索激活子集 → 循环激活扩散 → 完整模式回忆

    关键机制:
    1. 苔藓纤维 (DG→CA3): 强去极化器, 附加 MOSSY_FIBER_STP
       - 低基线释放 (p0=0.05) + 强易化 → 频率越高效能越强
       - 确保 DG 的稀疏输出能有效驱动 CA3
    2. 循环连接 (CA3→CA3): ClassicalSTDP 学习
       - 共活跃细胞间权重增强 (Hebbian)
       - 初始权重低 (0.1), 学习后增强
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

        # === 创建神经元 ===
        # CA3 锥体细胞: ID 范围 [3000, 3000+n_pyramidal)
        self.pyramidal_neurons: List[NeuronBase] = []
        for i in range(n_pyramidal):
            n = NeuronBase(neuron_id=3000 + i, params=PLACE_CELL_PARAMS)
            self.pyramidal_neurons.append(n)

        # PV 抑制性神经元: ID 范围 [3500, 3500+n_inhibitory)
        self.pv_neurons: List[NeuronBase] = []
        for i in range(n_inhibitory):
            n = NeuronBase(neuron_id=3500 + i, params=BASKET_PV_PARAMS)
            self.pv_neurons.append(n)

        # === 内部 SpikeBus ===
        self._bus = SpikeBus()

        # === 突触存储 ===
        self._synapses: List[SynapseBase] = []
        self._recurrent_synapses: List[SynapseBase] = []  # CA3→CA3 (STDP)

        # === STP 实例 (DG→CA3 苔藓纤维) ===
        # 每个 DG granule → CA3 连接有独立 STP
        self._mossy_stp: Dict[int, ShortTermPlasticity] = {}

        # === 连接矩阵 ===
        # DG→CA3 苔藓纤维连接: mossy_conn[granule_idx] = [ca3_idx_list]
        self._mossy_conn: Dict[int, List[int]] = {}
        # EC→CA3 直接通路连接矩阵
        self._ec_conn = np.zeros((n_ec_inputs, n_pyramidal))

        # === 构建连接 ===
        self._build_mossy_connections()
        self._build_ec_connections()
        self._build_recurrent_connections()
        self._build_inhibitory_connections()

        # === 追踪 ===
        self._step_count = 0

    def _build_mossy_connections(self) -> None:
        """构建 DG→CA3 苔藓纤维连接

        每个 granule 细胞 → 随机 3~5 个 CA3 锥体细胞
        附加 MOSSY_FIBER_STP (强易化型)
        """
        for g_idx in range(self.n_dg_granule):
            n_targets = self._rng.randint(3, 6)  # 3~5 个目标
            targets = self._rng.choice(
                self.n_pyramidal, size=min(n_targets, self.n_pyramidal),
                replace=False,
            )
            self._mossy_conn[g_idx] = list(targets)
            # 每个连接创建独立 STP
            for ca3_idx in targets:
                key = g_idx * 10000 + ca3_idx
                self._mossy_stp[key] = ShortTermPlasticity(MOSSY_FIBER_STP)

    def _build_ec_connections(self) -> None:
        """构建 EC→CA3 直接通路 (p=0.25, 中等强度)"""
        for i in range(self.n_ec_inputs):
            for j in range(self.n_pyramidal):
                if self._rng.random() < 0.25:
                    self._ec_conn[i, j] = 0.3 + 0.4 * self._rng.random()

    def _build_recurrent_connections(self) -> None:
        """构建 CA3→CA3 循环连接 (p=0.2, ClassicalSTDP)"""
        stdp = ClassicalSTDP(ClassicalSTDPParams(
            a_plus=0.02,
            a_minus=0.01,
            tau_plus=20.0,
            tau_minus=20.0,
        ))

        for i, pre in enumerate(self.pyramidal_neurons):
            for j, post in enumerate(self.pyramidal_neurons):
                if i == j:
                    continue  # 无自连接
                if self._rng.random() < self.recurrent_prob:
                    syn = SynapseBase(
                        pre_id=pre.id,
                        post_id=post.id,
                        weight=0.1,  # 初始低权重, 通过 STDP 学习
                        delay=1,
                        synapse_type=SynapseType.AMPA,
                        target_compartment=CompartmentType.BASAL,
                        plasticity_rule=stdp,
                        params=AMPA_PARAMS,
                        w_max=1.0,
                        w_min=0.0,
                    )
                    post.add_synapse(syn)
                    self._bus.register_synapse(syn)
                    self._synapses.append(syn)
                    self._recurrent_synapses.append(syn)

    def _build_inhibitory_connections(self) -> None:
        """构建 CA3→PV 和 PV→CA3 连接"""
        inh_stdp = InhibitorySTDP()

        # CA3 → PV (全连接, AMPA)
        for ca3 in self.pyramidal_neurons:
            for pv in self.pv_neurons:
                syn = SynapseBase(
                    pre_id=ca3.id,
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

        # PV → CA3 (全连接, GABA_A + InhibitorySTDP)
        # weight=0.3: 比皇层柱的 0.6 弱, 因为 CA3 模式补全需要循环兴奋胜出抑制
        # (文献: ACh 撤退期 PV 抵抑减弱, Cutsuridis 2010)
        for pv in self.pv_neurons:
            for ca3 in self.pyramidal_neurons:
                syn = SynapseBase(
                    pre_id=pv.id,
                    post_id=ca3.id,
                    weight=0.3,
                    delay=1,
                    synapse_type=SynapseType.GABA_A,
                    target_compartment=CompartmentType.SOMA,
                    plasticity_rule=inh_stdp,
                    params=GABA_A_PARAMS,
                )
                ca3.add_synapse(syn)
                self._bus.register_synapse(syn)
                self._synapses.append(syn)

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
           补偿 DG 极稀疏发放 (~1%) 导致的 STP 易化不足

        Args:
            dg_spikes: DG 颗粒细胞的输出脉冲列表
            dg_rates: DG 颗粒细胞发放率向量, shape=(n_dg_granule,)
        """
        # 对每个活跃的 granule, 通过苔藓纤维注入电流到目标 CA3
        active_granules = set()
        for spike in dg_spikes:
            g_idx = spike.source_id - 1000  # granule ID 从 1000 开始
            if 0 <= g_idx < self.n_dg_granule:
                active_granules.add(g_idx)

        for g_idx in range(self.n_dg_granule):
            targets = self._mossy_conn.get(g_idx, [])
            rate = dg_rates[g_idx] if g_idx < len(dg_rates) else 0.0

            for ca3_idx in targets:
                key = g_idx * 10000 + ca3_idx
                stp = self._mossy_stp.get(key)
                if stp is None:
                    continue

                # STP 每步恢复
                stp.step()

                if g_idx in active_granules:
                    # 脉冲到达 → STP 释放 (强去极化)
                    efficacy = stp.on_spike()
                    current = self.mossy_gain * efficacy
                    self.pyramidal_neurons[ca3_idx].inject_basal_current(current)

                # 速率通道: 持续电流 (补偿稀疏发放)
                # DG 发放率通常 < 5Hz, 归一化到 [0, 1] 后乘以增益
                if rate > 0:
                    rate_current = self.mossy_gain * 0.3 * min(rate / 10.0, 1.0)
                    self.pyramidal_neurons[ca3_idx].inject_basal_current(rate_current)

    def inject_ec_input(self, pattern: np.ndarray) -> None:
        """注入 EC-II 直接通路输入 (弱)

        Args:
            pattern: EC 输入向量, shape=(n_ec_inputs,)
        """
        currents = pattern @ self._ec_conn
        for i, neuron in enumerate(self.pyramidal_neurons):
            if currents[i] > 0:
                neuron.inject_basal_current(currents[i] * self.ec_direct_gain)

    def step(self, t: int, enable_recurrent: bool = True) -> None:
        """推进一个时间步

        基于文献的相位依赖路由 (Cutsuridis 2010; PLOS CB 2025):
        - 编码期 (enable_recurrent=False): 循环连接沉默, 细胞仅由外部输入驱动
          STDP 通过 apply_recurrent_stdp() 异突触更新
        - 检索期 (enable_recurrent=True): 循环连接放大, 部分线索→完整模式

        Args:
            t: 当前仿真时间步
            enable_recurrent: 是否启用循环连接放大 (检索期=True, 编码期=False)
        """
        self._step_count += 1

        # 更新 CA3 锥体细胞
        fired_indices = set()
        for i, neuron in enumerate(self.pyramidal_neurons):
            spike_type = neuron.step(t)
            if spike_type.is_active:
                self._bus.emit(Spike(neuron.id, t, spike_type))
                fired_indices.add(i)

        # === 循环连接电流放大 (仅检索期) ===
        # 编码期沉默: 避免循环兴奋干扰 STDP 学习
        # 检索期放大: 部分线索通过已学习的强连接扩散激活完整模式
        if enable_recurrent and fired_indices:
            for syn in self._recurrent_synapses:
                pre_idx = syn.pre_id - 3000
                if pre_idx in fired_indices:
                    post_idx = syn.post_id - 3000
                    if 0 <= post_idx < self.n_pyramidal:
                        current = self.recurrent_gain * syn.weight
                        if current > 0:
                            self.pyramidal_neurons[post_idx].inject_basal_current(current)

        # 更新 PV 细胞 (仅检索期)
        # 编码期: ACh 高浓度抑制 PV 中间神经元 (Cutsuridis 2010)
        #   → CA3 锥体细胞自由发放, STDP 纯净学习
        # 检索期: ACh 低浓度, PV 活跃 → 防止循环兴奋失控
        if enable_recurrent:
            for neuron in self.pv_neurons:
                spike_type = neuron.step(t)
                if spike_type.is_active:
                    self._bus.emit(Spike(neuron.id, t, spike_type))

        # SpikeBus 分发
        self._bus.step(t)

    def apply_recurrent_stdp(self, t: int) -> None:
        """对循环连接应用 STDP 更新

        仅在编码期调用。遍历所有循环突触,
        用突触前/后神经元的最近脉冲时间计算 STDP。

        Args:
            t: 当前时间步
        """
        for syn in self._recurrent_synapses:
            pre_neuron = self._find_pyramidal(syn.pre_id)
            post_neuron = self._find_pyramidal(syn.post_id)
            if pre_neuron is None or post_neuron is None:
                continue

            pre_times = pre_neuron.spike_train.get_recent_times(window_ms=50)
            post_times = post_neuron.spike_train.get_recent_times(window_ms=50)

            if pre_times and post_times:
                syn.update_weight_stdp(pre_times, post_times)

    def _find_pyramidal(self, neuron_id: int) -> Optional[NeuronBase]:
        """通过 ID 查找锥体细胞"""
        idx = neuron_id - 3000
        if 0 <= idx < self.n_pyramidal:
            return self.pyramidal_neurons[idx]
        return None

    # =========================================================================
    # 状态查询
    # =========================================================================

    def get_activity(self) -> np.ndarray:
        """获取 CA3 锥体细胞活跃状态 (binary)

        Returns:
            binary 向量, shape=(n_pyramidal,)
        """
        return np.array([
            1.0 if n.current_spike_type.is_active else 0.0
            for n in self.pyramidal_neurons
        ])

    def get_output_spikes(self) -> List[Spike]:
        """获取当前时间步的 CA3 输出脉冲 (→ CA1 Schaffer collateral)

        Returns:
            活跃 CA3 锥体细胞的 Spike 列表
        """
        spikes = []
        for n in self.pyramidal_neurons:
            if n.current_spike_type.is_active:
                spikes.append(Spike(
                    source_id=n.id,
                    timestamp=self._step_count,
                    spike_type=n.current_spike_type,
                ))
        return spikes

    def get_firing_rates(self) -> np.ndarray:
        """获取 CA3 锥体细胞发放率向量

        Returns:
            发放率向量, shape=(n_pyramidal,), 单位 Hz
        """
        return np.array([n.firing_rate for n in self.pyramidal_neurons])

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
        for syn in self._recurrent_synapses:
            i = syn.pre_id - 3000
            j = syn.post_id - 3000
            if 0 <= i < self.n_pyramidal and 0 <= j < self.n_pyramidal:
                W[i, j] = syn.weight
        return W

    def reset(self) -> None:
        """重置所有状态"""
        for n in self.pyramidal_neurons:
            n.reset()
        for n in self.pv_neurons:
            n.reset()
        for s in self._synapses:
            s.reset()
        for stp in self._mossy_stp.values():
            stp.reset()
        self._bus.reset()
        self._step_count = 0

    def __repr__(self) -> str:
        return (
            f"CA3Network(pyramidal={self.n_pyramidal}, pv={self.n_inhibitory}, "
            f"recurrent_syns={len(self._recurrent_synapses)}, "
            f"mean_rate={self.get_mean_rate():.1f}Hz)"
        )