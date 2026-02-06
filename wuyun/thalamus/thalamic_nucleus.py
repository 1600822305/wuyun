"""
丘脑核团 (ThalamicNucleus) — 单个丘脑核团

一个核团 = TC 中继神经元群 + TRN 门控神经元群 + 内部 SpikeBus。

内部连接 (解剖硬编码):
  TC → TRN  (basal, AMPA, p=0.3)  TC 侧支激活 TRN
  TRN → TC  (soma, GABA_A, p=0.5) TRN 抑制 TC (门控)

外部接口:
  输入: inject_sensory_current, inject_cortical_feedback_current,
        inject_trn_drive_current, receive_cortical_feedback
  输出: get_tc_output, get_trn_output
  仿真: step, reset

依赖:
- wuyun.spike (Spike, SpikeType, SpikeBus)
- wuyun.synapse (SynapseBase)
- wuyun.neuron (NeuronBase, THALAMIC_RELAY_PARAMS, TRN_PARAMS)
"""

from typing import List, Optional
import numpy as np

from wuyun.spike.signal_types import (
    SpikeType,
    SynapseType,
    CompartmentType,
    NeuronType,
)
from wuyun.spike.spike import Spike
from wuyun.spike.spike_bus import SpikeBus
from wuyun.synapse.synapse_base import SynapseBase
from wuyun.synapse.plasticity.classical_stdp import ClassicalSTDP
from wuyun.synapse.plasticity.inhibitory_stdp import InhibitorySTDP
from wuyun.neuron.neuron_base import (
    NeuronBase,
    THALAMIC_RELAY_PARAMS,
    TRN_PARAMS,
)


# 可塑性规则单例
_EXCITATORY_STDP = ClassicalSTDP()
_INHIBITORY_STDP = InhibitorySTDP()


def _make_thalamic_neuron_id(nucleus_id: int, local_id: int) -> int:
    """生成丘脑神经元全局唯一 ID

    编码规则: nucleus_id * 10000 + 80 * 100 + local_id
    用 80 区分于皮层柱的 4/5/6/23 层编号。

    Args:
        nucleus_id: 核团 ID
        local_id: 核团内局部 ID

    Returns:
        全局唯一 ID
    """
    return nucleus_id * 10000 + 80 * 100 + local_id


class ThalamicNucleus:
    """单个丘脑核团

    包含 TC 中继神经元群和 TRN 门控神经元群。
    TC 接收感觉输入 (basal) 和皮层反馈 (apical)，
    TRN 接收 TC 侧支和外部驱动，抑制 TC 实现门控。

    Attributes:
        nucleus_id: 核团 ID
        tc_neurons: TC 中继神经元列表
        trn_neurons: TRN 门控神经元列表
        bus: 核团内部脉冲总线
        synapses: 所有核团内突触
    """

    def __init__(
        self,
        nucleus_id: int,
        tc_neurons: List[NeuronBase],
        trn_neurons: List[NeuronBase],
        bus: SpikeBus,
        synapses: List[SynapseBase],
    ):
        self.nucleus_id = nucleus_id
        self.tc_neurons = tc_neurons
        self.trn_neurons = trn_neurons
        self.bus = bus
        self.synapses = synapses

        # 所有神经元索引
        self._all_neurons = {n.id: n for n in tc_neurons + trn_neurons}

        # 当前时间步输出缓存
        self._tc_output: List[Spike] = []
        self._trn_output: List[Spike] = []
        self._current_time: int = 0

    # =========================================================================
    # 外部输入接口
    # =========================================================================

    def inject_sensory_current(self, current: float) -> None:
        """向所有 TC 的 basal 注入感觉驱动电流

        Args:
            current: 注入电流强度
        """
        for tc in self.tc_neurons:
            tc.inject_basal_current(current)

    def inject_cortical_feedback_current(self, current: float) -> None:
        """向所有 TC 的 apical 注入皮层反馈电流 (L6 预测)

        Args:
            current: 注入电流强度
        """
        for tc in self.tc_neurons:
            tc.inject_apical_current(current)

    def inject_trn_drive_current(self, current: float) -> None:
        """向所有 TRN 的 basal 注入外部驱动电流

        用于跨核团竞争: 其他核团的 TRN 活动可驱动本核团 TRN。

        Args:
            current: 注入电流强度
        """
        for trn in self.trn_neurons:
            trn.inject_basal_current(current)

    def receive_cortical_feedback(self, spikes: List[Spike]) -> None:
        """接收皮层反馈脉冲 → 通过 bus emit

        需预先注册 L6→TC apical 突触。

        Args:
            spikes: 皮层反馈脉冲列表
        """
        for spike in spikes:
            self.bus.emit(spike)

    # =========================================================================
    # 仿真步进
    # =========================================================================

    def step(self, current_time: int, dt: float = 1.0) -> None:
        """推进一个时间步

        执行顺序:
        1. 所有神经元 step (TC + TRN)
        2. 发放 → emit 到 SpikeBus
        3. SpikeBus deliver

        Args:
            current_time: 当前仿真时间步
            dt: 时间步长 (ms)
        """
        self._current_time = current_time
        self._tc_output.clear()
        self._trn_output.clear()

        # Phase 1: 所有神经元 step — TC 先, TRN 后
        all_spikes = {}
        for neuron in self.tc_neurons:
            spike_type = neuron.step(current_time, dt)
            if spike_type.is_active:
                all_spikes[neuron.id] = spike_type

        for neuron in self.trn_neurons:
            spike_type = neuron.step(current_time, dt)
            if spike_type.is_active:
                all_spikes[neuron.id] = spike_type

        # Phase 2: 发放 → emit 到 SpikeBus + 收集输出
        for neuron_id, spike_type in all_spikes.items():
            spike = Spike(
                source_id=neuron_id,
                timestamp=current_time,
                spike_type=spike_type,
            )
            self.bus.emit(spike)

            neuron = self._all_neurons[neuron_id]
            if neuron.params.neuron_type == NeuronType.THALAMIC_RELAY:
                self._tc_output.append(spike)
            elif neuron.params.neuron_type == NeuronType.TRN:
                self._trn_output.append(spike)

        # Phase 3: SpikeBus deliver
        self.bus.step(current_time)

    # =========================================================================
    # 输出查询
    # =========================================================================

    def get_tc_output(self) -> List[Spike]:
        """获取当前步 TC 发放的脉冲"""
        return list(self._tc_output)

    def get_trn_output(self) -> List[Spike]:
        """获取当前步 TRN 发放的脉冲"""
        return list(self._trn_output)

    def get_tc_firing_rate(self) -> float:
        """TC 群体平均发放率 (Hz)"""
        if not self.tc_neurons:
            return 0.0
        return sum(n.firing_rate for n in self.tc_neurons) / len(self.tc_neurons)

    def get_trn_firing_rate(self) -> float:
        """TRN 群体平均发放率 (Hz)"""
        if not self.trn_neurons:
            return 0.0
        return sum(n.firing_rate for n in self.trn_neurons) / len(self.trn_neurons)

    def get_tc_burst_ratio(self) -> float:
        """TC 群体平均 burst 比率"""
        if not self.tc_neurons:
            return 0.0
        return sum(n.burst_ratio for n in self.tc_neurons) / len(self.tc_neurons)

    # =========================================================================
    # 生命周期
    # =========================================================================

    def reset(self) -> None:
        """重置核团到初始状态"""
        for n in self.tc_neurons:
            n.reset()
        for n in self.trn_neurons:
            n.reset()
        for syn in self.synapses:
            syn.reset()
        self.bus.reset()
        self._tc_output.clear()
        self._trn_output.clear()
        self._current_time = 0

    def __repr__(self) -> str:
        return (
            f"ThalamicNucleus(id={self.nucleus_id}, "
            f"TC={len(self.tc_neurons)}, "
            f"TRN={len(self.trn_neurons)}, "
            f"synapses={len(self.synapses)})"
        )


# =============================================================================
# 工厂函数
# =============================================================================

def create_thalamic_nucleus(
    nucleus_id: int,
    n_tc: int = 10,
    n_trn: int = 5,
    seed: int = None,
) -> ThalamicNucleus:
    """创建一个丘脑核团

    Args:
        nucleus_id: 核团 ID
        n_tc: TC 中继神经元数量
        n_trn: TRN 门控神经元数量
        seed: 随机种子

    Returns:
        配置好的 ThalamicNucleus
    """
    rng = np.random.RandomState(seed)

    # === 创建神经元 ===
    tc_neurons = []
    for i in range(n_tc):
        nid = _make_thalamic_neuron_id(nucleus_id, i)
        neuron = NeuronBase(
            neuron_id=nid,
            params=THALAMIC_RELAY_PARAMS,
            region_id=nucleus_id,
        )
        tc_neurons.append(neuron)

    trn_neurons = []
    for i in range(n_trn):
        nid = _make_thalamic_neuron_id(nucleus_id, n_tc + i)
        neuron = NeuronBase(
            neuron_id=nid,
            params=TRN_PARAMS,
            region_id=nucleus_id,
        )
        trn_neurons.append(neuron)

    # === 创建内部突触连接 ===
    all_synapses: List[SynapseBase] = []

    # TC → TRN (basal, AMPA, p=0.3) — TC 侧支激活 TRN
    for tc in tc_neurons:
        for trn in trn_neurons:
            if rng.random() < 0.3:
                w = rng.uniform(0.3, 0.7)
                syn = SynapseBase(
                    pre_id=tc.id,
                    post_id=trn.id,
                    weight=w,
                    delay=1,
                    synapse_type=SynapseType.AMPA,
                    target_compartment=CompartmentType.BASAL,
                    plasticity_rule=_EXCITATORY_STDP,
                )
                trn.add_synapse(syn)
                all_synapses.append(syn)

    # TRN → TC (soma, GABA_A, p=0.5) — TRN 抑制 TC (门控)
    for trn in trn_neurons:
        for tc in tc_neurons:
            if rng.random() < 0.5:
                w = rng.uniform(0.3, 0.7)
                syn = SynapseBase(
                    pre_id=trn.id,
                    post_id=tc.id,
                    weight=w,
                    delay=1,
                    synapse_type=SynapseType.GABA_A,
                    target_compartment=CompartmentType.SOMA,
                    plasticity_rule=_INHIBITORY_STDP,
                )
                tc.add_synapse(syn)
                all_synapses.append(syn)

    # === 创建 SpikeBus 并注册突触 ===
    bus = SpikeBus()
    bus.register_synapses(all_synapses)

    return ThalamicNucleus(
        nucleus_id=nucleus_id,
        tc_neurons=tc_neurons,
        trn_neurons=trn_neurons,
        bus=bus,
        synapses=all_synapses,
    )