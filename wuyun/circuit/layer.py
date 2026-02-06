"""
Layer 3: Layer — 皮层柱的单层神经元群

管理该层内的所有神经元，提供层级步进和统计接口。

每个 Layer 包含:
- 兴奋性神经元 (锥体/星形)
- 抑制性神经元 (PV+/SST+/VIP)
- 层内突触 (通过 SpikeBus 统一管理)

依赖:
- wuyun.spike (SpikeType, Spike)
- wuyun.neuron (NeuronBase)
"""

from typing import List, Dict, Optional
from wuyun.spike.signal_types import SpikeType, NeuronType
from wuyun.spike.spike import Spike, SpikeTrain
from wuyun.neuron.neuron_base import NeuronBase


# 兴奋性神经元类型集合
_EXCITATORY_TYPES = {
    NeuronType.L23_PYRAMIDAL,
    NeuronType.L5_PYRAMIDAL,
    NeuronType.L6_PYRAMIDAL,
    NeuronType.STELLATE,
    NeuronType.GRANULE,
    NeuronType.MOSSY,
    NeuronType.THALAMIC_RELAY,
    NeuronType.PLACE_CELL,
    NeuronType.GRID_CELL,
    NeuronType.HEAD_DIRECTION,
}


class Layer:
    """皮层柱的一层 (如 L2/3, L4, L5, L6)

    管理该层内的所有神经元。
    步进时按顺序更新所有神经元，收集发放结果。

    Attributes:
        layer_id: 层编号 (1=L1, 23=L2/3, 4=L4, 5=L5, 6=L6)
        neurons: 该层所有神经元列表
        excitatory: 兴奋性神经元子集
        inhibitory: 抑制性神经元子集
    """

    def __init__(self, layer_id: int, neurons: List[NeuronBase]):
        """
        Args:
            layer_id: 层编号
            neurons: 该层的神经元列表
        """
        self.layer_id = layer_id
        self.neurons = neurons

        # 按类型分组
        self.excitatory: List[NeuronBase] = []
        self.inhibitory: List[NeuronBase] = []

        for n in neurons:
            if n.params.neuron_type in _EXCITATORY_TYPES:
                self.excitatory.append(n)
            else:
                self.inhibitory.append(n)

        # 神经元 ID 索引
        self._id_to_neuron: Dict[int, NeuronBase] = {
            n.id: n for n in neurons
        }

        # 当前时间步发放缓存
        self._last_spikes: Dict[int, SpikeType] = {}

    def step(self, current_time: int, dt: float = 1.0) -> Dict[int, SpikeType]:
        """推进一个时间步

        更新该层所有神经元，返回有发放的神经元映射。

        Args:
            current_time: 当前仿真时间步
            dt: 时间步长 (ms)

        Returns:
            {neuron_id: spike_type} 仅包含 is_active 的神经元
        """
        self._last_spikes.clear()
        for neuron in self.neurons:
            spike_type = neuron.step(current_time, dt)
            if spike_type.is_active:
                self._last_spikes[neuron.id] = spike_type
        return self._last_spikes

    def get_neuron(self, neuron_id: int) -> Optional[NeuronBase]:
        """按 ID 获取神经元"""
        return self._id_to_neuron.get(neuron_id)

    def get_spike_trains(self) -> Dict[int, SpikeTrain]:
        """获取该层所有神经元的脉冲序列"""
        return {n.id: n.spike_train for n in self.neurons}

    def get_last_spikes(self) -> Dict[int, SpikeType]:
        """获取最近一次 step 的发放结果"""
        return self._last_spikes

    @property
    def n_excitatory(self) -> int:
        """兴奋性神经元数量"""
        return len(self.excitatory)

    @property
    def n_inhibitory(self) -> int:
        """抑制性神经元数量"""
        return len(self.inhibitory)

    @property
    def n_total(self) -> int:
        """总神经元数量"""
        return len(self.neurons)

    def reset(self) -> None:
        """重置该层所有神经元"""
        for n in self.neurons:
            n.reset()
        self._last_spikes.clear()

    def __repr__(self) -> str:
        return (f"Layer(id={self.layer_id}, "
                f"E={self.n_excitatory}, I={self.n_inhibitory}, "
                f"total={self.n_total})")