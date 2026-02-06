"""
Layer 3: CorticalColumn — 6 层预测编码计算单元

这是悟韵 (WuYun) 最核心的计算单元。

6 层结构:
  L1:   分子层 — 无细胞体，只有顶端树突纤维 (接收反馈)
  L2/3: 浅层 — L23锥体 + PV+ + SST+ → 预测误差(regular) / 匹配(burst)
  L4:   输入层 — Stellate 星形细胞 → 接收丘脑前馈/低层 regular
  L5:   输出层 — L5锥体(κ=0.6, 最强burst) + PV+ → 驱动输出
  L6:   多形层 — L6锥体 → 生成预测信号 (→ 丘脑/低层柱 L1)

内部信号流:
  外部前馈 → L4 basal
  外部反馈 → L2/3 apical + L5 apical (通过 L1)
  L4 → L2/3 basal (层间前馈)
  L2/3 → L5 basal (层间前馈)
  L5 → L6 basal
  L6 → L2/3 apical + L5 apical (柱内反馈预测) ★ 关键
  PV+ → 锥体 soma (快速抑制)
  SST+ → 锥体 apical (树突抑制, 控制burst)

预测编码对应:
  L2/3 regular spike = 预测误差 → 向高层传递
  L2/3 burst = 预测匹配 → 触发注意力/学习
  L5 burst = 驱动输出 → 影响运动/丘脑
  L6 output = 预测信号 → 反馈给低层

依赖:
- wuyun.spike (Spike, SpikeType, SpikeBus)
- wuyun.synapse (SynapseBase)
- wuyun.neuron (NeuronBase)
- wuyun.circuit.layer (Layer)
"""

from typing import List, Dict, Optional
from wuyun.spike.signal_types import SpikeType
from wuyun.spike.spike import Spike
from wuyun.spike.spike_bus import SpikeBus
from wuyun.synapse.synapse_base import SynapseBase
from wuyun.neuron.neuron_base import NeuronBase
from wuyun.circuit.layer import Layer


class CorticalColumn:
    """皮层柱 — 6 层预测编码计算单元

    所有新皮质区域（感觉/联合/运动）使用同一个 CorticalColumn 类，
    功能差异通过参数/连接/可塑性规则的不同来实现 (不是不同的类!)。

    Attributes:
        column_id: 柱全局 ID
        layers: {层编号: Layer} 字典 (23, 4, 5, 6)
        bus: 柱内脉冲总线
        synapses: 所有柱内突触
    """

    def __init__(
        self,
        column_id: int,
        layers: Dict[int, Layer],
        bus: SpikeBus,
        synapses: List[SynapseBase],
    ):
        """
        Args:
            column_id: 柱全局 ID
            layers: {层编号: Layer} (23=L2/3, 4=L4, 5=L5, 6=L6)
            bus: 柱内脉冲总线 (所有柱内突触已注册)
            synapses: 柱内所有突触列表 (用于统计/重置)
        """
        self.column_id = column_id
        self.layers = layers
        self.bus = bus
        self.synapses = synapses

        # 所有神经元的 ID 索引
        self._all_neurons: Dict[int, NeuronBase] = {}
        for layer in layers.values():
            for neuron in layer.neurons:
                self._all_neurons[neuron.id] = neuron

        # 当前时间步发放缓存
        self._current_spikes: Dict[int, SpikeType] = {}
        self._current_time: int = 0

    # =========================================================================
    # 外部输入接口
    # =========================================================================

    def receive_feedforward(self, spikes: List[Spike]) -> None:
        """接收前馈输入 → L4 stellate 的 basal

        前馈输入来自:
        - 丘脑中继 (感觉信号)
        - 低层柱的 L2/3 regular spikes (预测误差上传)

        通过 SpikeBus emit → 需要预先注册外部→L4 的突触。
        如果没有预注册突触，直接注入电流到 L4 神经元。

        Args:
            spikes: 前馈脉冲列表
        """
        for spike in spikes:
            self.bus.emit(spike)

    def receive_feedback(self, spikes: List[Spike]) -> None:
        """接收反馈预测 → L2/3 和 L5 锥体的 apical

        反馈输入来自:
        - 高层柱的 L6 输出 (预测信号)
        - 通过 L1 层传递到 L2/3 和 L5 的顶端树突

        Args:
            spikes: 反馈脉冲列表
        """
        for spike in spikes:
            self.bus.emit(spike)

    def inject_feedforward_current(self, current: float) -> None:
        """直接向 L4 所有兴奋性神经元注入前馈电流 (测试/外部刺激用)

        Args:
            current: 注入电流强度
        """
        if 4 in self.layers:
            for neuron in self.layers[4].excitatory:
                neuron.inject_basal_current(current)

    def inject_feedback_current(self, current: float) -> None:
        """直接向 L2/3 和 L5 的锥体神经元注入反馈电流 (测试用)

        Args:
            current: 注入电流强度
        """
        if 23 in self.layers:
            for neuron in self.layers[23].excitatory:
                neuron.inject_apical_current(current)
        if 5 in self.layers:
            for neuron in self.layers[5].excitatory:
                neuron.inject_apical_current(current)

    # =========================================================================
    # 仿真步进
    # =========================================================================

    def step(self, current_time: int, dt: float = 1.0) -> None:
        """推进一个时间步

        执行顺序 (关键!):
        1. 所有神经元 step (双区室更新 + 收集突触电流)
           — 突触的 step() 在 NeuronBase._collect_synaptic_currents() 中调用
        2. 发放的神经元 → emit 到 SpikeBus
        3. SpikeBus deliver → 将脉冲送达下游突触的延迟缓冲区
           — 这些脉冲会在下一个时间步被突触 step() 处理

        Args:
            current_time: 当前仿真时间步
            dt: 时间步长 (ms)
        """
        self._current_time = current_time
        self._current_spikes.clear()

        # === Phase 1: 所有层的神经元 step ===
        # 按层顺序: L4 → L23 → L5 → L6 (符合信号流方向)
        layer_order = [4, 23, 5, 6]
        for layer_id in layer_order:
            if layer_id in self.layers:
                layer_spikes = self.layers[layer_id].step(current_time, dt)
                self._current_spikes.update(layer_spikes)

        # === Phase 2: 发放 → emit 到 SpikeBus ===
        for neuron_id, spike_type in self._current_spikes.items():
            spike = Spike(
                source_id=neuron_id,
                timestamp=current_time,
                spike_type=spike_type,
            )
            self.bus.emit(spike)

        # === Phase 3: SpikeBus deliver ===
        self.bus.step(current_time)

    # =========================================================================
    # 输出查询
    # =========================================================================

    def get_prediction_error(self) -> List[Spike]:
        """L2/3 的 regular spikes → 预测误差, 向上传递

        Returns:
            L2/3 兴奋性神经元的 regular spikes 列表
        """
        result = []
        if 23 in self.layers:
            for neuron in self.layers[23].excitatory:
                st = neuron.current_spike_type
                if st == SpikeType.REGULAR:
                    result.append(Spike(neuron.id, self._current_time, st))
        return result

    def get_match_signal(self) -> List[Spike]:
        """L2/3 的 burst spikes → 预测匹配, 触发注意力/学习

        Returns:
            L2/3 兴奋性神经元的 burst spikes 列表
        """
        result = []
        if 23 in self.layers:
            for neuron in self.layers[23].excitatory:
                st = neuron.current_spike_type
                if st.is_burst:
                    result.append(Spike(neuron.id, self._current_time, st))
        return result

    def get_drive_output(self) -> List[Spike]:
        """L5 的 burst spikes → 驱动下游 (丘脑/脑干/脊髓)

        Returns:
            L5 兴奋性神经元的 burst spikes 列表
        """
        result = []
        if 5 in self.layers:
            for neuron in self.layers[5].excitatory:
                st = neuron.current_spike_type
                if st.is_burst:
                    result.append(Spike(neuron.id, self._current_time, st))
        return result

    def get_prediction(self) -> List[Spike]:
        """L6 的所有输出 → 预测信号, 反馈给低层柱/丘脑

        Returns:
            L6 兴奋性神经元的所有 active spikes
        """
        result = []
        if 6 in self.layers:
            for neuron in self.layers[6].excitatory:
                st = neuron.current_spike_type
                if st.is_active:
                    result.append(Spike(neuron.id, self._current_time, st))
        return result

    def get_all_spikes(self) -> Dict[int, SpikeType]:
        """获取当前时间步所有发放

        Returns:
            {neuron_id: spike_type}
        """
        return dict(self._current_spikes)

    # =========================================================================
    # 统计查询
    # =========================================================================

    def get_layer_firing_rates(self) -> Dict[int, float]:
        """获取各层平均发放率 (Hz)

        Returns:
            {layer_id: avg_firing_rate_hz}
        """
        rates = {}
        for layer_id, layer in self.layers.items():
            if layer.n_total > 0:
                total_rate = sum(n.firing_rate for n in layer.neurons)
                rates[layer_id] = total_rate / layer.n_total
        return rates

    def get_layer_burst_ratios(self) -> Dict[int, float]:
        """获取各层平均 burst 比率

        Returns:
            {layer_id: avg_burst_ratio}
        """
        ratios = {}
        for layer_id, layer in self.layers.items():
            exc_neurons = layer.excitatory
            if exc_neurons:
                total_ratio = sum(n.burst_ratio for n in exc_neurons)
                ratios[layer_id] = total_ratio / len(exc_neurons)
        return ratios

    @property
    def n_neurons(self) -> int:
        """总神经元数"""
        return len(self._all_neurons)

    @property
    def n_synapses(self) -> int:
        """总突触数"""
        return len(self.synapses)

    def get_neuron(self, neuron_id: int) -> Optional[NeuronBase]:
        """按 ID 获取神经元"""
        return self._all_neurons.get(neuron_id)

    def reset(self) -> None:
        """重置柱到初始状态"""
        for layer in self.layers.values():
            layer.reset()
        for syn in self.synapses:
            syn.reset()
        self.bus.reset()
        self._current_spikes.clear()
        self._current_time = 0

    def __repr__(self) -> str:
        layer_info = ", ".join(
            f"L{lid}={layer.n_total}"
            for lid, layer in sorted(self.layers.items())
        )
        return (f"CorticalColumn(id={self.column_id}, "
                f"neurons={self.n_neurons}, "
                f"synapses={self.n_synapses}, "
                f"layers=[{layer_info}])")