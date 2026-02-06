"""
Layer 3: CorticalColumn — 6 层预测编码计算单元 (向量化版本)

这是悟韵 (WuYun) 最核心的计算单元。

6 层结构:
  L1:   分子层 — 无细胞体，只有顶端树突纤维 (接收反馈)
  L2/3: 浅层 — L23锥体 + PV+ + SST+ → 预测误差(regular) / 匹配(burst)
  L4:   输入层 — Stellate 星形细胞 → 接收丘脑前馈/低层 regular
  L5:   输出层 — L5锥体(κ=0.6, 最强burst) + PV+ → 驱动输出
  L6:   多形层 — L6锥体 → 生成预测信号 (→ 丘脑/低层柱 L1)

依赖:
- wuyun.spike (Spike, SpikeType)
- wuyun.core (NeuronPopulation, SynapseGroup)
- wuyun.circuit.layer (Layer)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from wuyun.spike.signal_types import SpikeType, CompartmentType
from wuyun.spike.spike import Spike
from wuyun.core.population import NeuronPopulation
from wuyun.core.synapse_group import SynapseGroup
from wuyun.circuit.layer import Layer


class CorticalColumn:
    """皮层柱 — 6 层预测编码计算单元 (向量化版本)

    Attributes:
        column_id: 柱全局 ID
        layers: {层编号: Layer} 字典 (23, 4, 5, 6)
        _connections: [(SynapseGroup, src_pop, tgt_pop), ...] 所有柱内突触连接
    """

    def __init__(
        self,
        column_id: int,
        layers: Dict[int, Layer],
        connections: List[Tuple[SynapseGroup, NeuronPopulation, NeuronPopulation]],
    ):
        """
        Args:
            column_id: 柱全局 ID
            layers: {层编号: Layer} (23=L2/3, 4=L4, 5=L5, 6=L6)
            connections: [(SynapseGroup, src_pop, tgt_pop), ...] 柱内突触连接
        """
        self.column_id = column_id
        self.layers = layers
        self._connections = connections

        # 当前时间步发放缓存
        self._current_spikes: Dict[int, SpikeType] = {}
        self._current_time: int = 0

    # =========================================================================
    # 外部输入接口
    # =========================================================================

    def receive_feedforward(self, spikes: List[Spike]) -> None:
        """接收前馈输入 → L4 的 basal (向量化模式: no-op)"""
        pass

    def receive_feedback(self, spikes: List[Spike]) -> None:
        """接收反馈预测 (向量化模式: no-op, 用 inject_feedback_current)"""
        pass

    def inject_feedforward_current(self, current: float) -> None:
        """直接向 L4 兴奋性群体注入前馈电流"""
        if 4 in self.layers:
            self.layers[4].exc_pop.i_basal += current

    def inject_feedback_current(self, current: float) -> None:
        """直接向 L2/3 和 L5 的兴奋性群体注入反馈电流"""
        if 23 in self.layers:
            self.layers[23].exc_pop.i_apical += current
        if 5 in self.layers:
            self.layers[5].exc_pop.i_apical += current

    # =========================================================================
    # 仿真步进
    # =========================================================================

    def step(self, current_time: int, dt: float = 1.0) -> None:
        """推进一个时间步

        执行顺序:
        1. 计算所有 SynapseGroup 的突触电流 (from previous delivered spikes)
        2. 将电流加到目标 population
        3. 步进所有层 (各 population step)
        4. 将本步发放投递到 SynapseGroup delay buffer

        Args:
            current_time: 当前仿真时间步
            dt: 时间步长 (ms)
        """
        self._current_time = current_time
        self._current_spikes.clear()

        # === Phase 1: 计算突触电流 + 加到目标 ===
        for sg, src_pop, tgt_pop in self._connections:
            # 使用目标区室对应的电位计算电导
            if sg.target == CompartmentType.APICAL:
                v_post = tgt_pop.v_apical if tgt_pop.has_apical else tgt_pop.v_soma
            else:
                v_post = tgt_pop.v_soma

            i_syn = sg.step_and_compute(v_post)

            if sg.target == CompartmentType.BASAL:
                tgt_pop.i_basal += i_syn
            elif sg.target == CompartmentType.APICAL:
                tgt_pop.i_apical += i_syn
            else:  # SOMA
                tgt_pop.i_soma += i_syn

        # === Phase 2: 步进所有层 (L4 → L23 → L5 → L6) ===
        layer_order = [4, 23, 5, 6]
        for layer_id in layer_order:
            if layer_id in self.layers:
                layer_spikes = self.layers[layer_id].step(current_time, dt)
                self._current_spikes.update(layer_spikes)

        # === Phase 3: 投递本步发放到 SynapseGroup ===
        for sg, src_pop, tgt_pop in self._connections:
            sg.deliver_spikes(src_pop.fired, src_pop.spike_type)

    # =========================================================================
    # 输出查询
    # =========================================================================

    def get_prediction_error(self) -> List[Spike]:
        """L2/3 的 regular spikes → 预测误差"""
        result = []
        if 23 in self.layers:
            layer = self.layers[23]
            pop = layer.exc_pop
            fired_idx = np.nonzero(pop.fired)[0]
            for i in fired_idx:
                st = SpikeType(int(pop.spike_type[i]))
                if st == SpikeType.REGULAR:
                    result.append(Spike(layer.exc_id_base + i,
                                        self._current_time, st))
        return result

    def get_match_signal(self) -> List[Spike]:
        """L2/3 的 burst spikes → 预测匹配"""
        result = []
        if 23 in self.layers:
            layer = self.layers[23]
            pop = layer.exc_pop
            fired_idx = np.nonzero(pop.fired)[0]
            for i in fired_idx:
                st = SpikeType(int(pop.spike_type[i]))
                if st.is_burst:
                    result.append(Spike(layer.exc_id_base + i,
                                        self._current_time, st))
        return result

    def get_drive_output(self) -> List[Spike]:
        """L5 的 burst spikes → 驱动下游"""
        result = []
        if 5 in self.layers:
            layer = self.layers[5]
            pop = layer.exc_pop
            fired_idx = np.nonzero(pop.fired)[0]
            for i in fired_idx:
                st = SpikeType(int(pop.spike_type[i]))
                if st.is_burst:
                    result.append(Spike(layer.exc_id_base + i,
                                        self._current_time, st))
        return result

    def get_prediction(self) -> List[Spike]:
        """L6 的所有输出 → 预测信号"""
        result = []
        if 6 in self.layers:
            layer = self.layers[6]
            pop = layer.exc_pop
            fired_idx = np.nonzero(pop.fired)[0]
            for i in fired_idx:
                st = SpikeType(int(pop.spike_type[i]))
                if st.is_active:
                    result.append(Spike(layer.exc_id_base + i,
                                        self._current_time, st))
        return result

    def get_all_spikes(self) -> Dict[int, SpikeType]:
        """获取当前时间步所有发放"""
        return dict(self._current_spikes)

    # =========================================================================
    # 统计查询
    # =========================================================================

    def get_layer_firing_rates(self) -> Dict[int, float]:
        """获取各层平均发放率 (Hz)"""
        rates = {}
        for layer_id, layer in self.layers.items():
            if layer.n_total > 0:
                rates[layer_id] = layer.get_mean_firing_rate(self._current_time)
        return rates

    def get_layer_burst_ratios(self) -> Dict[int, float]:
        """获取各层兴奋性群体 burst 比率"""
        ratios = {}
        for layer_id, layer in self.layers.items():
            ratios[layer_id] = layer.get_exc_burst_ratio()
        return ratios

    @property
    def n_neurons(self) -> int:
        """总神经元数"""
        return sum(layer.n_total for layer in self.layers.values())

    @property
    def n_synapses(self) -> int:
        """总突触数"""
        return sum(sg.K for sg, _, _ in self._connections)

    @property
    def synapse_groups(self) -> List[SynapseGroup]:
        """所有 SynapseGroup (供外部检查权重)"""
        return [sg for sg, _, _ in self._connections]

    def get_neuron(self, neuron_id: int) -> None:
        """按 ID 获取神经元 (向量化版本不支持, 返回 None)"""
        return None

    # =========================================================================
    # 侧向输入接口 (Phase 1.8 新增)
    # =========================================================================

    def receive_lateral(self, spikes: List[Spike]) -> None:
        """接收侧向输入 (向量化模式: no-op)"""
        pass

    def inject_lateral_current(self, current: float) -> None:
        """直接向 L2/3 兴奋性群体注入侧向电流"""
        if 23 in self.layers:
            self.layers[23].exc_pop.i_basal += current

    # =========================================================================
    # 统一输出汇总 (Phase 1.8 新增)
    # =========================================================================

    def get_output_summary(self) -> dict:
        """获取当前时间步的所有输出信号汇总"""
        rates = self.get_layer_firing_rates()
        return {
            'prediction_error': self.get_prediction_error(),
            'match_signal': self.get_match_signal(),
            'prediction': self.get_prediction(),
            'drive': self.get_drive_output(),
            'l23_firing_rate': rates.get(23, 0.0),
            'l5_firing_rate': rates.get(5, 0.0),
            'l6_firing_rate': rates.get(6, 0.0),
        }

    def get_neuron_ids(self, layer_id: int, excitatory_only: bool = True) -> List[int]:
        """获取指定层的神经元 ID 列表"""
        if layer_id not in self.layers:
            return []
        layer = self.layers[layer_id]
        if excitatory_only:
            return list(range(layer.exc_id_base,
                              layer.exc_id_base + layer.exc_pop.n))
        ids = list(range(layer.exc_id_base,
                         layer.exc_id_base + layer.exc_pop.n))
        if layer.pv_pop is not None:
            ids.extend(range(layer.pv_id_base,
                             layer.pv_id_base + layer.pv_pop.n))
        if layer.sst_pop is not None:
            ids.extend(range(layer.sst_id_base,
                             layer.sst_id_base + layer.sst_pop.n))
        return ids

    # =========================================================================
    # 稳态可塑性接口 (Phase 1.8 新增)
    # =========================================================================

    def apply_homeostatic_scaling(self, homeostatic) -> None:
        """应用稳态可塑性 — 每隔 N 步调用一次

        对每个兴奋性 SynapseGroup 中的每个突触:
          1. 读取突触后神经元的发放率
          2. 缩放权重

        Args:
            homeostatic: HomeostaticPlasticity 规则实例
        """
        for sg, src_pop, tgt_pop in self._connections:
            if not sg.is_excitatory:
                continue
            rates = tgt_pop.get_firing_rates(
                window_ms=1000, current_time=self._current_time)
            for k in range(sg.K):
                post_idx = sg.post_ids[k]
                rate = float(rates[post_idx])
                sg.weights[k] = homeostatic.scale_weight(
                    sg.weights[k], rate, sg.w_min, sg.w_max
                )

    # =========================================================================
    # 生命周期
    # =========================================================================

    def reset(self) -> None:
        """重置柱到初始状态"""
        for layer in self.layers.values():
            layer.reset()
        for sg, _, _ in self._connections:
            sg.reset()
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