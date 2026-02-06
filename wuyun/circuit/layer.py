"""
Layer 3: Layer — 皮层柱的单层神经元群 (向量化版本)

管理该层内的所有 NeuronPopulation，提供层级步进和统计接口。

每个 Layer 包含:
- 兴奋性 NeuronPopulation (锥体/星形)
- 抑制性 NeuronPopulation (PV+, 可选 SST+)

依赖:
- wuyun.spike (SpikeType)
- wuyun.core (NeuronPopulation)
"""

from typing import Dict, Optional
import numpy as np
from wuyun.spike.signal_types import SpikeType
from wuyun.core.population import NeuronPopulation


class Layer:
    """皮层柱的一层 (向量化版本)

    管理该层内的多个 NeuronPopulation。

    Attributes:
        layer_id: 层编号 (23=L2/3, 4=L4, 5=L5, 6=L6)
        exc_pop: 兴奋性群体
        pv_pop: PV+ 抑制性群体 (可选)
        sst_pop: SST+ 抑制性群体 (可选)
        exc_id_base: 兴奋性神经元全局 ID 起始
        pv_id_base: PV+ 神经元全局 ID 起始
        sst_id_base: SST+ 神经元全局 ID 起始
    """

    def __init__(
        self,
        layer_id: int,
        exc_pop: NeuronPopulation,
        pv_pop: Optional[NeuronPopulation] = None,
        sst_pop: Optional[NeuronPopulation] = None,
        exc_id_base: int = 0,
        pv_id_base: int = 0,
        sst_id_base: int = 0,
    ):
        self.layer_id = layer_id
        self.exc_pop = exc_pop
        self.pv_pop = pv_pop
        self.sst_pop = sst_pop
        self.exc_id_base = exc_id_base
        self.pv_id_base = pv_id_base
        self.sst_id_base = sst_id_base

        # 当前时间步发放缓存
        self._last_spikes: Dict[int, SpikeType] = {}

        # burst 比率窗口统计
        self._exc_active_count = 0
        self._exc_burst_count = 0

    def step(self, current_time: int, dt: float = 1.0) -> Dict[int, SpikeType]:
        """推进一个时间步: 步进所有群体, 返回活跃神经元映射

        Returns:
            {neuron_id: spike_type} 仅包含发放的神经元
        """
        self._last_spikes.clear()

        # --- 兴奋性群体 ---
        self.exc_pop.step(current_time)
        exc_fired = np.nonzero(self.exc_pop.fired)[0]
        for i in exc_fired:
            st = SpikeType(int(self.exc_pop.spike_type[i]))
            self._last_spikes[self.exc_id_base + i] = st
        # burst 统计
        self._exc_active_count += len(exc_fired)
        for i in exc_fired:
            sv = int(self.exc_pop.spike_type[i])
            if sv in (SpikeType.BURST_START.value,
                      SpikeType.BURST_CONTINUE.value,
                      SpikeType.BURST_END.value):
                self._exc_burst_count += 1

        # --- PV+ ---
        if self.pv_pop is not None:
            self.pv_pop.step(current_time)
            pv_fired = np.nonzero(self.pv_pop.fired)[0]
            for i in pv_fired:
                self._last_spikes[self.pv_id_base + i] = \
                    SpikeType(int(self.pv_pop.spike_type[i]))

        # --- SST+ ---
        if self.sst_pop is not None:
            self.sst_pop.step(current_time)
            sst_fired = np.nonzero(self.sst_pop.fired)[0]
            for i in sst_fired:
                self._last_spikes[self.sst_id_base + i] = \
                    SpikeType(int(self.sst_pop.spike_type[i]))

        return self._last_spikes

    def get_last_spikes(self) -> Dict[int, SpikeType]:
        """获取最近一次 step 的发放结果"""
        return self._last_spikes

    def get_mean_firing_rate(self, current_time: int) -> float:
        """所有神经元的平均发放率 (Hz)"""
        total_rate = 0.0
        total_n = 0

        rates = self.exc_pop.get_firing_rates(
            window_ms=1000, current_time=current_time)
        total_rate += float(rates.sum())
        total_n += self.exc_pop.n

        if self.pv_pop is not None:
            rates = self.pv_pop.get_firing_rates(
                window_ms=1000, current_time=current_time)
            total_rate += float(rates.sum())
            total_n += self.pv_pop.n

        if self.sst_pop is not None:
            rates = self.sst_pop.get_firing_rates(
                window_ms=1000, current_time=current_time)
            total_rate += float(rates.sum())
            total_n += self.sst_pop.n

        return total_rate / total_n if total_n > 0 else 0.0

    def get_exc_burst_ratio(self) -> float:
        """兴奋性群体的累积 burst 比率"""
        if self._exc_active_count == 0:
            return 0.0
        return self._exc_burst_count / self._exc_active_count

    @property
    def n_excitatory(self) -> int:
        return self.exc_pop.n

    @property
    def n_inhibitory(self) -> int:
        n = 0
        if self.pv_pop is not None:
            n += self.pv_pop.n
        if self.sst_pop is not None:
            n += self.sst_pop.n
        return n

    @property
    def n_total(self) -> int:
        return self.n_excitatory + self.n_inhibitory

    def reset(self) -> None:
        """重置该层所有群体"""
        self.exc_pop.reset()
        if self.pv_pop is not None:
            self.pv_pop.reset()
        if self.sst_pop is not None:
            self.sst_pop.reset()
        self._last_spikes.clear()
        self._exc_active_count = 0
        self._exc_burst_count = 0

    def __repr__(self) -> str:
        return (f"Layer(id={self.layer_id}, "
                f"E={self.n_excitatory}, I={self.n_inhibitory}, "
                f"total={self.n_total})")