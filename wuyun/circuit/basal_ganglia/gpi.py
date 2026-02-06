"""
4-B: GPi — 苍白球内侧 (基底节输出核, 向量化版本)

默认状态: 持续高频抑制性发放 (~60-80Hz) → 压制丘脑
直接通路 (D1-MSN GABA) → 抑制 GPi → 去抑制丘脑 → 动作执行
间接通路 (经 GPe→STN) → 兴奋 GPi → 增强抑制 → 动作抑制

依赖: core/ (NeuronPopulation)
"""

from typing import List
import numpy as np

from wuyun.spike.spike import Spike
from wuyun.spike.signal_types import SpikeType
from wuyun.neuron.neuron_base import BASKET_PV_PARAMS, NeuronParams
from wuyun.neuron.compartment import SomaticParams
from wuyun.core.population import NeuronPopulation


# GPi 神经元参数: 持续高频 tonic 发放
_GPI_PARAMS = NeuronParams(
    somatic=SomaticParams(
        tau_m=10.0,             # 快速响应
        a=0.0,                  # 无适应 → 持续发放
        b=0.0,
        tau_w=50.0,
        v_threshold=-50.0,
        v_reset=-65.0,
        refractory_period=2,
    ),
    kappa=0.0,                  # 单区室 (抑制性输出核)
    neuron_type=BASKET_PV_PARAMS.neuron_type,  # 复用 PV 类型枚举
)


class GPi:
    """苍白球内侧 — 基底节输出核 (向量化版本)

    默认状态: 持续高频抑制性发放 (~60-80Hz) → 压制丘脑
    直接通路 (D1-MSN GABA) → 抑制 GPi → 去抑制丘脑 → 动作执行
    间接通路 (经 GPe→STN) → 兴奋 GPi → 增强抑制 → 动作抑制

    Attributes:
        pop: NeuronPopulation (GPi 神经元群体)
        tonic_drive: 持续注入电流, 维持 tonic 发放
    """

    def __init__(
        self,
        n_neurons: int = 10,
        tonic_drive: float = 25.0,
        seed: int = None,
    ):
        self.n_neurons = n_neurons
        self.tonic_drive = tonic_drive
        self._rng = np.random.RandomState(seed)

        # 向量化神经元群体
        self.pop = NeuronPopulation(n_neurons, _GPI_PARAMS)

        # 发放率追踪
        self._spike_counts = np.zeros(n_neurons)
        self._step_count = 0
        self._rate_window = 100

    def inject_direct_inhibition(
        self, d1_spikes: List[Spike], d1_rates: np.ndarray,
        gain: float = 15.0,
    ) -> None:
        """注入直接通路抑制 (D1-MSN → GPi, GABA)

        D1 活跃 → 抑制 GPi → GPi 发放率下降 → 去抑制丘脑

        Args:
            d1_spikes: D1-MSN 的输出脉冲列表
            d1_rates: D1-MSN 发放率向量
            gain: 抑制增益
        """
        mean_rate = np.mean(d1_rates) if len(d1_rates) > 0 else 0.0
        normalized = min(mean_rate / 200.0, 1.0)
        inhibition = -gain * normalized
        self.pop.i_soma += inhibition

    def inject_channel_inhibition(
        self, channel_rates: np.ndarray,
        n_channels: int,
        gain: float = 15.0,
    ) -> None:
        """注入按通道的直接通路抑制

        每个动作通道的 D1 发放率分别映射到对应 GPi 子群。

        Args:
            channel_rates: 每通道 D1 平均发放率, shape=(n_channels,)
            n_channels: 动作通道数
            gain: 抑制增益
        """
        n_per_ch = max(1, self.n_neurons // n_channels)
        for ch in range(n_channels):
            rate = channel_rates[ch] if ch < len(channel_rates) else 0.0
            normalized = min(rate / 200.0, 1.0)
            inhibition = -gain * normalized
            start = ch * n_per_ch
            end = min(start + n_per_ch, self.n_neurons)
            self.pop.i_soma[start:end] += inhibition

    def inject_indirect_excitation(
        self, stn_spikes: List[Spike], stn_rates: np.ndarray,
        gain: float = 10.0,
    ) -> None:
        """注入间接通路兴奋 (STN → GPi, 谷氨酸)

        STN 活跃 → 兴奋 GPi → GPi 发放率上升 → 增强丘脑抑制

        Args:
            stn_spikes: STN 输出脉冲列表
            stn_rates: STN 发放率向量
            gain: 兴奋增益
        """
        mean_rate = np.mean(stn_rates) if len(stn_rates) > 0 else 0.0
        normalized = min(mean_rate / 50.0, 1.0)
        excitation = gain * normalized
        self.pop.i_basal += excitation

    def step(self, t: int) -> None:
        """推进一个时间步

        注入 tonic drive 维持持续高频发放, 然后更新所有神经元。

        Args:
            t: 当前仿真时间步
        """
        self._step_count += 1

        # tonic drive: 持续注入, 维持 ~60-80Hz 发放
        self.pop.i_basal += self.tonic_drive

        self.pop.step(t)
        self._spike_counts += self.pop.fired.astype(np.float64)

    def get_output_rates(self) -> np.ndarray:
        """输出发放率 (Hz)

        高发放率 = 强抑制丘脑, 低发放率 = 去抑制丘脑
        """
        if self._step_count == 0:
            return np.zeros(self.n_neurons)
        return self._spike_counts * (1000.0 / self._step_count)

    def get_thalamic_inhibition(self) -> float:
        """对丘脑的平均抑制强度

        返回 0~1 范围: 1=完全抑制, 0=完全去抑制
        """
        rates = self.get_output_rates()
        mean_rate = np.mean(rates)
        return float(min(mean_rate / 80.0, 1.0))

    def get_output_spikes(self) -> List[Spike]:
        """获取最近一步的所有脉冲"""
        spikes = []
        fired_idx = np.nonzero(self.pop.fired)[0]
        for i in fired_idx:
            spikes.append(Spike(
                5600 + i, self._step_count,
                SpikeType(int(self.pop.spike_type[i]))))
        return spikes

    def reset_spike_counts(self) -> None:
        """重置发放计数器"""
        self._spike_counts = np.zeros(self.n_neurons)
        self._step_count = 0

    def get_state(self) -> dict:
        """返回当前状态供调试"""
        return {
            "output_rates": self.get_output_rates().tolist(),
            "thalamic_inhibition": self.get_thalamic_inhibition(),
            "mean_v": float(np.mean(self.pop.v_soma)),
        }