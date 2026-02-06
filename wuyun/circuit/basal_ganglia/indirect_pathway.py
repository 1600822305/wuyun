"""
4-C: GPe + STN — 间接通路 + 超直接通路

GPe (苍白球外侧):
  - 间接通路中继
  - 持续 tonic 发放 (~40Hz)
  - 接收: D2-MSN 的 GABA 抑制 → D2 激活时 GPe 被抑制
  - 输出: GABA 抑制 STN

STN (丘脑底核):
  - 基底节唯一的兴奋性核团
  - 超直接通路: 皮层 → STN (快速, 延迟 ~1-2ms)
  - 间接通路: GPe → STN (GABA 抑制)
  - 输出: 谷氨酸兴奋 GPi → 增强抑制 (全局刹车)

依赖: spike/ ← synapse/ ← neuron/ ← 本模块
"""

from typing import List
import numpy as np

from wuyun.spike.spike import Spike
from wuyun.spike.spike_bus import SpikeBus
from wuyun.spike.signal_types import SpikeType
from wuyun.neuron.neuron_base import (
    NeuronBase,
    NeuronParams,
    BASKET_PV_PARAMS,
    STN_PARAMS,
)
from wuyun.neuron.compartment import SomaticParams


# GPe 神经元参数: 持续中频 tonic 发放
_GPE_PARAMS = NeuronParams(
    somatic=SomaticParams(
        tau_m=10.0,
        a=0.0,
        b=0.0,
        tau_w=50.0,
        v_threshold=-50.0,
        v_reset=-65.0,
        refractory_period=2,
    ),
    kappa=0.0,
    neuron_type=BASKET_PV_PARAMS.neuron_type,
)


class GPe:
    """苍白球外侧 — 间接通路中继

    持续 tonic 发放 (~40Hz)。
    D2-MSN 激活 → GABA 抑制 GPe → GPe 沉默 → STN 脱抑制 → GPi 增强。

    Attributes:
        neurons: GPe 神经元列表
        tonic_drive: 维持 tonic 发放的注入电流
    """

    def __init__(
        self,
        n_neurons: int = 10,
        tonic_drive: float = 22.0,
        seed: int = None,
    ):
        self.n_neurons = n_neurons
        self.tonic_drive = tonic_drive
        self._rng = np.random.RandomState(seed)

        # GPe 神经元: ID 范围 [5800, 5800+n_neurons)
        self.neurons: List[NeuronBase] = []
        for i in range(n_neurons):
            n = NeuronBase(neuron_id=5800 + i, params=_GPE_PARAMS)
            self.neurons.append(n)

        self._bus = SpikeBus()
        self._spike_counts = np.zeros(n_neurons)
        self._step_count = 0
        self._rate_window = 100

    def inject_d2_inhibition(
        self, d2_spikes: List[Spike], d2_rates: np.ndarray,
        gain: float = 10.0,
    ) -> None:
        """注入 D2-MSN → GPe 抑制 (GABA)

        D2 活跃 → 抑制 GPe → GPe 发放下降

        Args:
            d2_spikes: D2-MSN 输出脉冲
            d2_rates: D2-MSN 发放率
            gain: 抑制增益
        """
        mean_rate = np.mean(d2_rates) if len(d2_rates) > 0 else 0.0
        normalized = min(mean_rate / 50.0, 1.0)
        inhibition = -gain * normalized

        for neuron in self.neurons:
            neuron.inject_somatic_current(inhibition)

    def step(self, t: int) -> None:
        """推进一个时间步"""
        self._step_count += 1

        for i, neuron in enumerate(self.neurons):
            neuron.inject_basal_current(self.tonic_drive)
            spike_type = neuron.step(t)
            if spike_type.is_active:
                self._bus.emit(Spike(neuron.id, t, spike_type))
                self._spike_counts[i] += 1

        self._bus.step(t)

    def get_output_rates(self) -> np.ndarray:
        """输出发放率 (Hz)"""
        if self._step_count == 0:
            return np.zeros(self.n_neurons)
        return self._spike_counts * (1000.0 / self._step_count)

    def get_output_spikes(self) -> List[Spike]:
        """获取最近一步的输出脉冲"""
        spikes = []
        for neuron in self.neurons:
            if neuron.current_spike_type.is_active:
                spikes.append(Spike(neuron.id, self._step_count, neuron.current_spike_type))
        return spikes

    def reset_spike_counts(self) -> None:
        """重置发放计数器"""
        self._spike_counts = np.zeros(self.n_neurons)
        self._step_count = 0


class STN:
    """丘脑底核 — 超直接通路 + 间接通路汇聚

    基底节唯一的兴奋性核团。接收:
    - 皮层直接输入 (超直接通路, 快速, 延迟 ~1-2ms)
    - GPe 抑制性输入 (间接通路)
    输出: 谷氨酸兴奋 GPi → 增强抑制 (全局刹车)

    Attributes:
        neurons: STN 神经元列表
    """

    def __init__(
        self,
        n_neurons: int = 8,
        seed: int = None,
    ):
        self.n_neurons = n_neurons
        self._rng = np.random.RandomState(seed)

        # STN 神经元: ID 范围 [6000, 6000+n_neurons)
        self.neurons: List[NeuronBase] = []
        for i in range(n_neurons):
            n = NeuronBase(neuron_id=6000 + i, params=STN_PARAMS)
            self.neurons.append(n)

        self._bus = SpikeBus()
        self._spike_counts = np.zeros(n_neurons)
        self._step_count = 0
        self._rate_window = 100

    def inject_cortical_input(
        self, cortical_input: np.ndarray,
        gain: float = 20.0,
    ) -> None:
        """注入皮层 → STN 输入 (超直接通路, 谷氨酸)

        超直接通路: 皮层直接驱动 STN, 延迟最短 (~1ms)
        → STN 兴奋 → GPi 增强 → 全局刹车 (先于 Go/NoGo 通路)

        Args:
            cortical_input: 皮层输入向量
            gain: 兴奋增益 (超直接通路最强: 20.0)
        """
        for i, neuron in enumerate(self.neurons):
            idx = i % len(cortical_input)
            if cortical_input[idx] > 0:
                neuron.inject_basal_current(gain * float(cortical_input[idx]))

    def inject_gpe_inhibition(
        self, gpe_spikes: List[Spike], gpe_rates: np.ndarray,
        gain: float = 8.0,
    ) -> None:
        """注入 GPe → STN 抑制 (GABA)

        GPe 活跃 → 抑制 STN → STN 沉默 → GPi 减弱
        GPe 被 D2 抑制 → STN 脱抑制 → GPi 增强

        Args:
            gpe_spikes: GPe 输出脉冲
            gpe_rates: GPe 发放率
            gain: 抑制增益
        """
        mean_rate = np.mean(gpe_rates) if len(gpe_rates) > 0 else 0.0
        normalized = min(mean_rate / 50.0, 1.0)
        inhibition = -gain * normalized

        for neuron in self.neurons:
            neuron.inject_somatic_current(inhibition)

    def step(self, t: int) -> None:
        """推进一个时间步"""
        self._step_count += 1

        for i, neuron in enumerate(self.neurons):
            spike_type = neuron.step(t)
            if spike_type.is_active:
                self._bus.emit(Spike(neuron.id, t, spike_type))
                self._spike_counts[i] += 1

        self._bus.step(t)

    def get_output_rates(self) -> np.ndarray:
        """输出发放率 (Hz)"""
        if self._step_count == 0:
            return np.zeros(self.n_neurons)
        return self._spike_counts * (1000.0 / self._step_count)

    def get_output_spikes(self) -> List[Spike]:
        """获取最近一步的输出脉冲"""
        spikes = []
        for neuron in self.neurons:
            if neuron.current_spike_type.is_active:
                spikes.append(Spike(neuron.id, self._step_count, neuron.current_spike_type))
        return spikes

    def reset_spike_counts(self) -> None:
        """重置发放计数器"""
        self._spike_counts = np.zeros(self.n_neurons)
        self._step_count = 0