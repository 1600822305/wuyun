"""
Layer 0: Spike 事件定义

Spike 是悟韵 (WuYun) 系统的"原子"信号单元。

神经元通过 Spike 事件进行通信，每个 Spike 携带:
- 源神经元 ID
- 时间戳
- 脉冲类型 (REGULAR / BURST_START / BURST_CONTINUE / BURST_END)

脉冲类型直接编码预测编码信息:
- REGULAR: 有前馈输入但未被预测 → 预测误差 → 向高层传递
- BURST:   前馈+反馈同时激活 → 预测匹配 → 驱动学习+注意力
"""

from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional
import numpy as np

from wuyun.spike.signal_types import SpikeType


# =============================================================================
# Spike 事件
# =============================================================================

@dataclass(frozen=True, slots=True)
class Spike:
    """单个脉冲事件

    这是系统中最基本的通信单元。
    frozen=True 确保 Spike 一旦创建不可修改 (事件不可变)。
    slots=True 优化内存使用。

    Attributes:
        source_id: 发放神经元的全局唯一 ID
        timestamp: 发放时间 (以时间步为单位, 1步 = 1ms)
        spike_type: 脉冲类型 (REGULAR / BURST_START / ...)
    """
    source_id: int
    timestamp: int
    spike_type: SpikeType = SpikeType.REGULAR

    @property
    def is_burst(self) -> bool:
        """是否为 burst 类型脉冲"""
        return self.spike_type.is_burst

    @property
    def is_active(self) -> bool:
        """是否有实际脉冲发放"""
        return self.spike_type.is_active


# =============================================================================
# SpikeTrain — 脉冲序列
# =============================================================================

class SpikeTrain:
    """单个神经元的脉冲序列记录

    用于:
    - STDP 计算 (需要精确的前后脉冲时间)
    - 发放率统计
    - burst 检测

    Attributes:
        neuron_id: 所属神经元 ID
        capacity: 最大记录长度 (滑动窗口, 防止内存膨胀)
    """

    def __init__(self, neuron_id: int, capacity: int = 1000):
        self.neuron_id = neuron_id
        self.capacity = capacity
        self._timestamps: deque = deque(maxlen=capacity)
        self._types: deque = deque(maxlen=capacity)

    def record(self, spike: Spike) -> None:
        """记录一个脉冲事件"""
        self._timestamps.append(spike.timestamp)
        self._types.append(spike.spike_type)
        # deque(maxlen) 自动淘汰最旧记录, 无需手动 pop

    def record_spike(self, timestamp: int, spike_type: SpikeType) -> None:
        """直接记录脉冲参数 (避免创建 Spike 对象的开销)"""
        self._timestamps.append(timestamp)
        self._types.append(spike_type)
        # deque(maxlen) 自动淘汰最旧记录, 无需手动 pop

    @property
    def last_spike_time(self) -> Optional[int]:
        """最近一次脉冲的时间戳, 无记录时返回 None"""
        return self._timestamps[-1] if self._timestamps else None

    @property
    def last_spike_type(self) -> SpikeType:
        """最近一次脉冲的类型, 无记录时返回 NONE"""
        return self._types[-1] if self._types else SpikeType.NONE

    @property
    def count(self) -> int:
        """记录的脉冲总数"""
        return len(self._timestamps)

    def firing_rate(self, window_ms: int = 1000) -> float:
        """计算指定时间窗口内的平均发放率 (Hz)

        Args:
            window_ms: 统计窗口大小 (毫秒/时间步)

        Returns:
            发放率 (Hz). 窗口内无脉冲则返回 0.0.
        """
        if not self._timestamps:
            return 0.0
        latest = self._timestamps[-1]
        window_start = latest - window_ms
        # 统计窗口内的脉冲数
        count = sum(1 for t in self._timestamps if t > window_start)
        # Hz = spikes_per_second = count / (window_ms / 1000)
        return count * 1000.0 / window_ms

    def burst_ratio(self, window_ms: int = 1000) -> float:
        """计算指定窗口内的 burst/total 比率

        这个比率反映了预测匹配程度:
        - 高 burst 比率 → 预测准确, 预测编码高效
        - 低 burst 比率 → 大量预测误差, 需要学习

        Returns:
            burst 比率 [0.0, 1.0]. 无脉冲返回 0.0.
        """
        if not self._timestamps:
            return 0.0
        latest = self._timestamps[-1]
        window_start = latest - window_ms
        total = 0
        bursts = 0
        for t, st in zip(self._timestamps, self._types):
            if t > window_start:
                total += 1
                if st in (SpikeType.BURST_START,
                          SpikeType.BURST_CONTINUE,
                          SpikeType.BURST_END):
                    bursts += 1
        return bursts / total if total > 0 else 0.0

    def get_recent_times(self, window_ms: int = 100) -> List[int]:
        """获取最近窗口内的所有脉冲时间戳 (用于 STDP 计算)

        Args:
            window_ms: 回溯窗口 (STDP 时间窗口典型 ±20ms, 留余量)

        Returns:
            时间戳列表 (从早到晚)
        """
        if not self._timestamps:
            return []
        latest = self._timestamps[-1]
        window_start = latest - window_ms
        return [t for t in self._timestamps if t > window_start]

    def clear(self) -> None:
        """清空所有记录"""
        self._timestamps.clear()
        self._types.clear()

    def __len__(self) -> int:
        return len(self._timestamps)

    def __repr__(self) -> str:
        rate = self.firing_rate() if self._timestamps else 0.0
        return (f"SpikeTrain(neuron={self.neuron_id}, "
                f"count={self.count}, rate={rate:.1f}Hz)")