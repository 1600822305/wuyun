"""
Layer 0: Spike + Signal — 脉冲与信号原语

这是悟韵 (WuYun) 系统的最底层，定义了所有信号的基本类型。
不依赖任何其他悟韵模块。

主要组件:
- SpikeType: 脉冲类型枚举 (NONE/REGULAR/BURST)
- Spike: 脉冲事件数据结构
- SpikeTrain: 脉冲序列记录器
- CompartmentType: 突触目标区室 (SOMA/BASAL/APICAL)
- 各种信号类型枚举
"""

from wuyun.spike.signal_types import (
    SpikeType,
    CompartmentType,
    NeuronType,
    SynapseType,
    PlasticityType,
    OscillationBand,
    NeuromodulatorType,
    NeuromodulatorLevels,
    OscillationState,
)

from wuyun.spike.spike import (
    Spike,
    SpikeTrain,
)

from wuyun.spike.spike_bus import SpikeBus, SpikeReceiver

__all__ = [
    # 核心脉冲
    "SpikeType",
    "Spike",
    "SpikeTrain",
    # 脉冲总线
    "SpikeBus",
    "SpikeReceiver",
    # 区室
    "CompartmentType",
    # 类型枚举
    "NeuronType",
    "SynapseType",
    "PlasticityType",
    "OscillationBand",
    "NeuromodulatorType",
    # 数据结构
    "NeuromodulatorLevels",
    "OscillationState",
]