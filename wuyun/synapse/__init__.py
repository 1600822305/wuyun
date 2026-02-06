"""
Layer 1: Synapse + Channel — 突触与通道抽象

突触是神经元间通信的基本连接单元:
- 接收突触前脉冲 (Spike)
- 经过传导延迟后
- 产生突触电流注入突触后神经元的特定区室

关键设计:
  target_compartment 决定电流注入 BASAL (前馈) 还是 APICAL (反馈),
  这是预测编码的硬件基础。
"""

from wuyun.synapse.synapse_base import (
    SynapseBase,
    SynapseParams,
    AMPA_PARAMS,
    NMDA_PARAMS,
    GABA_A_PARAMS,
    GABA_B_PARAMS,
)

__all__ = [
    "SynapseBase",
    "SynapseParams",
    "AMPA_PARAMS",
    "NMDA_PARAMS",
    "GABA_A_PARAMS",
    "GABA_B_PARAMS",
]