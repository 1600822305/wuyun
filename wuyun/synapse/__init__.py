"""
Layer 1: Synapse + Channel — 突触与通道抽象

突触是神经元间通信的基本连接单元:
- 接收突触前脉冲 (Spike)
- 经过传导延迟后
- 产生突触电流注入突触后神经元的特定区室

关键设计:
  target_compartment 决定电流注入 BASAL (前馈) 还是 APICAL (反馈),
  这是预测编码的硬件基础。

可塑性规则:
  - ClassicalSTDP: 经典 STDP (兴奋性突触, 皮层主要规则)
  - DAModulatedSTDP: 三因子 STDP (DA调制, PFC/纹状体)
  - InhibitorySTDP: 抑制性 STDP (对称型, E/I平衡)
"""

from wuyun.synapse.synapse_base import (
    SynapseBase,
    SynapseParams,
    AMPA_PARAMS,
    NMDA_PARAMS,
    GABA_A_PARAMS,
    GABA_B_PARAMS,
)

from wuyun.synapse.plasticity import (
    PlasticityRule,
    ClassicalSTDP,
    ClassicalSTDPParams,
    DAModulatedSTDP,
    DAModulatedSTDPParams,
    InhibitorySTDP,
    InhibitorySTDPParams,
)

__all__ = [
    # 突触
    "SynapseBase",
    "SynapseParams",
    "AMPA_PARAMS",
    "NMDA_PARAMS",
    "GABA_A_PARAMS",
    "GABA_B_PARAMS",
    # 可塑性规则
    "PlasticityRule",
    "ClassicalSTDP",
    "ClassicalSTDPParams",
    "DAModulatedSTDP",
    "DAModulatedSTDPParams",
    "InhibitorySTDP",
    "InhibitorySTDPParams",
]