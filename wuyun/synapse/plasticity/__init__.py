"""
Layer 1: 突触可塑性规则模块

提供多种可塑性规则，通过组合模式与 SynapseBase 集成:
- PlasticityRule: 基类 (抽象接口)
- ClassicalSTDP: 经典 STDP (兴奋性突触)
- DAModulatedSTDP: 三因子 STDP (DA调制 + 资格痕迹)
- InhibitorySTDP: 抑制性 STDP (对称型, E/I平衡)

依赖约束:
- 只依赖 wuyun.spike.signal_types (PlasticityType 枚举)
- 不依赖 synapse_base.py (避免循环依赖)
- 不依赖 neuron/ 或 circuit/
"""

from wuyun.synapse.plasticity.plasticity_base import PlasticityRule
from wuyun.synapse.plasticity.classical_stdp import ClassicalSTDP, ClassicalSTDPParams
from wuyun.synapse.plasticity.da_modulated_stdp import DAModulatedSTDP, DAModulatedSTDPParams
from wuyun.synapse.plasticity.inhibitory_stdp import InhibitorySTDP, InhibitorySTDPParams

__all__ = [
    "PlasticityRule",
    "ClassicalSTDP",
    "ClassicalSTDPParams",
    "DAModulatedSTDP",
    "DAModulatedSTDPParams",
    "InhibitorySTDP",
    "InhibitorySTDPParams",
]