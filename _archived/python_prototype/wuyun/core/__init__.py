"""
wuyun.core — 向量化计算核心引擎

提供 NeuronPopulation 和 SynapseGroup 两大基础组件,
将逐对象 Python 循环替换为 NumPy 批量矩阵运算。

设计文档: docs/02_neuron_system_design.md
"""

from wuyun.core.population import NeuronPopulation
from wuyun.core.synapse_group import SynapseGroup

__all__ = [
    'NeuronPopulation',
    'SynapseGroup',
]
