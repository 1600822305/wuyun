"""
Layer 3: Circuit — 环路与皮层柱

提供皮层柱组装和层级管理:
- Layer: 单层神经元群
- CorticalColumn: 6 层预测编码计算单元
- create_sensory_column: 感觉皮层柱工厂函数

所有新皮质区域使用同一个 CorticalColumn 类，
功能差异通过参数/连接/可塑性的不同实现。
"""

from wuyun.circuit.layer import Layer
from wuyun.circuit.cortical_column import CorticalColumn
from wuyun.circuit.column_factory import create_sensory_column

__all__ = [
    "Layer",
    "CorticalColumn",
    "create_sensory_column",
]