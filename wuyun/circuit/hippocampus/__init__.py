"""
Phase 3: 海马记忆系统

DG→CA3→CA1 三突触环路:
  - DentateGyrus: 模式分离 (EC 输入 → 稀疏正交化表示)
  - CA3Network: 自联想记忆 (循环连接 + STDP → 模式存储/补全)
  - CA1Network: 比较/输出 (CA3 回忆 vs EC-III 感知 → 匹配/新奇)
  - HippocampalLoop: 全环路 + Theta 门控 (编码/检索自动交替)

依赖方向: spike/ ← synapse/ ← neuron/ ← circuit/hippocampus/
"""

from wuyun.circuit.hippocampus.dentate_gyrus import DentateGyrus
from wuyun.circuit.hippocampus.ca3_network import CA3Network
from wuyun.circuit.hippocampus.ca1_network import CA1Network
from wuyun.circuit.hippocampus.hippocampal_loop import (
    HippocampalLoop,
    create_hippocampal_loop,
)

__all__ = [
    "DentateGyrus",
    "CA3Network",
    "CA1Network",
    "HippocampalLoop",
    "create_hippocampal_loop",
]