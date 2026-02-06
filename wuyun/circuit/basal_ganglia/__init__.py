"""
基底神经节环路 (Basal Ganglia Circuit)

三条通路:
1. 直接 (Go):   皮层 → D1-MSN ─┤ GPi → (去抑制) → 丘脑
2. 间接 (NoGo): 皮层 → D2-MSN ─┤ GPe ─┤ STN → GPi → (增强抑制)
3. 超直接 (Stop): 皮层 → STN → GPi → (全局刹车)

─┤ 表示 GABA 抑制, → 表示谷氨酸兴奋

模块:
- Striatum: 纹状体 (D1/D2-MSN + FSI)
- GPi: 苍白球内侧 (输出核)
- GPe: 苍白球外侧 (间接通路中继)
- STN: 丘脑底核 (超直接通路 + 间接通路汇聚)
- BasalGangliaCircuit: 完整基底节环路
"""

from wuyun.circuit.basal_ganglia.striatum import Striatum
from wuyun.circuit.basal_ganglia.gpi import GPi
from wuyun.circuit.basal_ganglia.indirect_pathway import GPe, STN
from wuyun.circuit.basal_ganglia.basal_ganglia import BasalGangliaCircuit

__all__ = [
    "Striatum",
    "GPi",
    "GPe",
    "STN",
    "BasalGangliaCircuit",
]