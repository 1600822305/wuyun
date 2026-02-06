"""
Layer 2: Neuron — 双区室神经元模型

悟韵 (WuYun) 系统的"心脏"。

核心创新: 双区室模型 (顶端树突 + 胞体/基底树突)
- 基底树突接收前馈输入 (来自丘脑/低层柱)
- 顶端树突接收反馈预测 (来自高层皮层)
- 两者是否同时激活决定 burst vs regular spike
- 这是预测编码的硬件基础

主要组件:
- NeuronBase: 双区室 AdLIF+ 神经元基类
- SomaticCompartment: 胞体区室
- ApicalCompartment: 顶端树突区室
- NeuronParams: 参数包
"""

from wuyun.neuron.compartment import (
    SomaticCompartment,
    ApicalCompartment,
    SomaticParams,
    ApicalParams,
)

from wuyun.neuron.neuron_base import (
    NeuronBase,
    NeuronParams,
    L23_PYRAMIDAL_PARAMS,
    L5_PYRAMIDAL_PARAMS,
    L6_PYRAMIDAL_PARAMS,
    STELLATE_PARAMS,
    BASKET_PV_PARAMS,
    MARTINOTTI_SST_PARAMS,
    VIP_PARAMS,
)

__all__ = [
    # 神经元
    "NeuronBase",
    "NeuronParams",
    # 区室
    "SomaticCompartment",
    "ApicalCompartment",
    "SomaticParams",
    "ApicalParams",
    # 预定义参数
    "L23_PYRAMIDAL_PARAMS",
    "L5_PYRAMIDAL_PARAMS",
    "L6_PYRAMIDAL_PARAMS",
    "STELLATE_PARAMS",
    "BASKET_PV_PARAMS",
    "MARTINOTTI_SST_PARAMS",
    "VIP_PARAMS",
]