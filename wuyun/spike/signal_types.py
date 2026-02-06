"""
Layer 0: 信号类型枚举与消息格式

定义悟韵 (WuYun) 系统中所有信号的基本类型:
- SpikeType: 脉冲类型 (NONE/REGULAR/BURST)
- CompartmentType: 突触目标区室 (SOMA/BASAL/APICAL)
- 各总线消息格式的数据结构

这些是整个系统最底层的"原子"定义，不依赖任何其他模块。
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# =============================================================================
# 脉冲类型枚举
# =============================================================================

class SpikeType(IntEnum):
    """脉冲类型枚举

    双区室神经元的发放模式直接编码预测编码信息:
    - NONE:    无脉冲 (沉默)
    - REGULAR: 普通脉冲 — 有前馈输入但未被预测 = 预测误差
    - BURST_START:    爆发发放起始 — 前馈+反馈同时激活 = 预测匹配
    - BURST_CONTINUE: 爆发发放持续
    - BURST_END:      爆发发放结束

    预测编码对应关系:
    | 基底树突(前馈) | 顶端树突(反馈) | 发放模式    | 含义       |
    |---------------|---------------|------------|-----------|
    | ✔ 激活        | ✖ 未激活      | REGULAR    | 预测误差   |
    | ✔ 激活        | ✔ 激活        | BURST      | 预测匹配   |
    | ✖ 未激活      | ✔ 激活        | NONE       | 无事发生   |
    | ✖ 未激活      | ✖ 未激活      | NONE       | 沉默       |
    """
    NONE = 0
    REGULAR = 1
    BURST_START = 2
    BURST_CONTINUE = 3
    BURST_END = 4

    @property
    def is_burst(self) -> bool:
        """是否为 burst 类型的脉冲"""
        return self in (SpikeType.BURST_START,
                        SpikeType.BURST_CONTINUE,
                        SpikeType.BURST_END)

    @property
    def is_active(self) -> bool:
        """是否有实际脉冲发放"""
        return self != SpikeType.NONE


# =============================================================================
# 区室类型枚举
# =============================================================================

class CompartmentType(IntEnum):
    """突触目标区室类型

    双区室神经元模型中，突触输入根据目标区室被分流:
    - SOMA:   胞体 — 直接影响 V_s (如 PV+ 篮状细胞的抑制)
    - BASAL:  基底树突 — 前馈输入 (来自丘脑/低层柱 L2/3)
    - APICAL: 顶端树突 — 反馈输入 (来自高层皮层 L6 预测)

    这个分流是预测编码的硬件基础:
    - basal 激活但 apical 未激活 → regular spike (预测误差)
    - basal + apical 同时激活 → burst (预测匹配)
    """
    SOMA = 0
    BASAL = 1
    APICAL = 2


# =============================================================================
# 神经元类型枚举
# =============================================================================

class NeuronType(IntEnum):
    """神经元类型枚举

    对应设计文档 02_neuron_system_design.md 1.3 节的类型体系。
    每种类型有不同的参数集 (τ_m, τ_a, κ, a, b 等)。
    """
    # 兴奋性 (双区室, κ > 0)
    L23_PYRAMIDAL = 10       # 浅层锥体 — 跨柱通信 κ=0.3
    L5_PYRAMIDAL = 11        # 深层锥体 — 驱动输出 κ=0.6
    L6_PYRAMIDAL = 12        # 多形层锥体 — 丘脑反馈 κ=0.2
    STELLATE = 13            # L4 星形细胞 κ=0.1
    GRANULE = 14             # 颗粒细胞 (海马DG/小脑) κ=0
    MOSSY = 15               # 苔藓细胞 (海马DG) κ=0

    # 抑制性 (单区室, κ = 0)
    BASKET_PV = 20           # PV+ 篮状细胞 → 靶向胞体
    MARTINOTTI_SST = 21      # SST+ Martinotti → 靶向顶端树突
    VIP_INTERNEURON = 22     # VIP → 去抑制 (抑制SST)
    CHANDELIER_PV = 23       # PV+ 枝形烛台 → 轴突起始段
    NGF = 24                 # 慢抑制 GABA_B

    # 调制性 (单区室, κ = 0)
    DOPAMINE = 30            # VTA/SNc DA 神经元
    SEROTONIN = 31           # 中缝核 5-HT 神经元
    NOREPINEPHRINE = 32      # 蓝斑核 NE 神经元
    CHOLINERGIC = 33         # 基底前脑 ACh 神经元

    # 特化型 (κ 值各异)
    THALAMIC_RELAY = 40      # 丘脑中继 (Tonic/Burst切换) κ=0.2~0.5
    TRN = 41                 # 丘脑网状核 (纯抑制门控) κ=0
    MEDIUM_SPINY_D1 = 42     # 纹状体中棘神经元 D1型
    MEDIUM_SPINY_D2 = 43     # 纹状体中棘神经元 D2型
    PURKINJE = 44            # 小脑浦肯野细胞
    PLACE_CELL = 45          # 海马位置细胞 κ=0.3
    GRID_CELL = 46           # 内嗅皮层网格细胞 κ=0.2
    HEAD_DIRECTION = 47      # 头朝向细胞 κ=0.1
    STN = 48                 # 丘脑底核 (兴奋性, 持续高频发放) κ=0


# =============================================================================
# 突触类型枚举
# =============================================================================

class SynapseType(IntEnum):
    """突触类型枚举"""
    # 兴奋性 (谷氨酸)
    AMPA = 0                 # 快速兴奋 τ ~2ms
    NMDA = 1                 # 慢速兴奋 + 电压门控 τ ~50-150ms
    KAINATE = 2              # 调制性兴奋

    # 抑制性 (GABA)
    GABA_A = 10              # 快速抑制 τ ~5-10ms
    GABA_B = 11              # 慢速抑制 τ ~100-300ms

    # 调制性
    DOPAMINE_D1 = 20         # D1 受体 (兴奋性调制)
    DOPAMINE_D2 = 21         # D2 受体 (抑制性调制)
    SEROTONIN_5HT = 22       # 5-HT 受体
    ACETYLCHOLINE_M = 23     # 毒蕈碱型 (慢调制)
    ACETYLCHOLINE_N = 24     # 烟碱型 (快兴奋)
    NOREPINEPHRINE_A = 25    # α 受体
    NOREPINEPHRINE_B = 26    # β 受体

    # 电突触
    GAP_JUNCTION = 30        # 缝隙连接


# =============================================================================
# 可塑性规则枚举
# =============================================================================

class PlasticityType(IntEnum):
    """可塑性规则类型"""
    NONE = 0                 # 无可塑性 (固定权重)
    STDP = 1                 # 经典 STDP
    VOLTAGE_STDP = 2         # 电压依赖 STDP
    DA_MODULATED_STDP = 3    # 多巴胺调制三因子 STDP
    REWARD_MODULATED_STDP = 4  # 奖励调制 STDP
    BCM = 5                  # BCM 理论
    INHIBITORY_STDP = 6      # 抑制性 STDP (对称型)
    HOMEOSTATIC = 7          # 稳态可塑性
    STRUCTURAL = 8           # 结构可塑性


# =============================================================================
# 振荡频段枚举
# =============================================================================

class OscillationBand(IntEnum):
    """神经振荡频段"""
    DELTA = 0                # δ 0.5-4 Hz   深睡眠; 记忆巩固
    THETA = 1                # θ 4-8 Hz     导航; 记忆编码; 序列压缩
    ALPHA = 2                # α 8-13 Hz    空闲抑制; 注意力选择
    BETA = 3                 # β 13-30 Hz   运动规划; 状态维持; 长距离反馈
    GAMMA = 4                # γ 30-100 Hz  特征绑定; 局部处理; 注意力增强


# =============================================================================
# 神经调质类型
# =============================================================================

class NeuromodulatorType(IntEnum):
    """神经调质类型"""
    DA = 0                   # 多巴胺 — 奖励预测误差 / 动机
    NE = 1                   # 去甲肾上腺素 — 警觉 / 探索-利用权衡
    SEROTONIN = 2            # 5-羟色胺 — 耐心 / 时间折扣
    ACH = 3                  # 乙酰胆碱 — 注意力 / 学习率


# =============================================================================
# 调质水平数据结构
# =============================================================================

@dataclass
class NeuromodulatorLevels:
    """神经调质浓度水平

    每种调质的浓度范围 [0.0, 1.0]:
    - DA:  相位性(phasic) RPE + 紧张性(tonic) 基线
    - NE:  低=困倦, 中=聚焦, 高=广泛警觉
    - 5-HT: 高=耐心, 低=冲动
    - ACh: 高=学习模式(自下而上), 低=推理模式(自上而下)
    """
    da: float = 0.5          # 多巴胺水平
    ne: float = 0.3          # 去甲肾上腺素水平
    serotonin: float = 0.5   # 5-HT 水平
    ach: float = 0.3         # 乙酰胆碱水平

    def __post_init__(self):
        """确保所有值在 [0, 1] 范围内"""
        self.da = float(np.clip(self.da, 0.0, 1.0))
        self.ne = float(np.clip(self.ne, 0.0, 1.0))
        self.serotonin = float(np.clip(self.serotonin, 0.0, 1.0))
        self.ach = float(np.clip(self.ach, 0.0, 1.0))


# =============================================================================
# 振荡状态数据结构
# =============================================================================

@dataclass
class OscillationState:
    """脑区振荡状态

    每个脑区维护 5 个频段的当前相位和功率。
    振荡状态影响该区域内神经元的发放概率:
    只有在"兴奋相位窗口"内的脉冲才能有效传递
    (Communication Through Coherence)。
    """
    # 各频段当前相位 [0, 2π)
    delta_phase: float = 0.0
    theta_phase: float = 0.0
    alpha_phase: float = 0.0
    beta_phase: float = 0.0
    gamma_phase: float = 0.0

    # 各频段功率 [0, 1]
    delta_power: float = 0.0
    theta_power: float = 0.0
    alpha_power: float = 0.0
    beta_power: float = 0.0
    gamma_power: float = 0.0