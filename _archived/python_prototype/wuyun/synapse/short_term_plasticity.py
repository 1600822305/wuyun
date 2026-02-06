"""
短时程可塑性 (Short-Term Plasticity, STP)

Tsodyks-Markram 模型 — 囊泡耗竭 + 释放概率易化:

  n' = (1 - n) / τ_r - p · n · δ(t - t_spike)    囊泡池恢复 + 耗竭
  p' = (p₀ - p) / τ_f + a_f · (1 - p) · δ(t)     释放概率易化

  有效突触效能 = p(t) · n(t)

预定义模式:
  - MOSSY_FIBER: 强易化型 (DG→CA3 "去极化器突触")
    低 p₀ + 强易化 → 频率越高效能越强
  - SCHAFFER_COLLATERAL: 轻度抑制型 (CA3→CA1)
    高 p₀ + 弱易化 → 高频时效能下降
  - DEPRESSING: 纯抑制型 (通用, 皮层内连接)
  - FACILITATING: 纯易化型 (通用)

依赖约束:
  - 不依赖 neuron/ 或 circuit/
  - 可作为独立组件附加到任何 SynapseBase
"""

import math
from dataclasses import dataclass


@dataclass
class STPParams:
    """短时程可塑性参数

    Attributes:
        p0:    基线释放概率 [0, 1]
        tau_r: 囊泡池恢复时间常数 (ms), 典型 500-1500ms
        tau_f: 易化恢复时间常数 (ms), 典型 50-500ms
        a_f:   每次脉冲的易化增量, 0 = 无易化
    """
    p0: float = 0.5
    tau_r: float = 800.0
    tau_f: float = 200.0
    a_f: float = 0.1


# --- 预定义参数集 ---

MOSSY_FIBER_STP = STPParams(
    p0=0.05,                # ★ 极低基线释放概率
    tau_r=800.0,            # 中等恢复
    tau_f=500.0,            # 慢易化衰减 → 易化累积持久
    a_f=0.15,               # ★ 渐进易化 → 多次脉冲后达到去极化效应
)

SCHAFFER_COLLATERAL_STP = STPParams(
    p0=0.5,                 # 高基线释放概率
    tau_r=500.0,            # 较快恢复
    tau_f=50.0,             # 极快易化衰减 → 易化几乎不累积
    a_f=0.02,               # 极弱易化 → 抑制主导
)

DEPRESSING_STP = STPParams(
    p0=0.6,                 # 高释放概率
    tau_r=800.0,            # 标准恢复
    tau_f=100.0,
    a_f=0.0,                # 无易化 → 纯抑制
)

FACILITATING_STP = STPParams(
    p0=0.1,                 # 低释放概率
    tau_r=1200.0,           # 慢恢复
    tau_f=300.0,            # 慢易化恢复
    a_f=0.3,                # 强易化
)


class ShortTermPlasticity:
    """短时程可塑性 (Tsodyks-Markram 模型)

    管理两个状态变量:
    - n: 可释放囊泡池占有率 [0, 1]
    - p: 当前释放概率 [p0, 1]

    每个时间步调用 step() 推进衰减，
    脉冲到达时调用 on_spike() 触发释放+更新。
    get_efficacy() 返回当前有效突触效能 p*n ∈ [0, 1]。

    使用示例:
        stp = ShortTermPlasticity(MOSSY_FIBER_STP)
        for t in range(1000):
            stp.step()
            if presynaptic_spike:
                efficacy = stp.on_spike()
                effective_weight = base_weight * efficacy
    """

    def __init__(self, params: STPParams = None):
        self.params = params or STPParams()
        self._n: float = 1.0           # 囊泡池满
        self._p: float = self.params.p0  # 基线释放概率

        # 预计算衰减因子 (dt=1ms)
        self._n_recovery = 1.0 / self.params.tau_r if self.params.tau_r > 0 else 0.0
        self._p_decay = math.exp(-1.0 / self.params.tau_f) if self.params.tau_f > 0 else 0.0

    @property
    def n(self) -> float:
        """当前囊泡池占有率"""
        return self._n

    @property
    def p(self) -> float:
        """当前释放概率"""
        return self._p

    def get_efficacy(self) -> float:
        """当前有效突触效能 = p * n

        Returns:
            有效效能 [0, 1], 乘以基础权重得到实际传递强度
        """
        return self._p * self._n

    def step(self, dt: float = 1.0) -> None:
        """推进一个时间步 (无脉冲时的恢复动力学)

        n 向 1.0 恢复: dn = (1 - n) / τ_r · dt
        p 向 p0 衰减: p *= exp(-dt / τ_f)  (+ p0 恢复项)

        Args:
            dt: 时间步长 (ms)
        """
        # 囊泡池恢复: n → 1.0
        if dt == 1.0:
            self._n += (1.0 - self._n) * self._n_recovery
        else:
            recovery_rate = 1.0 / self.params.tau_r if self.params.tau_r > 0 else 0.0
            self._n += (1.0 - self._n) * recovery_rate * dt

        # 释放概率衰减: p → p0
        p0 = self.params.p0
        if dt == 1.0:
            self._p = p0 + (self._p - p0) * self._p_decay
        else:
            decay = math.exp(-dt / self.params.tau_f) if self.params.tau_f > 0 else 0.0
            self._p = p0 + (self._p - p0) * decay

    def on_spike(self) -> float:
        """处理突触前脉冲到达

        1. 计算本次释放的有效效能 = p * n
        2. 消耗囊泡: n -= p * n
        3. 易化释放概率: p += a_f * (1 - p)

        Returns:
            本次脉冲的有效突触效能 [0, 1]
        """
        # 有效效能 (本次释放)
        efficacy = self._p * self._n

        # 囊泡耗竭
        self._n -= self._p * self._n
        self._n = max(0.0, self._n)

        # 释放概率易化
        a_f = self.params.a_f
        if a_f > 0:
            self._p += a_f * (1.0 - self._p)
            self._p = min(1.0, self._p)

        return efficacy

    def reset(self) -> None:
        """重置到初始状态"""
        self._n = 1.0
        self._p = self.params.p0

    def __repr__(self) -> str:
        return (f"STP(p0={self.params.p0}, τ_r={self.params.tau_r}ms, "
                f"τ_f={self.params.tau_f}ms, a_f={self.params.a_f}, "
                f"n={self._n:.3f}, p={self._p:.3f}, eff={self.get_efficacy():.3f})")
