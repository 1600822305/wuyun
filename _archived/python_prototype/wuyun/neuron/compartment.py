"""
Layer 2: 区室模型 (Compartment Models)

双区室神经元的硬件基础:
- SomaticCompartment: 胞体+基底树突区室 — 接收前馈输入, 产生动作电位
- ApicalCompartment:  顶端树突区室 — 接收反馈预测, 产生 Ca²⁺ 树突脉冲

胞体区室方程:
  τ_m · dV_s/dt = -(V_s - V_rest) + R_s · I_basal - w + κ · (V_a - V_s)
  τ_w · dw/dt   = a · (V_s - V_rest) - w

顶端树突方程:
  τ_a · dV_a/dt = -(V_a - V_rest) + R_a · I_apical + κ · (V_s - V_a)
  当 V_a ≥ V_ca_threshold 时触发 Ca²⁺ 树突脉冲

两个区室通过耦合系数 κ 连接:
- κ = 0: 退化为单区室模型 (抑制性中间神经元等)
- κ > 0: 双区室模型 (锥体细胞等)
- κ 越大, 顶端对胞体影响越强, 越容易 burst
"""

from dataclasses import dataclass
import numpy as np


# =============================================================================
# 区室参数
# =============================================================================

@dataclass
class SomaticParams:
    """胞体区室参数 (生物物理常数, 允许硬编码)

    这些参数由离子通道特性决定, 是"基因决定"的部分。
    不同神经元类型有不同的参数集。
    """
    v_rest: float = -70.0       # 静息电位 (mV)
    v_threshold: float = -50.0  # Na⁺ 发放阈值 (mV)
    v_reset: float = -65.0      # 发放后重置电位 (mV)
    tau_m: float = 20.0         # 膜时间常数 (ms), 10-30ms
    r_s: float = 1.0            # 胞体膜电阻 (归一化)
    a: float = 0.02             # 亚阈值适应耦合
    b: float = 0.5              # 脉冲后适应增量
    tau_w: float = 200.0        # 适应时间常数 (ms), 50-500ms
    refractory_period: int = 2  # 不应期 (时间步/ms)


@dataclass
class ApicalParams:
    """顶端树突区室参数

    顶端树突时间常数比胞体更长 (20-50ms vs 10-30ms),
    因为树突膜电容更大。
    Ca²⁺ 脉冲阈值比 Na⁺ 脉冲阈值更高 (-30mV vs -50mV)。
    """
    tau_a: float = 30.0            # 顶端树突时间常数 (ms), 20-50ms
    r_a: float = 1.0               # 顶端树突膜电阻 (归一化)
    v_ca_threshold: float = -30.0  # Ca²⁺ 树突脉冲阈值 (mV)
    ca_boost: float = 20.0         # Ca²⁺ 反冲增强 (mV)
    ca_duration: int = 30          # Ca²⁺ 脉冲持续时间 (ms/时间步), 20-50ms


# =============================================================================
# 胞体区室
# =============================================================================

class SomaticCompartment:
    """胞体+基底树突区室

    标准 AdLIF+ 动力学:
      τ_m · dV_s/dt = -(V_s - V_rest) + R_s · I_basal - w + κ · (V_a - V_s)
      τ_w · dw/dt   = a · (V_s - V_rest) - w

    发放条件: V_s ≥ V_threshold
    发放后: V_s → V_reset, w → w + b, 进入不应期

    Attributes:
        v: 膜电位 (mV)
        w: 适应变量
        refractory_counter: 不应期倒计时
        params: 胞体参数
    """

    def __init__(self, params: SomaticParams = None):
        self.params = params or SomaticParams()
        # 动态状态
        self.v: float = self.params.v_rest    # 膜电位
        self.w: float = 0.0                    # 适应变量
        self.refractory_counter: int = 0       # 不应期倒计时

    def update(
        self,
        i_basal: float,
        v_apical: float,
        kappa: float,
        dt: float = 1.0,
    ) -> bool:
        """推进一个时间步

        Args:
            i_basal: 基底树突突触电流 (前馈输入)
            v_apical: 顶端树突膜电位 (用于耦合电流)
            kappa: 区室耦合系数 κ
            dt: 时间步长 (ms)

        Returns:
            是否发放了动作电位 (True/False)
        """
        p = self.params

        # 不应期中: 不更新膜电位, 只倒计时
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            return False

        # === 胞体膜电位动力学 ===
        # τ_m · dV_s/dt = -(V_s - V_rest) + R_s · I_basal - w + κ · (V_a - V_s)
        leak = -(self.v - p.v_rest)
        input_current = p.r_s * i_basal
        coupling = kappa * (v_apical - self.v)
        dv = (leak + input_current - self.w + coupling) / p.tau_m * dt
        self.v += dv

        # === 适应变量动力学 ===
        # τ_w · dw/dt = a · (V_s - V_rest) - w
        dw = (p.a * (self.v - p.v_rest) - self.w) / p.tau_w * dt
        self.w += dw

        # === 发放检测 ===
        fired = self.v >= p.v_threshold
        if fired:
            self.v = p.v_reset
            self.w += p.b
            self.refractory_counter = p.refractory_period

        return fired

    def reset(self) -> None:
        """重置到初始状态"""
        self.v = self.params.v_rest
        self.w = 0.0
        self.refractory_counter = 0


# =============================================================================
# 顶端树突区室
# =============================================================================

class ApicalCompartment:
    """顶端树突区室

    动力学:
      τ_a · dV_a/dt = -(V_a - V_rest) + R_a · I_apical + κ · (V_s - V_a)

    Ca²⁺ 树突脉冲:
      当 V_a ≥ V_ca_threshold 时:
        触发 Ca²⁺ 脉冲 (持续 20-50ms, 比 Na⁺ 脉冲慢得多)
        V_a += Ca_boost (增强胞体去极化)

    这个区室是预测编码的硬件基础:
    - 接收高层反馈预测 → I_apical
    - Ca²⁺ 脉冲 → 通过耦合增强胞体去极化 → 促进 burst

    Attributes:
        v: 顶端树突膜电位 (mV)
        ca_spike: 当前是否有 Ca²⁺ 脉冲
        ca_timer: Ca²⁺ 脉冲剩余持续时间
        params: 顶端树突参数
    """

    def __init__(self, v_rest: float = -70.0, params: ApicalParams = None):
        self.params = params or ApicalParams()
        self._v_rest = v_rest
        # 动态状态
        self.v: float = v_rest           # 顶端树突膜电位
        self.ca_spike: bool = False       # Ca²⁺ 脉冲状态
        self.ca_timer: int = 0            # Ca²⁺ 脉冲持续时间倒计时

    def update(
        self,
        i_apical: float,
        v_soma: float,
        kappa: float,
        dt: float = 1.0,
    ) -> bool:
        """推进一个时间步

        Args:
            i_apical: 顶端树突突触电流 (反馈预测输入)
            v_soma: 胞体膜电位 (用于耦合电流)
            kappa: 区室耦合系数 κ
            dt: 时间步长 (ms)

        Returns:
            当前是否处于 Ca²⁺ 脉冲状态
        """
        p = self.params

        # === 顶端树突膜电位动力学 ===
        # τ_a · dV_a/dt = -(V_a - V_rest) + R_a · I_apical + κ · (V_s - V_a)
        leak = -(self.v - self._v_rest)
        input_current = p.r_a * i_apical
        coupling = kappa * (v_soma - self.v)
        dv = (leak + input_current + coupling) / p.tau_a * dt
        self.v += dv

        # === Ca²⁺ 树突脉冲检测 ===
        if self.ca_timer > 0:
            # Ca²⁺ 脉冲正在进行中
            self.ca_timer -= 1
            if self.ca_timer == 0:
                self.ca_spike = False
        elif self.v >= p.v_ca_threshold:
            # 触发新的 Ca²⁺ 脉冲
            self.ca_spike = True
            self.ca_timer = p.ca_duration
            self.v += p.ca_boost  # Ca²⁺ 反冲 → 增强胞体去极化 (通过耦合)

        return self.ca_spike

    def reset(self) -> None:
        """重置到初始状态"""
        self.v = self._v_rest
        self.ca_spike = False
        self.ca_timer = 0