"""
三因子 STDP (DA-Modulated STDP)

多巴胺调制的可塑性规则，解决奖励延迟的信用分配问题:

  核心公式:
    Δw = DA_signal × eligibility_trace

  资格痕迹更新:
    de/dt = -e / τ_e + STDP(Δt)

  工作流:
    1. pre/post 脉冲发生 → STDP 计算 → 存入资格痕迹 (不立即改权重)
    2. 资格痕迹随时间指数衰减 (τ_e ≈ 1000ms)
    3. DA 信号到达时: Δw = da_level × eligibility
    4. 这样，即使奖励延迟 ~1秒，相关突触仍能被正确强化

应用脑区:
  - PFC (A-01/03): 工作记忆相关的三因子学习
  - 纹状体 (BG-01/02): D1 增强 LTP, D2 增强 LTD
  - 运动皮层: 强化学习驱动

参考: docs/02_neuron_system_design.md 3.2 节
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from wuyun.synapse.plasticity.plasticity_base import PlasticityRule
# Note: numpy still needed for exp() in update_eligibility decay


@dataclass
class DAModulatedSTDPParams:
    """三因子 STDP 参数

    包含 STDP 窗口参数 + 资格痕迹衰减时间。

    Attributes:
        a_plus:   LTP 幅度
        a_minus:  LTD 幅度
        tau_plus:  LTP 时间窗口 (ms)
        tau_minus:  LTD 时间窗口 (ms)
        tau_eligibility: 资格痕迹衰减时间常数 (ms)
            ~1000ms 允许奖励延迟最多 ~1秒仍能分配信用
    """
    a_plus: float = 0.005
    a_minus: float = 0.00525
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    tau_eligibility: float = 1000.0


class DAModulatedSTDP(PlasticityRule):
    """三因子 STDP: DA调制 + 资格痕迹

    与经典 STDP 的关键区别:
    - 经典 STDP: 脉冲配对 → 立即改权重
    - 三因子 STDP: 脉冲配对 → 存入资格痕迹 → DA 到达时才改权重

    compute_weight_update() 返回 0 (不直接改权重)
    update_eligibility() 更新资格痕迹 (STDP 增量 + 衰减)
    apply_modulated_update() 将 DA × eligibility → Δw

    使用示例:
        rule = DAModulatedSTDP()

        # 每个时间步: 更新资格痕迹
        new_elig = rule.update_eligibility(
            pre_times, post_times, current_elig, dt=1.0)

        # DA 信号到达时: 转化为权重变化
        dw = rule.apply_modulated_update(
            eligibility=new_elig, modulation=da_level,
            current_weight=w, w_min=0.0, w_max=1.0)
    """

    def __init__(self, params: DAModulatedSTDPParams = None):
        self.params = params or DAModulatedSTDPParams()

    def compute_weight_update(
        self,
        pre_spike_times: List[int],
        post_spike_times: List[int],
        current_weight: float,
        w_min: float,
        w_max: float,
    ) -> float:
        """三因子 STDP 不直接更新权重

        权重更新通过 update_eligibility() + apply_modulated_update() 路径。
        此方法始终返回 0。
        """
        return 0.0

    def _compute_stdp_increment(
        self,
        pre_spike_times: List[int],
        post_spike_times: List[int],
    ) -> float:
        """计算 STDP 增量 (不含软边界, 纯 STDP 窗口)

        对所有 pre-post 配对计算 Δt, 求和。
        此增量将存入资格痕迹。

        Returns:
            STDP 增量 (可正可负)
        """
        p = self.params
        return self._compute_stdp_window(
            pre_spike_times, post_spike_times,
            p.a_plus, p.a_minus, p.tau_plus, p.tau_minus,
        )

    def update_eligibility(
        self,
        pre_spike_times: List[int],
        post_spike_times: List[int],
        current_eligibility: float,
        dt: float = 1.0,
    ) -> float:
        """更新资格痕迹

        de/dt = -e / τ_e + STDP(Δt)

        每个时间步:
        1. 资格痕迹指数衰减
        2. 如果有 pre/post 脉冲配对, 加上 STDP 增量

        Args:
            pre_spike_times: 当前时间步突触前脉冲时间
            post_spike_times: 当前时间步突触后脉冲时间
            current_eligibility: 当前资格痕迹值
            dt: 时间步长 (ms)

        Returns:
            更新后的资格痕迹值
        """
        p = self.params

        # 1. 指数衰减
        decay = np.exp(-dt / p.tau_eligibility)
        new_eligibility = current_eligibility * decay

        # 2. 加上 STDP 增量
        stdp_increment = self._compute_stdp_increment(
            pre_spike_times, post_spike_times
        )
        new_eligibility += stdp_increment

        return float(new_eligibility)

    def apply_modulated_update(
        self,
        eligibility: float,
        modulation: float,
        current_weight: float,
        w_min: float,
        w_max: float,
    ) -> float:
        """将调制信号 (DA) × 资格痕迹 → 权重变化

        Δw = modulation × eligibility × 软边界

        Args:
            eligibility: 当前资格痕迹值
            modulation: DA 调制信号 [0, 1]
                0.0 = 无 DA → 无学习
                0.5 = 基线 DA → 弱学习
                1.0 = 强 DA (正向 RPE) → 强学习
            current_weight: 当前权重
            w_min: 权重下界
            w_max: 权重上界

        Returns:
            权重变化量 Δw
        """
        raw_dw = modulation * eligibility
        return float(self._apply_soft_bound(raw_dw, current_weight, w_min, w_max))

    def __repr__(self) -> str:
        p = self.params
        return (f"DAModulatedSTDP(A+={p.a_plus}, A-={p.a_minus}, "
                f"τ+={p.tau_plus}ms, τ_e={p.tau_eligibility}ms)")