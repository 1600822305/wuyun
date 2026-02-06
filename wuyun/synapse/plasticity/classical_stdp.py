"""
经典 STDP (Spike-Timing-Dependent Plasticity)

兴奋性突触的标准学习规则:
  Δw = {  A+ · exp(-Δt / τ+)   当 Δt > 0 (pre 先于 post → LTP)
       { -A- · exp(+Δt / τ-)   当 Δt < 0 (post 先于 pre → LTD)

其中 Δt = t_post - t_pre

特性:
- A- > A+ (0.00525 > 0.005) → 略偏向 LTD, 维持稀疏性
- 软边界: 接近上/下限时变化减小, 防止权重饱和
- 对所有最近 pre×post 脉冲配对求和

参考: docs/02_neuron_system_design.md 3.2 节
"""

from dataclasses import dataclass
from typing import List

from wuyun.synapse.plasticity.plasticity_base import PlasticityRule


@dataclass
class ClassicalSTDPParams:
    """经典 STDP 参数

    Attributes:
        a_plus:  LTP 幅度 (pre→post 顺序)
        a_minus: LTD 幅度 (post→pre 顺序), 略大于 a_plus 维持稀疏性
        tau_plus:  LTP 时间窗口 (ms)
        tau_minus: LTD 时间窗口 (ms)
    """
    a_plus: float = 0.005
    a_minus: float = 0.00525
    tau_plus: float = 20.0
    tau_minus: float = 20.0


class ClassicalSTDP(PlasticityRule):
    """经典 STDP 规则

    对所有最近的 pre-post 脉冲配对计算 Δt,
    应用指数窗口函数求和得到总 Δw,
    然后通过软边界约束权重范围。

    使用示例:
        rule = ClassicalSTDP()
        dw = rule.compute_weight_update(
            pre_spike_times=[100],
            post_spike_times=[110],
            current_weight=0.5, w_min=0.0, w_max=1.0
        )
        # dw > 0 (LTP, pre 先于 post)
    """

    def __init__(self, params: ClassicalSTDPParams = None):
        self.params = params or ClassicalSTDPParams()

    def compute_weight_update(
        self,
        pre_spike_times: List[int],
        post_spike_times: List[int],
        current_weight: float,
        w_min: float,
        w_max: float,
    ) -> float:
        """计算经典 STDP 权重变化

        对 pre_spike_times × post_spike_times 所有配对:
        1. 计算 Δt = t_post - t_pre
        2. 应用指数 STDP 窗口
        3. 求和
        4. 应用软边界

        Args:
            pre_spike_times: 突触前最近脉冲时间 (ms)
            post_spike_times: 突触后最近脉冲时间 (ms)
            current_weight: 当前权重
            w_min: 权重下界
            w_max: 权重上界

        Returns:
            权重变化量 Δw
        """
        p = self.params
        raw_dw = self._compute_stdp_window(
            pre_spike_times, post_spike_times,
            p.a_plus, p.a_minus, p.tau_plus, p.tau_minus,
        )
        return self._apply_soft_bound(raw_dw, current_weight, w_min, w_max)

    def __repr__(self) -> str:
        p = self.params
        return (f"ClassicalSTDP(A+={p.a_plus}, A-={p.a_minus}, "
                f"τ+={p.tau_plus}ms, τ-={p.tau_minus}ms)")