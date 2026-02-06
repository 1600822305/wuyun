"""
抑制性 STDP (对称型)

维持兴奋-抑制 (E/I) 平衡的自动调节规则:

  Δw = { A · exp(-|Δt| / τ)   当 |Δt| < τ_window (相关 → 增强抑制)
       { -B                    当无相关脉冲 (不相关 → 减弱抑制)

关键特性:
- 对称窗口: 不区分 pre/post 顺序, 只看时间接近程度
- 相关活动 → 增强抑制 (防止过度兴奋)
- 不相关活动 → 减弱抑制 (允许活动)
- 效果: 自动维持 E/I 平衡

应用:
- PV+ 篮状细胞 → 胞体的抑制突触
- SST+ Martinotti → 顶端树突的抑制突触
- 所有抑制性突触的默认可塑性规则

参考: docs/02_neuron_system_design.md 3.2 节
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from wuyun.synapse.plasticity.plasticity_base import PlasticityRule
# Note: numpy still needed for exp() in symmetric window


@dataclass
class InhibitorySTDPParams:
    """抑制性 STDP 参数

    Attributes:
        a_corr: 相关活动时的增强幅度
        b_uncorr: 不相关活动时的减弱幅度
        tau: 时间窗口宽度 (ms)
        tau_window: 判定"相关/不相关"的时间阈值 (ms)
            |Δt| < tau_window → 相关
            |Δt| ≥ tau_window → 不相关
    """
    a_corr: float = 0.001
    b_uncorr: float = 0.0001
    tau: float = 20.0
    tau_window: float = 50.0   # 超过此阈值视为不相关


class InhibitorySTDP(PlasticityRule):
    """抑制性 STDP 规则 (对称型)

    与经典 STDP 的关键区别:
    - 经典 STDP: 不对称 (pre→post LTP, post→pre LTD)
    - 抑制性 STDP: 对称 (只看 |Δt|, 不看方向)

    效果:
    - 同步活动 (|Δt| 小) → 抑制增强 → 防止兴奋性爆发
    - 不同步 (|Δt| 大或无配对) → 抑制减弱 → 允许正常活动

    使用示例:
        rule = InhibitorySTDP()
        dw = rule.compute_weight_update(
            pre_spike_times=[100],
            post_spike_times=[105],  # 同步活动
            current_weight=0.5, w_min=0.0, w_max=1.0
        )
        # dw > 0 (增强抑制)
    """

    def __init__(self, params: InhibitorySTDPParams = None):
        self.params = params or InhibitorySTDPParams()

    def compute_weight_update(
        self,
        pre_spike_times: List[int],
        post_spike_times: List[int],
        current_weight: float,
        w_min: float,
        w_max: float,
    ) -> float:
        """计算抑制性 STDP 权重变化

        对称窗口逻辑:
        - 有 pre 和 post 配对且 |Δt| 在窗口内 → 增强抑制
        - 有 pre 或 post 脉冲但无配对或 |Δt| 过大 → 减弱抑制
        - 无脉冲 → 无变化

        Args:
            pre_spike_times: 突触前脉冲时间
            post_spike_times: 突触后脉冲时间
            current_weight: 当前权重
            w_min: 权重下界
            w_max: 权重上界

        Returns:
            权重变化量 Δw
        """
        p = self.params

        # 无任何脉冲 → 无变化
        has_pre = len(pre_spike_times) > 0
        has_post = len(post_spike_times) > 0

        if not has_pre and not has_post:
            return 0.0

        raw_dw = 0.0
        found_correlated = False

        # 计算所有 pre-post 配对的对称 STDP
        if has_pre and has_post:
            for t_pre in pre_spike_times:
                for t_post in post_spike_times:
                    abs_dt = abs(t_post - t_pre)

                    if abs_dt < p.tau_window:
                        # 相关活动 → 增强抑制
                        raw_dw += p.a_corr * np.exp(-abs_dt / p.tau)
                        found_correlated = True

        # 有脉冲但不相关 → 减弱抑制
        if not found_correlated and (has_pre or has_post):
            raw_dw = -p.b_uncorr

        return float(self._apply_soft_bound(raw_dw, current_weight, w_min, w_max))

    def __repr__(self) -> str:
        p = self.params
        return (f"InhibitorySTDP(A={p.a_corr}, B={p.b_uncorr}, "
                f"τ={p.tau}ms, window={p.tau_window}ms)")