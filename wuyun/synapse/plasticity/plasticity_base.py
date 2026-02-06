"""
可塑性规则基类

定义所有可塑性规则的统一接口。
规则对象是无状态的参数容器 + 计算方法，
状态 (eligibility, weight) 存储在 SynapseBase 中。

依赖约束:
- 不依赖 synapse_base.py (避免循环依赖)
- 不依赖 neuron/ 或 circuit/
"""

from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
import math
import numpy as np


class PlasticityRule(ABC):
    """可塑性规则基类

    所有可塑性规则的统一接口。
    规则负责计算，状态由 SynapseBase 维护。

    使用模式:
        rule = ClassicalSTDP(a_plus=0.005, ...)
        dw = rule.compute_weight_update(pre_times, post_times, w, w_min, w_max)
        new_w = w + dw
    """

    @abstractmethod
    def compute_weight_update(
        self,
        pre_spike_times: List[int],
        post_spike_times: List[int],
        current_weight: float,
        w_min: float,
        w_max: float,
    ) -> float:
        """计算权重变化量 Δw

        基于突触前/后的脉冲时间列表，计算权重更新。

        Args:
            pre_spike_times: 突触前最近脉冲时间列表 (ms)
            post_spike_times: 突触后最近脉冲时间列表 (ms)
            current_weight: 当前突触权重
            w_min: 权重下界
            w_max: 权重上界

        Returns:
            权重变化量 Δw (可正可负, 已含软边界)
        """
        raise NotImplementedError

    def update_eligibility(
        self,
        pre_spike_times: List[int],
        post_spike_times: List[int],
        current_eligibility: float,
        dt: float = 1.0,
    ) -> float:
        """更新资格痕迹 (三因子规则用)

        默认实现: 不使用资格痕迹 (直接返回 0)。
        三因子规则子类覆盖此方法。

        Args:
            pre_spike_times: 突触前最近脉冲时间列表
            post_spike_times: 突触后最近脉冲时间列表
            current_eligibility: 当前资格痕迹值
            dt: 时间步长 (ms)

        Returns:
            新的资格痕迹值
        """
        return 0.0

    def apply_modulated_update(
        self,
        eligibility: float,
        modulation: float,
        current_weight: float,
        w_min: float,
        w_max: float,
    ) -> float:
        """应用调制因子将资格痕迹转化为权重变化 (三因子规则用)

        默认实现: Δw = modulation × eligibility

        Args:
            eligibility: 当前资格痕迹值
            modulation: 调制因子 (如 DA 浓度, 范围 [0, 1])
            current_weight: 当前权重
            w_min: 权重下界
            w_max: 权重上界

        Returns:
            权重变化量 Δw
        """
        return modulation * eligibility

    # =========================================================================
    # 工具方法
    # =========================================================================

    @staticmethod
    def _compute_stdp_window(
        pre_spike_times: List[int],
        post_spike_times: List[int],
        a_plus: float,
        a_minus: float,
        tau_plus: float,
        tau_minus: float,
    ) -> float:
        """计算 STDP 时间窗口函数的加权和

        对所有 pre × post 脉冲配对:
          Δt = t_post - t_pre
          Δt > 0 → A+ · exp(-Δt / τ+)  (LTP)
          Δt < 0 → -A- · exp(Δt / τ-)  (LTD)

        Args:
            pre_spike_times: 突触前脉冲时间列表
            post_spike_times: 突触后脉冲时间列表
            a_plus: LTP 幅度
            a_minus: LTD 幅度
            tau_plus: LTP 时间窗口 (ms)
            tau_minus: LTD 时间窗口 (ms)

        Returns:
            STDP 增量 (可正可负, 不含软边界)
        """
        if not pre_spike_times or not post_spike_times:
            return 0.0

        n_pre = len(pre_spike_times)
        n_post = len(post_spike_times)

        # 小规模配对: 纯 Python 更快 (避免 numpy 小数组开销)
        if n_pre * n_post <= 64:
            increment = 0.0
            inv_tau_plus = 1.0 / tau_plus
            inv_tau_minus = 1.0 / tau_minus
            _exp = math.exp
            for t_pre in pre_spike_times:
                for t_post in post_spike_times:
                    delta_t = t_post - t_pre
                    if delta_t > 0:
                        increment += a_plus * _exp(-delta_t * inv_tau_plus)
                    elif delta_t < 0:
                        increment -= a_minus * _exp(delta_t * inv_tau_minus)
            return increment

        # 大规模配对: numpy 向量化
        pre = np.asarray(pre_spike_times, dtype=np.float64)
        post = np.asarray(post_spike_times, dtype=np.float64)
        dt = post[:, None] - pre[None, :]

        increment = 0.0
        ltp_vals = dt[dt > 0]
        if len(ltp_vals):
            increment += a_plus * float(np.exp(-ltp_vals / tau_plus).sum())
        ltd_vals = dt[dt < 0]
        if len(ltd_vals):
            increment -= a_minus * float(np.exp(ltd_vals / tau_minus).sum())
        return increment

    @staticmethod
    def _apply_soft_bound(raw_dw: float, current_weight: float,
                          w_min: float, w_max: float) -> float:
        """应用软边界: 接近上/下限时变化减小

        Args:
            raw_dw: 原始权重变化量
            current_weight: 当前权重
            w_min: 权重下界
            w_max: 权重上界

        Returns:
            经过软边界调整的 Δw
        """
        w_range = w_max - w_min
        if w_range <= 0:
            return raw_dw
        if raw_dw > 0:
            raw_dw *= (w_max - current_weight) / w_range
        elif raw_dw < 0:
            raw_dw *= (current_weight - w_min) / w_range
        return raw_dw