"""
稳态可塑性 — 突触缩放 (Synaptic Scaling)

原理: 如果一个神经元的平均发放率偏离目标发放率，
其所有传入兴奋性突触的权重按比例缩放：
  - 发放率太高 → 所有传入权重乘以 <1 的因子 → 降低兴奋性
  - 发放率太低 → 所有传入权重乘以 >1 的因子 → 增加兴奋性

关键参数:
  - target_rate: 目标发放率 (Hz), 典型 5-10 Hz
  - tau_homeostatic: 稳态时间常数 (ms), 典型 10000-100000ms (很慢!)
    比 STDP 慢 10-100 倍 — 稳态是"慢调节"，不干扰快速学习
  - scaling_strength: 每次调整的缩放强度

生物学对应: 星形胶质细胞介导的突触缩放 (Turrigiano 2008)

使用模式 (不是每个时间步调用! 是每隔 tau_homeostatic 调用一次):
    if t % homeostatic_interval == 0:
        for neuron in all_neurons:
            rate = neuron.firing_rate  # 最近 1 秒的平均发放率
            for syn in neuron.afferent_excitatory_synapses:
                homeostatic.scale_weight(syn, rate)

依赖约束:
- 不依赖 neuron/ 或 circuit/ (低层不导入高层)
- 不依赖 synapse_base.py (避免循环依赖)
- 只使用基础 Python 类型
"""

from dataclasses import dataclass


@dataclass
class HomeostaticParams:
    """稳态可塑性参数"""
    target_rate: float = 5.0          # 目标发放率 (Hz)
    tau_homeostatic: float = 50000.0  # 稳态时间常数 (ms) = 50 秒
    scaling_strength: float = 0.001   # 缩放强度 (每次调整的最大比例变化)


class HomeostaticPlasticity:
    """突触缩放稳态可塑性

    STDP 是正反馈循环：强突触 → 更多脉冲 → 更强突触 → 失控。
    稳态可塑性是负反馈机制，通过全局缩放所有传入突触权重，
    将神经元发放率拉回到目标范围。

    关键特性:
    - 比 STDP 慢 10-100 倍 (τ_homeostatic >> τ_STDP)
    - 乘性缩放: 保持权重的相对大小关系
    - 只影响兴奋性突触 (由调用方保证)

    典型用法:
        h = HomeostaticPlasticity()

        # 每隔 N 步调用一次 (不是每步!)
        if t % 1000 == 0:
            for neuron in neurons:
                rate = neuron.firing_rate
                for syn in neuron.afferent_excitatory_synapses:
                    syn.weight = h.scale_weight(
                        syn.weight, rate, syn.w_min, syn.w_max
                    )
    """

    def __init__(self, params: HomeostaticParams = None):
        self.params = params or HomeostaticParams()

    def compute_scaling_factor(self, current_rate: float) -> float:
        """计算缩放因子

        Args:
            current_rate: 当前发放率 (Hz)

        Returns:
            缩放因子: >1.0 需要增强, <1.0 需要减弱, =1.0 不变
        """
        p = self.params
        # 发放率差异 (正=太高, 负=太低)
        rate_error = current_rate - p.target_rate
        # 缩放因子: 1 - strength * error / target
        # 太高 → factor < 1 → 权重缩小
        # 太低 → factor > 1 → 权重放大
        if p.target_rate > 0:
            factor = 1.0 - p.scaling_strength * (rate_error / p.target_rate)
        else:
            factor = 1.0
        # 限制缩放范围 [0.9, 1.1] 防止突变
        return max(0.9, min(factor, 1.1))

    def scale_weight(
        self,
        current_weight: float,
        current_rate: float,
        w_min: float,
        w_max: float,
    ) -> float:
        """计算缩放后的新权重

        Args:
            current_weight: 当前权重
            current_rate: 突触后神经元的当前发放率 (Hz)
            w_min: 权重下限
            w_max: 权重上限

        Returns:
            缩放后的新权重
        """
        factor = self.compute_scaling_factor(current_rate)
        new_weight = current_weight * factor
        return max(w_min, min(new_weight, w_max))