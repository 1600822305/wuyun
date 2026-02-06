"""
Layer 1: 突触基类与 AMPA 突触实现

突触是神经元间通信的基本单元:
- 接收突触前脉冲 (Spike)
- 产生突触电流 (I_syn) 注入突触后神经元的特定区室

突触电流模型:
  I_syn = g_max * w * s * (V_post - E_rev)

其中:
  g_max: 最大电导 (固定, 由突触类型决定)
  w:     突触权重 (可塑性学习调整)
  s:     门控变量 (脉冲到达时跳增, 指数衰减)
  V_post: 突触后膜电位
  E_rev:  反转电位 (兴奋性=0mV, 抑制性=-75mV)

关键设计:
  - target_compartment 决定电流注入哪个区室 (BASAL/APICAL/SOMA)
  - 这是预测编码的硬件基础: 前馈→basal, 反馈→apical
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from typing import List

from wuyun.spike.signal_types import (
    SpikeType,
    SynapseType,
    CompartmentType,
    PlasticityType,
)


# =============================================================================
# 突触参数数据类
# =============================================================================

@dataclass
class SynapseParams:
    """突触类型参数

    不同突触类型 (AMPA/NMDA/GABA_A/GABA_B) 有不同的参数集。
    这些参数是"生物物理常数"，可以硬编码 (参见 00_design_principles.md)。

    Attributes:
        tau_decay: 衰减时间常数 (ms). AMPA~2ms, NMDA~100ms, GABA_A~6ms
        tau_rise:  上升时间常数 (ms). 大部分突触 ~0.5-2ms
        e_rev:     反转电位 (mV). 兴奋性=0mV, 抑制性=-75mV
        g_max:     最大电导 (nS). 突触类型依赖
    """
    tau_decay: float = 2.0      # 衰减时间常数 (ms)
    tau_rise: float = 0.5       # 上升时间常数 (ms)
    e_rev: float = 0.0          # 反转电位 (mV), 兴奋性=0, GABA_A=-70, GABA_B=-95
    g_max: float = 1.0          # 最大电导 (nS)


# 预定义参数集 — 生物物理常数 (允许硬编码)
AMPA_PARAMS = SynapseParams(tau_decay=2.0, tau_rise=0.5, e_rev=0.0, g_max=1.0)
NMDA_PARAMS = SynapseParams(tau_decay=100.0, tau_rise=5.0, e_rev=0.0, g_max=0.5)
GABA_A_PARAMS = SynapseParams(tau_decay=6.0, tau_rise=0.5, e_rev=-70.0, g_max=1.0)
GABA_B_PARAMS = SynapseParams(tau_decay=200.0, tau_rise=5.0, e_rev=-95.0, g_max=0.3)


# =============================================================================
# 突触基类
# =============================================================================

class SynapseBase:
    """突触基类

    每个突触连接一个突触前神经元和一个突触后神经元的特定区室。
    核心职责: 接收脉冲 → 更新门控变量 → 计算突触电流

    Attributes:
        pre_id:  突触前神经元 ID
        post_id: 突触后神经元 ID
        target_compartment: 目标区室 (BASAL/APICAL/SOMA)
        weight:  突触权重 [0, w_max] (由可塑性规则调整)
        delay:   传导延迟 (时间步, 1步=1ms)
        synapse_type: 突触类型 (AMPA/NMDA/GABA_A 等)
        params:  突触参数 (时间常数/反转电位/最大电导)
    """

    __slots__ = (
        'pre_id', 'post_id', 'target_compartment', 'synapse_type',
        'plasticity_type', '_plasticity_rule', 'weight', 'w_max', 'w_min',
        'delay', 'params', '_s', '_x', '_eligibility', '_delay_buffer',
        '_decay_factor', '_elig_decay_factor', '_is_nmda', '_gw', '_e_rev',
    )

    def __init__(
        self,
        pre_id: int,
        post_id: int,
        weight: float = 0.5,
        delay: int = 1,
        synapse_type: SynapseType = SynapseType.AMPA,
        target_compartment: CompartmentType = CompartmentType.BASAL,
        plasticity_type: PlasticityType = PlasticityType.NONE,
        plasticity_rule: 'Optional[PlasticityRule]' = None,
        params: Optional[SynapseParams] = None,
        w_max: float = 1.0,
        w_min: float = 0.0,
    ):
        # 连接信息
        self.pre_id = pre_id
        self.post_id = post_id
        self.target_compartment = target_compartment
        self.synapse_type = synapse_type
        self.plasticity_type = plasticity_type

        # 可塑性规则对象 (可选, None = 无可塑性)
        # 使用字符串类型注解避免循环依赖 (PlasticityRule 在 plasticity/ 子模块中)
        self._plasticity_rule = plasticity_rule

        # 突触权重 (可塑性学习的对象)
        self.weight = max(w_min, min(weight, w_max))
        self.w_max = w_max
        self.w_min = w_min

        # 传导延迟
        self.delay = max(1, delay)

        # 突触参数 (生物物理常数)
        if params is None:
            self.params = self._default_params()
        else:
            self.params = params

        # === 动态状态变量 ===
        self._s: float = 0.0          # 门控变量 (突触激活程度)
        self._x: float = 0.0          # 上升变量 (用于双指数模型)
        self._eligibility: float = 0.0  # 资格痕迹 (三因子学习)

        # 延迟缓冲: 存储待传递的脉冲 (timestamp → spike_type)
        self._delay_buffer: deque = deque()

        # 预计算衰减因子 (dt=1.0 快速路径)
        self._decay_factor: float = math.exp(-1.0 / self.params.tau_decay)
        self._elig_decay_factor: float = math.exp(-1.0 / 1000.0)

        # 性能缓存: 避免每步重复计算
        self._is_nmda: bool = (synapse_type == SynapseType.NMDA)
        self._gw: float = self.params.g_max * self.weight  # g_max * weight 缓存
        self._e_rev: float = self.params.e_rev              # 反转电位缓存

    def _default_params(self) -> SynapseParams:
        """根据突触类型返回默认参数"""
        defaults = {
            SynapseType.AMPA: AMPA_PARAMS,
            SynapseType.NMDA: NMDA_PARAMS,
            SynapseType.GABA_A: GABA_A_PARAMS,
            SynapseType.GABA_B: GABA_B_PARAMS,
        }
        return defaults.get(self.synapse_type, AMPA_PARAMS)

    # =========================================================================
    # 核心接口
    # =========================================================================

    def receive_spike(self, timestamp: int, spike_type: SpikeType) -> None:
        """接收突触前脉冲

        脉冲不会立即生效，而是放入延迟缓冲区。
        经过 self.delay 个时间步后才会到达突触后端。

        Args:
            timestamp: 脉冲发放时间
            spike_type: 脉冲类型 (REGULAR/BURST)
        """
        if spike_type.is_active:
            arrival_time = timestamp + self.delay
            self._delay_buffer.append((arrival_time, spike_type))

    def step(self, current_time: int, dt: float = 1.0) -> None:
        """推进一个时间步

        1. 检查延迟缓冲区中是否有到达的脉冲
        2. 如果有，增加门控变量 s
        3. 门控变量指数衰减

        Args:
            current_time: 当前时间步
            dt: 时间步长 (ms), 默认 1.0
        """
        # 检查是否有脉冲到达 (利用 deque FIFO 顺序)
        arrived = False
        spike_type_arrived = SpikeType.NONE
        while self._delay_buffer and self._delay_buffer[0][0] <= current_time:
            _, spike_type = self._delay_buffer.popleft()
            arrived = True
            spike_type_arrived = spike_type

        # 门控变量动力学 (双指数模型简化为单指数)
        # ds/dt = -s / τ_decay
        if dt == 1.0:
            self._s *= self._decay_factor
        else:
            self._s *= math.exp(-dt / self.params.tau_decay)

        # 脉冲到达时: s 跳增
        if arrived:
            # burst 脉冲产生更强的突触激活 (每个 burst 脉冲都增加)
            increment = 1.0
            if spike_type_arrived.is_burst:
                increment = 1.5  # burst 增强效应
            self._s += increment
            # 门控变量上限为 1.0 (饱和)
            self._s = min(self._s, 1.0)

        # 资格痕迹衰减 (仅当有可塑性规则或痕迹非零时才计算)
        if self._eligibility != 0.0:
            if dt == 1.0:
                self._eligibility *= self._elig_decay_factor
            else:
                self._eligibility *= math.exp(-dt / 1000.0)

    def step_and_compute(self, current_time: int, v_post: float,
                         dt: float = 1.0) -> float:
        """合并 step + compute_current 的高性能版本

        比分开调用减少一次 Python 函数调用开销,
        并在 _s < 1e-7 时跳过电流计算 (绝大多数突触大部分时间处于非活跃状态).

        Args:
            current_time: 当前时间步
            v_post: 突触后神经元膜电位 (mV)
            dt: 时间步长

        Returns:
            突触电流
        """
        # --- step 部分 ---
        arrived = False
        spike_type_arrived = SpikeType.NONE
        buf = self._delay_buffer
        while buf and buf[0][0] <= current_time:
            _, spike_type_arrived = buf.popleft()
            arrived = True

        s = self._s * self._decay_factor if dt == 1.0 else self._s * math.exp(-dt / self.params.tau_decay)

        if arrived:
            increment = 1.5 if spike_type_arrived.is_burst else 1.0
            s = min(s + increment, 1.0)

        self._s = s

        # 资格痕迹衰减 (仅非零时)
        if self._eligibility != 0.0:
            self._eligibility *= self._elig_decay_factor if dt == 1.0 else math.exp(-dt / 1000.0)

        # --- compute_current 部分 (非活跃时跳过) ---
        if s < 1e-7:
            return 0.0

        conductance = self._gw * s

        if self._is_nmda:
            conductance /= (1.0 + 0.28011204 * math.exp(-0.062 * v_post))

        return conductance * (self._e_rev - v_post)

    def compute_current(self, v_post: float) -> float:
        """计算突触电流

        I_syn = g_max * weight * s * (V_post - E_rev)

        NMDA 突触额外应用 Mg²⁺ 电压门控因子 B(V):
          B(V) = 1 / (1 + [Mg²⁺]/3.57 · exp(-0.062 · V))
        这使 NMDA 成为重合检测器: 只有突触后细胞已部分去极化时才传导。
        (Jahr & Stevens, 1990; Mayer et al., 1984)

        注意: 对于兴奋性突触 (E_rev=0mV), 当 V_post < 0 时,
        (V_post - E_rev) < 0, 所以 I_syn < 0 (内向电流, 去极化方向)。
        在神经元方程中用 -I_syn 或直接用正值表示去极化电流。

        为简化, 这里返回的是突触电导×驱动力的绝对值,
        符号由反转电位和膜电位的关系自然决定:
        - 兴奋性 (E_rev > V_rest): 产生去极化(正)电流
        - 抑制性 (E_rev < V_rest): 产生超极化(负)电流

        Args:
            v_post: 突触后神经元的膜电位 (mV)

        Returns:
            突触电流 (归一化单位). 正值=去极化, 负值=超极化.
        """
        # 电导: g = g_max * weight * s
        conductance = self.params.g_max * self.weight * self._s

        # NMDA 电压门控: Mg²⁺ 阻断因子 (Jahr & Stevens, 1990)
        # B(V) ≈ 0.1 at V=-65mV (静息态, 几乎完全阻断)
        # B(V) ≈ 0.5 at V=-35mV (部分去极化, 开始传导)
        # B(V) ≈ 0.9 at V=-10mV (强去极化, 几乎完全传导)
        if self.synapse_type == SynapseType.NMDA:
            mg_concentration = 1.0  # mM, 生理浓度
            mg_block = 1.0 / (1.0 + (mg_concentration / 3.57)
                              * math.exp(-0.062 * v_post))
            conductance *= mg_block

        # 驱动力: (E_rev - V_post) — 注意符号方向
        # 当 E_rev > V_post → 正电流 → 去极化 (兴奋)
        # 当 E_rev < V_post → 负电流 → 超极化 (抑制)
        driving_force = self.params.e_rev - v_post

        return conductance * driving_force

    # =========================================================================
    # 可塑性接口
    # =========================================================================

    @property
    def plasticity_rule(self):
        """当前可塑性规则对象 (可为 None)"""
        return self._plasticity_rule

    @plasticity_rule.setter
    def plasticity_rule(self, rule) -> None:
        """设置可塑性规则"""
        self._plasticity_rule = rule

    def update_weight_stdp(
        self,
        pre_spike_times: List[int],
        post_spike_times: List[int],
    ) -> float:
        """通过 STDP 规则直接更新权重 (经典 STDP 路径)

        委托给 plasticity_rule.compute_weight_update()。

        Args:
            pre_spike_times: 突触前最近脉冲时间列表 (ms)
            post_spike_times: 突触后最近脉冲时间列表 (ms)

        Returns:
            实际权重变化量 Δw
        """
        if self._plasticity_rule is None:
            return 0.0
        dw = self._plasticity_rule.compute_weight_update(
            pre_spike_times, post_spike_times,
            self.weight, self.w_min, self.w_max,
        )
        new_w = self.weight + dw
        self.weight = float(max(self.w_min, min(new_w, self.w_max)))
        self._gw = self.params.g_max * self.weight
        return dw

    def update_eligibility(
        self,
        pre_spike_times: List[int],
        post_spike_times: List[int],
        dt: float = 1.0,
    ) -> None:
        """更新资格痕迹 (三因子 STDP 路径)

        委托给 plasticity_rule.update_eligibility()。
        STDP 产生的权重变化先存储在资格痕迹中,
        等到调质信号 (如 DA) 到来时才转化为实际权重变化。

        Args:
            pre_spike_times: 突触前最近脉冲时间列表 (ms)
            post_spike_times: 突触后最近脉冲时间列表 (ms)
            dt: 时间步长 (ms)
        """
        if self._plasticity_rule is None:
            return
        self._eligibility = self._plasticity_rule.update_eligibility(
            pre_spike_times, post_spike_times,
            self._eligibility, dt,
        )

    def apply_plasticity(self, modulation: float = 1.0) -> float:
        """应用可塑性规则, 将资格痕迹转化为权重变化

        委托给 plasticity_rule.apply_modulated_update()。

        注意: 此方法不会重置 eligibility。资格痕迹通过
        update_eligibility() 中的指数衰减自然消退 (τ_e=1000ms)。

        正确使用模式:
            # 每个时间步: 更新资格痕迹 (衰减 + STDP 增量)
            syn.update_eligibility(pre_times, post_times, dt=1.0)

            # 仅在 DA 事件到达时调用 (不是每个时间步!)
            if da_event:
                syn.apply_plasticity(modulation=da_level)

        Args:
            modulation: 调制因子 (如 DA 水平). 1.0 = 无调制.

        Returns:
            实际权重变化量 Δw
        """
        if self._plasticity_rule is None:
            return 0.0
        dw = self._plasticity_rule.apply_modulated_update(
            self._eligibility, modulation,
            self.weight, self.w_min, self.w_max,
        )
        new_w = self.weight + dw
        self.weight = float(max(self.w_min, min(new_w, self.w_max)))
        self._gw = self.params.g_max * self.weight
        return dw

    # =========================================================================
    # 状态查询
    # =========================================================================

    @property
    def gate(self) -> float:
        """当前门控变量值"""
        return self._s

    @property
    def eligibility(self) -> float:
        """当前资格痕迹值"""
        return self._eligibility

    @property
    def is_excitatory(self) -> bool:
        """是否为兴奋性突触"""
        return self.params.e_rev > -50.0  # 兴奋性反转电位 ~0mV

    @property
    def is_inhibitory(self) -> bool:
        """是否为抑制性突触"""
        return self.params.e_rev < -50.0  # 抑制性反转电位 ~-75mV

    def reset(self) -> None:
        """重置突触动态状态 (不改变权重)"""
        self._s = 0.0
        self._x = 0.0
        self._eligibility = 0.0
        self._delay_buffer.clear()

    def __repr__(self) -> str:
        return (
            f"SynapseBase(pre={self.pre_id}→post={self.post_id}, "
            f"type={self.synapse_type.name}, "
            f"target={self.target_compartment.name}, "
            f"w={self.weight:.4f}, s={self._s:.4f})"
        )