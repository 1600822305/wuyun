"""
Layer 0: OscillationClock — 神经振荡时钟

为脑区提供振荡相位驱动，实现相位依赖的信息路由。

核心应用: 海马 Theta 振荡 (4-8 Hz)
  - 编码相位 (θ谷): EC→DG→CA3 通路激活
  - 检索相位 (θ峰): CA3→CA1→EC 通路激活
  - 这种相位切换是海马记忆编码/检索分离的基础

设计:
  - 简单正弦振荡器, 每步推进相位
  - 支持多频段 (delta/theta/alpha/beta/gamma)
  - 提供 encoding_gate / retrieval_gate 二值门控信号
  - 可选: 相位调制神经元发放概率 (Communication Through Coherence)

依赖约束:
  - 仅依赖 signal_types (OscillationBand, OscillationState)
  - 不依赖 synapse/ 或 neuron/

参考:
  - Hasselmo et al., 2002: theta phase separation of encoding/retrieval
  - PLOS Comp Bio 2025: binary on/off gating tied to theta cycle
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

from wuyun.spike.signal_types import OscillationBand, OscillationState


@dataclass
class OscillationParams:
    """振荡参数

    Attributes:
        frequency: 振荡频率 (Hz)
        power: 振荡功率/强度 [0, 1], 0=关闭
        phase_offset: 初始相位偏移 (rad)
    """
    frequency: float = 6.0       # Hz
    power: float = 1.0           # [0, 1]
    phase_offset: float = 0.0    # rad


# 预定义频段参数
THETA_PARAMS = OscillationParams(frequency=6.0, power=1.0)
GAMMA_PARAMS = OscillationParams(frequency=40.0, power=0.5)
ALPHA_PARAMS = OscillationParams(frequency=10.0, power=0.3)
BETA_PARAMS = OscillationParams(frequency=20.0, power=0.3)
DELTA_PARAMS = OscillationParams(frequency=2.0, power=0.5)


class OscillationClock:
    """神经振荡时钟

    管理一个脑区的多频段振荡状态。
    每个时间步 (1ms) 推进所有活跃频段的相位。

    核心功能:
    1. 相位推进: phase += 2π × f × dt/1000
    2. 门控信号: encoding_gate / retrieval_gate (基于 theta 相位)
    3. 调制因子: 返回 [0, 1] 的发放概率调制值
    4. 状态快照: 输出 OscillationState 数据结构

    使用示例:
        clock = OscillationClock()
        clock.add_band(OscillationBand.THETA, THETA_PARAMS)

        for t in range(1000):
            clock.step()
            if clock.is_encoding_phase():
                # EC→DG→CA3 通路激活
                ...
            else:
                # CA3→CA1→EC 通路激活
                ...
    """

    def __init__(self, dt: float = 1.0):
        """
        Args:
            dt: 时间步长 (ms), 默认 1.0
        """
        self._dt = dt
        self._bands: Dict[OscillationBand, OscillationParams] = {}
        self._phases: Dict[OscillationBand, float] = {}
        self._time: int = 0

    def add_band(self, band: OscillationBand, params: OscillationParams = None) -> None:
        """添加一个振荡频段

        Args:
            band: 频段枚举 (THETA/GAMMA/...)
            params: 振荡参数, None 则使用默认
        """
        if params is None:
            defaults = {
                OscillationBand.DELTA: DELTA_PARAMS,
                OscillationBand.THETA: THETA_PARAMS,
                OscillationBand.ALPHA: ALPHA_PARAMS,
                OscillationBand.BETA: BETA_PARAMS,
                OscillationBand.GAMMA: GAMMA_PARAMS,
            }
            params = defaults.get(band, THETA_PARAMS)

        self._bands[band] = params
        self._phases[band] = params.phase_offset

    def step(self) -> None:
        """推进一个时间步

        所有活跃频段的相位前进:
          phase += 2π × frequency × dt / 1000
        相位保持在 [0, 2π) 范围内。
        """
        self._time += 1
        dt_sec = self._dt / 1000.0  # ms → s
        two_pi = 2.0 * math.pi

        for band, params in self._bands.items():
            if params.power > 0:
                self._phases[band] += two_pi * params.frequency * dt_sec
                # 保持在 [0, 2π)
                if self._phases[band] >= two_pi:
                    self._phases[band] -= two_pi

    def get_phase(self, band: OscillationBand) -> float:
        """获取指定频段的当前相位

        Args:
            band: 频段枚举

        Returns:
            当前相位 [0, 2π), 如果频段未添加返回 0.0
        """
        return self._phases.get(band, 0.0)

    def get_power(self, band: OscillationBand) -> float:
        """获取指定频段的功率

        Returns:
            功率 [0, 1], 如果频段未添加返回 0.0
        """
        params = self._bands.get(band)
        return params.power if params else 0.0

    def get_modulation(self, band: OscillationBand) -> float:
        """获取指定频段的发放概率调制因子

        基于 Communication Through Coherence (CTC) 理论:
        - 兴奋相位窗口 (cos > 0): 调制因子 > 0.5
        - 抑制相位窗口 (cos < 0): 调制因子 < 0.5

        Returns:
            调制因子 [0, 1], 如果频段不活跃返回 0.5 (无调制)
        """
        params = self._bands.get(band)
        if not params or params.power <= 0:
            return 0.5

        phase = self._phases.get(band, 0.0)
        # 调制 = 0.5 + 0.5 * power * cos(phase)
        # power=1 时: cos=1 → 1.0, cos=-1 → 0.0, cos=0 → 0.5
        return 0.5 + 0.5 * params.power * math.cos(phase)

    # =========================================================================
    # Theta 专用: 编码/检索相位门控
    # =========================================================================

    def is_encoding_phase(self) -> bool:
        """当前是否处于 theta 编码相位

        编码相位 = theta 波谷附近 (cos(θ) < 0)
        生物学: theta 谷时 EC→DG→CA3 通路最活跃

        Returns:
            True 如果在编码相位, False 如果在检索相位
            如果 THETA 未添加, 默认返回 True (始终编码)
        """
        if OscillationBand.THETA not in self._phases:
            return True
        phase = self._phases[OscillationBand.THETA]
        return math.cos(phase) < 0

    def is_retrieval_phase(self) -> bool:
        """当前是否处于 theta 检索相位

        检索相位 = theta 波峰附近 (cos(θ) > 0)
        生物学: theta 峰时 CA3→CA1 通路最活跃

        Returns:
            True 如果在检索相位
        """
        return not self.is_encoding_phase()

    def get_encoding_strength(self) -> float:
        """编码通路强度 [0, 1]

        平滑版本的编码门控, 用于调制突触电流而非二值开关。
        = max(0, -cos(theta_phase)) * power

        Returns:
            编码强度 [0, 1], 谷时最大=1, 峰时=0
        """
        params = self._bands.get(OscillationBand.THETA)
        if not params or params.power <= 0:
            return 1.0  # 无 theta → 编码通路始终全开

        phase = self._phases.get(OscillationBand.THETA, 0.0)
        raw = -math.cos(phase)  # 谷时=+1, 峰时=-1
        return max(0.0, raw) * params.power

    def get_retrieval_strength(self) -> float:
        """检索通路强度 [0, 1]

        = max(0, cos(theta_phase)) * power

        Returns:
            检索强度 [0, 1], 峰时最大=1, 谷时=0
        """
        params = self._bands.get(OscillationBand.THETA)
        if not params or params.power <= 0:
            return 1.0  # 无 theta → 检索通路始终全开

        phase = self._phases.get(OscillationBand.THETA, 0.0)
        raw = math.cos(phase)  # 峰时=+1, 谷时=-1
        return max(0.0, raw) * params.power

    # =========================================================================
    # 状态查询
    # =========================================================================

    def get_state(self) -> OscillationState:
        """获取当前振荡状态快照

        Returns:
            OscillationState 数据结构, 包含 5 个频段的相位和功率
        """
        def _phase(band):
            return self._phases.get(band, 0.0)

        def _power(band):
            p = self._bands.get(band)
            return p.power if p else 0.0

        return OscillationState(
            delta_phase=_phase(OscillationBand.DELTA),
            theta_phase=_phase(OscillationBand.THETA),
            alpha_phase=_phase(OscillationBand.ALPHA),
            beta_phase=_phase(OscillationBand.BETA),
            gamma_phase=_phase(OscillationBand.GAMMA),
            delta_power=_power(OscillationBand.DELTA),
            theta_power=_power(OscillationBand.THETA),
            alpha_power=_power(OscillationBand.ALPHA),
            beta_power=_power(OscillationBand.BETA),
            gamma_power=_power(OscillationBand.GAMMA),
        )

    @property
    def time(self) -> int:
        """当前时间步"""
        return self._time

    def reset(self) -> None:
        """重置所有相位到初始偏移"""
        self._time = 0
        for band, params in self._bands.items():
            self._phases[band] = params.phase_offset

    def __repr__(self) -> str:
        bands_str = ", ".join(
            f"{band.name}({self._phases[band]:.2f}rad)"
            for band in self._bands
        )
        return f"OscillationClock(t={self._time}, {bands_str})"
