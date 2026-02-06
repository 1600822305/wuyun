"""
NeuronPopulation — 向量化双区室 AdLIF+ 神经元群体

将 NeuronBase + SomaticCompartment + ApicalCompartment 的逐对象循环
替换为 NumPy 批量矩阵运算。

数学模型 (与 neuron_base.py / compartment.py 完全等价):

胞体:
  τ_m · dV_s/dt = -(V_s - V_rest) + R_s · I_total - w + κ · (V_a - V_s)
  τ_w · dw/dt   = a · (V_s - V_rest) - w
  发放: V_s ≥ V_threshold → reset, w += b, 进入不应期

顶端树突:
  τ_a · dV_a/dt = -(V_a - V_rest) + R_a · I_apical + κ_back · (V_s - V_a)
  Ca²⁺ 脉冲: V_a ≥ V_ca_threshold → ca_spike=True, V_a += ca_boost

Burst 判定:
  fired & ca_spike → BURST_START (后续 CONTINUE/END 由状态机驱动)
  fired & ~ca_spike → REGULAR

设计文档: docs/05_vectorization_design.md §3.1
"""

import numpy as np
from typing import Optional, List

from wuyun.spike.signal_types import SpikeType


# SpikeType 整数编码 (与 SpikeType enum 对应)
_NONE = 0
_REGULAR = SpikeType.REGULAR.value
_BURST_START = SpikeType.BURST_START.value
_BURST_CONTINUE = SpikeType.BURST_CONTINUE.value
_BURST_END = SpikeType.BURST_END.value


class NeuronPopulation:
    """向量化神经元群体 — 同参数神经元的批量计算

    所有状态变量存储为 numpy 数组, step() 一次处理整个群体。

    支持:
    - 双区室 (κ > 0) 和单区室 (κ = 0) 混合
    - burst / regular / silence 三种模式
    - Ca²⁺ 树突脉冲 + burst 状态机
    - 不应期
    - 脉冲历史环形缓冲 (用于 STDP)

    使用示例:
        from wuyun.neuron.neuron_base import L23_PYRAMIDAL_PARAMS
        pop = NeuronPopulation(100, L23_PYRAMIDAL_PARAMS)
        pop.i_basal[:] = 5.0  # 注入电流
        fired = pop.step(t=0)
    """

    # 环形缓冲大小 (记录最近 K 次脉冲时间, 用于 STDP)
    SPIKE_RING_SIZE = 32

    def __init__(self, n: int, params):
        """
        Args:
            n: 群体中的神经元数量
            params: NeuronParams 对象 (来自 neuron_base.py)
        """
        self.n = n
        self.params = params

        # === 参数向量 (长度 N, 全部为同值 — 后续可支持异构) ===
        p_s = params.somatic
        p_a = params.apical

        self.v_rest = np.full(n, p_s.v_rest)
        self.v_threshold = np.full(n, p_s.v_threshold)
        self.v_reset = np.full(n, p_s.v_reset)
        self.tau_m = np.full(n, p_s.tau_m)
        self.r_s = np.full(n, p_s.r_s)
        self.a_adapt = np.full(n, p_s.a)           # 亚阈值适应耦合
        self.b_adapt = np.full(n, p_s.b)           # 脉冲后适应增量
        self.tau_w = np.full(n, p_s.tau_w)
        self.refrac_period = np.full(n, p_s.refractory_period, dtype=np.int32)

        self.kappa = np.full(n, params.kappa)       # apical→soma 正向耦合
        self.kappa_back = np.full(n, params.kappa_backward)  # soma→apical 反向耦合
        self.has_apical = params.kappa > 0          # 标量 bool (同构群体)

        self.tau_a = np.full(n, p_a.tau_a)
        self.r_a = np.full(n, p_a.r_a)
        self.v_ca_thresh = np.full(n, p_a.v_ca_threshold)
        self.ca_boost_val = np.full(n, p_a.ca_boost)
        self.ca_dur = np.full(n, p_a.ca_duration, dtype=np.int32)

        self.burst_spike_count = np.full(n, params.burst_spike_count, dtype=np.int32)
        self.burst_isi_val = np.full(n, params.burst_isi, dtype=np.int32)

        # === 动态状态向量 ===
        self.v_soma = np.full(n, p_s.v_rest)
        self.v_apical = np.full(n, p_s.v_rest)
        self.w_adapt = np.zeros(n)
        self.refrac_count = np.zeros(n, dtype=np.int32)
        self.ca_spike = np.zeros(n, dtype=bool)
        self.ca_timer = np.zeros(n, dtype=np.int32)
        self.burst_remain = np.zeros(n, dtype=np.int32)
        self.burst_isi_ct = np.zeros(n, dtype=np.int32)

        # === 输入累积 (每步清零) ===
        self.i_basal = np.zeros(n)
        self.i_apical = np.zeros(n)
        self.i_soma = np.zeros(n)

        # === 输出 ===
        self.fired = np.zeros(n, dtype=bool)
        self.spike_type = np.zeros(n, dtype=np.int8)

        # === 脉冲历史 (环形缓冲, 用于 STDP) ===
        self._spike_ring = np.full((n, self.SPIKE_RING_SIZE), -1, dtype=np.int32)
        self._spike_ptr = np.zeros(n, dtype=np.int32)

    # =========================================================================
    # 核心仿真
    # =========================================================================

    def step(self, t: int, dt: float = 1.0) -> np.ndarray:
        """向量化推进一个时间步

        执行顺序 (与 NeuronBase.step 完全对应):
        1. 顶端树突更新 + Ca²⁺ 脉冲检测
        2. 处理正在进行的 burst (状态机)
        3. 胞体更新 + 发放检测
        4. burst/regular 类型判定
        5. 记录脉冲
        6. 清空输入

        Args:
            t: 当前仿真时间步
            dt: 时间步长 (ms)

        Returns:
            fired: bool[N], 本步发放状态
        """
        n = self.n

        # === Step 1: 顶端树突更新 (向量化) ===
        if self.has_apical:
            self._update_apical(dt)

        # === Step 2: 处理正在进行的 burst (向量化状态机) ===
        # 找出正在 burst 中的神经元
        in_burst = self.burst_remain > 0
        spike_type = np.full(n, _NONE, dtype=np.int8)

        if in_burst.any():
            spike_type = self._continue_burst(in_burst, spike_type, t, dt)

        # === Step 3: 胞体更新 — 只对非 burst 中的神经元 ===
        not_burst = ~in_burst
        if not_burst.any():
            spike_type = self._update_soma_and_fire(not_burst, spike_type, t, dt)

        # === Step 4: 输出 ===
        self.spike_type = spike_type
        self.fired = spike_type != _NONE

        # === Step 5: 记录脉冲到环形缓冲 ===
        fired_idx = np.nonzero(self.fired)[0]
        if len(fired_idx) > 0:
            ptrs = self._spike_ptr[fired_idx]
            ring_idx = ptrs % self.SPIKE_RING_SIZE
            self._spike_ring[fired_idx, ring_idx] = t
            self._spike_ptr[fired_idx] = ptrs + 1

        # === Step 6: 清空输入 ===
        self.i_basal[:] = 0.0
        self.i_apical[:] = 0.0
        self.i_soma[:] = 0.0

        return self.fired

    def _update_apical(self, dt: float):
        """向量化顶端树突更新 + Ca²⁺ 脉冲检测

        等价于 ApicalCompartment.update()
        """
        # τ_a · dV_a/dt = -(V_a - V_rest) + R_a · I_apical + κ_back · (V_s - V_a)
        leak = -(self.v_apical - self.v_rest)
        inp = self.r_a * self.i_apical
        coupling = self.kappa_back * (self.v_soma - self.v_apical)
        dv = (leak + inp + coupling) / self.tau_a * dt
        self.v_apical += dv

        # Ca²⁺ 脉冲状态机
        # 正在进行中的 Ca²⁺ 脉冲: 倒计时
        active_ca = self.ca_timer > 0
        self.ca_timer[active_ca] -= 1
        ended_ca = active_ca & (self.ca_timer == 0)
        self.ca_spike[ended_ca] = False

        # 新的 Ca²⁺ 脉冲触发
        can_trigger = (~active_ca) & (self.v_apical >= self.v_ca_thresh)
        if can_trigger.any():
            self.ca_spike[can_trigger] = True
            self.ca_timer[can_trigger] = self.ca_dur[can_trigger]
            self.v_apical[can_trigger] += self.ca_boost_val[can_trigger]

    def _continue_burst(
        self,
        in_burst: np.ndarray,
        spike_type: np.ndarray,
        t: int,
        dt: float,
    ) -> np.ndarray:
        """向量化 burst 状态机

        等价于 NeuronBase._continue_burst()

        NeuronBase._continue_burst 的执行顺序:
        1. ISI 倒计时递减
        2. apical.update() (已在 _update_apical 中完成)
        3. soma.update() — ★ 包含不应期检查 (refractory)
        4. if ISI <= 0: 发放 burst 脉冲, 重置胞体
        """
        idx = np.nonzero(in_burst)[0]

        # --- 1. ISI 倒计时 (NeuronBase 先递减 ISI 再更新区室) ---
        self.burst_isi_ct[idx] -= 1

        # --- 2. apical 已在 _update_apical 中完成 ---

        # --- 3. 胞体更新 (含不应期检查, 与 soma.update 等价) ---
        # 不应期中: 只递减计数器, 不更新膜电位
        in_refrac = self.refrac_count[idx] > 0
        refrac_idx = idx[in_refrac]
        self.refrac_count[refrac_idx] -= 1

        # 只更新非不应期的 burst 神经元
        active_mask = ~in_refrac
        active_idx = idx[active_mask]

        if len(active_idx) > 0:
            total_input = self.i_basal[active_idx] + self.i_soma[active_idx]
            v = self.v_soma[active_idx]
            v_a = self.v_apical[active_idx] if self.has_apical else self.v_rest[active_idx]

            leak = -(v - self.v_rest[active_idx])
            inp = self.r_s[active_idx] * total_input
            coupling = self.kappa[active_idx] * (v_a - v)
            dv = (leak + inp - self.w_adapt[active_idx] + coupling) / self.tau_m[active_idx] * dt
            self.v_soma[active_idx] += dv

            dw = (self.a_adapt[active_idx] * (self.v_soma[active_idx] - self.v_rest[active_idx]) - self.w_adapt[active_idx]) / self.tau_w[active_idx] * dt
            self.w_adapt[active_idx] += dw

        # --- 4. ISI 到期 → 发放 burst 中的下一个脉冲 ---
        fire_now = in_burst.copy()
        fire_now[idx] &= (self.burst_isi_ct[idx] <= 0)
        fire_idx = np.nonzero(fire_now)[0]

        if len(fire_idx) > 0:
            self.burst_remain[fire_idx] -= 1
            self.burst_isi_ct[fire_idx] = self.burst_isi_val[fire_idx]

            # 强制重置胞体 (模拟 burst 发放)
            self.v_soma[fire_idx] = self.v_reset[fire_idx]
            self.w_adapt[fire_idx] += self.b_adapt[fire_idx] * 0.5  # burst 内适应较弱

            # 判定 BURST_CONTINUE 还是 BURST_END
            is_end = self.burst_remain[fire_idx] <= 0
            spike_type[fire_idx] = np.where(is_end, _BURST_END, _BURST_CONTINUE)

        return spike_type

    def _update_soma_and_fire(
        self,
        mask: np.ndarray,
        spike_type: np.ndarray,
        t: int,
        dt: float,
    ) -> np.ndarray:
        """向量化胞体更新 + 发放检测

        等价于 SomaticCompartment.update() + NeuronBase 发放判定
        """
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return spike_type

        # 不应期处理
        in_refrac = self.refrac_count[idx] > 0
        refrac_idx = idx[in_refrac]
        self.refrac_count[refrac_idx] -= 1

        # 只更新非不应期的神经元
        active_mask = ~in_refrac
        active_idx = idx[active_mask]
        if len(active_idx) == 0:
            return spike_type

        # τ_m · dV_s/dt = -(V_s - V_rest) + R_s · I_total - w + κ · (V_a - V_s)
        total_input = self.i_basal[active_idx] + self.i_soma[active_idx]
        v = self.v_soma[active_idx]
        v_a = self.v_apical[active_idx] if self.has_apical else self.v_rest[active_idx]

        leak = -(v - self.v_rest[active_idx])
        inp = self.r_s[active_idx] * total_input
        coupling = self.kappa[active_idx] * (v_a - v)
        dv = (leak + inp - self.w_adapt[active_idx] + coupling) / self.tau_m[active_idx] * dt
        self.v_soma[active_idx] += dv

        # 适应变量
        dw = (self.a_adapt[active_idx] * (self.v_soma[active_idx] - self.v_rest[active_idx]) - self.w_adapt[active_idx]) / self.tau_w[active_idx] * dt
        self.w_adapt[active_idx] += dw

        # 发放检测
        fired = self.v_soma[active_idx] >= self.v_threshold[active_idx]
        fired_global = active_idx[fired]

        if len(fired_global) > 0:
            # 重置
            self.v_soma[fired_global] = self.v_reset[fired_global]
            self.w_adapt[fired_global] += self.b_adapt[fired_global]
            self.refrac_count[fired_global] = self.refrac_period[fired_global]

            # burst / regular 判定
            ca_active = self.ca_spike[fired_global]
            burst_start = fired_global[ca_active]
            regular = fired_global[~ca_active]

            if len(burst_start) > 0:
                spike_type[burst_start] = _BURST_START
                self.burst_remain[burst_start] = self.burst_spike_count[burst_start] - 1
                self.burst_isi_ct[burst_start] = self.burst_isi_val[burst_start]

            if len(regular) > 0:
                spike_type[regular] = _REGULAR

        return spike_type

    # =========================================================================
    # 输入注入
    # =========================================================================

    def inject_basal_current(self, indices: np.ndarray, currents: np.ndarray):
        """向指定神经元注入基底树突电流

        Args:
            indices: 目标神经元索引 (int 数组)
            currents: 对应的电流值 (float 数组)
        """
        np.add.at(self.i_basal, indices, currents)

    def inject_basal_current_all(self, current: float):
        """向所有神经元注入相同的基底树突电流"""
        self.i_basal += current

    def inject_apical_current_all(self, current: float):
        """向所有神经元注入相同的顶端树突电流"""
        self.i_apical += current

    # =========================================================================
    # 脉冲历史查询 (用于 STDP)
    # =========================================================================

    def get_recent_spike_times(self, neuron_idx: int, window_ms: int = 50) -> np.ndarray:
        """获取单个神经元最近脉冲时间

        Args:
            neuron_idx: 神经元在群体中的索引
            window_ms: 回溯窗口 (ms)

        Returns:
            时间戳数组 (从早到晚排序)
        """
        ring = self._spike_ring[neuron_idx]
        valid = ring[ring >= 0]
        if len(valid) == 0:
            return np.array([], dtype=np.int32)
        latest = valid.max()
        result = valid[valid > latest - window_ms]
        result.sort()
        return result

    def get_all_recent_spike_times(self, window_ms: int = 50):
        """批量获取所有神经元的最近脉冲时间缓存

        Returns:
            dict: {neuron_idx: np.ndarray of spike times}
            只包含有脉冲的神经元
        """
        cache = {}
        for i in range(self.n):
            times = self.get_recent_spike_times(i, window_ms)
            if len(times) > 0:
                cache[i] = times
        return cache

    # =========================================================================
    # 状态查询
    # =========================================================================

    def get_activity(self) -> np.ndarray:
        """获取当前活跃状态 (binary)

        Returns:
            bool[N], True=本步发放
        """
        return self.fired.copy()

    def get_spike_types(self) -> np.ndarray:
        """获取当前脉冲类型向量

        Returns:
            int8[N], SpikeType.value
        """
        return self.spike_type.copy()

    def get_firing_rates(self, window_ms: int = 1000, current_time: int = 0) -> np.ndarray:
        """估算发放率 (Hz)

        基于环形缓冲中最近 window_ms 内的脉冲数。

        Args:
            window_ms: 统计窗口 (ms)
            current_time: 当前时间 (用于窗口计算)

        Returns:
            float[N], 发放率 Hz
        """
        rates = np.zeros(self.n)
        if current_time <= 0:
            return rates
        cutoff = max(current_time - window_ms, -1)
        for i in range(self.n):
            ring = self._spike_ring[i]
            valid = ring[ring > cutoff]
            rates[i] = len(valid) * 1000.0 / window_ms
        return rates

    # =========================================================================
    # 重置
    # =========================================================================

    def reset(self):
        """重置所有动态状态 (保留参数和连接)"""
        p_s = self.params.somatic
        self.v_soma[:] = p_s.v_rest
        self.v_apical[:] = p_s.v_rest
        self.w_adapt[:] = 0.0
        self.refrac_count[:] = 0
        self.ca_spike[:] = False
        self.ca_timer[:] = 0
        self.burst_remain[:] = 0
        self.burst_isi_ct[:] = 0
        self.i_basal[:] = 0.0
        self.i_apical[:] = 0.0
        self.i_soma[:] = 0.0
        self.fired[:] = False
        self.spike_type[:] = 0
        self._spike_ring[:] = -1
        self._spike_ptr[:] = 0

    def __repr__(self) -> str:
        active = int(self.fired.sum()) if self.fired.any() else 0
        return (f"NeuronPopulation(n={self.n}, "
                f"type={self.params.neuron_type.name}, "
                f"κ={self.params.kappa}, "
                f"active={active})")
