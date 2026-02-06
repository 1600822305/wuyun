"""
SynapseGroup — 向量化突触组

将 List[SynapseBase] + SpikeBus 的逐对象循环替换为批量矩阵运算。

同一 SynapseGroup 内的突触共享:
  - 突触类型 (AMPA/NMDA/GABA_A/GABA_B)
  - 目标区室 (BASAL/APICAL/SOMA)
  - 生物物理参数 (tau_decay, e_rev, g_max)

核心数据:
  pre_ids[K]:   突触前神经元在源 Population 中的索引
  post_ids[K]:  突触后神经元在目标 Population 中的索引
  weights[K]:   突触权重
  s[K]:         门控变量 (突触激活程度)
  delays[K]:    传导延迟 (时间步)

电流计算 (电导模型):
  I_syn[k] = g_max * weights[k] * s[k] * (E_rev - V_post[post_ids[k]])
  I_post[i] = Σ_{k: post_ids[k]==i} I_syn[k]

设计文档: docs/05_vectorization_design.md §3.2
"""

import math
import numpy as np
from typing import Optional

from wuyun.spike.signal_types import SynapseType, CompartmentType


class SynapseGroup:
    """向量化突触组 — 同类型突触的批量计算

    使用示例:
        # 创建 AMPA 突触组: 10 个突触连接 pre_pop → post_pop
        group = SynapseGroup(
            pre_ids=np.array([0,0,1,1,2,2,3,3,4,4]),
            post_ids=np.array([0,1,0,1,0,1,0,1,0,1]),
            weights=np.full(10, 0.5),
            delays=np.ones(10, dtype=np.int32),
            synapse_type=SynapseType.AMPA,
            target=CompartmentType.BASAL,
            tau_decay=2.0, e_rev=0.0, g_max=1.0,
            n_post=2,
        )

        # 每步:
        group.deliver_spikes(pre_pop.fired, pre_pop.spike_type)
        I_post = group.step_and_compute(post_pop.v_soma)
        post_pop.i_basal += I_post
    """

    def __init__(
        self,
        pre_ids: np.ndarray,
        post_ids: np.ndarray,
        weights: np.ndarray,
        delays: np.ndarray,
        synapse_type: SynapseType,
        target: CompartmentType,
        tau_decay: float,
        e_rev: float,
        g_max: float,
        n_post: int,
        w_max: float = 1.0,
        w_min: float = 0.0,
    ):
        """
        Args:
            pre_ids: int[K], 突触前神经元索引 (在源 Population 中)
            post_ids: int[K], 突触后神经元索引 (在目标 Population 中)
            weights: float[K], 初始突触权重
            delays: int[K], 传导延迟 (时间步)
            synapse_type: 突触类型
            target: 目标区室
            tau_decay: 衰减时间常数 (ms)
            e_rev: 反转电位 (mV)
            g_max: 最大电导 (nS)
            n_post: 目标群体大小
            w_max: 权重上界
            w_min: 权重下界
        """
        self.K = len(pre_ids)
        self.pre_ids = pre_ids.astype(np.int32)
        self.post_ids = post_ids.astype(np.int32)
        self.weights = weights.astype(np.float64).copy()
        self.delays = delays.astype(np.int32)
        self.synapse_type = synapse_type
        self.target = target
        self.n_post = n_post
        self.w_max = w_max
        self.w_min = w_min

        # 生物物理参数
        self.tau_decay = tau_decay
        self.e_rev = e_rev
        self.g_max = g_max
        self.is_nmda = (synapse_type == SynapseType.NMDA)

        # 预计算衰减因子
        self.decay_factor = math.exp(-1.0 / tau_decay)

        # === 动态状态 ===
        self.s = np.zeros(self.K)                     # 门控变量
        self.eligibility = np.zeros(self.K)            # 资格痕迹 (三因子学习)
        self._elig_decay = math.exp(-1.0 / 1000.0)    # 资格痕迹衰减因子

        # === 延迟缓冲 (环形矩阵) ===
        # delay_buffer[slot, k]: 0=无, 1=regular, 2=burst
        self.max_delay = int(delays.max()) if self.K > 0 else 1
        self._buf_size = self.max_delay + 1
        self.delay_buffer = np.zeros((self._buf_size, self.K), dtype=np.int8)
        self._time_ptr = 0

    # =========================================================================
    # 脉冲传递
    # =========================================================================

    def deliver_spikes(self, pre_fired: np.ndarray, pre_spike_type: np.ndarray):
        """将突触前群体的脉冲送入延迟缓冲

        Args:
            pre_fired: bool[N_pre], 突触前群体本步发放状态
            pre_spike_type: int8[N_pre], 脉冲类型 (SpikeType.value)
        """
        if self.K == 0:
            return

        # 找出哪些突触的 pre 发放了
        active_mask = pre_fired[self.pre_ids]
        if not active_mask.any():
            return

        active_idx = np.nonzero(active_mask)[0]
        spike_vals = pre_spike_type[self.pre_ids[active_idx]]

        # burst 脉冲编码为 2, 其他活跃脉冲编码为 1
        is_burst = spike_vals >= 4  # BURST_START=4, CONTINUE=5, END=6
        buf_vals = np.where(is_burst, np.int8(2), np.int8(1))

        # 放入对应延迟槽
        arrival_slots = (self._time_ptr + self.delays[active_idx]) % self._buf_size
        # 同一槽内可能有多个突触, 直接赋值即可 (每个突触独立)
        self.delay_buffer[arrival_slots, active_idx] = buf_vals

    # =========================================================================
    # 门控更新 + 电流计算
    # =========================================================================

    def step_and_compute(self, v_post: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """向量化 step + compute_current

        1. 门控变量衰减
        2. 检查到达的脉冲 → 增加门控
        3. 计算电导 × 驱动力 → 聚合到 post neurons

        Args:
            v_post: float[N_post], 突触后群体膜电位

        Returns:
            I_post: float[N_post], 聚合后的突触电流
        """
        if self.K == 0:
            return np.zeros(self.n_post)

        # --- 1. 门控衰减 ---
        self.s *= self.decay_factor

        # --- 2. 检查到达的脉冲 ---
        arrived = self.delay_buffer[self._time_ptr]
        spike_mask = arrived > 0
        if spike_mask.any():
            spike_idx = np.nonzero(spike_mask)[0]
            increment = np.where(arrived[spike_idx] == 2, 1.5, 1.0)
            self.s[spike_idx] = np.minimum(self.s[spike_idx] + increment, 1.0)
            self.delay_buffer[self._time_ptr, spike_idx] = 0

        # 推进时间指针
        self._time_ptr = (self._time_ptr + 1) % self._buf_size

        # --- 3. 资格痕迹衰减 ---
        has_elig = np.any(self.eligibility != 0.0)
        if has_elig:
            self.eligibility *= self._elig_decay

        # --- 4. 计算突触电流 ---
        # 跳过全部非活跃的情况
        active = self.s > 1e-7
        if not active.any():
            return np.zeros(self.n_post)

        # conductance = g_max * weight * s
        conductance = self.g_max * self.weights * self.s

        # NMDA 电压门控 (Mg²⁺ 阻断)
        if self.is_nmda:
            v_at_post = v_post[self.post_ids]
            mg_block = 1.0 / (1.0 + 0.28011204 * np.exp(-0.062 * v_at_post))
            conductance *= mg_block

        # driving force
        driving = self.e_rev - v_post[self.post_ids]

        # per-synapse current
        i_syn = conductance * driving

        # 非活跃突触置零 (避免浮点噪声)
        i_syn[~active] = 0.0

        # --- 5. 聚合到 post neurons ---
        I_post = np.zeros(self.n_post)
        np.add.at(I_post, self.post_ids, i_syn)

        return I_post

    # =========================================================================
    # 可塑性
    # =========================================================================

    def update_weights_stdp(
        self,
        pre_spike_cache: dict,
        post_spike_cache: dict,
        a_plus: float = 0.005,
        a_minus: float = 0.00525,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ):
        """向量化 STDP 权重更新

        Args:
            pre_spike_cache: {pre_idx: np.ndarray of spike times}
            post_spike_cache: {post_idx: np.ndarray of spike times}
            a_plus: LTP 幅度
            a_minus: LTD 幅度
            tau_plus: LTP 时间常数 (ms)
            tau_minus: LTD 时间常数 (ms)
        """
        if not pre_spike_cache or not post_spike_cache:
            return

        # 找出 pre 和 post 都有脉冲的突触
        pre_has = np.zeros(self.K, dtype=bool)
        post_has = np.zeros(self.K, dtype=bool)

        pre_set = set(pre_spike_cache.keys())
        post_set = set(post_spike_cache.keys())

        for k in range(self.K):
            if self.pre_ids[k] in pre_set:
                pre_has[k] = True
            if self.post_ids[k] in post_set:
                post_has[k] = True

        relevant = pre_has & post_has
        if not relevant.any():
            return

        rel_idx = np.nonzero(relevant)[0]
        w_range = self.w_max - self.w_min

        for syn_k in rel_idx:
            pre_times = pre_spike_cache[self.pre_ids[syn_k]]
            post_times = post_spike_cache[self.post_ids[syn_k]]

            # 向量化 STDP 窗口计算
            dt_matrix = post_times[:, None].astype(np.float64) - pre_times[None, :].astype(np.float64)

            raw_dw = 0.0
            ltp_vals = dt_matrix[dt_matrix > 0]
            if len(ltp_vals):
                raw_dw += a_plus * float(np.exp(-ltp_vals / tau_plus).sum())
            ltd_vals = dt_matrix[dt_matrix < 0]
            if len(ltd_vals):
                raw_dw -= a_minus * float(np.exp(ltd_vals / tau_minus).sum())

            if raw_dw == 0.0:
                continue

            # 软边界
            w = self.weights[syn_k]
            if w_range > 0:
                if raw_dw > 0:
                    raw_dw *= (self.w_max - w) / w_range
                else:
                    raw_dw *= (w - self.w_min) / w_range

            self.weights[syn_k] = float(np.clip(w + raw_dw, self.w_min, self.w_max))

    # =========================================================================
    # 状态管理
    # =========================================================================

    def reset(self):
        """重置所有动态状态"""
        self.s[:] = 0.0
        self.eligibility[:] = 0.0
        self.delay_buffer[:] = 0
        self._time_ptr = 0

    @property
    def is_excitatory(self) -> bool:
        return self.synapse_type in (SynapseType.AMPA, SynapseType.NMDA)

    def get_mean_weight(self) -> float:
        return float(self.weights.mean()) if self.K > 0 else 0.0

    def __repr__(self) -> str:
        return (f"SynapseGroup(K={self.K}, type={self.synapse_type.name}, "
                f"target={self.target.name}, w_mean={self.get_mean_weight():.4f})")
