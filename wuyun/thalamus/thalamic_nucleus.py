"""
丘脑核团 (ThalamicNucleus) — 单个丘脑核团 (向量化版本)

一个核团 = TC 中继神经元群 + TRN 门控神经元群 + SynapseGroup。

内部连接 (向量化 SynapseGroup):
  TC → TRN  (basal, AMPA, p=0.3)  TC 侧支激活 TRN
  TRN → TC  (soma, GABA_A, p=0.5) TRN 抑制 TC (门控)

依赖: core/ (NeuronPopulation, SynapseGroup)
"""

from typing import List, Optional
import numpy as np

from wuyun.spike.signal_types import (
    SpikeType,
    SynapseType,
    CompartmentType,
    NeuronType,
)
from wuyun.spike.spike import Spike
from wuyun.synapse.synapse_base import AMPA_PARAMS, GABA_A_PARAMS
from wuyun.neuron.neuron_base import (
    THALAMIC_RELAY_PARAMS,
    TRN_PARAMS,
)
from wuyun.core.population import NeuronPopulation
from wuyun.core.synapse_group import SynapseGroup


def _make_thalamic_neuron_id(nucleus_id: int, local_id: int) -> int:
    """生成丘脑神经元全局唯一 ID

    编码规则: nucleus_id * 10000 + 80 * 100 + local_id
    用 80 区分于皮层柱的 4/5/6/23 层编号。
    """
    return nucleus_id * 10000 + 80 * 100 + local_id


class ThalamicNucleus:
    """单个丘脑核团 (向量化版本)

    包含 TC 中继神经元群和 TRN 门控神经元群。
    TC 接收感觉输入 (basal) 和皮层反馈 (apical)，
    TRN 接收 TC 侧支和外部驱动，抑制 TC 实现门控。

    Attributes:
        nucleus_id: 核团 ID
        n_tc: TC 数量
        n_trn: TRN 数量
        tc_pop: NeuronPopulation (TC 中继)
        trn_pop: NeuronPopulation (TRN 门控)
        tc_trn_syn: SynapseGroup (TC→TRN)
        trn_tc_syn: SynapseGroup (TRN→TC)
    """

    def __init__(
        self,
        nucleus_id: int,
        n_tc: int,
        n_trn: int,
        tc_pop: NeuronPopulation,
        trn_pop: NeuronPopulation,
        tc_trn_syn: SynapseGroup,
        trn_tc_syn: SynapseGroup,
    ):
        self.nucleus_id = nucleus_id
        self.n_tc = n_tc
        self.n_trn = n_trn
        self.tc_pop = tc_pop
        self.trn_pop = trn_pop
        self.tc_trn_syn = tc_trn_syn
        self.trn_tc_syn = trn_tc_syn

        # ID 基址
        self._tc_id_base = _make_thalamic_neuron_id(nucleus_id, 0)
        self._trn_id_base = _make_thalamic_neuron_id(nucleus_id, n_tc)

        # 当前时间步输出缓存
        self._tc_output: List[Spike] = []
        self._trn_output: List[Spike] = []
        self._current_time: int = 0

        # burst 比率窗口统计
        self._tc_active_count = 0
        self._tc_burst_count = 0

    # =========================================================================
    # 外部输入接口
    # =========================================================================

    def inject_sensory_current(self, current: float) -> None:
        """向所有 TC 的 basal 注入感觉驱动电流"""
        self.tc_pop.i_basal += current

    def inject_cortical_feedback_current(self, current: float) -> None:
        """向所有 TC 的 apical 注入皮层反馈电流 (L6 预测)"""
        self.tc_pop.i_apical += current

    def inject_trn_drive_current(self, current: float) -> None:
        """向所有 TRN 的 basal 注入外部驱动电流"""
        self.trn_pop.i_basal += current

    def receive_cortical_feedback(self, spikes: List[Spike]) -> None:
        """接收皮层反馈脉冲

        向量化模式下不再通过 SpikeBus，此方法保留兼容但为 no-op。
        外部应通过 inject_cortical_feedback_current 注入电流。
        """
        pass

    # =========================================================================
    # 仿真步进
    # =========================================================================

    def step(self, current_time: int, dt: float = 1.0) -> None:
        """推进一个时间步

        执行顺序:
        1. TC step
        2. TC→TRN 传递
        3. TRN step
        4. TRN→TC 传递
        5. 收集输出

        Args:
            current_time: 当前仿真时间步
            dt: 时间步长 (ms)
        """
        self._current_time = current_time
        self._tc_output.clear()
        self._trn_output.clear()

        # --- 1. TC step ---
        self.tc_pop.step(current_time)

        # --- 2. TC→TRN 传递 ---
        self.tc_trn_syn.deliver_spikes(self.tc_pop.fired, self.tc_pop.spike_type)
        i_tc_trn = self.tc_trn_syn.step_and_compute(self.trn_pop.v_soma)
        self.trn_pop.i_basal += i_tc_trn

        # --- 3. TRN step ---
        self.trn_pop.step(current_time)

        # --- 4. TRN→TC 传递 ---
        self.trn_tc_syn.deliver_spikes(self.trn_pop.fired, self.trn_pop.spike_type)
        i_trn_tc = self.trn_tc_syn.step_and_compute(self.tc_pop.v_soma)
        self.tc_pop.i_soma += i_trn_tc

        # --- 5. 收集输出 + burst 统计 ---
        tc_fired = np.nonzero(self.tc_pop.fired)[0]
        tc_st = self.tc_pop.spike_type
        self._tc_active_count += len(tc_fired)
        for i in tc_fired:
            st_val = int(tc_st[i])
            if st_val in (SpikeType.BURST_START.value,
                         SpikeType.BURST_CONTINUE.value,
                         SpikeType.BURST_END.value):
                self._tc_burst_count += 1
            self._tc_output.append(Spike(
                self._tc_id_base + i, current_time,
                SpikeType(st_val)))

        trn_fired = np.nonzero(self.trn_pop.fired)[0]
        for i in trn_fired:
            self._trn_output.append(Spike(
                self._trn_id_base + i, current_time,
                SpikeType(int(self.trn_pop.spike_type[i]))))

    # =========================================================================
    # 输出查询
    # =========================================================================

    def get_tc_output(self) -> List[Spike]:
        """获取当前步 TC 发放的脉冲"""
        return list(self._tc_output)

    def get_trn_output(self) -> List[Spike]:
        """获取当前步 TRN 发放的脉冲"""
        return list(self._trn_output)

    def get_tc_firing_rate(self) -> float:
        """TC 群体平均发放率 (Hz)"""
        if self.n_tc == 0:
            return 0.0
        rates = self.tc_pop.get_firing_rates(
            window_ms=1000, current_time=self._current_time)
        return float(np.mean(rates))

    def get_trn_firing_rate(self) -> float:
        """TRN 群体平均发放率 (Hz)"""
        if self.n_trn == 0:
            return 0.0
        rates = self.trn_pop.get_firing_rates(
            window_ms=1000, current_time=self._current_time)
        return float(np.mean(rates))

    def get_tc_burst_ratio(self) -> float:
        """TC 群体累积 burst 比率 (窗口统计)"""
        if self._tc_active_count == 0:
            return 0.0
        return self._tc_burst_count / self._tc_active_count

    # =========================================================================
    # 生命周期
    # =========================================================================

    def reset(self) -> None:
        """重置核团到初始状态"""
        self.tc_pop.reset()
        self.trn_pop.reset()
        self.tc_trn_syn.reset()
        self.trn_tc_syn.reset()
        self._tc_output.clear()
        self._trn_output.clear()
        self._current_time = 0
        self._tc_active_count = 0
        self._tc_burst_count = 0

    def __repr__(self) -> str:
        return (
            f"ThalamicNucleus(id={self.nucleus_id}, "
            f"TC={self.n_tc}, "
            f"TRN={self.n_trn}, "
            f"synapses={self.tc_trn_syn.K + self.trn_tc_syn.K})"
        )


# =============================================================================
# 工厂函数
# =============================================================================

def create_thalamic_nucleus(
    nucleus_id: int,
    n_tc: int = 10,
    n_trn: int = 5,
    seed: int = None,
) -> ThalamicNucleus:
    """创建一个丘脑核团

    Args:
        nucleus_id: 核团 ID
        n_tc: TC 中继神经元数量
        n_trn: TRN 门控神经元数量
        seed: 随机种子

    Returns:
        配置好的 ThalamicNucleus
    """
    rng = np.random.RandomState(seed)

    # === 向量化神经元群体 ===
    tc_pop = NeuronPopulation(n_tc, THALAMIC_RELAY_PARAMS)
    trn_pop = NeuronPopulation(n_trn, TRN_PARAMS)

    # === TC → TRN (basal, AMPA, p=0.3) ===
    tc_trn_pre_list, tc_trn_post_list, tc_trn_w_list = [], [], []
    for i in range(n_tc):
        for j in range(n_trn):
            if rng.random() < 0.3:
                tc_trn_pre_list.append(i)
                tc_trn_post_list.append(j)
                tc_trn_w_list.append(rng.uniform(0.3, 0.7))

    n_tc_trn = len(tc_trn_pre_list)
    if n_tc_trn > 0:
        tc_trn_pre = np.array(tc_trn_pre_list, dtype=np.int32)
        tc_trn_post = np.array(tc_trn_post_list, dtype=np.int32)
        tc_trn_w = np.array(tc_trn_w_list)
    else:
        tc_trn_pre = np.zeros(0, dtype=np.int32)
        tc_trn_post = np.zeros(0, dtype=np.int32)
        tc_trn_w = np.zeros(0)

    tc_trn_syn = SynapseGroup(
        pre_ids=tc_trn_pre, post_ids=tc_trn_post,
        weights=tc_trn_w,
        delays=np.ones(n_tc_trn, dtype=np.int32),
        synapse_type=SynapseType.AMPA,
        target=CompartmentType.BASAL,
        tau_decay=AMPA_PARAMS.tau_decay,
        e_rev=AMPA_PARAMS.e_rev,
        g_max=AMPA_PARAMS.g_max,
        n_post=n_trn,
    )

    # === TRN → TC (soma, GABA_A, p=0.5) ===
    trn_tc_pre_list, trn_tc_post_list, trn_tc_w_list = [], [], []
    for i in range(n_trn):
        for j in range(n_tc):
            if rng.random() < 0.5:
                trn_tc_pre_list.append(i)
                trn_tc_post_list.append(j)
                trn_tc_w_list.append(rng.uniform(0.3, 0.7))

    n_trn_tc = len(trn_tc_pre_list)
    if n_trn_tc > 0:
        trn_tc_pre = np.array(trn_tc_pre_list, dtype=np.int32)
        trn_tc_post = np.array(trn_tc_post_list, dtype=np.int32)
        trn_tc_w = np.array(trn_tc_w_list)
    else:
        trn_tc_pre = np.zeros(0, dtype=np.int32)
        trn_tc_post = np.zeros(0, dtype=np.int32)
        trn_tc_w = np.zeros(0)

    trn_tc_syn = SynapseGroup(
        pre_ids=trn_tc_pre, post_ids=trn_tc_post,
        weights=trn_tc_w,
        delays=np.ones(n_trn_tc, dtype=np.int32),
        synapse_type=SynapseType.GABA_A,
        target=CompartmentType.SOMA,
        tau_decay=GABA_A_PARAMS.tau_decay,
        e_rev=GABA_A_PARAMS.e_rev,
        g_max=GABA_A_PARAMS.g_max,
        n_post=n_tc,
    )

    return ThalamicNucleus(
        nucleus_id=nucleus_id,
        n_tc=n_tc,
        n_trn=n_trn,
        tc_pop=tc_pop,
        trn_pop=trn_pop,
        tc_trn_syn=tc_trn_syn,
        trn_tc_syn=trn_tc_syn,
    )