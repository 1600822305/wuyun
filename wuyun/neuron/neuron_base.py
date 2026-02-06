"""
Layer 2: 双区室 AdLIF+ 神经元基类

这是悟韵 (WuYun) 系统的"心脏"。

双区室模型结构:
        反馈/预测输入 (来自高层 L6/L1)
              │
              ▼
        ┌──────────┐
        │  apical   │ ← 顶端树突区室
        │ dendrite  │    独立 Ca²⁺ 动力学
        └────┬─────┘
             │  耦合系数 κ (0.0 ~ 1.0)
             ▼
        ┌──────────┐
        │   soma    │ ← 胞体区室
        │  + basal  │    标准 AdLIF+ 动力学
        └────┬─────┘
             │
             ▼
        轴突输出 (spike / burst)
             ↑
        前馈输入 (来自丘脑/低层 L2/3)

发放模式与预测编码的对应关系 (★核心创新):
  - 基底(前馈)激活 + 顶端(反馈)未激活 → REGULAR (预测误差)
  - 基底+顶端同时激活               → BURST   (预测匹配)
  - 仅顶端激活                       → 沉默   (无事发生)
  - 均未激活                         → 沉默
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

from wuyun.spike.signal_types import (
    SpikeType,
    CompartmentType,
    NeuronType,
)
from wuyun.spike.spike import Spike, SpikeTrain
from wuyun.synapse.synapse_base import SynapseBase
from wuyun.neuron.compartment import (
    SomaticCompartment,
    ApicalCompartment,
    SomaticParams,
    ApicalParams,
)


# =============================================================================
# 神经元参数包
# =============================================================================

@dataclass
class NeuronParams:
    """神经元完整参数包

    包含胞体参数、顶端树突参数和耦合系数。
    不同神经元类型通过不同参数实现功能分化。

    κ (kappa) — apical→soma 正向耦合:
      κ = 0.0: 无顶端树突 (退化为单区室, 如抑制性中间神经元)
      κ = 0.1: 弱耦合 (L4 星形细胞)
      κ = 0.2: 中弱耦合 (L6 锥体, 丘脑中继)
      κ = 0.3: 中等耦合 (L2/3 锥体, 位置细胞)
      κ = 0.6: 强耦合 (L5 大锥体 — 最易产生 burst, 驱动输出)

    κ_back (kappa_backward) — soma→apical 反向耦合:
      生物学事实: L5 锥体细胞顶端树突长达 ~1mm,
      反传动作电位沿树突强烈衰减, 所以 soma→apical 耦合远弱于反方向。
      如果对称, 胞体每次 reset 到 -65mV 会把 V_a 拉低, 阻止 Ca²⁺ 脉冲。
      不对称耦合解决这个问题: apical→soma 强 (Ca²⁺ boost), soma→apical 弱。
    """
    somatic: SomaticParams = field(default_factory=SomaticParams)
    apical: ApicalParams = field(default_factory=ApicalParams)
    kappa: float = 0.3           # apical→soma 正向耦合系数
    kappa_backward: float = 0.1  # soma→apical 反向耦合系数 (默认弱)
    neuron_type: NeuronType = NeuronType.L23_PYRAMIDAL

    # burst 参数
    burst_spike_count: int = 3   # burst 中的脉冲数 (2-5)
    burst_isi: int = 5           # burst 内脉冲间隔 (ms), ~200Hz


# 预定义参数集 (对应设计文档 1.2 节参数表)
L23_PYRAMIDAL_PARAMS = NeuronParams(
    somatic=SomaticParams(tau_m=20.0, a=0.02, b=0.5, tau_w=200.0),
    apical=ApicalParams(tau_a=30.0),
    kappa=0.3,
    kappa_backward=0.1,       # 弱反向耦合
    neuron_type=NeuronType.L23_PYRAMIDAL,
)

L5_PYRAMIDAL_PARAMS = NeuronParams(
    somatic=SomaticParams(tau_m=15.0, a=0.01, b=1.0, tau_w=100.0),
    apical=ApicalParams(tau_a=40.0, ca_boost=25.0),
    kappa=0.6,                # 强正向: Ca²⁺ → 胞体 burst
    kappa_backward=0.05,      # 极弱反向: 防止胞体 reset 拉低 V_a
    neuron_type=NeuronType.L5_PYRAMIDAL,
)

L6_PYRAMIDAL_PARAMS = NeuronParams(
    somatic=SomaticParams(tau_m=25.0, a=0.03, b=0.3, tau_w=300.0),
    apical=ApicalParams(tau_a=35.0),
    kappa=0.2,
    kappa_backward=0.08,
    neuron_type=NeuronType.L6_PYRAMIDAL,
)

STELLATE_PARAMS = NeuronParams(
    somatic=SomaticParams(tau_m=15.0, a=0.01, b=0.2, tau_w=150.0),
    apical=ApicalParams(tau_a=25.0),
    kappa=0.1,
    neuron_type=NeuronType.STELLATE,
)

BASKET_PV_PARAMS = NeuronParams(
    somatic=SomaticParams(tau_m=10.0, a=0.0, b=0.0, tau_w=50.0,
                          v_threshold=-45.0, refractory_period=1),
    kappa=0.0,  # 无顶端树突 — 退化为单区室
    neuron_type=NeuronType.BASKET_PV,
)

MARTINOTTI_SST_PARAMS = NeuronParams(
    somatic=SomaticParams(tau_m=20.0, a=0.03, b=0.3, tau_w=300.0,
                          v_threshold=-48.0),
    kappa=0.0,  # 抑制性, 单区室
    neuron_type=NeuronType.MARTINOTTI_SST,
)

VIP_PARAMS = NeuronParams(
    somatic=SomaticParams(tau_m=15.0, a=0.01, b=0.1, tau_w=100.0,
                          v_threshold=-48.0),
    kappa=0.0,  # 抑制性, 单区室
    neuron_type=NeuronType.VIP_INTERNEURON,
)


# =============================================================================
# 双区室神经元基类
# =============================================================================

class NeuronBase:
    """双区室 AdLIF+ 神经元

    所有悟韵神经元的基类。通过参数差异实现类型分化。

    核心能力:
    1. 接收突触输入 (分流到 basal/apical/soma 区室)
    2. 双区室并行更新 (胞体 + 顶端树突)
    3. Ca²⁺ 树突脉冲检测
    4. 发放判定 + burst/regular 类型分类
    5. 脉冲序列记录 (用于 STDP)

    Attributes:
        id: 全局唯一神经元 ID
        params: 神经元参数包
        soma: 胞体区室
        apical: 顶端树突区室 (κ>0 时存在)
        spike_train: 脉冲序列记录器
        afferent_synapses: 传入突触列表 (按区室分组)
    """

    def __init__(
        self,
        neuron_id: int,
        params: NeuronParams = None,
        region_id: int = 0,
        column_id: int = 0,
        layer: int = 0,
    ):
        self.id = neuron_id
        self.params = params or NeuronParams()
        self.region_id = region_id
        self.column_id = column_id
        self.layer = layer

        # === 创建区室 ===
        self.soma = SomaticCompartment(self.params.somatic)

        # 只有 κ > 0 时才创建顶端树突区室
        self._has_apical = self.params.kappa > 0.0
        if self._has_apical:
            self.apical = ApicalCompartment(
                v_rest=self.params.somatic.v_rest,
                params=self.params.apical,
            )
        else:
            self.apical = None

        # === 脉冲记录 ===
        self.spike_train = SpikeTrain(neuron_id=neuron_id)

        # === 传入突触 (按目标区室分组) ===
        self._synapses_basal: List[SynapseBase] = []
        self._synapses_apical: List[SynapseBase] = []
        self._synapses_soma: List[SynapseBase] = []

        # === 当前时间步的输入电流累积 ===
        self._i_basal: float = 0.0
        self._i_apical: float = 0.0
        self._i_soma: float = 0.0

        # === Burst 状态机 ===
        self._burst_remaining: int = 0    # burst 中剩余脉冲数
        self._burst_isi_counter: int = 0  # burst 内脉冲间隔倒计时

        # === 当前时间步的输出 ===
        self._current_spike_type: SpikeType = SpikeType.NONE

    # =========================================================================
    # 突触连接管理
    # =========================================================================

    def add_synapse(self, synapse: SynapseBase) -> None:
        """添加一个传入突触

        根据突触的 target_compartment 自动分流到对应区室。
        """
        if synapse.target_compartment == CompartmentType.BASAL:
            self._synapses_basal.append(synapse)
        elif synapse.target_compartment == CompartmentType.APICAL:
            if self._has_apical:
                self._synapses_apical.append(synapse)
            else:
                # 无顶端树突的神经元: apical 输入重定向到 soma
                self._synapses_soma.append(synapse)
        elif synapse.target_compartment == CompartmentType.SOMA:
            self._synapses_soma.append(synapse)

    # =========================================================================
    # 输入注入 (外部直接注入电流, 不经过突触)
    # =========================================================================

    def inject_basal_current(self, current: float) -> None:
        """直接向基底树突注入电流 (用于外部刺激/测试)"""
        self._i_basal += current

    def inject_apical_current(self, current: float) -> None:
        """直接向顶端树突注入电流 (用于外部刺激/测试)"""
        self._i_apical += current

    def inject_somatic_current(self, current: float) -> None:
        """直接向胞体注入电流"""
        self._i_soma += current

    # =========================================================================
    # 核心仿真步骤
    # =========================================================================

    def step(self, current_time: int, dt: float = 1.0) -> SpikeType:
        """推进一个时间步 — 这是神经元的核心计算

        执行顺序:
        1. 收集突触输入 (分流到各区室)
        2. 更新顶端树突区室 (检测 Ca²⁺ 脉冲)
        3. 更新胞体区室 (检测 Na⁺ 脉冲)
        4. burst/regular 类型判定
        5. 记录脉冲
        6. 清空输入累积

        Args:
            current_time: 当前仿真时间 (时间步)
            dt: 时间步长 (ms)

        Returns:
            本时间步的脉冲类型 (NONE / REGULAR / BURST_*)
        """
        # === Step 1: 收集突触电流 ===
        self._collect_synaptic_currents(current_time, dt)

        # 总胞体输入 = basal 突触 + soma 突触 + 直接注入
        total_soma_input = self._i_basal + self._i_soma

        # === Step 2: 处理正在进行的 burst ===
        if self._burst_remaining > 0:
            spike_type = self._continue_burst(current_time, dt, total_soma_input)
            self._clear_inputs()
            return spike_type

        # === Step 3: 更新顶端树突区室 ===
        # 注意不对称耦合: soma→apical 使用 kappa_backward (弱)
        ca_spike = False
        v_apical = self.params.somatic.v_rest  # 默认: 无顶端树突时用静息电位
        if self._has_apical:
            ca_spike = self.apical.update(
                i_apical=self._i_apical,
                v_soma=self.soma.v,
                kappa=self.params.kappa_backward,  # ★ 弱反向耦合
                dt=dt,
            )
            v_apical = self.apical.v

        # === Step 4: 更新胞体区室 ===
        # 注意不对称耦合: apical→soma 使用 kappa (强)
        fired = self.soma.update(
            i_basal=total_soma_input,
            v_apical=v_apical,
            kappa=self.params.kappa,                # ★ 强正向耦合
            dt=dt,
        )

        # === Step 5: 发放类型判定 ===
        if fired:
            if ca_spike:
                # 前馈 + 反馈同时激活 → BURST (预测匹配!)
                spike_type = SpikeType.BURST_START
                # 启动 burst 状态机
                self._burst_remaining = self.params.burst_spike_count - 1
                self._burst_isi_counter = self.params.burst_isi
            else:
                # 只有前馈输入 → REGULAR (预测误差!)
                spike_type = SpikeType.REGULAR
        else:
            spike_type = SpikeType.NONE

        # === Step 6: 记录脉冲 ===
        if spike_type.is_active:
            self.spike_train.record_spike(current_time, spike_type)

        self._current_spike_type = spike_type

        # === Step 7: 清空输入累积 ===
        self._clear_inputs()

        return spike_type

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _collect_synaptic_currents(self, current_time: int, dt: float) -> None:
        """收集所有传入突触的电流

        每个突触:
        1. 先 step() 更新门控变量 (检查延迟缓冲区)
        2. 再 compute_current() 计算电流
        3. 累加到对应区室的输入
        """
        v_soma = self.soma.v
        v_apical = self.apical.v if self._has_apical else self.params.somatic.v_rest

        # 基底树突突触
        for syn in self._synapses_basal:
            syn.step(current_time, dt)
            self._i_basal += syn.compute_current(v_soma)

        # 顶端树突突触
        for syn in self._synapses_apical:
            syn.step(current_time, dt)
            self._i_apical += syn.compute_current(v_apical)

        # 胞体突触
        for syn in self._synapses_soma:
            syn.step(current_time, dt)
            self._i_soma += syn.compute_current(v_soma)

    def _continue_burst(
        self, current_time: int, dt: float, total_soma_input: float
    ) -> SpikeType:
        """处理进行中的 burst

        Burst = 2-5 个脉冲 @ >100Hz (间隔 ~5ms)
        burst 期间胞体强制发放 (模拟 Ca²⁺ 驱动的持续去极化)
        """
        self._burst_isi_counter -= 1

        # 即使在 burst 中也要更新区室 (维持物理一致性)
        # 不对称耦合: apical.update 用 kappa_backward, soma.update 用 kappa
        v_apical = self.params.somatic.v_rest
        if self._has_apical:
            self.apical.update(
                i_apical=self._i_apical,
                v_soma=self.soma.v,
                kappa=self.params.kappa_backward,  # ★ 弱反向耦合
                dt=dt,
            )
            v_apical = self.apical.v

        self.soma.update(
            i_basal=total_soma_input,
            v_apical=v_apical,
            kappa=self.params.kappa,                # ★ 强正向耦合
            dt=dt,
        )

        if self._burst_isi_counter <= 0:
            # 发放 burst 中的下一个脉冲
            self._burst_remaining -= 1
            self._burst_isi_counter = self.params.burst_isi

            # 强制重置胞体 (模拟 burst 发放)
            self.soma.v = self.params.somatic.v_reset
            self.soma.w += self.params.somatic.b * 0.5  # burst 内适应较弱

            if self._burst_remaining <= 0:
                spike_type = SpikeType.BURST_END
            else:
                spike_type = SpikeType.BURST_CONTINUE

            self.spike_train.record_spike(current_time, spike_type)
            self._current_spike_type = spike_type
            return spike_type

        # burst 内脉冲间隔中: 不发放
        self._current_spike_type = SpikeType.NONE
        return SpikeType.NONE

    def _clear_inputs(self) -> None:
        """清空输入电流累积 (每个时间步重置)"""
        self._i_basal = 0.0
        self._i_apical = 0.0
        self._i_soma = 0.0

    # =========================================================================
    # 状态查询
    # =========================================================================

    @property
    def v_soma(self) -> float:
        """胞体膜电位"""
        return self.soma.v

    @property
    def v_apical(self) -> float:
        """顶端树突膜电位 (无顶端树突则返回静息电位)"""
        return self.apical.v if self._has_apical else self.params.somatic.v_rest

    @property
    def ca_spike(self) -> bool:
        """顶端树突是否有 Ca²⁺ 脉冲"""
        return self.apical.ca_spike if self._has_apical else False

    @property
    def has_apical(self) -> bool:
        """是否有顶端树突区室"""
        return self._has_apical

    @property
    def kappa(self) -> float:
        """区室耦合系数"""
        return self.params.kappa

    @property
    def current_spike_type(self) -> SpikeType:
        """当前时间步的脉冲类型"""
        return self._current_spike_type

    @property
    def is_bursting(self) -> bool:
        """是否正在进行 burst"""
        return self._burst_remaining > 0

    @property
    def firing_rate(self) -> float:
        """最近 1 秒平均发放率 (Hz)"""
        return self.spike_train.firing_rate()

    @property
    def burst_ratio(self) -> float:
        """最近 1 秒 burst/total 比率"""
        return self.spike_train.burst_ratio()

    def reset(self) -> None:
        """重置神经元到初始状态"""
        self.soma.reset()
        if self._has_apical:
            self.apical.reset()
        self.spike_train.clear()
        self._i_basal = 0.0
        self._i_apical = 0.0
        self._i_soma = 0.0
        self._burst_remaining = 0
        self._burst_isi_counter = 0
        self._current_spike_type = SpikeType.NONE

    def __repr__(self) -> str:
        return (
            f"NeuronBase(id={self.id}, "
            f"type={self.params.neuron_type.name}, "
            f"κ={self.params.kappa}, "
            f"V_s={self.soma.v:.1f}mV, "
            f"V_a={self.v_apical:.1f}mV, "
            f"ca={self.ca_spike})"
        )