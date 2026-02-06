"""
Layer 3: Column Factory — 皮层柱工厂函数

创建预配置的皮层柱实例:
- 分配神经元 (每层的类型/数量)
- 建立柱内突触连接 (解剖拓扑, 可硬编码)
- 挂载可塑性规则
- 注册到 SpikeBus

连接拓扑 (解剖结构 = 基因决定, 允许硬编码):
  L4 stellate → L23 pyramidal  (basal)  AMPA  p=0.3   层间前馈
  L23 pyramidal → L5 pyramidal (basal)  AMPA  p=0.3   层间前馈
  L5 pyramidal → L6 pyramidal  (basal)  AMPA  p=0.2   层间前馈
  L6 pyramidal → L23 pyramidal (apical) AMPA  p=0.2   ★ 柱内预测反馈
  L6 pyramidal → L5 pyramidal  (apical) AMPA  p=0.2   ★ 柱内预测反馈
  PV+ → 同层锥体              (soma)   GABA_A p=0.5  胞体抑制
  SST+ → 同层锥体             (apical) GABA_A p=0.3  树突抑制 (控制burst)

权重: 随机初始化 (不能硬编码! 由 STDP 学习)
"""

from typing import List, Tuple
import numpy as np

from wuyun.spike.signal_types import (
    SpikeType,
    SynapseType,
    CompartmentType,
    NeuronType,
)
from wuyun.spike.spike_bus import SpikeBus
from wuyun.synapse.synapse_base import SynapseBase
from wuyun.synapse.plasticity.classical_stdp import ClassicalSTDP
from wuyun.synapse.plasticity.inhibitory_stdp import InhibitorySTDP
from wuyun.neuron.neuron_base import (
    NeuronBase,
    NeuronParams,
    L23_PYRAMIDAL_PARAMS,
    L5_PYRAMIDAL_PARAMS,
    L6_PYRAMIDAL_PARAMS,
    STELLATE_PARAMS,
    BASKET_PV_PARAMS,
    MARTINOTTI_SST_PARAMS,
)
from wuyun.circuit.layer import Layer
from wuyun.circuit.cortical_column import CorticalColumn


# 可塑性规则单例 (同类型突触共享规则对象, 节省内存)
_EXCITATORY_STDP = ClassicalSTDP()
_INHIBITORY_STDP = InhibitorySTDP()


def _make_neuron_id(column_id: int, layer_id: int, local_id: int) -> int:
    """生成全局唯一神经元 ID

    编码规则: column_id * 10000 + layer_id * 100 + local_id

    Args:
        column_id: 柱 ID (0-999)
        layer_id: 层 ID (1, 4, 5, 6, 23)
        local_id: 层内局部 ID (0-99)

    Returns:
        全局唯一 ID
    """
    return column_id * 10000 + layer_id * 100 + local_id


def _create_layer_neurons(
    column_id: int,
    layer_id: int,
    exc_params: NeuronParams,
    n_exc: int,
    n_pv: int,
    n_sst: int = 0,
) -> Tuple[List[NeuronBase], List[NeuronBase], List[NeuronBase], List[NeuronBase]]:
    """创建一层的神经元

    Returns:
        (all_neurons, excitatory, pv_neurons, sst_neurons)
    """
    neurons = []
    excitatory = []
    pv_neurons = []
    sst_neurons = []
    local_id = 0

    # 兴奋性
    for i in range(n_exc):
        nid = _make_neuron_id(column_id, layer_id, local_id)
        neuron = NeuronBase(
            neuron_id=nid,
            params=exc_params,
            column_id=column_id,
            layer=layer_id,
        )
        neurons.append(neuron)
        excitatory.append(neuron)
        local_id += 1

    # PV+
    for i in range(n_pv):
        nid = _make_neuron_id(column_id, layer_id, local_id)
        neuron = NeuronBase(
            neuron_id=nid,
            params=BASKET_PV_PARAMS,
            column_id=column_id,
            layer=layer_id,
        )
        neurons.append(neuron)
        pv_neurons.append(neuron)
        local_id += 1

    # SST+
    for i in range(n_sst):
        nid = _make_neuron_id(column_id, layer_id, local_id)
        neuron = NeuronBase(
            neuron_id=nid,
            params=MARTINOTTI_SST_PARAMS,
            column_id=column_id,
            layer=layer_id,
        )
        neurons.append(neuron)
        sst_neurons.append(neuron)
        local_id += 1

    return neurons, excitatory, pv_neurons, sst_neurons


def _connect_populations(
    pre_neurons: List[NeuronBase],
    post_neurons: List[NeuronBase],
    target_compartment: CompartmentType,
    synapse_type: SynapseType,
    probability: float,
    w_init_range: Tuple[float, float] = (0.3, 0.7),
    delay: int = 1,
    is_inhibitory: bool = False,
    rng: np.random.RandomState = None,
) -> List[SynapseBase]:
    """按概率连接两个神经元群

    Args:
        pre_neurons: 突触前神经元列表
        post_neurons: 突触后神经元列表
        target_compartment: 目标区室 (BASAL/APICAL/SOMA)
        synapse_type: 突触类型 (AMPA/GABA_A)
        probability: 连接概率 [0, 1]
        w_init_range: 初始权重范围 (随机均匀)
        delay: 传导延迟 (ms)
        is_inhibitory: 是否为抑制性突触
        rng: 随机数生成器

    Returns:
        创建的突触列表
    """
    if rng is None:
        rng = np.random.RandomState()

    plasticity_rule = _INHIBITORY_STDP if is_inhibitory else _EXCITATORY_STDP
    synapses = []

    for pre in pre_neurons:
        for post in post_neurons:
            # 不允许自连接
            if pre.id == post.id:
                continue
            # 按概率连接
            if rng.random() < probability:
                w = rng.uniform(w_init_range[0], w_init_range[1])
                syn = SynapseBase(
                    pre_id=pre.id,
                    post_id=post.id,
                    weight=w,
                    delay=delay,
                    synapse_type=synapse_type,
                    target_compartment=target_compartment,
                    plasticity_rule=plasticity_rule,
                )
                # 注册到突触后神经元
                post.add_synapse(syn)
                synapses.append(syn)

    return synapses


def create_sensory_column(
    column_id: int,
    n_per_layer: int = 20,
    seed: int = None,
) -> CorticalColumn:
    """创建一个感觉皮层柱 (如 V1)

    Args:
        column_id: 柱全局 ID
        n_per_layer: 每层兴奋性神经元数量 (默认 20, 测试用小规模)
        seed: 随机种子 (可复现)

    神经元配置:
        L4:  n 个 Stellate (κ=0.1) + n//4 个 PV+
        L23: n 个 L23Pyramidal (κ=0.3) + n//4 个 PV+ + n//4 个 SST+
        L5:  n//2 个 L5Pyramidal (κ=0.6) + n//4 个 PV+
        L6:  n//2 个 L6Pyramidal (κ=0.2) + n//4 个 PV+

    Returns:
        配置好的 CorticalColumn (含所有突触和 SpikeBus)
    """
    rng = np.random.RandomState(seed)

    n_pv = max(1, n_per_layer // 4)
    n_sst = max(1, n_per_layer // 4)
    n_deep = max(1, n_per_layer // 2)

    # === 创建各层神经元 ===

    # L4: 输入层 — Stellate + PV+
    l4_all, l4_exc, l4_pv, _ = _create_layer_neurons(
        column_id, 4, STELLATE_PARAMS, n_per_layer, n_pv, 0,
    )

    # L2/3: 浅层 — L23 Pyramidal + PV+ + SST+
    l23_all, l23_exc, l23_pv, l23_sst = _create_layer_neurons(
        column_id, 23, L23_PYRAMIDAL_PARAMS, n_per_layer, n_pv, n_sst,
    )

    # L5: 输出层 — L5 Pyramidal + PV+
    l5_all, l5_exc, l5_pv, _ = _create_layer_neurons(
        column_id, 5, L5_PYRAMIDAL_PARAMS, n_deep, n_pv, 0,
    )

    # L6: 多形层 — L6 Pyramidal + PV+
    l6_all, l6_exc, l6_pv, _ = _create_layer_neurons(
        column_id, 6, L6_PYRAMIDAL_PARAMS, n_deep, n_pv, 0,
    )

    # === 组装 Layer 对象 ===
    layers = {
        4: Layer(4, l4_all),
        23: Layer(23, l23_all),
        5: Layer(5, l5_all),
        6: Layer(6, l6_all),
    }

    # === 创建柱内突触连接 ===
    all_synapses: List[SynapseBase] = []

    # --- 层间前馈连接 (兴奋性 AMPA → basal) ---

    # L4 stellate → L23 pyramidal (basal)
    syns = _connect_populations(
        l4_exc, l23_exc,
        CompartmentType.BASAL, SynapseType.AMPA,
        probability=0.3, delay=1, rng=rng,
    )
    all_synapses.extend(syns)

    # L23 pyramidal → L5 pyramidal (basal)
    syns = _connect_populations(
        l23_exc, l5_exc,
        CompartmentType.BASAL, SynapseType.AMPA,
        probability=0.3, delay=1, rng=rng,
    )
    all_synapses.extend(syns)

    # L5 pyramidal → L6 pyramidal (basal)
    syns = _connect_populations(
        l5_exc, l6_exc,
        CompartmentType.BASAL, SynapseType.AMPA,
        probability=0.2, delay=1, rng=rng,
    )
    all_synapses.extend(syns)

    # --- 柱内反馈连接 (兴奋性 AMPA → apical) ★关键 ---

    # L6 pyramidal → L23 pyramidal (apical) — 柱内预测
    syns = _connect_populations(
        l6_exc, l23_exc,
        CompartmentType.APICAL, SynapseType.AMPA,
        probability=0.2, delay=2, rng=rng,  # 反馈稍慢 delay=2ms
    )
    all_synapses.extend(syns)

    # L6 pyramidal → L5 pyramidal (apical) — 柱内预测
    syns = _connect_populations(
        l6_exc, l5_exc,
        CompartmentType.APICAL, SynapseType.AMPA,
        probability=0.2, delay=2, rng=rng,
    )
    all_synapses.extend(syns)

    # --- 层内抑制连接 ---

    # L4: PV+ → L4 stellate (soma) — 快速抑制
    syns = _connect_populations(
        l4_pv, l4_exc,
        CompartmentType.SOMA, SynapseType.GABA_A,
        probability=0.5, delay=1,
        is_inhibitory=True, rng=rng,
    )
    all_synapses.extend(syns)

    # L4: L4 stellate → L4 PV+ (basal) — 兴奋性驱动抑制
    syns = _connect_populations(
        l4_exc, l4_pv,
        CompartmentType.BASAL, SynapseType.AMPA,
        probability=0.3, delay=1, rng=rng,
    )
    all_synapses.extend(syns)

    # L23: PV+ → L23 pyramidal (soma) — 快速抑制
    syns = _connect_populations(
        l23_pv, l23_exc,
        CompartmentType.SOMA, SynapseType.GABA_A,
        probability=0.5, delay=1,
        is_inhibitory=True, rng=rng,
    )
    all_synapses.extend(syns)

    # L23: SST+ → L23 pyramidal (apical) — 树突抑制 ★控制 burst
    syns = _connect_populations(
        l23_sst, l23_exc,
        CompartmentType.APICAL, SynapseType.GABA_A,
        probability=0.3, delay=1,
        is_inhibitory=True, rng=rng,
    )
    all_synapses.extend(syns)

    # L23: L23 pyramidal → L23 PV+ (basal) — 驱动
    syns = _connect_populations(
        l23_exc, l23_pv,
        CompartmentType.BASAL, SynapseType.AMPA,
        probability=0.3, delay=1, rng=rng,
    )
    all_synapses.extend(syns)

    # L23: L23 pyramidal → L23 SST+ (basal) — 驱动
    syns = _connect_populations(
        l23_exc, l23_sst,
        CompartmentType.BASAL, SynapseType.AMPA,
        probability=0.2, delay=1, rng=rng,
    )
    all_synapses.extend(syns)

    # L5: PV+ → L5 pyramidal (soma)
    syns = _connect_populations(
        l5_pv, l5_exc,
        CompartmentType.SOMA, SynapseType.GABA_A,
        probability=0.5, delay=1,
        is_inhibitory=True, rng=rng,
    )
    all_synapses.extend(syns)

    # L5: L5 pyramidal → L5 PV+ (basal)
    syns = _connect_populations(
        l5_exc, l5_pv,
        CompartmentType.BASAL, SynapseType.AMPA,
        probability=0.3, delay=1, rng=rng,
    )
    all_synapses.extend(syns)

    # L6: PV+ → L6 pyramidal (soma)
    syns = _connect_populations(
        l6_pv, l6_exc,
        CompartmentType.SOMA, SynapseType.GABA_A,
        probability=0.5, delay=1,
        is_inhibitory=True, rng=rng,
    )
    all_synapses.extend(syns)

    # L6: L6 pyramidal → L6 PV+ (basal)
    syns = _connect_populations(
        l6_exc, l6_pv,
        CompartmentType.BASAL, SynapseType.AMPA,
        probability=0.3, delay=1, rng=rng,
    )
    all_synapses.extend(syns)

    # === 创建 SpikeBus 并注册所有突触 ===
    bus = SpikeBus()
    bus.register_synapses(all_synapses)

    # === 组装 CorticalColumn ===
    column = CorticalColumn(
        column_id=column_id,
        layers=layers,
        bus=bus,
        synapses=all_synapses,
    )

    return column