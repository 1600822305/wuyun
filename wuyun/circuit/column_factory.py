"""
Layer 3: Column Factory — 皮层柱工厂函数 (向量化版本)

创建预配置的皮层柱实例:
- 分配 NeuronPopulation (每层的类型/数量)
- 建立柱内 SynapseGroup 连接 (解剖拓扑, 可硬编码)

连接拓扑 (解剖结构 = 基因决定, 允许硬编码):
  L4 stellate → L23 pyramidal  (basal)  AMPA  p=0.3   层间前馈
  L23 pyramidal → L5 pyramidal (basal)  AMPA  p=0.3   层间前馈
  L5 pyramidal → L6 pyramidal  (basal)  AMPA  p=0.2   层间前馈
  L6 pyramidal → L23 pyramidal (apical) NMDA  p=0.3   ★ 柱内预测反馈
  L6 pyramidal → L5 pyramidal  (apical) NMDA  p=0.3   ★ 柱内预测反馈
  PV+ → 同层锥体              (soma)   GABA_A p=0.5  胞体抑制
  SST+ → 同层锥体             (apical) GABA_A p=0.3  树突抑制 (控制burst)

权重: 随机初始化 (不能硬编码! 由 STDP 学习)

依赖: core/ (NeuronPopulation, SynapseGroup)
"""

from typing import List, Tuple
import numpy as np

from wuyun.spike.signal_types import (
    SynapseType,
    CompartmentType,
)
from wuyun.synapse.synapse_base import AMPA_PARAMS, NMDA_PARAMS, GABA_A_PARAMS
from wuyun.neuron.neuron_base import (
    NeuronParams,
    L23_PYRAMIDAL_PARAMS,
    L5_PYRAMIDAL_PARAMS,
    L6_PYRAMIDAL_PARAMS,
    STELLATE_PARAMS,
    BASKET_PV_PARAMS,
    MARTINOTTI_SST_PARAMS,
)
from wuyun.core.population import NeuronPopulation
from wuyun.core.synapse_group import SynapseGroup
from wuyun.circuit.layer import Layer
from wuyun.circuit.cortical_column import CorticalColumn


def _make_neuron_id(column_id: int, layer_id: int, local_id: int) -> int:
    """生成全局唯一神经元 ID

    编码规则: column_id * 10000 + layer_id * 100 + local_id
    """
    return column_id * 10000 + layer_id * 100 + local_id


def _build_synapse_group(
    n_pre: int,
    n_post: int,
    synapse_type: SynapseType,
    target: CompartmentType,
    probability: float,
    w_init_range: Tuple[float, float],
    delay: int,
    rng: np.random.RandomState,
) -> SynapseGroup:
    """按概率创建 SynapseGroup

    Returns:
        SynapseGroup 实例
    """
    pre_list, post_list, w_list = [], [], []
    for i in range(n_pre):
        for j in range(n_post):
            if rng.random() < probability:
                pre_list.append(i)
                post_list.append(j)
                w_list.append(rng.uniform(w_init_range[0], w_init_range[1]))

    k = len(pre_list)
    if k > 0:
        pre_ids = np.array(pre_list, dtype=np.int32)
        post_ids = np.array(post_list, dtype=np.int32)
        weights = np.array(w_list)
    else:
        pre_ids = np.zeros(0, dtype=np.int32)
        post_ids = np.zeros(0, dtype=np.int32)
        weights = np.zeros(0)

    # 选择参数
    if synapse_type == SynapseType.AMPA:
        params = AMPA_PARAMS
    elif synapse_type == SynapseType.NMDA:
        params = NMDA_PARAMS
    elif synapse_type == SynapseType.GABA_A:
        params = GABA_A_PARAMS
    else:
        params = AMPA_PARAMS

    return SynapseGroup(
        pre_ids=pre_ids, post_ids=post_ids,
        weights=weights,
        delays=np.full(k, delay, dtype=np.int32),
        synapse_type=synapse_type,
        target=target,
        tau_decay=params.tau_decay,
        e_rev=params.e_rev,
        g_max=params.g_max,
        n_post=n_post,
    )


def create_sensory_column(
    column_id: int,
    n_per_layer: int = 20,
    seed: int = None,
    ff_connection_strength: float = 1.0,
) -> CorticalColumn:
    """创建一个感觉皮层柱 (如 V1)

    Args:
        column_id: 柱全局 ID
        n_per_layer: 每层兴奋性神经元数量 (默认 20, 测试用小规模)
        seed: 随机种子 (可复现)
        ff_connection_strength: 前馈连接强度倍率 (默认 1.0)

    神经元配置:
        L4:  n 个 Stellate (κ=0.1) + n//4 个 PV+
        L23: n 个 L23Pyramidal (κ=0.3) + n//4 个 PV+ + n//4 个 SST+
        L5:  n//2 个 L5Pyramidal (κ=0.6) + n//4 个 PV+
        L6:  n//2 个 L6Pyramidal (κ=0.2) + n//4 个 PV+

    Returns:
        配置好的 CorticalColumn
    """
    rng = np.random.RandomState(seed)

    ff_w_lo = min(0.3 * ff_connection_strength, 1.0)
    ff_w_hi = min(0.7 * ff_connection_strength, 1.0)
    ff_w_range = (ff_w_lo, ff_w_hi)
    fb_w_range = (ff_w_lo, ff_w_hi)

    n_pv = max(1, n_per_layer // 4)
    n_sst = max(1, n_per_layer // 4)
    n_deep = max(1, n_per_layer // 2)

    # === 创建各层 NeuronPopulation ===
    l4_exc_pop = NeuronPopulation(n_per_layer, STELLATE_PARAMS)
    l4_pv_pop = NeuronPopulation(n_pv, BASKET_PV_PARAMS)

    l23_exc_pop = NeuronPopulation(n_per_layer, L23_PYRAMIDAL_PARAMS)
    l23_pv_pop = NeuronPopulation(n_pv, BASKET_PV_PARAMS)
    l23_sst_pop = NeuronPopulation(n_sst, MARTINOTTI_SST_PARAMS)

    l5_exc_pop = NeuronPopulation(n_deep, L5_PYRAMIDAL_PARAMS)
    l5_pv_pop = NeuronPopulation(n_pv, BASKET_PV_PARAMS)

    l6_exc_pop = NeuronPopulation(n_deep, L6_PYRAMIDAL_PARAMS)
    l6_pv_pop = NeuronPopulation(n_pv, BASKET_PV_PARAMS)

    # === 计算全局 ID 基址 ===
    # ID 编码: column_id * 10000 + layer_id * 100 + local_id
    # 每层内: exc 从 local_id=0 开始, PV 从 n_exc 开始, SST 从 n_exc+n_pv 开始
    l4_exc_base = _make_neuron_id(column_id, 4, 0)
    l4_pv_base = _make_neuron_id(column_id, 4, n_per_layer)

    l23_exc_base = _make_neuron_id(column_id, 23, 0)
    l23_pv_base = _make_neuron_id(column_id, 23, n_per_layer)
    l23_sst_base = _make_neuron_id(column_id, 23, n_per_layer + n_pv)

    l5_exc_base = _make_neuron_id(column_id, 5, 0)
    l5_pv_base = _make_neuron_id(column_id, 5, n_deep)

    l6_exc_base = _make_neuron_id(column_id, 6, 0)
    l6_pv_base = _make_neuron_id(column_id, 6, n_deep)

    # === 组装 Layer 对象 ===
    layers = {
        4: Layer(4, l4_exc_pop, pv_pop=l4_pv_pop,
                 exc_id_base=l4_exc_base, pv_id_base=l4_pv_base),
        23: Layer(23, l23_exc_pop, pv_pop=l23_pv_pop, sst_pop=l23_sst_pop,
                  exc_id_base=l23_exc_base, pv_id_base=l23_pv_base,
                  sst_id_base=l23_sst_base),
        5: Layer(5, l5_exc_pop, pv_pop=l5_pv_pop,
                 exc_id_base=l5_exc_base, pv_id_base=l5_pv_base),
        6: Layer(6, l6_exc_pop, pv_pop=l6_pv_pop,
                 exc_id_base=l6_exc_base, pv_id_base=l6_pv_base),
    }

    # === 创建柱内 SynapseGroup 连接 ===
    # 每个连接: (SynapseGroup, src_pop, tgt_pop)
    connections = []

    # --- 层间前馈 (AMPA → basal) ---
    # L4 exc → L23 exc
    sg = _build_synapse_group(n_per_layer, n_per_layer, SynapseType.AMPA,
                              CompartmentType.BASAL, 0.3, ff_w_range, 1, rng)
    connections.append((sg, l4_exc_pop, l23_exc_pop))

    # L23 exc → L5 exc
    sg = _build_synapse_group(n_per_layer, n_deep, SynapseType.AMPA,
                              CompartmentType.BASAL, 0.3, ff_w_range, 1, rng)
    connections.append((sg, l23_exc_pop, l5_exc_pop))

    # L5 exc → L6 exc
    sg = _build_synapse_group(n_deep, n_deep, SynapseType.AMPA,
                              CompartmentType.BASAL, 0.2, ff_w_range, 1, rng)
    connections.append((sg, l5_exc_pop, l6_exc_pop))

    # --- 柱内反馈 (NMDA → apical) ★关键 ---
    # L6 exc → L23 exc (apical)
    sg = _build_synapse_group(n_deep, n_per_layer, SynapseType.NMDA,
                              CompartmentType.APICAL, 0.3, fb_w_range, 2, rng)
    connections.append((sg, l6_exc_pop, l23_exc_pop))

    # L6 exc → L5 exc (apical)
    sg = _build_synapse_group(n_deep, n_deep, SynapseType.NMDA,
                              CompartmentType.APICAL, 0.3, fb_w_range, 2, rng)
    connections.append((sg, l6_exc_pop, l5_exc_pop))

    # --- 层内抑制 ---
    # L4: PV → exc (soma GABA_A)
    sg = _build_synapse_group(n_pv, n_per_layer, SynapseType.GABA_A,
                              CompartmentType.SOMA, 0.5, (0.3, 0.7), 1, rng)
    connections.append((sg, l4_pv_pop, l4_exc_pop))
    # L4: exc → PV (basal AMPA)
    sg = _build_synapse_group(n_per_layer, n_pv, SynapseType.AMPA,
                              CompartmentType.BASAL, 0.3, (0.3, 0.7), 1, rng)
    connections.append((sg, l4_exc_pop, l4_pv_pop))

    # L23: PV → exc (soma GABA_A)
    sg = _build_synapse_group(n_pv, n_per_layer, SynapseType.GABA_A,
                              CompartmentType.SOMA, 0.5, (0.3, 0.7), 1, rng)
    connections.append((sg, l23_pv_pop, l23_exc_pop))
    # L23: SST → exc (apical GABA_A) ★控制 burst
    sg = _build_synapse_group(n_sst, n_per_layer, SynapseType.GABA_A,
                              CompartmentType.APICAL, 0.3, (0.3, 0.7), 1, rng)
    connections.append((sg, l23_sst_pop, l23_exc_pop))
    # L23: exc → PV (basal AMPA)
    sg = _build_synapse_group(n_per_layer, n_pv, SynapseType.AMPA,
                              CompartmentType.BASAL, 0.3, (0.3, 0.7), 1, rng)
    connections.append((sg, l23_exc_pop, l23_pv_pop))
    # L23: exc → SST (basal AMPA)
    sg = _build_synapse_group(n_per_layer, n_sst, SynapseType.AMPA,
                              CompartmentType.BASAL, 0.2, (0.3, 0.7), 1, rng)
    connections.append((sg, l23_exc_pop, l23_sst_pop))

    # L5: PV → exc (soma GABA_A)
    sg = _build_synapse_group(n_pv, n_deep, SynapseType.GABA_A,
                              CompartmentType.SOMA, 0.5, (0.3, 0.7), 1, rng)
    connections.append((sg, l5_pv_pop, l5_exc_pop))
    # L5: exc → PV (basal AMPA)
    sg = _build_synapse_group(n_deep, n_pv, SynapseType.AMPA,
                              CompartmentType.BASAL, 0.3, (0.3, 0.7), 1, rng)
    connections.append((sg, l5_exc_pop, l5_pv_pop))

    # L6: PV → exc (soma GABA_A)
    sg = _build_synapse_group(n_pv, n_deep, SynapseType.GABA_A,
                              CompartmentType.SOMA, 0.5, (0.3, 0.7), 1, rng)
    connections.append((sg, l6_pv_pop, l6_exc_pop))
    # L6: exc → PV (basal AMPA)
    sg = _build_synapse_group(n_deep, n_pv, SynapseType.AMPA,
                              CompartmentType.BASAL, 0.3, (0.3, 0.7), 1, rng)
    connections.append((sg, l6_exc_pop, l6_pv_pop))

    # === 组装 CorticalColumn ===
    column = CorticalColumn(
        column_id=column_id,
        layers=layers,
        connections=connections,
    )

    return column