"""
Layer 3: MultiColumnNetwork — 多柱 + 丘脑路由网络

管理多个 CorticalColumn + ThalamicRouter + 柱间连接。

跨模块通信通过电流注入实现 (不创建跨模块突触):
  丘脑→皮层: TC spike count × gain → column L4 inject_feedforward_current
  皮层→丘脑: L6 firing rate × gain → TC apical inject_cortical_feedback_current
  皮层→TRN:  L6 firing rate × gain → TRN basal inject_trn_drive_current
  低柱→高柱: L2/3 regular count × gain → high col L4 inject_feedforward_current
  高柱→低柱: L6 firing rate × gain → low col apical inject_feedback_current
  侧向:      L2/3 firing rate × gain → col_b L2/3 inject_lateral_current

依赖:
- wuyun.circuit.cortical_column (CorticalColumn)
- wuyun.circuit.column_factory (create_sensory_column)
- wuyun.thalamus (ThalamicRouter, ThalamicNucleus, create_thalamic_nucleus)
- wuyun.synapse.plasticity.homeostatic (HomeostaticPlasticity)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from wuyun.circuit.cortical_column import CorticalColumn
from wuyun.circuit.column_factory import create_sensory_column
from wuyun.thalamus.thalamic_nucleus import create_thalamic_nucleus
from wuyun.thalamus.thalamic_router import ThalamicRouter


@dataclass
class GainParams:
    """跨模块连接增益参数

    默认值已调优，确保在 n_per_layer=20, ff_connection_strength=1.5 下:
    - L6 能被激活 (闭环反馈)
    - TC 发放率不无限增长 (有负反馈)
    - 高层柱能被低层误差驱动
    """
    tc_to_column_gain: float = 30.0       # 丘脑→皮层: TC 每个 spike 对 L4 的驱动强度
    column_to_tc_gain: float = 25.0       # 皮层→丘脑反馈 (L6→TC apical, 预测信号)
    column_to_trn_gain: float = 15.0      # 皮层→TRN (L6→TRN, 负反馈关键!)
    error_forward_gain: float = 20.0      # 误差上传 (低柱 L2/3 regular → 高柱 L4)
    prediction_backward_gain: float = 10.0  # 预测下传 (高柱 L6 → 低柱 apical)
    lateral_gain: float = 5.0             # 侧向上下文


class MultiColumnNetwork:
    """多柱 + 丘脑路由网络

    管理多个 CorticalColumn + ThalamicRouter + 柱间连接。

    Attributes:
        columns: {column_id: CorticalColumn}
        router: ThalamicRouter
        hierarchy: [(low_col_id, high_col_id), ...] 层级关系
        lateral_pairs: [(col_a_id, col_b_id), ...] 侧向连接对
        gains: 增益参数
    """

    def __init__(
        self,
        router: ThalamicRouter = None,
        gains: GainParams = None,
    ):
        self.columns: Dict[int, CorticalColumn] = {}
        self.router = router or ThalamicRouter()
        self.hierarchy: List[Tuple[int, int]] = []
        self.lateral_pairs: List[Tuple[int, int]] = []
        self.gains = gains or GainParams()

        # 步进计数器
        self._step_count: int = 0

    # =========================================================================
    # 网络构建
    # =========================================================================

    def add_column(self, column: CorticalColumn) -> None:
        """添加皮层柱

        Args:
            column: CorticalColumn 实例
        """
        self.columns[column.column_id] = column

    def set_hierarchy(self, pairs: List[Tuple[int, int]]) -> None:
        """设置层级关系

        Args:
            pairs: [(low_col_id, high_col_id), ...]
        """
        self.hierarchy = list(pairs)

    def set_lateral(self, pairs: List[Tuple[int, int]]) -> None:
        """设置侧向连接对

        Args:
            pairs: [(col_a_id, col_b_id), ...]
        """
        self.lateral_pairs = list(pairs)

    # =========================================================================
    # 核心仿真步进
    # =========================================================================

    def step(self, current_time: int, dt: float = 1.0) -> None:
        """推进一个时间步 — 核心仿真循环

        执行顺序:
        1. router.step() — 丘脑核团更新
        2. router.apply_trn_competition() — TRN 竞争
        3. 路由 TC 输出 → 目标柱 inject_feedforward_current
        4. 所有柱 step() — 皮层处理
        5. 收集柱输出 (L6 prediction, L2/3 error)
        6. 路由: L6 → 丘脑反馈, L2/3 error → 高层柱, L6 prediction → 低层柱
        7. 侧向连接

        Args:
            current_time: 当前仿真时间步
            dt: 时间步长
        """
        g = self.gains
        self._step_count += 1

        # === Step 1: 丘脑核团更新 ===
        self.router.step(current_time, dt)

        # === Step 2: TRN 竞争 ===
        self.router.apply_trn_competition()

        # === Step 3: 路由 TC 输出 → 目标柱 ===
        routed = self.router.get_routed_outputs()
        for col_id, spikes in routed.items():
            if col_id in self.columns:
                tc_count = len(spikes)
                if tc_count > 0:
                    self.columns[col_id].inject_feedforward_current(
                        tc_count * g.tc_to_column_gain
                    )

        # === Step 4: 所有柱 step ===
        for column in self.columns.values():
            column.step(current_time, dt)

        # === Step 5+6: 收集柱输出并路由 ===
        # 收集所有柱的输出摘要
        summaries: Dict[int, dict] = {}
        for col_id, column in self.columns.items():
            summaries[col_id] = column.get_output_summary()

        # 皮层→丘脑反馈: L6 firing rate → TC apical + TRN drive
        for nid, nucleus in self.router.nuclei.items():
            targets = self.router.routing_table.get(nid, [])
            for col_id in targets:
                if col_id in summaries:
                    l6_rate = summaries[col_id].get('l6_firing_rate', 0.0)
                    if l6_rate > 0:
                        nucleus.inject_cortical_feedback_current(
                            l6_rate * g.column_to_tc_gain
                        )
                        nucleus.inject_trn_drive_current(
                            l6_rate * g.column_to_trn_gain
                        )

        # 层级连接: 低柱 L2/3 error → 高柱 L4, 高柱 L6 → 低柱 apical
        for low_id, high_id in self.hierarchy:
            if low_id in summaries and high_id in self.columns:
                # 低柱 L2/3 regular (预测误差) → 高柱 L4
                error_count = len(summaries[low_id].get('prediction_error', []))
                if error_count > 0:
                    self.columns[high_id].inject_feedforward_current(
                        error_count * g.error_forward_gain
                    )

            if high_id in summaries and low_id in self.columns:
                # 高柱 L6 (预测) → 低柱 apical
                pred_rate = summaries[high_id].get('l6_firing_rate', 0.0)
                if pred_rate > 0:
                    self.columns[low_id].inject_feedback_current(
                        pred_rate * g.prediction_backward_gain
                    )

        # 侧向连接
        for col_a_id, col_b_id in self.lateral_pairs:
            if col_a_id in summaries and col_b_id in self.columns:
                rate_a = summaries[col_a_id].get('l23_firing_rate', 0.0)
                if rate_a > 0:
                    self.columns[col_b_id].inject_lateral_current(
                        rate_a * g.lateral_gain
                    )
            if col_b_id in summaries and col_a_id in self.columns:
                rate_b = summaries[col_b_id].get('l23_firing_rate', 0.0)
                if rate_b > 0:
                    self.columns[col_a_id].inject_lateral_current(
                        rate_b * g.lateral_gain
                    )

    # =========================================================================
    # 状态查询
    # =========================================================================

    def get_network_state(self) -> dict:
        """获取网络状态: 所有柱和核团的发放率、burst 比率等

        Returns:
            dict with keys: columns, nuclei, step_count
        """
        col_states = {}
        for col_id, column in self.columns.items():
            summary = column.get_output_summary()
            rates = column.get_layer_firing_rates()
            burst_ratios = column.get_layer_burst_ratios()
            col_states[col_id] = {
                'firing_rates': rates,
                'burst_ratios': burst_ratios,
                'l23_firing_rate': summary.get('l23_firing_rate', 0.0),
                'l5_firing_rate': summary.get('l5_firing_rate', 0.0),
                'l6_firing_rate': summary.get('l6_firing_rate', 0.0),
                'prediction_error_count': len(summary.get('prediction_error', [])),
                'match_signal_count': len(summary.get('match_signal', [])),
            }

        nuc_states = {}
        for nid, nucleus in self.router.nuclei.items():
            nuc_states[nid] = {
                'tc_firing_rate': nucleus.get_tc_firing_rate(),
                'trn_firing_rate': nucleus.get_trn_firing_rate(),
                'tc_burst_ratio': nucleus.get_tc_burst_ratio(),
            }

        return {
            'columns': col_states,
            'nuclei': nuc_states,
            'step_count': self._step_count,
        }

    # =========================================================================
    # 生命周期
    # =========================================================================

    def reset(self) -> None:
        """重置整个网络"""
        for column in self.columns.values():
            column.reset()
        self.router.reset()
        self._step_count = 0

    def __repr__(self) -> str:
        return (
            f"MultiColumnNetwork(columns={len(self.columns)}, "
            f"nuclei={len(self.router.nuclei)}, "
            f"hierarchy={len(self.hierarchy)}, "
            f"lateral={len(self.lateral_pairs)}, "
            f"steps={self._step_count})"
        )


# =============================================================================
# 工厂函数
# =============================================================================

def create_hierarchical_network(
    n_columns: int = 2,
    n_per_layer: int = 20,
    n_tc: int = 10,
    n_trn: int = 5,
    seed: int = None,
    gains: GainParams = None,
    ff_connection_strength: float = 1.5,
) -> MultiColumnNetwork:
    """创建层级多柱网络

    创建 n_columns 个 CorticalColumn + n_columns 个 ThalamicNucleus，
    每柱配一个核团，线性层级连接。

    Args:
        n_columns: 柱数量
        n_per_layer: 每层兴奋性神经元数量
        n_tc: 每个核团 TC 数量
        n_trn: 每个核团 TRN 数量
        seed: 随机种子
        gains: 增益参数
        ff_connection_strength: 柱内前馈连接强度 (默认 1.5, 确保深层 L5/L6 能被激活)

    Returns:
        配置好的 MultiColumnNetwork
    """
    import numpy as np
    rng_base = seed if seed is not None else 42

    router = ThalamicRouter()
    network = MultiColumnNetwork(router=router, gains=gains)

    # 创建柱和核团
    for i in range(n_columns):
        col_seed = rng_base + i * 1000
        column = create_sensory_column(
            column_id=i,
            n_per_layer=n_per_layer,
            seed=col_seed,
            ff_connection_strength=ff_connection_strength,
        )
        network.add_column(column)

        nuc_seed = rng_base + i * 1000 + 500
        nucleus = create_thalamic_nucleus(
            nucleus_id=i,
            n_tc=n_tc,
            n_trn=n_trn,
            seed=nuc_seed,
        )
        router.add_nucleus(nucleus)
        router.add_route(i, i)  # nucleus[i] → column[i]

    # 线性层级: column_0 (低层) → column_1 (高层) → ...
    hierarchy = [(i, i + 1) for i in range(n_columns - 1)]
    network.set_hierarchy(hierarchy)

    return network