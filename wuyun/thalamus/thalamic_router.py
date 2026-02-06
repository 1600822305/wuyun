"""
丘脑路由器 (ThalamicRouter) — 多核团路由 + TRN 竞争抑制

管理多个 ThalamicNucleus，实现:
1. 多核团统一步进
2. 跨核团 TRN 竞争抑制 (选择性注意力)
3. TC 输出路由到目标皮层柱

TRN 竞争机制:
  收集所有核团的 TRN 发放率 → 发放率最高的核团"获胜" →
  向其他核团的 TRN 注入额外兴奋电流 → 使其 TRN 更活跃 →
  抑制那些核团的 TC → winner-take-all 注意力

依赖:
- wuyun.thalamus.thalamic_nucleus (ThalamicNucleus)
- wuyun.spike (Spike)
"""

from typing import Dict, List, Tuple
from wuyun.spike.spike import Spike
from wuyun.thalamus.thalamic_nucleus import ThalamicNucleus


class ThalamicRouter:
    """多核团路由器

    管理多个 ThalamicNucleus，实现跨核团 TRN 竞争抑制。

    Attributes:
        nuclei: {nucleus_id: ThalamicNucleus}
        routing_table: {nucleus_id: [target_column_ids]}
    """

    def __init__(self, trn_competition_gain: float = 30.0):
        """
        Args:
            trn_competition_gain: TRN 竞争抑制的增益系数
                获胜核团向其他核团 TRN 注入的电流 = gain * winner_trn_rate
        """
        self.nuclei: Dict[int, ThalamicNucleus] = {}
        self.routing_table: Dict[int, List[int]] = {}
        self._trn_competition_gain = trn_competition_gain

    # =========================================================================
    # 核团管理
    # =========================================================================

    def add_nucleus(self, nucleus: ThalamicNucleus) -> None:
        """注册核团

        Args:
            nucleus: 丘脑核团实例
        """
        self.nuclei[nucleus.nucleus_id] = nucleus

    def add_route(self, nucleus_id: int, target_column_id: int) -> None:
        """添加路由规则: 核团 → 目标柱

        Args:
            nucleus_id: 核团 ID
            target_column_id: 目标皮层柱 ID
        """
        if nucleus_id not in self.routing_table:
            self.routing_table[nucleus_id] = []
        if target_column_id not in self.routing_table[nucleus_id]:
            self.routing_table[nucleus_id].append(target_column_id)

    # =========================================================================
    # 仿真步进
    # =========================================================================

    def step(self, current_time: int, dt: float = 1.0) -> None:
        """步进所有核团

        Args:
            current_time: 当前仿真时间步
            dt: 时间步长
        """
        for nucleus in self.nuclei.values():
            nucleus.step(current_time, dt)

    def apply_trn_competition(self) -> None:
        """跨核团 TRN 竞争抑制 — winner-take-all 注意力机制

        1. 收集所有核团的 TRN 发放率
        2. 找到发放率最高的核团 (获胜者)
        3. 向其他核团的 TRN 注入额外兴奋电流
           → 使其 TRN 更活跃 → 抑制那些核团的 TC
        """
        if len(self.nuclei) < 2:
            return

        # 收集所有核团的 TRN 发放率
        trn_rates: Dict[int, float] = {}
        for nid, nucleus in self.nuclei.items():
            trn_rates[nid] = nucleus.get_trn_firing_rate()

        # 找到获胜者 (TRN 发放率最高 = TC 最活跃的核团)
        # 注意: 我们用 TC 发放率来判断哪个核团最活跃
        tc_rates: Dict[int, float] = {}
        for nid, nucleus in self.nuclei.items():
            tc_rates[nid] = nucleus.get_tc_firing_rate()

        if not tc_rates:
            return

        winner_id = max(tc_rates, key=tc_rates.get)
        winner_tc_rate = tc_rates[winner_id]

        if winner_tc_rate <= 0:
            return

        # 向非获胜核团的 TRN 注入额外兴奋电流
        for nid, nucleus in self.nuclei.items():
            if nid != winner_id:
                # 注入电流与获胜者 TC 活跃度成正比
                inhibition_current = self._trn_competition_gain * (
                    winner_tc_rate / 100.0  # 归一化到 [0, ~1]
                )
                nucleus.inject_trn_drive_current(inhibition_current)

    def get_routed_outputs(self) -> Dict[int, List[Spike]]:
        """获取路由后的输出: {target_column_id: spikes}

        将每个核团的 TC 输出按路由表分发到目标柱。

        Returns:
            {target_column_id: [Spike, ...]}
        """
        result: Dict[int, List[Spike]] = {}
        for nid, nucleus in self.nuclei.items():
            tc_spikes = nucleus.get_tc_output()
            targets = self.routing_table.get(nid, [])
            for col_id in targets:
                if col_id not in result:
                    result[col_id] = []
                result[col_id].extend(tc_spikes)
        return result

    # =========================================================================
    # 状态查询
    # =========================================================================

    def get_all_tc_rates(self) -> Dict[int, float]:
        """获取所有核团的 TC 发放率"""
        return {nid: n.get_tc_firing_rate() for nid, n in self.nuclei.items()}

    def get_all_trn_rates(self) -> Dict[int, float]:
        """获取所有核团的 TRN 发放率"""
        return {nid: n.get_trn_firing_rate() for nid, n in self.nuclei.items()}

    # =========================================================================
    # 生命周期
    # =========================================================================

    def reset(self) -> None:
        """重置所有核团"""
        for nucleus in self.nuclei.values():
            nucleus.reset()

    def __repr__(self) -> str:
        return (
            f"ThalamicRouter(nuclei={len(self.nuclei)}, "
            f"routes={sum(len(v) for v in self.routing_table.values())})"
        )