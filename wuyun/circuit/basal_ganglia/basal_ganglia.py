"""
4-D: BasalGangliaCircuit — 基底节完整环路

三条通路:
1. 直接 (Go):   皮层 → D1-MSN ─┤ GPi → (去抑制) → 丘脑
2. 间接 (NoGo): 皮层 → D2-MSN ─┤ GPe ─┤ STN → GPi → (增强抑制)
3. 超直接 (Stop): 皮层 → STN → GPi → (全局刹车)

─┤ 表示 GABA 抑制, → 表示谷氨酸兴奋

动作通道拓扑:
- n_actions 个并行动作通道
- 每个通道有独立的 D1 和 D2 亚群
- GPi 按通道组织
- 通道间通过 FSI 侧向抑制竞争

依赖: striatum, gpi, indirect_pathway
"""

from typing import List, Optional
import numpy as np

from wuyun.spike.spike import Spike
from wuyun.circuit.basal_ganglia.striatum import Striatum
from wuyun.circuit.basal_ganglia.gpi import GPi
from wuyun.circuit.basal_ganglia.indirect_pathway import GPe, STN


class BasalGangliaCircuit:
    """基底节完整环路

    整合 Striatum + GPi + GPe + STN, 实现三条通路竞争。

    Attributes:
        n_actions: 动作通道数
        striatum: 纹状体 (D1/D2-MSN + FSI)
        gpi: 苍白球内侧 (输出核)
        gpe: 苍白球外侧 (间接通路中继)
        stn: 丘脑底核 (超直接/间接通路)
    """

    def __init__(
        self,
        n_actions: int = 4,
        n_d1_per_action: int = 5,
        n_d2_per_action: int = 5,
        n_gpi: int = 10,
        n_gpe: int = 10,
        n_stn: int = 8,
        n_fsi: int = 8,
        da_gain_d1: float = 10.0,
        da_gain_d2: float = 10.0,
        direct_gain: float = 15.0,
        indirect_gain: float = 10.0,
        hyperdirect_gain: float = 20.0,
        seed: int = None,
    ):
        self.n_actions = n_actions
        self.n_d1_per_action = n_d1_per_action
        self.n_d2_per_action = n_d2_per_action
        self.direct_gain = direct_gain
        self.indirect_gain = indirect_gain
        self.hyperdirect_gain = hyperdirect_gain
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        # 总 D1/D2 数量
        total_d1 = n_actions * n_d1_per_action
        total_d2 = n_actions * n_d2_per_action

        # === 创建子模块 ===
        self.striatum = Striatum(
            n_d1=total_d1,
            n_d2=total_d2,
            n_fsi=n_fsi,
            da_gain_d1=da_gain_d1,
            da_gain_d2=da_gain_d2,
            seed=seed,
        )

        self.gpi = GPi(
            n_neurons=n_gpi,
            tonic_drive=25.0,
            seed=seed + 1 if seed is not None else None,
        )

        self.gpe = GPe(
            n_neurons=n_gpe,
            tonic_drive=22.0,
            seed=seed + 2 if seed is not None else None,
        )

        self.stn = STN(
            n_neurons=n_stn,
            seed=seed + 3 if seed is not None else None,
        )

        # === 追踪 ===
        self._step_count = 0

    def step(
        self,
        t: int,
        cortical_input: np.ndarray,
        da_level: float = 0.0,
    ) -> None:
        """完整一步

        执行顺序 (考虑延迟差异):
        1. 皮层 → STN (超直接通路, 最快)
        2. 皮层 → Striatum (D1 + D2)
        3. DA 调制 → Striatum
        4. Striatum step → D1, D2 输出
        5. D1 → GPi (直接抑制)
        6. D2 → GPe (间接)
        7. GPe step
        8. GPe → STN (间接)
        9. STN step
        10. STN → GPi (兴奋)
        11. GPi step → 输出

        Args:
            t: 当前仿真时间步
            cortical_input: 皮层输入向量, shape=(n_actions,)
                每个元素 = 对应动作通道的输入强度
            da_level: 多巴胺水平 (>0=奖励, <0=惩罚, 0=基线)
        """
        self._step_count += 1

        # === 1. 超直接通路: 皮层 → STN (最快, 延迟 1ms) ===
        self.stn.inject_cortical_input(cortical_input, gain=self.hyperdirect_gain)

        # === 2. 皮层 → Striatum ===
        # 将 n_actions 维输入扩展到 D1/D2 维度
        # 每个动作通道的输入映射到对应的 D1/D2 亚群
        expanded_input = np.repeat(cortical_input, self.n_d1_per_action)
        self.striatum.inject_cortical_input(expanded_input)

        # === 3. DA 调制 ===
        if abs(da_level) > 0.001:
            self.striatum.apply_dopamine(da_level)

        # === 4. Striatum step ===
        self.striatum.step(t)

        # === 5. D1 → GPi (直接通路抑制, 按通道) ===
        d1_spikes = self.striatum.get_d1_spikes()
        d1_rates = self.striatum.get_d1_rates()
        # 按动作通道聚合 D1 发放率
        channel_d1_rates = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            start = a * self.n_d1_per_action
            end = start + self.n_d1_per_action
            channel_d1_rates[a] = np.mean(d1_rates[start:end])
        self.gpi.inject_channel_inhibition(
            channel_d1_rates, self.n_actions, gain=self.direct_gain,
        )

        # === 6. D2 → GPe (间接通路) ===
        d2_spikes = self.striatum.get_d2_spikes()
        d2_rates = self.striatum.get_d2_rates()
        self.gpe.inject_d2_inhibition(
            d2_spikes, d2_rates, gain=self.indirect_gain,
        )

        # === 7. GPe step ===
        self.gpe.step(t)

        # === 8. GPe → STN (间接通路) ===
        gpe_spikes = self.gpe.get_output_spikes()
        gpe_rates = self.gpe.get_output_rates()
        self.stn.inject_gpe_inhibition(gpe_spikes, gpe_rates)

        # === 9. STN step ===
        self.stn.step(t)

        # === 10. STN → GPi (兴奋) ===
        stn_spikes = self.stn.get_output_spikes()
        stn_rates = self.stn.get_output_rates()
        self.gpi.inject_indirect_excitation(
            stn_spikes, stn_rates, gain=self.indirect_gain,
        )

        # === 11. GPi step → 输出 ===
        self.gpi.step(t)

    def get_action_values(self) -> np.ndarray:
        """每个动作通道的"去抑制程度"

        GPi 高发放率 = 丘脑被抑制 = 动作被压制
        GPi 低发放率 = 丘脑被释放 = 动作可执行

        返回: shape=(n_actions,), 越高越倾向执行
        """
        gpi_rates = self.gpi.get_output_rates()

        # tonic 基线发放率 (~180Hz for tonic_drive=25)
        # 用固定基线而非动态计算, 避免全通道统一 GPi 导致 action_value=0
        tonic_baseline = 200.0  # 归一化基线

        # 按动作通道分组 GPi 输出
        n_per_channel = max(1, self.gpi.n_neurons // self.n_actions)
        action_values = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            start = a * n_per_channel
            end = min(start + n_per_channel, self.gpi.n_neurons)
            if start < self.gpi.n_neurons:
                channel_rate = np.mean(gpi_rates[start:end])
                # 去抑制程度 = 1 - (通道 GPi 发放率 / tonic 基线)
                action_values[a] = max(0.0, 1.0 - channel_rate / tonic_baseline)

        return action_values

    def select_action(self) -> int:
        """返回去抑制程度最高的动作 index (winner-take-all)

        Returns:
            动作通道 index (0 ~ n_actions-1)
        """
        return int(np.argmax(self.get_action_values()))

    def apply_reward(self, reward: float, da_level: float) -> None:
        """应用 DA 调制的三因子 STDP

        Args:
            reward: 奖励信号 (正=奖励, 负=惩罚)
            da_level: DA 水平
        """
        self.striatum.apply_da_modulated_plasticity(da_level)

    def get_state(self) -> dict:
        """返回各核团状态供调试"""
        return {
            "step_count": self._step_count,
            "action_values": self.get_action_values().tolist(),
            "selected_action": self.select_action(),
            "striatum": self.striatum.get_state(),
            "gpi": self.gpi.get_state(),
            "d1_rates": self.striatum.get_d1_rates().tolist(),
            "d2_rates": self.striatum.get_d2_rates().tolist(),
            "gpe_rates": self.gpe.get_output_rates().tolist(),
            "stn_rates": self.stn.get_output_rates().tolist(),
        }

    def reset(self) -> None:
        """重置所有子模块"""
        self.striatum.reset_spike_counts()
        self.gpi.reset_spike_counts()
        self.gpe.reset_spike_counts()
        self.stn.reset_spike_counts()
        self._step_count = 0

        # 重置所有神经元状态
        for n in self.striatum.d1_neurons:
            n.reset()
        for n in self.striatum.d2_neurons:
            n.reset()
        for n in self.striatum.fsi_neurons:
            n.reset()
        for n in self.gpi.neurons:
            n.reset()
        for n in self.gpe.neurons:
            n.reset()
        for n in self.stn.neurons:
            n.reset()