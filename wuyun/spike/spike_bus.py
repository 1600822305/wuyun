"""
Layer 0: SpikeBus — 脉冲总线

SpikeBus 是神经元间通信的中枢调度器:
1. 神经元发放时，将 Spike 事件提交到总线 (emit)
2. 总线根据突触连接关系，将脉冲分发给所有突触后神经元的对应突触 (deliver)
3. 支持按时间步批量处理

设计原则:
- 只做"脉冲转发"，不做任何基于脑区/类型的路由判断 (anti-cheating)
- 用 defaultdict(list) 做 pre_id → 突触列表索引，O(1) 查找
- 支持动态注册/注销突触 (为未来结构可塑性预留)

依赖关系:
- wuyun.spike.spike (Spike 数据结构)
- wuyun.spike.signal_types (SpikeType 枚举)
- wuyun.synapse.synapse_base (SynapseBase.receive_spike 接口)

典型使用模式:
```python
bus = SpikeBus()
for syn in all_synapses:
    bus.register_synapse(syn)

for t in range(duration):
    # 1. 所有神经元 step → 发放 → emit 到总线
    for neuron in neurons:
        spike_type = neuron.step(t)
        if spike_type.is_active:
            bus.emit(Spike(neuron.id, t, spike_type))

    # 2. 总线分发脉冲到突触延迟缓冲区
    bus.step(t)
```
"""

from typing import Dict, List, Protocol, runtime_checkable
from collections import defaultdict

from wuyun.spike.spike import Spike
from wuyun.spike.signal_types import SpikeType


# =============================================================================
# 脉冲接收者协议 (依赖反转: spike/ 不直接导入 synapse/)
# =============================================================================

@runtime_checkable
class SpikeReceiver(Protocol):
    """任何能接收脉冲的对象 (如 SynapseBase)

    使用 Protocol 实现依赖反转:
    - spike/ (Layer 0) 定义接口
    - synapse/ (Layer 1) 的 SynapseBase 天然满足此接口 (结构化子类型)
    - spike/ 不需要 import synapse/
    """
    pre_id: int

    def receive_spike(self, timestamp: int, spike_type: SpikeType) -> None: ...


class SpikeBus:
    """脉冲总线 — 神经元间通信调度器

    核心数据结构:
        _pre_to_synapses: dict[int, list[SpikeReceiver]]
            pre_id → 以该神经元为突触前的所有突触列表
            避免每次遍历全部突触，实现 O(fanout) 分发

        _pending: list[Spike]
            当前时间步累积的待分发脉冲

    不变量:
        - 总线只做转发，不修改 Spike 内容
        - 总线不做任何 IF-ELSE 路由判断
        - 分发顺序 = emit 顺序 (FIFO)
    """

    def __init__(self):
        # pre_id → 以该神经元为突触前的所有突触
        self._pre_to_synapses: Dict[int, List[SpikeReceiver]] = defaultdict(list)

        # 当前时间步待分发的脉冲缓冲
        self._pending: List[Spike] = []

        # 统计计数器
        self._total_emitted: int = 0
        self._total_delivered: int = 0

    # =========================================================================
    # 突触注册
    # =========================================================================

    def register_synapse(self, synapse: SpikeReceiver) -> None:
        """注册一个突触连接

        内部建立 pre_id → [synapse] 的索引，
        使得 deliver() 时可以 O(1) 找到该源神经元的所有下游突触。

        Args:
            synapse: 要注册的突触 (必须有 pre_id 属性)
        """
        self._pre_to_synapses[synapse.pre_id].append(synapse)

    def unregister_synapse(self, synapse: SpikeReceiver) -> None:
        """注销一个突触连接 (为结构可塑性预留)

        Args:
            synapse: 要注销的突触
        """
        synapses = self._pre_to_synapses.get(synapse.pre_id)
        if synapses and synapse in synapses:
            synapses.remove(synapse)
            # 清理空列表
            if not synapses:
                del self._pre_to_synapses[synapse.pre_id]

    def register_synapses(self, synapses: List[SpikeReceiver]) -> None:
        """批量注册突触连接

        Args:
            synapses: 突触列表
        """
        for syn in synapses:
            self.register_synapse(syn)

    # =========================================================================
    # 脉冲提交与分发
    # =========================================================================

    def emit(self, spike: Spike) -> None:
        """神经元发放后调用：将脉冲提交到总线

        只有 is_active 的脉冲才会被接受 (SpikeType.NONE 被忽略)。
        脉冲暂存在缓冲区，等待 deliver() 统一分发。

        Args:
            spike: 脉冲事件 (source_id, timestamp, spike_type)
        """
        if spike.spike_type.is_active:
            self._pending.append(spike)
            self._total_emitted += 1

    def deliver(self) -> int:
        """将所有待分发的脉冲送达对应突触

        对每个 pending spike:
        1. 通过 source_id 在索引中找到所有下游突触
        2. 调用 synapse.receive_spike() 将脉冲放入突触延迟缓冲区

        分发后清空 pending 缓冲。

        Returns:
            本次分发的脉冲-突触对数量 (一个脉冲发送给 N 个突触 = N)
        """
        count = 0
        for spike in self._pending:
            targets = self._pre_to_synapses.get(spike.source_id)
            if targets:
                for syn in targets:
                    syn.receive_spike(spike.timestamp, spike.spike_type)
                    count += 1

        self._total_delivered += count
        self._pending.clear()
        return count

    def step(self, current_time: int) -> int:
        """每个时间步调用：分发所有待处理脉冲并清空缓冲

        Args:
            current_time: 当前仿真时间步 (保留参数，未来可扩展时间相关逻辑)

        Returns:
            本次分发的脉冲-突触对数量
        """
        return self.deliver()

    def clear(self) -> None:
        """清空当前时间步的脉冲缓冲 (不清除注册的突触)"""
        self._pending.clear()

    # =========================================================================
    # 状态查询
    # =========================================================================

    @property
    def pending_count(self) -> int:
        """当前待分发的脉冲数量"""
        return len(self._pending)

    @property
    def synapse_count(self) -> int:
        """已注册的突触总数"""
        return sum(len(syns) for syns in self._pre_to_synapses.values())

    @property
    def total_emitted(self) -> int:
        """累计提交的脉冲总数"""
        return self._total_emitted

    @property
    def total_delivered(self) -> int:
        """累计分发的脉冲-突触对总数"""
        return self._total_delivered

    def get_fanout(self, pre_id: int) -> int:
        """查询指定神经元的扇出数 (下游突触数量)

        Args:
            pre_id: 突触前神经元 ID

        Returns:
            该神经元的下游突触数量
        """
        return len(self._pre_to_synapses.get(pre_id, []))

    def reset_stats(self) -> None:
        """重置统计计数器 (保留突触注册和缓冲)"""
        self._total_emitted = 0
        self._total_delivered = 0

    def reset(self) -> None:
        """完全重置: 清空缓冲 + 统计 (保留突触注册)"""
        self._pending.clear()
        self.reset_stats()

    def __repr__(self) -> str:
        return (
            f"SpikeBus(synapses={self.synapse_count}, "
            f"pending={self.pending_count}, "
            f"emitted={self.total_emitted}, "
            f"delivered={self.total_delivered})"
        )