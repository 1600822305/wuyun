"""
SpikeBus 验证测试

测试脉冲总线的核心功能:
1. 突触注册和索引
2. 脉冲提交和分发
3. 端到端: A 发放 → SpikeBus → B 收到 → B 发放
4. 扇出: 一个源 → 多个目标
5. 无连接 → 无分发
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wuyun.spike.spike import Spike
from wuyun.spike.signal_types import SpikeType, CompartmentType, SynapseType
from wuyun.spike.spike_bus import SpikeBus
from wuyun.synapse.synapse_base import SynapseBase
from wuyun.neuron.neuron_base import NeuronBase, NeuronParams, L23_PYRAMIDAL_PARAMS


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_case_1_bus_mechanics():
    """Case 1: SpikeBus 基本机制 — 注册/提交/分发"""
    print_header("Case 1: SpikeBus 基本机制")

    bus = SpikeBus()

    # 创建突触: neuron 0 → neuron 1 (BASAL)
    syn_01 = SynapseBase(pre_id=0, post_id=1, weight=1.0,
                         target_compartment=CompartmentType.BASAL)
    # 创建突触: neuron 0 → neuron 2 (APICAL)
    syn_02 = SynapseBase(pre_id=0, post_id=2, weight=0.8,
                         target_compartment=CompartmentType.APICAL)

    # 注册
    bus.register_synapse(syn_01)
    bus.register_synapse(syn_02)

    print(f"  注册后: {bus}")
    assert bus.synapse_count == 2, f"期望 2 个突触, 得到 {bus.synapse_count}"
    assert bus.get_fanout(0) == 2, f"neuron 0 扇出应为 2, 得到 {bus.get_fanout(0)}"
    assert bus.get_fanout(1) == 0, f"neuron 1 扇出应为 0, 得到 {bus.get_fanout(1)}"

    # 提交脉冲
    spike = Spike(source_id=0, timestamp=10, spike_type=SpikeType.REGULAR)
    bus.emit(spike)
    assert bus.pending_count == 1, f"pending 应为 1, 得到 {bus.pending_count}"

    # NONE 脉冲应被忽略
    none_spike = Spike(source_id=0, timestamp=10, spike_type=SpikeType.NONE)
    bus.emit(none_spike)
    assert bus.pending_count == 1, "NONE 脉冲不应被接受"

    # 分发
    delivered = bus.step(10)
    assert delivered == 2, f"应分发到 2 个突触, 得到 {delivered}"
    assert bus.pending_count == 0, "分发后 pending 应为 0"
    assert bus.total_emitted == 1, f"累计提交应为 1, 得到 {bus.total_emitted}"
    assert bus.total_delivered == 2, f"累计分发应为 2, 得到 {bus.total_delivered}"

    # 检查突触延迟缓冲区已收到脉冲
    assert len(syn_01._delay_buffer) == 1, "syn_01 应有 1 个待处理脉冲"
    assert len(syn_02._delay_buffer) == 1, "syn_02 应有 1 个待处理脉冲"

    print(f"  分发后: {bus}")
    print(f"  syn_01 延迟缓冲: {syn_01._delay_buffer}")
    print(f"  syn_02 延迟缓冲: {syn_02._delay_buffer}")
    print(f"  ✅ PASS: SpikeBus 基本机制正确")
    return True


def test_case_2_a_to_b_propagation():
    """Case 2: 端到端 — A 发放 → SpikeBus → B 发放"""
    print_header("Case 2: A → SpikeBus → B 传播")

    bus = SpikeBus()

    # 创建两个 L2/3 锥体神经元
    neuron_a = NeuronBase(neuron_id=0, params=L23_PYRAMIDAL_PARAMS)
    neuron_b = NeuronBase(neuron_id=1, params=L23_PYRAMIDAL_PARAMS)

    # 创建 10 个突触: A → B (BASAL), 模拟汇聚输入
    # 生物学事实: 单个 AMPA 突触 (g_max=1.0, τ=2ms) 只能产生 ~7mV EPSP,
    # 不足以驱动突触后神经元发放 (需要 ~20mV)。
    # 真实大脑中, 一个神经元接收 ~1000-10000 个突触的汇聚输入。
    n_synapses = 10
    synapses_ab = []
    for i in range(n_synapses):
        syn = SynapseBase(
            pre_id=0, post_id=1,
            weight=1.0,
            delay=1,
            synapse_type=SynapseType.AMPA,
            target_compartment=CompartmentType.BASAL,
        )
        neuron_b.add_synapse(syn)
        bus.register_synapse(syn)
        synapses_ab.append(syn)

    print(f"  neuron_a: {neuron_a}")
    print(f"  neuron_b: {neuron_b}")
    print(f"  突触数量: {n_synapses} (A→B BASAL, 模拟汇聚输入)")
    print(f"  bus: {bus}")
    print()

    # 仿真参数
    duration = 200  # ms
    basal_inject = 30.0  # 给 A 的直接注入电流 (足够触发)

    a_spikes = []  # A 的发放时间和类型
    b_spikes = []  # B 的发放时间和类型

    for t in range(duration):
        # === Phase 1: 给 A 注入电流 ===
        neuron_a.inject_basal_current(basal_inject)

        # === Phase 2: 所有神经元 step ===
        spike_a = neuron_a.step(t)
        spike_b = neuron_b.step(t)

        # === Phase 3: A 发放 → emit 到 bus ===
        if spike_a.is_active:
            bus.emit(Spike(neuron_a.id, t, spike_a))
            a_spikes.append((t, spike_a))

        if spike_b.is_active:
            b_spikes.append((t, spike_b))

        # === Phase 4: bus 分发 ===
        bus.step(t)

    # 输出结果
    print(f"  A 发放次数: {len(a_spikes)}")
    if a_spikes:
        print(f"    首次: t={a_spikes[0][0]}ms, type={a_spikes[0][1].name}")
    print(f"  B 发放次数: {len(b_spikes)}")
    if b_spikes:
        print(f"    首次: t={b_spikes[0][0]}ms, type={b_spikes[0][1].name}")
    print(f"  bus 统计: emitted={bus.total_emitted}, delivered={bus.total_delivered}")

    # 断言
    assert len(a_spikes) > 0, "A 应该有发放"
    assert len(b_spikes) > 0, "B 应该通过 SpikeBus 收到输入后发放"

    # B 的首次发放应晚于 A (因为需要 A 发放 → bus → 突触延迟 → B 积累)
    if a_spikes and b_spikes:
        assert b_spikes[0][0] > a_spikes[0][0], \
            f"B 首次发放 ({b_spikes[0][0]}ms) 应晚于 A ({a_spikes[0][0]}ms)"
        print(f"  B 首次发放比 A 晚 {b_spikes[0][0] - a_spikes[0][0]}ms (含突触延迟+积累)")

    # B 应该有 regular spike (只有 basal 输入, 无 apical)
    b_regulars = [s for _, s in b_spikes if s == SpikeType.REGULAR]
    assert len(b_regulars) > 0, "B 应该有 REGULAR spike (只有前馈输入)"
    print(f"  B regular spikes: {len(b_regulars)}")

    print(f"  ✅ PASS: A→SpikeBus→B 脉冲传播成功")
    return True


def test_case_3_fanout():
    """Case 3: 扇出 — 一个源 → 多个目标"""
    print_header("Case 3: 扇出 (1→N)")

    bus = SpikeBus()

    # 3 个突触: neuron 0 → {1, 2, 3}
    synapses = []
    for post_id in [1, 2, 3]:
        syn = SynapseBase(pre_id=0, post_id=post_id, weight=0.5,
                          target_compartment=CompartmentType.BASAL)
        synapses.append(syn)
        bus.register_synapse(syn)

    assert bus.get_fanout(0) == 3

    # 提交一个脉冲
    spike = Spike(source_id=0, timestamp=5, spike_type=SpikeType.BURST_START)
    bus.emit(spike)
    delivered = bus.step(5)

    assert delivered == 3, f"应分发到 3 个突触, 得到 {delivered}"

    # 每个突触都应收到
    for i, syn in enumerate(synapses):
        assert len(syn._delay_buffer) == 1, \
            f"syn[{i}] (→post {syn.post_id}) 应有 1 个待处理脉冲"
        arrival_time, spike_type = syn._delay_buffer[0]
        assert spike_type == SpikeType.BURST_START, \
            f"脉冲类型应为 BURST_START, 得到 {spike_type.name}"

    print(f"  1→3 扇出: 所有目标突触都收到 BURST_START")
    print(f"  ✅ PASS: 扇出分发正确")
    return True


def test_case_4_no_connection():
    """Case 4: 无连接 → 无分发"""
    print_header("Case 4: 无连接 → 无分发")

    bus = SpikeBus()

    # 注册 neuron 0 → neuron 1 的突触
    syn = SynapseBase(pre_id=0, post_id=1, weight=0.5)
    bus.register_synapse(syn)

    # 但发放的是 neuron 99 (无下游突触)
    spike = Spike(source_id=99, timestamp=10, spike_type=SpikeType.REGULAR)
    bus.emit(spike)
    delivered = bus.step(10)

    assert delivered == 0, f"无连接应分发 0, 得到 {delivered}"
    assert len(syn._delay_buffer) == 0, "不相关突触不应收到脉冲"

    print(f"  neuron 99 无下游突触, 脉冲被安全丢弃")
    print(f"  ✅ PASS: 无连接时不分发")
    return True


def test_case_5_unregister():
    """Case 5: 注销突触"""
    print_header("Case 5: 注销突触 (结构可塑性预留)")

    bus = SpikeBus()

    syn_a = SynapseBase(pre_id=0, post_id=1, weight=0.5)
    syn_b = SynapseBase(pre_id=0, post_id=2, weight=0.5)
    bus.register_synapse(syn_a)
    bus.register_synapse(syn_b)

    assert bus.synapse_count == 2
    assert bus.get_fanout(0) == 2

    # 注销 syn_a
    bus.unregister_synapse(syn_a)
    assert bus.synapse_count == 1, f"注销后应剩 1, 得到 {bus.synapse_count}"
    assert bus.get_fanout(0) == 1

    # 发放应只到达 syn_b
    spike = Spike(source_id=0, timestamp=10, spike_type=SpikeType.REGULAR)
    bus.emit(spike)
    delivered = bus.step(10)
    assert delivered == 1
    assert len(syn_a._delay_buffer) == 0, "注销的突触不应收到脉冲"
    assert len(syn_b._delay_buffer) == 1, "保留的突触应收到脉冲"

    print(f"  注销 syn_a 后, 脉冲只到达 syn_b")
    print(f"  ✅ PASS: 注销突触功能正确")
    return True


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  悟韵 (WuYun) SpikeBus 脉冲总线验证测试                ║")
    print("║  测试神经元间通信调度机制                               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    results = {}
    tests = [
        ("Case 1: 基本机制", test_case_1_bus_mechanics),
        ("Case 2: A→B 传播", test_case_2_a_to_b_propagation),
        ("Case 3: 扇出 1→N", test_case_3_fanout),
        ("Case 4: 无连接", test_case_4_no_connection),
        ("Case 5: 注销突触", test_case_5_unregister),
    ]

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = "PASS" if passed else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()

    # 总结
    print_header("总结")
    all_pass = True
    for name, result in results.items():
        icon = "✅" if result == "PASS" else "❌"
        if result != "PASS":
            all_pass = False
        print(f"  {icon} {result}: {name}")

    print()
    if all_pass:
        print("🎉 所有测试通过! SpikeBus 脉冲总线功能验证完毕。")
        print("   神经元间通信调度机制工作正常。")
    else:
        print("❌ 存在失败的测试，请检查。")
        sys.exit(1)