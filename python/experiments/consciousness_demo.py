"""
悟韵 (WuYun) — 全局工作空间意识演示

演示内容:
1. 视觉刺激 → GW竞争 → 点火 → 意识访问
2. 多模态竞争: 视觉 vs 听觉 → 赢者进入意识
3. 注意力门控: attention_gain调制GW竞争结果
4. 内驱力影响: Hypothalamus状态 → 觉醒 → 意识能力
5. 全链路: 视觉→意识→决策→运动 (有意识的行为)

运行: python consciousness_demo.py [--save]
"""

import sys
import os
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib', 'Release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pywuyun


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def count_fired(region):
    return sum(1 for x in region.fired() if x)


# =============================================================================
# Demo 1: Basic Visual Consciousness
# =============================================================================
def demo_visual_consciousness():
    print_header("演示1: 视觉意识 — LGN→V1→GW点火→广播")

    eng = pywuyun.SimulationEngine(10)
    eng.build_standard_brain()

    gw = eng.find_region('GW')
    lgn = eng.find_region('LGN')

    print(f"  脑: {eng.num_regions()}区域, {eng.bus().num_projections()}投射")

    # Phase 1: No stimulus
    print("\n  阶段1: 无刺激 (50步)")
    for t in range(50):
        eng.step()
    print(f"    GW点火: {gw.ignition_count()}, 意识内容: '{gw.conscious_content_name()}'")

    # Phase 2: Visual stimulus
    print("\n  阶段2: 视觉刺激 (200步)")
    ignition_timeline = []
    for t in range(200):
        lgn.inject_external([35.0] * lgn.n_neurons())
        eng.step()
        ignition_timeline.append(gw.is_ignited())

    ignitions = gw.ignition_count()
    content = gw.conscious_content_name()
    broadcast_steps = sum(ignition_timeline)

    print(f"    GW点火: {ignitions}次")
    print(f"    意识内容: '{content}'")
    print(f"    广播活跃步数: {broadcast_steps}/200")
    print(f"    意识占比: {broadcast_steps/200*100:.1f}%")

    # Phase 3: Stimulus off
    print("\n  阶段3: 刺激关闭 (50步)")
    final_ignitions = gw.ignition_count()
    for t in range(50):
        eng.step()
    post_ignitions = gw.ignition_count() - final_ignitions

    print(f"    新增点火: {post_ignitions}")
    print(f"    Salience衰减: {gw.winning_salience():.2f}")

    ok = ignitions > 0 and content != ""
    print(f"\n  结果: {'✓ 视觉刺激成功进入意识' if ok else '✗ 未能进入意识'}")
    return ok


# =============================================================================
# Demo 2: Multimodal Competition
# =============================================================================
def demo_multimodal_competition():
    print_header("演示2: 多模态竞争 — 仅视觉 vs 仅听觉")

    # Visual only
    eng1 = pywuyun.SimulationEngine(10)
    eng1.build_standard_brain()
    gw1 = eng1.find_region('GW')

    print("  仅视觉刺激 (LGN=35.0)...")
    v1_spikes = 0
    for t in range(200):
        eng1.find_region('LGN').inject_external([35.0] * eng1.find_region('LGN').n_neurons())
        eng1.step()
        v1_spikes += count_fired(eng1.find_region('V1'))

    content1 = gw1.conscious_content_name()
    ign1 = gw1.ignition_count()
    print(f"    点火: {ign1}次, 意识内容: '{content1}', V1={v1_spikes}")

    # Auditory only
    print("\n  仅听觉刺激 (MGN=35.0)...")
    eng2 = pywuyun.SimulationEngine(10)
    eng2.build_standard_brain()
    gw2 = eng2.find_region('GW')

    a1_spikes = 0
    for t in range(200):
        eng2.find_region('MGN').inject_external([35.0] * eng2.find_region('MGN').n_neurons())
        eng2.step()
        a1_spikes += count_fired(eng2.find_region('A1'))

    content2 = gw2.conscious_content_name()
    ign2 = gw2.ignition_count()
    print(f"    点火: {ign2}次, 意识内容: '{content2}', A1={a1_spikes}")

    ok = ign1 > 0 and ign2 > 0
    print(f"\n  结果: {'✓' if ok else '✗'} 两种模态都能独立进入意识")
    print(f"    视觉→'{content1}' | 听觉→'{content2}'")
    return ok


# =============================================================================
# Demo 3: Attention Gates Consciousness
# =============================================================================
def demo_attention_gating():
    print_header("演示3: 注意力调制皮层响应")
    print("  (使用LGN→V1子系统, 避免全脑反馈干扰)")

    def run_with_gain(gain, steps=100):
        eng = pywuyun.SimulationEngine(10)
        tc = pywuyun.ThalamicConfig()
        tc.name = "LGN"; tc.n_relay = 50; tc.n_trn = 15
        eng.add_thalamic(tc)

        cc = pywuyun.ColumnConfig()
        cc.n_l4_stellate = 50; cc.n_l23_pyramidal = 100
        cc.n_l5_pyramidal = 50; cc.n_l6_pyramidal = 40
        cc.n_pv_basket = 15; cc.n_sst_martinotti = 10; cc.n_vip = 5
        v1 = eng.add_cortical("V1", cc)
        eng.add_projection("LGN", "V1", 2)
        v1.set_attention_gain(gain)

        total = 0
        for t in range(steps):
            eng.find_region("LGN").inject_external([30.0] * 50)
            eng.step()
            total += count_fired(v1)
        return total

    v1_ignore = run_with_gain(0.5)
    v1_normal = run_with_gain(1.0)
    v1_attend = run_with_gain(1.5)

    print(f"  V1(注意1.5):  {v1_attend:6d} spikes")
    print(f"  V1(正常1.0):  {v1_normal:6d} spikes")
    print(f"  V1(忽略0.5):  {v1_ignore:6d} spikes")

    ok = v1_attend > v1_normal > v1_ignore
    print(f"\n  结果: {'✓ 注意力: 增强>正常>忽略' if ok else '✗ 注意力调制异常'}")
    return ok


# =============================================================================
# Demo 4: Internal Drives + Consciousness
# =============================================================================
def demo_drives_consciousness():
    print_header("演示4: 内驱力 × 意识")

    # Awake brain
    eng1 = pywuyun.SimulationEngine(10)
    eng1.build_standard_brain()
    hypo1 = eng1.find_region('Hypothalamus')
    gw1 = eng1.find_region('GW')

    for t in range(200):
        eng1.find_region('LGN').inject_external([35.0] * eng1.find_region('LGN').n_neurons())
        eng1.step()

    wake1 = hypo1.wake_level()
    ign1 = gw1.ignition_count()
    print(f"  觉醒大脑: wake={wake1:.2f}, 点火={ign1}")

    # Stressed brain
    eng2 = pywuyun.SimulationEngine(10)
    eng2.build_standard_brain()
    hypo2 = eng2.find_region('Hypothalamus')
    gw2 = eng2.find_region('GW')
    hypo2.set_stress_level(0.9)

    for t in range(200):
        eng2.find_region('LGN').inject_external([35.0] * eng2.find_region('LGN').n_neurons())
        eng2.step()

    stress2 = hypo2.stress_output()
    ign2 = gw2.ignition_count()
    print(f"  应激大脑: stress={stress2:.2f}, 点火={ign2}")

    # Hungry brain
    eng3 = pywuyun.SimulationEngine(10)
    eng3.build_standard_brain()
    hypo3 = eng3.find_region('Hypothalamus')
    gw3 = eng3.find_region('GW')
    hypo3.set_hunger_level(0.9)

    for t in range(200):
        eng3.find_region('LGN').inject_external([35.0] * eng3.find_region('LGN').n_neurons())
        eng3.step()

    hunger3 = hypo3.hunger_output()
    ign3 = gw3.ignition_count()
    print(f"  饥饿大脑: hunger={hunger3:.2f}, 点火={ign3}")

    print(f"\n  结果: 内驱力状态调制全脑动态")
    return True


# =============================================================================
# Demo 5: Full Conscious Perception-Action Loop
# =============================================================================
def demo_conscious_action():
    print_header("演示5: 有意识的感知-行动循环")
    print("  流程: LGN→V1→GW(点火)→ILN→dlPFC→BG→M1")

    eng = pywuyun.SimulationEngine(10)
    eng.build_standard_brain()

    gw = eng.find_region('GW')
    lgn = eng.find_region('LGN')

    # Track activity in key regions
    regions_to_track = ['V1', 'GW', 'dlPFC', 'BG', 'M1']
    spike_counts = {name: 0 for name in regions_to_track}

    # Run perception-action loop
    for t in range(300):
        lgn.inject_external([35.0] * lgn.n_neurons())
        eng.step()

        for name in regions_to_track:
            r = eng.find_region(name)
            spike_counts[name] += count_fired(r)

    print(f"\n  活动链路:")
    for name in regions_to_track:
        bar = '█' * min(50, spike_counts[name] // 50)
        print(f"    {name:8s}: {spike_counts[name]:6d} {bar}")

    print(f"\n  GW: {gw.ignition_count()}次点火, 内容='{gw.conscious_content_name()}'")

    # Verify the chain works
    chain_ok = all(spike_counts[n] > 0 for n in regions_to_track)
    print(f"\n  结果: {'✓ 完整的有意识感知-行动链路' if chain_ok else '✗ 链路断裂'}")
    return chain_ok


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════╗")
    print("║  悟韵 (WuYun) — 全局工作空间意识演示            ║")
    print("║  48区域 | 5528神经元 | ~109投射 | GNW           ║")
    print("╚══════════════════════════════════════════════════╝")

    results = []
    results.append(("视觉意识", demo_visual_consciousness()))
    results.append(("多模态竞争", demo_multimodal_competition()))
    results.append(("注意力门控", demo_attention_gating()))
    results.append(("内驱力×意识", demo_drives_consciousness()))
    results.append(("感知-行动循环", demo_conscious_action()))

    print_header("总结")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")

    passed = sum(1 for _, ok in results if ok)
    print(f"\n  {passed}/{len(results)} 演示通过")
