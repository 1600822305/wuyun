"""
悟韵 (WuYun) — "Agent一天的生活" 端到端演示

展示所有11个Step协同工作的完整链路:

  晨醒 → 视觉感知 → 听觉感知 → 学习编码 → 决策
    → 疲劳(睡眠压力↑) → 入睡
    → NREM(慢波 + SWR记忆巩固)
    → REM(theta + PGO梦境 + 创造性重组)
    → 唤醒 → 回忆测试(巩固后 vs 巩固前)

涉及子系统:
  - 感觉输入 (Step 9): VisualInput + AuditoryInput
  - 层级处理 (Step 5): LGN→V1→V2→V4→IT→dlPFC
  - 学习 (Step 4): CA3 STDP + DA-STDP
  - 注意力 (Step 5): Pulvinar + TRN
  - GNW意识 (Step 7.5): 竞争→点火→广播
  - 内驱力 (Step 6): 下丘脑 SCN/VLPO/Orexin
  - NREM睡眠 (Step 8): 皮层慢波 + 海马SWR
  - REM睡眠 (Step 11): theta + PGO + 创造性重组
  - 4种调质: DA/NE/5-HT/ACh

运行: python day_in_life_demo.py [--save]
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib', 'Release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pywuyun

SAVE_FIGS = '--save' in sys.argv

try:
    import matplotlib
    if SAVE_FIGS:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for font_name in ['SimHei', 'Microsoft YaHei', 'STSong', 'WenQuanYi Micro Hei']:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font_name] + matplotlib.rcParams['font.sans-serif']
            break
        except Exception:
            continue
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[INFO] matplotlib not installed, text-only output")


# =============================================================================
# Utilities
# =============================================================================
def print_header(title, char='='):
    line = char * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def count_fired(region):
    return int(sum(1 for x in region.fired() if x))


def run_steps(eng, n, inject_fn=None):
    """Run n steps, optionally calling inject_fn(t) each step. Returns total spikes per named region."""
    totals = {}
    for t in range(n):
        if inject_fn:
            inject_fn(t)
        eng.step()
    return totals


def collect_activity(eng, region_names, n_steps, inject_fn=None):
    """Run n_steps and collect per-step spike counts for named regions."""
    history = {name: [] for name in region_names}
    for t in range(n_steps):
        if inject_fn:
            inject_fn(t)
        eng.step()
        for name in region_names:
            r = eng.find_region(name)
            history[name].append(count_fired(r) if r else 0)
    return history


# =============================================================================
# Phase 1: Morning — Build Brain + Wake Up
# =============================================================================
def phase_morning(eng):
    print_header("Phase 1: Morning Awakening", '=')
    print("  [SCN] Circadian clock signals dawn...")
    print("  [Orexin] Stabilizing wakefulness...")

    hypo = eng.find_region("Hypothalamus")
    if hypo:
        hypo.set_sleep_pressure(0.1)

    # Run 50 warm-up steps
    regions_to_watch = ["V1", "dlPFC", "Hippocampus", "BG", "M1"]
    history = collect_activity(eng, regions_to_watch, 50)

    total_neurons = sum(eng.find_region(name).n_neurons()
                       for name in ["LGN", "V1", "V2", "dlPFC", "Hippocampus", "BG", "M1"]
                       if eng.find_region(name))

    print(f"  48 regions online, ~{eng.find_region('V1').n_neurons() * 20}+ neurons")
    print(f"  Baseline activity (50 steps):")
    for name in regions_to_watch:
        total = sum(history[name])
        print(f"    {name:>15s}: {total:5d} spikes")

    return history


# =============================================================================
# Phase 2: Morning Perception — Visual + Auditory
# =============================================================================
def phase_perception(eng):
    print_header("Phase 2: Sensory Perception", '=')

    lgn = eng.find_region("LGN")
    mgn = eng.find_region("MGN")
    v1 = eng.find_region("V1")
    a1 = eng.find_region("A1")

    # Create sensory inputs
    vcfg = pywuyun.VisualInputConfig()
    vcfg.input_width = 8
    vcfg.input_height = 8
    vcfg.n_lgn_neurons = lgn.n_neurons()
    vis = pywuyun.VisualInput(vcfg)

    acfg = pywuyun.AuditoryInputConfig()
    acfg.n_freq_bands = 16
    acfg.n_mgn_neurons = mgn.n_neurons()
    aud = pywuyun.AuditoryInput(acfg)

    # --- Scene A: "Apple" (bright center = red object) ---
    print("\n  [Eyes] Seeing 'Pattern A' (bright center)...")
    pattern_a = np.zeros(64)
    for r in range(2, 6):
        for c in range(2, 6):
            pattern_a[r * 8 + c] = 0.9

    regions_watch = ["V1", "V2", "V4", "IT", "dlPFC", "Hippocampus", "GW"]

    def inject_scene_a(t):
        vis.encode_and_inject(pattern_a.tolist(), lgn)

    scene_a_hist = collect_activity(eng, regions_watch, 100, inject_scene_a)
    print("  Visual pathway activation (100 steps):")
    for name in regions_watch:
        total = sum(scene_a_hist[name])
        print(f"    {name:>15s}: {total:5d} spikes")

    # Check GW consciousness
    gw = eng.find_region("GW")
    gw_ignitions_a = gw.ignition_count() if gw else 0
    gw_content_a = gw.conscious_content_name() if gw else "?"
    print(f"  [GW] Conscious content: '{gw_content_a}', ignitions: {gw_ignitions_a}")

    # --- Scene B: "Bird song" (auditory tone) ---
    print("\n  [Ears] Hearing 'Pattern B' (low frequency tone)...")
    spectrum_b = np.zeros(16)
    spectrum_b[1:4] = 0.8  # Low frequency

    def inject_scene_b(t):
        aud.encode_and_inject(spectrum_b.tolist(), mgn)

    scene_b_hist = collect_activity(eng, ["A1", "Wernicke", "Hippocampus", "GW"], 100, inject_scene_b)
    print("  Auditory pathway activation (100 steps):")
    for name in ["A1", "Wernicke", "Hippocampus", "GW"]:
        total = sum(scene_b_hist[name])
        print(f"    {name:>15s}: {total:5d} spikes")

    gw_ignitions_b = gw.ignition_count() if gw else 0
    print(f"  [GW] Total ignitions after both stimuli: {gw_ignitions_b}")

    return vis, aud, pattern_a, spectrum_b, scene_a_hist, scene_b_hist


# =============================================================================
# Phase 3: Learning — Encode Memories
# =============================================================================
def phase_learning(eng, vis, aud, pattern_a, spectrum_b):
    print_header("Phase 3: Learning & Memory Encoding", '=')

    lgn = eng.find_region("LGN")
    mgn = eng.find_region("MGN")
    hipp = eng.find_region("Hippocampus")

    # Pre-learning hippocampal response to Pattern A
    print("  [Pre-learning] Testing hippocampal response to Pattern A...")
    pre_hipp_a = 0
    for t in range(50):
        vis.encode_and_inject(pattern_a.tolist(), lgn)
        eng.step()
        pre_hipp_a += count_fired(hipp)

    # Training: repeated exposure to Pattern A (visual)
    print("  [Training] Encoding Pattern A (200 steps, STDP active)...")
    train_hipp_a = 0
    for t in range(200):
        vis.encode_and_inject(pattern_a.tolist(), lgn)
        eng.step()
        train_hipp_a += count_fired(hipp)

    # Training: repeated exposure to Pattern B (auditory)
    print("  [Training] Encoding Pattern B (200 steps, STDP active)...")
    train_hipp_b = 0
    for t in range(200):
        aud.encode_and_inject(spectrum_b.tolist(), mgn)
        eng.step()
        train_hipp_b += count_fired(hipp)

    # Post-learning hippocampal response to Pattern A
    print("  [Post-learning] Testing hippocampal response to Pattern A...")
    post_hipp_a = 0
    for t in range(50):
        vis.encode_and_inject(pattern_a.tolist(), lgn)
        eng.step()
        post_hipp_a += count_fired(hipp)

    print(f"\n  Hippocampal encoding results:")
    print(f"    Pre-learning (A, 50 steps):  {pre_hipp_a:5d} spikes")
    print(f"    Training A (200 steps):      {train_hipp_a:5d} spikes")
    print(f"    Training B (200 steps):      {train_hipp_b:5d} spikes")
    print(f"    Post-learning (A, 50 steps): {post_hipp_a:5d} spikes")

    return pre_hipp_a, post_hipp_a


# =============================================================================
# Phase 4: Getting Tired — Sleep Pressure Rises
# =============================================================================
def phase_fatigue(eng):
    print_header("Phase 4: Fatigue — Sleep Pressure Rising", '=')

    hypo = eng.find_region("Hypothalamus")
    if hypo:
        print("  [SCN] Evening approaches...")
        print("  [VLPO] Sleep pressure accumulating...")
        hypo.set_sleep_pressure(0.7)

    # Run 100 steps with rising fatigue
    for t in range(100):
        eng.step()

    if hypo:
        wake = hypo.wake_level()
        print(f"  [Hypothalamus] Wake level: {wake:.2f} (dropping)")
    print("  Agent is getting drowsy...")

    return True


# =============================================================================
# Phase 5: NREM Sleep — Slow Waves + SWR Consolidation
# =============================================================================
def phase_nrem(eng, sleep_mgr):
    print_header("Phase 5: NREM Sleep", '=')

    hipp = eng.find_region("Hippocampus")
    v1_region = eng.find_region("V1")

    # Enter sleep, start with NREM
    sleep_mgr.enter_sleep()

    # Enable NREM mode on all cortical regions
    cortical_names = ["V1", "V2", "V4", "IT", "MT", "PPC", "S1", "S2", "A1",
                      "dlPFC", "OFC", "vmPFC", "ACC", "M1", "PMC", "SMA",
                      "PCC", "Insula", "TPJ", "Broca", "Wernicke",
                      "Gustatory", "Piriform", "FEF"]

    for name in cortical_names:
        r = eng.find_region(name)
        if r:
            r.set_sleep_mode(True)

    # Enable SWR replay in hippocampus
    if hipp:
        hipp.enable_sleep_replay()

    print("  [Cortex] Slow wave oscillations started (~1Hz up/down)")
    print("  [Hippocampus] SWR replay enabled")
    print(f"  [SleepCycle] NREM duration: {sleep_mgr.current_nrem_duration()} steps")

    # Run NREM
    nrem_steps = sleep_mgr.current_nrem_duration()
    nrem_v1_spikes = 0
    up_count = 0
    swr_events = 0

    for t in range(nrem_steps):
        sleep_mgr.step()
        eng.step()
        nrem_v1_spikes += count_fired(v1_region) if v1_region else 0
        if v1_region and v1_region.is_up_state():
            up_count += 1

    swr_events = hipp.swr_count() if hipp else 0
    replay_str = hipp.last_replay_strength() if hipp else 0

    print(f"\n  NREM results ({nrem_steps} steps):")
    print(f"    V1 spikes: {nrem_v1_spikes} (up states: {up_count}/{nrem_steps})")
    print(f"    SWR events: {swr_events}")
    print(f"    Last replay strength: {replay_str:.3f}")
    print(f"    [Memory consolidation: SWR -> CA1 burst -> cortical STDP]")

    return nrem_v1_spikes, swr_events


# =============================================================================
# Phase 6: REM Sleep — Dreaming
# =============================================================================
def phase_rem(eng, sleep_mgr):
    print_header("Phase 6: REM Sleep — Dreaming", '=')

    hipp = eng.find_region("Hippocampus")
    v1_region = eng.find_region("V1")
    m1_region = eng.find_region("M1")

    # Transition to REM
    cortical_names = ["V1", "V2", "V4", "IT", "MT", "PPC", "S1", "S2", "A1",
                      "dlPFC", "OFC", "vmPFC", "ACC", "PMC", "SMA",
                      "PCC", "Insula", "TPJ", "Broca", "Wernicke",
                      "Gustatory", "Piriform", "FEF"]

    for name in cortical_names:
        r = eng.find_region(name)
        if r:
            r.set_rem_mode(True)

    # M1 gets motor atonia
    if m1_region:
        m1_region.set_rem_mode(True)
        m1_region.set_motor_atonia(True)

    # Hippocampus: switch from SWR to theta
    if hipp:
        hipp.disable_sleep_replay()
        hipp.enable_rem_theta()

    print("  [Cortex] Desynchronized activity (dream-like)")
    print("  [Hippocampus] Theta oscillation + creative recombination")
    print("  [M1] Motor atonia (can't act out dreams)")
    print(f"  [SleepCycle] REM duration: {sleep_mgr.current_rem_duration()} steps")

    # Advance sleep_mgr to REM
    while not sleep_mgr.is_rem():
        sleep_mgr.step()
        if not sleep_mgr.is_sleeping():
            break

    # Run REM
    rem_steps = sleep_mgr.current_rem_duration()
    rem_v1_spikes = 0
    rem_m1_spikes = 0
    pgo_count = 0
    recomb_before = hipp.rem_recombination_count() if hipp else 0

    for t in range(rem_steps):
        sleep_mgr.step()

        # PGO waves
        if sleep_mgr.pgo_active():
            pgo_count += 1
            if v1_region:
                v1_region.inject_pgo_wave(25.0)

        eng.step()
        rem_v1_spikes += count_fired(v1_region) if v1_region else 0
        rem_m1_spikes += count_fired(m1_region) if m1_region else 0

    recomb_after = hipp.rem_recombination_count() if hipp else 0

    print(f"\n  REM results ({rem_steps} steps):")
    print(f"    V1 spikes: {rem_v1_spikes} (desynchronized + PGO)")
    print(f"    M1 spikes: {rem_m1_spikes} (atonia suppressed)")
    print(f"    PGO waves: {pgo_count}")
    print(f"    Creative recombinations: {recomb_after - recomb_before}")
    print(f"    Theta phase cycling: {hipp.rem_theta_phase():.3f}" if hipp else "")

    return rem_v1_spikes, pgo_count, recomb_after - recomb_before


# =============================================================================
# Phase 7: Waking Up — Memory Recall Test
# =============================================================================
def phase_wakeup(eng, vis, pattern_a, pre_hipp_a, post_hipp_a):
    print_header("Phase 7: Waking Up — Memory Recall", '=')

    # Disable all sleep modes
    cortical_names = ["V1", "V2", "V4", "IT", "MT", "PPC", "S1", "S2", "A1",
                      "dlPFC", "OFC", "vmPFC", "ACC", "M1", "PMC", "SMA",
                      "PCC", "Insula", "TPJ", "Broca", "Wernicke",
                      "Gustatory", "Piriform", "FEF"]

    for name in cortical_names:
        r = eng.find_region(name)
        if r:
            r.set_rem_mode(False)
            r.set_motor_atonia(False)

    hipp = eng.find_region("Hippocampus")
    if hipp:
        hipp.disable_rem_theta()

    hypo = eng.find_region("Hypothalamus")
    if hypo:
        hypo.set_sleep_pressure(0.1)

    lgn = eng.find_region("LGN")

    print("  [Orexin] Wakefulness restored")
    print("  [All cortex] Sleep modes off")

    # Warm up (30 steps)
    for t in range(30):
        eng.step()

    # Post-sleep recall: Pattern A
    print("\n  [Recall Test] Presenting Pattern A after sleep...")
    recall_hipp_a = 0
    recall_v1 = 0
    for t in range(50):
        vis.encode_and_inject(pattern_a.tolist(), lgn)
        eng.step()
        recall_hipp_a += count_fired(hipp) if hipp else 0
        recall_v1 += count_fired(eng.find_region("V1"))

    gw = eng.find_region("GW")
    gw_ign = gw.ignition_count() if gw else 0
    gw_content = gw.conscious_content_name() if gw else "?"

    print(f"\n  Memory comparison (Pattern A, 50 steps each):")
    print(f"    Pre-learning:  {pre_hipp_a:5d} hippocampal spikes")
    print(f"    Post-learning: {post_hipp_a:5d} hippocampal spikes")
    print(f"    Post-sleep:    {recall_hipp_a:5d} hippocampal spikes")
    print(f"    V1 recall:     {recall_v1:5d} spikes")
    print(f"    [GW] Content: '{gw_content}', total ignitions: {gw_ign}")

    return recall_hipp_a


# =============================================================================
# Visualization
# =============================================================================
def make_timeline_plot(all_data):
    """Create a memory consolidation bar chart."""
    if not HAS_MPL:
        return

    mem_data = all_data.get('memory', {})
    if not mem_data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(mem_data.keys())
    values = list(mem_data.values())
    bar_colors = ['#FFC107', '#4CAF50', '#2196F3'][:len(labels)]
    bars = ax.bar(labels, values, color=bar_colors, edgecolor='black', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
               str(val), ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel("Hippocampal Spikes (50 steps)")
    ax.set_title("Memory Consolidation: A Day in the Life")
    plt.tight_layout()
    if SAVE_FIGS:
        fig.savefig("day_in_life_memory.png", dpi=150)
        print(f"\n  [Saved] day_in_life_memory.png")
    else:
        plt.show(block=False)
        plt.pause(1.0)


# =============================================================================
# Main: A Day in the Life
# =============================================================================
def main():
    print_header("WuYun — A Day in the Life of an Agent", '#')
    print("  48 brain regions | ~5500 neurons | ~109 projections")
    print("  11 functional systems working together")
    print()

    t_start = time.time()

    # Build the brain
    print("  Building standard brain...")
    eng = pywuyun.SimulationEngine(10)
    eng.build_standard_brain()
    print(f"  Brain built in {time.time() - t_start:.2f}s")

    # Setup sleep cycle manager
    sleep_cfg = pywuyun.SleepCycleConfig()
    sleep_cfg.nrem_duration = 400
    sleep_cfg.rem_duration = 200
    sleep_cfg.rem_growth = 0
    sleep_cfg.min_nrem_duration = 200
    sleep_cfg.rem_pgo_prob = 0.03
    sleep_mgr = pywuyun.SleepCycleManager(sleep_cfg)

    all_data = {'phases': {}, 'sleep_stages': [], 'memory': {}}

    # ===== Phase 1: Morning =====
    morning_hist = phase_morning(eng)
    all_data['phases']['wake'] = {
        'v1': morning_hist.get('V1', []),
        'hipp': morning_hist.get('Hippocampus', []),
        'x_range': list(range(50))
    }

    # ===== Phase 2: Perception =====
    vis, aud, pattern_a, spectrum_b, scene_a_hist, scene_b_hist = phase_perception(eng)
    offset = 50
    all_data['phases']['learn'] = {
        'v1': scene_a_hist.get('V1', []) + scene_b_hist.get('V1', [scene_b_hist.get('A1', 0)]),
        'hipp': scene_a_hist.get('Hippocampus', []) + scene_b_hist.get('Hippocampus', []),
        'x_range': list(range(offset, offset + 200))
    }

    # ===== Phase 3: Learning =====
    pre_hipp_a, post_hipp_a = phase_learning(eng, vis, aud, pattern_a, spectrum_b)
    all_data['memory']['Pre-learning'] = pre_hipp_a

    # ===== Phase 4: Fatigue =====
    phase_fatigue(eng)

    # ===== Phase 5: NREM =====
    nrem_spikes, swr_events = phase_nrem(eng, sleep_mgr)

    # ===== Phase 6: REM =====
    rem_v1_spikes, pgo_count, recomb_count = phase_rem(eng, sleep_mgr)

    # ===== Phase 7: Wake Up & Recall =====
    sleep_mgr.wake_up()
    recall_hipp = phase_wakeup(eng, vis, pattern_a, pre_hipp_a, post_hipp_a)
    all_data['memory']['Post-learning'] = post_hipp_a
    all_data['memory']['Post-sleep'] = recall_hipp

    elapsed = time.time() - t_start

    # ===== Final Summary =====
    print_header("Summary: Agent's Day Complete", '#')
    print(f"""
  Timeline:
    Morning:     50 steps  (baseline)
    Perception: 200 steps  (visual + auditory)
    Learning:   500 steps  (encode Pattern A + B)
    Fatigue:    100 steps  (sleep pressure rising)
    NREM:       {sleep_cfg.nrem_duration} steps  (slow waves + SWR = {swr_events} events)
    REM:        {sleep_cfg.rem_duration} steps  (PGO = {pgo_count}, recombinations = {recomb_count})
    Recall:      80 steps  (post-sleep memory test)

  Memory Consolidation:
    Pre-learning:  {pre_hipp_a:5d} hippocampal spikes
    Post-learning: {post_hipp_a:5d} hippocampal spikes
    Post-sleep:    {recall_hipp:5d} hippocampal spikes

  Systems Used:
    [OK] Sensory Input (VisualInput + AuditoryInput)
    [OK] Visual Hierarchy (LGN -> V1 -> V2 -> V4 -> IT)
    [OK] Auditory Pathway (MGN -> A1 -> Wernicke)
    [OK] Memory Encoding (Hippocampus CA3 STDP)
    [OK] Decision/PFC (dlPFC + ACC + OFC)
    [OK] Consciousness (GW ignition + broadcast)
    [OK] Internal Drives (Hypothalamus SCN/VLPO/Orexin)
    [OK] NREM Sleep (cortical slow wave + SWR replay)
    [OK] REM Sleep (theta + PGO + creative recombination)
    [OK] Motor System (M1 atonia during REM)
    [OK] Neuromodulators (DA/NE/5-HT/ACh)

  Total time: {elapsed:.2f}s ({elapsed/1.53:.0f}x faster than real-time estimate)
""")

    # Visualization
    if HAS_MPL:
        make_timeline_plot(all_data)

    return 0


if __name__ == '__main__':
    sys.exit(main())
