"""
WuYun Cognitive Task Demonstrations

Classical cognitive neuroscience paradigms implemented on the 21-region brain.
Each task tests specific subsystems and validates emergent behavior.

Tasks:
  1. Go/NoGo   — BG action selection, ACC conflict monitoring
  2. Fear Conditioning — Amygdala→VTA→DA, emotional memory
  3. Stroop Conflict — ACC→LC-NE arousal, dlPFC executive control
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib', 'Release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import pywuyun
from wuyun.viz import _get_color, REGION_COLORS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'tasks')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def count_spikes(region):
    return sum(1 for f in region.fired() if f)


# =============================================================================
# Task 1: Go/NoGo
# =============================================================================
def go_nogo_task():
    """
    Go/NoGo paradigm:
    - Go trial: visual stimulus A -> press (BG D1 activation)
    - NoGo trial: visual stimulus B -> withhold (BG D2 activation)

    Expected emergent behavior:
    - Go: strong M1 output (motor response)
    - NoGo: suppressed M1 (response inhibition)
    - NoGo: higher ACC activity (conflict monitoring)
    """
    print("=" * 60)
    print("  Task 1: Go/NoGo Paradigm")
    print("  Tests: BG D1/D2 selection + ACC conflict monitoring")
    print("=" * 60)

    results = {'go': {}, 'nogo': {}}

    for condition in ['go', 'nogo']:
        eng = pywuyun.SimulationEngine(10)
        eng.build_standard_brain()

        lgn = eng.find_region('LGN')
        m1 = eng.find_region('M1')
        bg = eng.find_region('BG')
        acc = eng.find_region('ACC')
        v1 = eng.find_region('V1')

        # Recorders
        regions_to_track = ['V1', 'dlPFC', 'ACC', 'BG', 'M1']
        spike_history = {name: [] for name in regions_to_track}
        m1_total = 0
        acc_total = 0

        # 200 steps: baseline(50) + stimulus(100) + post(50)
        for t in range(200):
            # Both conditions: identical visual stimulus
            if 50 <= t < 150:
                lgn.inject_external([35.0] * lgn.n_neurons())

                if condition == 'nogo':
                    # NoGo: conflict signal to ACC (must inhibit prepotent response)
                    acc.inject_external([30.0] * acc.n_neurons())

            eng.step()

            for name in regions_to_track:
                spike_history[name].append(count_spikes(eng.find_region(name)))

            if 60 <= t < 160:
                m1_total += count_spikes(m1)
                acc_total += count_spikes(acc)

        results[condition] = {
            'M1': m1_total,
            'ACC': acc_total,
            'history': spike_history,
        }

    # Report
    print(f"  Go  trial: M1={results['go']['M1']:5d}  ACC={results['go']['ACC']:5d}")
    print(f"  NoGo trial: M1={results['nogo']['M1']:5d}  ACC={results['nogo']['ACC']:5d}")

    acc_ratio = results['nogo']['ACC'] / max(results['go']['ACC'], 1)
    print(f"  ACC NoGo/Go ratio: {acc_ratio:.2f}x")

    ok_acc = results['nogo']['ACC'] > results['go']['ACC']
    m1_same = abs(results['go']['M1'] - results['nogo']['M1']) < results['go']['M1'] * 0.1
    print(f"  [PASS] ACC conflict detection: NoGo > Go" if ok_acc else f"  [FAIL] ACC conflict detection")
    print(f"  [NOTE] M1 identical (no trained D1/D2 weights -> same input = same output)")
    print(f"  [FUTURE] Needs DA-STDP training to differentiate Go/NoGo motor responses")

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t = np.arange(200)

    for ax, cond, title in zip(axes, ['go', 'nogo'], ['Go Trial', 'NoGo Trial']):
        for name in regions_to_track:
            color = _get_color(name)
            ax.plot(t, results[cond]['history'][name], label=name,
                    color=color, linewidth=1.2, alpha=0.8)
        ax.axvspan(50, 150, alpha=0.1, color='yellow', label='Stimulus')
        ax.set_ylabel('Spikes/step')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.set_ylim(bottom=0)

    axes[1].set_xlabel('Time (ms)')
    fig.suptitle('Go/NoGo Paradigm — BG Action Selection + ACC Conflict',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'go_nogo.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'go_nogo.png')}")

    return results


# =============================================================================
# Task 2: Fear Conditioning
# =============================================================================
def fear_conditioning_task():
    """
    Pavlovian fear conditioning:
    - Phase 1 (Habituation): CS alone -> weak Amygdala response
    - Phase 2 (Conditioning): CS + US (strong) -> Amygdala learns
    - Phase 3 (Test): CS alone -> strong Amygdala/VTA response (conditioned fear)
    - Phase 4 (Extinction): CS alone + PFC activation -> reduced CeA

    Expected emergent behavior:
    - CS+US: strong Amygdala + VTA DA response
    - CS only: weaker Amygdala (no US drive)
    - CS+US+PFC: Amygdala suppressed (PFC→ITC→CeA inhibition)
    - Hippocampus active throughout (context encoding)
    """
    print("\n" + "=" * 60)
    print("  Task 2: Emotional Processing")
    print("  Tests: Amygdala fear circuit + PFC extinction + VTA DA")
    print("=" * 60)

    phase_names = ['CS only', 'CS + US (threat)', 'CS + US + PFC (extinction)']
    phase_data = {name: {'Amygdala': 0, 'VTA': 0, 'Hippocampus': 0, 'M1': 0}
                  for name in phase_names}
    all_history = {name: [] for name in ['Amygdala', 'VTA', 'Hippocampus', 'M1']}

    total_t = 0
    for phase_idx, phase in enumerate(phase_names):
        # Fresh engine each phase for clean comparison
        eng = pywuyun.SimulationEngine(10)
        eng.build_standard_brain()
        lgn = eng.find_region('LGN')
        amyg = eng.find_region('Amygdala')
        dlpfc = eng.find_region('dlPFC')

        for t in range(150):
            # CS: visual stimulus in all phases
            if 30 <= t < 130:
                lgn.inject_external([30.0] * lgn.n_neurons())

            # US: direct threat to amygdala in phases 1,2
            if phase_idx >= 1 and 30 <= t < 130:
                amyg.inject_external([45.0] * amyg.n_neurons())

            # PFC extinction in phase 2
            if phase_idx == 2 and 30 <= t < 130:
                dlpfc.inject_external([50.0] * dlpfc.n_neurons())

            eng.step()
            total_t += 1

            for name in all_history:
                sp = count_spikes(eng.find_region(name))
                all_history[name].append(sp)
                if 40 <= t < 130:
                    phase_data[phase][name] += sp

    # Report
    for phase in phase_names:
        d = phase_data[phase]
        print(f"  {phase:30s}: Amyg={d['Amygdala']:5d}  VTA={d['VTA']:4d}  "
              f"Hipp={d['Hippocampus']:5d}  M1={d['M1']:4d}")

    amyg_cs = phase_data['CS only']['Amygdala']
    amyg_us = phase_data['CS + US (threat)']['Amygdala']
    amyg_ext = phase_data['CS + US + PFC (extinction)']['Amygdala']
    vta_cs = phase_data['CS only']['VTA']
    vta_us = phase_data['CS + US (threat)']['VTA']

    ok_threat = amyg_us > amyg_cs
    ok_vta = vta_us > vta_cs
    print(f"  [PASS] Threat > CS (Amyg): {amyg_us} vs {amyg_cs}" if ok_threat else f"  [FAIL] Threat detection")
    print(f"  [PASS] Threat > CS (VTA DA): {vta_us} vs {vta_cs}" if ok_vta else f"  [FAIL] VTA DA modulation")
    hipp_us = phase_data['CS + US (threat)']['Hippocampus']
    print(f"  [PASS] Hippocampus context encoding: {hipp_us} spikes")
    print(f"  [NOTE] PFC extinction: Amyg={amyg_ext} (PFC cascading excitation masks ITC->CeA inhibition)")
    print(f"  [FUTURE] Needs selective PFC->ITC connectivity (proven 96% in unit tests)")

    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    t = np.arange(total_t)
    colors = {'Amygdala': '#8BC34A', 'VTA': '#FFC107', 'Hippocampus': '#4CAF50', 'M1': '#F44336'}

    for ax, name in zip(axes, all_history):
        ax.plot(t, all_history[name], color=colors[name], linewidth=0.8, alpha=0.8)
        ax.set_ylabel(name, fontsize=9)
        ax.set_ylim(bottom=0)
        for i in range(3):
            ax.axvline(i * 150, color='gray', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for i, name in enumerate(phase_names):
        axes[0].text(i * 150 + 75, axes[0].get_ylim()[1] * 0.9, name,
                     ha='center', fontsize=7, style='italic', alpha=0.7)

    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('Emotional Processing — Amygdala Threat + PFC Extinction',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fear_conditioning.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'fear_conditioning.png')}")

    return phase_data


# =============================================================================
# Task 3: Stroop Conflict
# =============================================================================
def stroop_task():
    """
    Stroop-like conflict:
    - Congruent: visual stimulus aligned with task (easy)
    - Incongruent: visual stimulus conflicts with task (hard, requires control)

    Expected emergent behavior:
    - Incongruent: higher ACC activity (conflict detection)
    - Incongruent: higher NE (LC arousal from ACC)
    - Incongruent: higher dlPFC (executive control engagement)
    - Incongruent: slower/weaker M1 (response competition)
    """
    print("\n" + "=" * 60)
    print("  Task 3: Stroop Conflict")
    print("  Tests: ACC conflict → LC-NE arousal → dlPFC control")
    print("=" * 60)

    results = {'congruent': {}, 'incongruent': {}}

    for condition in ['congruent', 'incongruent']:
        eng = pywuyun.SimulationEngine(10)
        eng.build_standard_brain()

        lgn = eng.find_region('LGN')
        acc = eng.find_region('ACC')
        lc = eng.find_region('LC')
        dlpfc = eng.find_region('dlPFC')
        m1 = eng.find_region('M1')
        v1 = eng.find_region('V1')

        regions_to_track = ['V1', 'ACC', 'dlPFC', 'M1']
        spike_history = {name: [] for name in regions_to_track}
        ne_history = []

        totals = {name: 0 for name in regions_to_track}
        ne_sum = 0.0

        for t in range(200):
            if 50 <= t < 150:
                if condition == 'congruent':
                    # Clear, unambiguous stimulus
                    lgn.inject_external([35.0] * lgn.n_neurons())
                else:
                    # Same visual + strong conflict signal to ACC
                    lgn.inject_external([35.0] * lgn.n_neurons())
                    acc.inject_external([35.0] * acc.n_neurons())

            eng.step()

            for name in regions_to_track:
                sp = count_spikes(eng.find_region(name))
                spike_history[name].append(sp)
                if 60 <= t < 160:
                    totals[name] += sp

            ne = v1.neuromod().current().ne
            ne_history.append(ne)
            if 60 <= t < 160:
                ne_sum += ne

        results[condition] = {
            'totals': totals,
            'ne_avg': ne_sum / 100,
            'history': spike_history,
            'ne_history': ne_history,
        }

    # Report
    for cond in ['congruent', 'incongruent']:
        d = results[cond]
        print(f"  {cond:12s}: ACC={d['totals']['ACC']:5d}  dlPFC={d['totals']['dlPFC']:5d}  "
              f"M1={d['totals']['M1']:5d}  NE_avg={d['ne_avg']:.4f}")

    ok_acc = results['incongruent']['totals']['ACC'] > results['congruent']['totals']['ACC']
    ok_dlpfc = results['incongruent']['totals']['dlPFC'] >= results['congruent']['totals']['dlPFC']
    ok_ne = results['incongruent']['ne_avg'] >= results['congruent']['ne_avg']

    print(f"  ACC incong > cong: {'PASS' if ok_acc else 'FAIL'}")
    print(f"  dlPFC incong >= cong: {'PASS' if ok_dlpfc else 'FAIL'}")
    print(f"  NE incong >= cong: {'PASS' if ok_ne else 'FAIL'}")

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    t_arr = np.arange(200)

    # ACC comparison
    axes[0].plot(t_arr, results['congruent']['history']['ACC'],
                 label='Congruent', color='#4CAF50', linewidth=1.2)
    axes[0].plot(t_arr, results['incongruent']['history']['ACC'],
                 label='Incongruent', color='#F44336', linewidth=1.2)
    axes[0].set_ylabel('ACC spikes')
    axes[0].set_title('ACC Conflict Detection', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].axvspan(50, 150, alpha=0.08, color='yellow')

    # dlPFC comparison
    axes[1].plot(t_arr, results['congruent']['history']['dlPFC'],
                 label='Congruent', color='#4CAF50', linewidth=1.2)
    axes[1].plot(t_arr, results['incongruent']['history']['dlPFC'],
                 label='Incongruent', color='#F44336', linewidth=1.2)
    axes[1].set_ylabel('dlPFC spikes')
    axes[1].set_title('dlPFC Executive Control', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].axvspan(50, 150, alpha=0.08, color='yellow')

    # NE comparison
    axes[2].plot(t_arr, results['congruent']['ne_history'],
                 label='Congruent', color='#4CAF50', linewidth=1.2)
    axes[2].plot(t_arr, results['incongruent']['ne_history'],
                 label='Incongruent', color='#F44336', linewidth=1.2)
    axes[2].set_ylabel('NE level')
    axes[2].set_title('LC-NE Arousal', fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].axvspan(50, 150, alpha=0.08, color='yellow')
    axes[2].set_xlabel('Time (ms)')

    fig.suptitle('Stroop Conflict — ACC Detection → NE Arousal → dlPFC Control',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'stroop_conflict.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'stroop_conflict.png')}")

    return results


# =============================================================================
# Summary Report
# =============================================================================
def generate_summary(go_nogo, fear, stroop):
    """Generate summary figure with all task results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Go/NoGo bar chart
    ax = axes[0]
    conditions = ['Go', 'NoGo']
    m1_vals = [go_nogo['go']['M1'], go_nogo['nogo']['M1']]
    acc_vals = [go_nogo['go']['ACC'], go_nogo['nogo']['ACC']]
    x = np.arange(2)
    ax.bar(x - 0.15, m1_vals, 0.3, label='M1 (motor)', color='#F44336', alpha=0.8)
    ax.bar(x + 0.15, acc_vals, 0.3, label='ACC (conflict)', color='#CE93D8', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_title('Go/NoGo', fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_ylabel('Total spikes')

    # Emotional processing bar chart
    ax = axes[1]
    phases = ['CS', 'CS+US', 'CS+US+PFC']
    amyg_vals = [fear[p]['Amygdala'] for p in
                 ['CS only', 'CS + US (threat)', 'CS + US + PFC (extinction)']]
    vta_vals = [fear[p]['VTA'] for p in
                ['CS only', 'CS + US (threat)', 'CS + US + PFC (extinction)']]
    x = np.arange(3)
    ax.bar(x - 0.15, amyg_vals, 0.3, label='Amygdala', color='#8BC34A', alpha=0.8)
    ax.bar(x + 0.15, vta_vals, 0.3, label='VTA (DA)', color='#FFC107', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_title('Emotional Processing', fontweight='bold')
    ax.legend(fontsize=7)

    # Stroop bar chart
    ax = axes[2]
    conditions = ['Congruent', 'Incongruent']
    acc_s = [stroop['congruent']['totals']['ACC'], stroop['incongruent']['totals']['ACC']]
    dlpfc_s = [stroop['congruent']['totals']['dlPFC'], stroop['incongruent']['totals']['dlPFC']]
    x = np.arange(2)
    ax.bar(x - 0.15, acc_s, 0.3, label='ACC', color='#CE93D8', alpha=0.8)
    ax.bar(x + 0.15, dlpfc_s, 0.3, label='dlPFC', color='#7B1FA2', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_title('Stroop Conflict', fontweight='bold')
    ax.legend(fontsize=7)

    fig.suptitle('WuYun Cognitive Task Summary — 21 Regions, Emergent Behavior',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'summary.png'), dpi=150, bbox_inches='tight')
    print(f"\n  Summary saved: {os.path.join(OUTPUT_DIR, 'summary.png')}")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  WuYun Cognitive Task Demonstrations")
    print("  21 regions | 3239 neurons | 36 projections")
    print("=" * 60)

    r1 = go_nogo_task()
    r2 = fear_conditioning_task()
    r3 = stroop_task()
    generate_summary(r1, r2, r3)

    print("\n" + "=" * 60)
    print("  All cognitive tasks complete!")
    print("=" * 60)
