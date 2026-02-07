"""
WuYun Visualization Tools

Spike raster plots, brain connectivity graphs, and learning curves
for the WuYun brain simulation engine.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

# Add pywuyun to path
_LIB_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib', 'Release')
if os.path.isdir(_LIB_DIR):
    sys.path.insert(0, os.path.abspath(_LIB_DIR))

import pywuyun


# =============================================================================
# Color scheme for brain regions
# =============================================================================
REGION_COLORS = {
    # Visual (blues)
    'LGN': '#1f77b4', 'V1': '#2196F3', 'V2': '#42A5F5',
    'V4': '#64B5F6', 'IT': '#90CAF9',
    # Dorsal (teals)
    'MT': '#009688', 'PPC': '#4DB6AC', 'FEF': '#00BCD4',
    # Sensory (light blues)
    'S1': '#03A9F4', 'S2': '#29B6F6', 'A1': '#4FC3F7',
    'Gustatory': '#81D4FA', 'Piriform': '#B3E5FC',
    # Decision (purples)
    'OFC': '#9C27B0', 'vmPFC': '#AB47BC', 'ACC': '#CE93D8',
    'dlPFC': '#7B1FA2',
    # Motor (reds)
    'M1': '#F44336', 'MotorThal': '#EF5350',
    'PMC': '#E53935', 'SMA': '#EF5350',
    # Language (pinks)
    'Broca': '#E91E63', 'Wernicke': '#F06292',
    # Association (indigos)
    'PCC': '#3F51B5', 'TPJ': '#5C6BC0', 'Insula': '#7986CB',
    # Subcortical (oranges/yellows)
    'BG': '#FF9800', 'VTA': '#FFC107',
    # Thalamic (amber)
    'VPL': '#FFB300', 'MGN': '#FFA000', 'MD': '#FF8F00',
    'VA': '#FF6F00', 'LP': '#F57F17', 'LD': '#F9A825',
    'Pulvinar': '#FDD835', 'CeM': '#FFEE58', 'ILN': '#FFF176',
    'ATN': '#FFD54F',
    # Limbic (greens)
    'Hippocampus': '#4CAF50', 'Amygdala': '#8BC34A',
    'SeptalNucleus': '#66BB6A', 'MammillaryBody': '#81C784',
    # Hypothalamus (deep green)
    'Hypothalamus': '#2E7D32',
    # Cerebellum (brown)
    'Cerebellum': '#795548',
    # Neuromod (grays)
    'LC': '#607D8B', 'DRN': '#78909C', 'NBM': '#90A4AE',
    # Consciousness (gold)
    'GW': '#FFD700',
}

def _get_color(name):
    return REGION_COLORS.get(name, '#333333')


# =============================================================================
# Spike Raster Plot
# =============================================================================
def plot_raster(engine, recorders, duration=None, figsize=(14, 8), save_path=None):
    """
    Plot spike raster for multiple regions.

    Args:
        engine: SimulationEngine instance
        recorders: dict of {region_name: SpikeRecorder}
        duration: total timesteps (for x-axis)
        figsize: figure size
        save_path: if provided, save figure to this path
    """
    n_regions = len(recorders)
    fig, axes = plt.subplots(n_regions, 1, figsize=figsize, sharex=True)
    if n_regions == 1:
        axes = [axes]

    for ax, (name, rec) in zip(axes, recorders.items()):
        times, neurons = rec.to_raster()
        times = np.array(times)
        neurons = np.array(neurons)

        color = _get_color(name)
        region = engine.find_region(name)
        n_neurons = region.n_neurons()

        ax.scatter(times, neurons, s=0.3, c=color, alpha=0.6, rasterized=True)
        ax.set_ylabel(f'{name}\n({n_neurons}n)', fontsize=8, rotation=0,
                      ha='right', va='center')
        ax.set_ylim(-1, n_neurons)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        total = rec.total_spikes()
        ax.text(0.98, 0.85, f'{total}', transform=ax.transAxes,
                fontsize=7, ha='right', color=color, alpha=0.8)

    if duration:
        axes[-1].set_xlim(0, duration)
    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('WuYun Spike Raster', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig


# =============================================================================
# Brain Connectivity Graph
# =============================================================================
def plot_connectivity(engine, figsize=(12, 10), save_path=None):
    """
    Plot brain region connectivity graph using networkx.

    Args:
        engine: SimulationEngine with regions and projections
        figsize: figure size
        save_path: if provided, save figure
    """
    G = nx.DiGraph()

    # Add nodes
    for i in range(engine.num_regions()):
        r = engine.region(i)
        name = r.name()
        G.add_node(name, n_neurons=r.n_neurons())

    # Add edges from projections (we'll parse from the standard brain setup)
    # Since SpikeBus doesn't expose projection details to Python yet,
    # use the known standard brain topology
    STANDARD_PROJECTIONS = [
        # Visual
        ('LGN', 'V1'), ('V1', 'V2'), ('V2', 'V4'), ('V4', 'IT'),
        ('V2', 'V1'), ('V4', 'V2'), ('IT', 'V4'),
        ('V1', 'MT'), ('V2', 'MT'), ('MT', 'PPC'), ('PPC', 'MT'),
        ('PPC', 'IT'), ('IT', 'PPC'),
        # Decision
        ('IT', 'OFC'), ('OFC', 'vmPFC'), ('vmPFC', 'BG'),
        ('vmPFC', 'Amygdala'), ('ACC', 'dlPFC'), ('ACC', 'LC'),
        ('dlPFC', 'ACC'), ('IT', 'dlPFC'), ('PPC', 'dlPFC'),
        ('PPC', 'M1'), ('dlPFC', 'BG'), ('BG', 'MotorThal'),
        ('MotorThal', 'M1'), ('M1', 'Cerebellum'),
        ('Cerebellum', 'MotorThal'),
        # Limbic
        ('V1', 'Amygdala'), ('dlPFC', 'Amygdala'), ('Amygdala', 'OFC'),
        ('dlPFC', 'Hippocampus'), ('Hippocampus', 'dlPFC'),
        ('Amygdala', 'VTA'), ('Amygdala', 'Hippocampus'),
        ('VTA', 'BG'),
        # Sensory
        ('VPL', 'S1'), ('S1', 'S2'), ('S2', 'PPC'),
        ('MGN', 'A1'), ('A1', 'Wernicke'), ('Wernicke', 'Broca'),
        # Motor hierarchy
        ('dlPFC', 'PMC'), ('dlPFC', 'SMA'), ('PMC', 'M1'), ('SMA', 'M1'),
        # DMN
        ('PCC', 'vmPFC'), ('vmPFC', 'PCC'), ('TPJ', 'PCC'),
        # GW broadcast
        ('V1', 'GW'), ('IT', 'GW'), ('dlPFC', 'GW'), ('ACC', 'GW'),
        ('GW', 'ILN'), ('GW', 'CeM'),
        # Hypothalamus
        ('Hypothalamus', 'LC'), ('Hypothalamus', 'DRN'),
        ('Hypothalamus', 'NBM'), ('Hypothalamus', 'VTA'),
        # Papez
        ('Hippocampus', 'MammillaryBody'), ('MammillaryBody', 'ATN'),
        ('ATN', 'ACC'),
    ]

    for src, dst in STANDARD_PROJECTIONS:
        if G.has_node(src) and G.has_node(dst):
            G.add_edge(src, dst)

    # Layout - hierarchical by subsystem
    pos = {
        # Sensory input (left)
        'LGN': (0, 0), 'VPL': (0, 2), 'MGN': (0, -2),
        # Primary sensory
        'V1': (1.5, 0), 'S1': (1.5, 2), 'A1': (1.5, -2),
        # Secondary sensory
        'V2': (3, 0), 'S2': (3, 2), 'Gustatory': (3, -1), 'Piriform': (3, -3),
        # Higher visual
        'V4': (4, -0.5), 'IT': (5, -0.5), 'MT': (3, 1), 'PPC': (4, 1.5),
        # Language
        'Wernicke': (5, -2.5), 'Broca': (6.5, -2.5),
        # Executive
        'dlPFC': (6, 0.5), 'OFC': (6, -1), 'vmPFC': (7, -0.5),
        'ACC': (6, 1.5), 'FEF': (5, 1.5),
        # DMN
        'PCC': (7.5, 1.5), 'TPJ': (5, 2.5), 'Insula': (4.5, -1.5),
        # Motor
        'PMC': (7.5, 0.5), 'SMA': (7.5, 0), 'M1': (9, 0.5),
        'MotorThal': (8.5, 0), 'BG': (7.5, -0.5),
        # Thalamic
        'Pulvinar': (2, 0.8), 'MD': (5.5, 0), 'VA': (7, 0),
        'LP': (3.5, 1.2), 'LD': (5, 2), 'ILN': (6.5, 2.5), 'CeM': (7, 2.5),
        'ATN': (6, 3),
        # Limbic (bottom)
        'Hippocampus': (4, -3.5), 'Amygdala': (3, -3.5),
        'SeptalNucleus': (3.5, -4.5), 'MammillaryBody': (5, -4),
        # Subcortical
        'VTA': (5.5, -3.5), 'Cerebellum': (10, -0.5),
        # Neuromod (top)
        'LC': (7, 3.5), 'DRN': (8, 3.5), 'NBM': (9, 3.5),
        # Hypothalamus (bottom-right)
        'Hypothalamus': (7, -3.5),
        # GW (center, prominent)
        'GW': (6, 3),
    }
    # Use available positions
    pos = {k: v for k, v in pos.items() if k in G.nodes}

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Node sizes proportional to neuron count
    sizes = [G.nodes[n].get('n_neurons', 50) * 3 for n in G.nodes]
    colors = [_get_color(n) for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                           node_color=colors, alpha=0.85, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight='bold')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#999999',
                           arrows=True, arrowsize=12, alpha=0.5,
                           connectionstyle='arc3,rad=0.1',
                           min_source_margin=15, min_target_margin=15)

    ax.set_title(f'WuYun Brain Connectivity\n'
                 f'{engine.num_regions()} regions | '
                 f'{engine.bus().num_projections()} projections',
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig


# =============================================================================
# Activity Bar Chart
# =============================================================================
def plot_activity_bars(spike_counts, figsize=(12, 5), save_path=None):
    """
    Plot bar chart of total spikes per region.

    Args:
        spike_counts: dict of {region_name: total_spikes}
    """
    names = list(spike_counts.keys())
    counts = list(spike_counts.values())
    colors = [_get_color(n) for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, counts, color=colors, alpha=0.85, edgecolor='white')

    ax.set_ylabel('Total Spikes')
    ax.set_title('WuYun Region Activity', fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    str(count), ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# =============================================================================
# Neuromodulator Timeline
# =============================================================================
def plot_neuromod_timeline(nm_history, figsize=(12, 4), save_path=None):
    """
    Plot neuromodulator levels over time.

    Args:
        nm_history: list of dicts with keys 'da', 'ne', 'sht', 'ach'
    """
    t = np.arange(len(nm_history))
    da  = [h['da']  for h in nm_history]
    ne  = [h['ne']  for h in nm_history]
    sht = [h['sht'] for h in nm_history]
    ach = [h['ach'] for h in nm_history]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, da,  label='DA',  color='#FF9800', linewidth=1.5)
    ax.plot(t, ne,  label='NE',  color='#2196F3', linewidth=1.5)
    ax.plot(t, sht, label='5-HT', color='#9C27B0', linewidth=1.5)
    ax.plot(t, ach, label='ACh', color='#4CAF50', linewidth=1.5)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Level')
    ax.set_title('Neuromodulator Dynamics', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# =============================================================================
# Quick Demo
# =============================================================================
def run_demo(duration=200, stim_duration=50, save_dir=None):
    """
    Run a quick demo: build 48-region brain, stimulate, visualize.

    Args:
        duration: total simulation steps
        stim_duration: how long to apply visual stimulus
        save_dir: directory to save figures (None = just show)
    """
    print("=== WuYun Brain Simulation Demo (48 regions) ===")

    # Build
    eng = pywuyun.SimulationEngine(10)
    eng.build_standard_brain()
    print(f"Built: {eng.num_regions()} regions, {eng.bus().num_projections()} projections")

    # Key regions to record (subset for clarity)
    region_names = ['LGN', 'V1', 'V2', 'IT', 'MT', 'PPC',
                    'dlPFC', 'ACC', 'OFC', 'BG', 'M1',
                    'A1', 'S1', 'Broca', 'Wernicke',
                    'Hippocampus', 'Amygdala', 'GW']
    recorders = {}
    for name in region_names:
        r = eng.find_region(name)
        if r:
            recorders[name] = pywuyun.SpikeRecorder()

    nm_history = []
    gw_history = []
    lgn = eng.find_region('LGN')

    # Try to get GW and Hypothalamus references
    gw = eng.find_region('GW')
    hypo = eng.find_region('Hypothalamus')

    # Run simulation
    print(f"Running {duration} steps (stim: 0-{stim_duration})...")
    for t in range(duration):
        if t < stim_duration:
            lgn.inject_external([35.0] * lgn.n_neurons())

        eng.step()

        for name, rec in recorders.items():
            rec.record(eng.find_region(name), t)

        # Record neuromod from V1
        v1 = eng.find_region('V1')
        nm = v1.neuromod().current()
        nm_history.append({'da': nm.da, 'ne': nm.ne, 'sht': nm.sht, 'ach': nm.ach})

        # Record GW state
        if gw:
            gw_history.append({
                'ignited': gw.is_ignited(),
                'salience': gw.winning_salience(),
                'content': gw.conscious_content_name(),
            })

    # Collect spike counts
    spike_counts = {name: rec.total_spikes() for name, rec in recorders.items()}
    active = {k: v for k, v in spike_counts.items() if v > 0}
    print(f"Activity: {active}")

    # GW summary
    if gw:
        print(f"GW: {gw.ignition_count()} ignitions, content='{gw.conscious_content_name()}'")
    if hypo:
        print(f"Hypothalamus: wake={hypo.wake_level():.2f}, stress={hypo.stress_output():.2f}")

    # Visualize
    prefix = save_dir + '/' if save_dir else None
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    fig1 = plot_raster(eng, recorders, duration,
                       save_path=f'{prefix}raster.png' if prefix else None)
    fig2 = plot_connectivity(eng,
                             save_path=f'{prefix}connectivity.png' if prefix else None)
    fig3 = plot_activity_bars(spike_counts,
                              save_path=f'{prefix}activity.png' if prefix else None)
    fig4 = plot_neuromod_timeline(nm_history,
                                  save_path=f'{prefix}neuromod.png' if prefix else None)

    if not save_dir:
        plt.show()

    print("=== Demo Complete ===")
    return eng, recorders, spike_counts


if __name__ == '__main__':
    save = 'output' if '--save' in sys.argv else None
    run_demo(save_dir=save)
