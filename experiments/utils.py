"""
实验辅助工具 — 统计/绘图/格式化

提供实验代码共用的工具函数，保持实验代码简洁。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple, Optional
from wuyun.spike.signal_types import SpikeType
from wuyun.circuit.cortical_column import CorticalColumn


# =============================================================================
# 时间窗口统计
# =============================================================================

def collect_window_stats(
    column: CorticalColumn,
    duration: int,
    ff_current: float,
    fb_current: float = 0.0,
    window_size: int = 100,
    start_time: int = 0,
) -> List[Dict]:
    """运行仿真并按时间窗口收集统计

    Args:
        column: 皮层柱实例
        duration: 仿真时长 (ms)
        ff_current: 前馈电流强度
        fb_current: 反馈电流强度 (默认 0 = 无反馈)
        window_size: 统计窗口大小 (ms)
        start_time: 起始时间步

    Returns:
        列表，每个元素是一个窗口的统计字典:
        {
            'window_start': int,
            'window_end': int,
            'l23_regular': int,
            'l23_burst': int,
            'l23_burst_ratio': float,
            'l5_regular': int,
            'l5_burst': int,
            'l6_spikes': int,
            'layer_spikes': {layer_id: int},
        }
    """
    windows = []
    current_window = _new_window_stats(start_time, start_time + window_size)

    for t in range(start_time, start_time + duration):
        # 注入输入
        column.inject_feedforward_current(ff_current)
        if fb_current > 0:
            column.inject_feedback_current(fb_current)

        # 仿真步进
        column.step(t)

        # 收集当前时间步的脉冲
        _accumulate_step(column, current_window)

        # 检查是否到达窗口边界
        if (t + 1 - start_time) % window_size == 0 and t > start_time:
            _finalize_window(current_window)
            windows.append(current_window)
            next_start = t + 1
            current_window = _new_window_stats(next_start, next_start + window_size)

    # 处理最后一个不完整窗口
    if current_window['l23_regular'] + current_window['l23_burst'] > 0:
        _finalize_window(current_window)
        windows.append(current_window)

    return windows


def _new_window_stats(start: int, end: int) -> Dict:
    """创建空白窗口统计"""
    return {
        'window_start': start,
        'window_end': end,
        'l23_regular': 0,
        'l23_burst': 0,
        'l23_burst_ratio': 0.0,
        'l5_regular': 0,
        'l5_burst': 0,
        'l6_spikes': 0,
        'layer_spikes': {4: 0, 23: 0, 5: 0, 6: 0},
    }


def _accumulate_step(column: CorticalColumn, window: Dict) -> None:
    """累积一个时间步的数据到窗口"""
    # L2/3 输出
    for neuron in column.layers[23].excitatory:
        st = neuron.current_spike_type
        if st == SpikeType.REGULAR:
            window['l23_regular'] += 1
        elif st.is_burst:
            window['l23_burst'] += 1

    # L5 输出
    if 5 in column.layers:
        for neuron in column.layers[5].excitatory:
            st = neuron.current_spike_type
            if st == SpikeType.REGULAR:
                window['l5_regular'] += 1
            elif st.is_burst:
                window['l5_burst'] += 1

    # L6 输出
    if 6 in column.layers:
        for neuron in column.layers[6].excitatory:
            st = neuron.current_spike_type
            if st.is_active:
                window['l6_spikes'] += 1

    # 各层发放计数
    for lid, layer in column.layers.items():
        spikes = layer.get_last_spikes()
        window['layer_spikes'][lid] += len(spikes)


def _finalize_window(window: Dict) -> None:
    """计算窗口的 burst 比率"""
    total = window['l23_regular'] + window['l23_burst']
    if total > 0:
        window['l23_burst_ratio'] = window['l23_burst'] / total
    else:
        window['l23_burst_ratio'] = 0.0


# =============================================================================
# 权重快照
# =============================================================================

def snapshot_weights(column: CorticalColumn) -> Dict[str, Dict]:
    """拍摄柱内所有突触的权重快照

    Returns:
        {
            'all': {'mean': float, 'std': float, 'min': float, 'max': float, 'count': int},
            'excitatory': {...},
            'inhibitory': {...},
            'ff_l4_l23': {...},    # L4→L23 前馈
            'ff_l23_l5': {...},    # L23→L5 前馈
            'ff_l5_l6': {...},     # L5→L6 前馈
            'fb_l6_l23': {...},    # L6→L23 反馈 (apical)
            'fb_l6_l5': {...},     # L6→L5 反馈 (apical)
            'inh_pv': {...},       # PV+→锥体 抑制
        }
    """
    import numpy as np
    from wuyun.spike.signal_types import CompartmentType, SynapseType

    categories = {
        'all': [],
        'excitatory': [],
        'inhibitory': [],
        'ff_l4_l23': [],
        'ff_l23_l5': [],
        'ff_l5_l6': [],
        'fb_l6_l23': [],
        'fb_l6_l5': [],
        'inh_pv': [],
    }

    # 神经元 ID → 层映射
    neuron_to_layer = {}
    for lid, layer in column.layers.items():
        for n in layer.neurons:
            neuron_to_layer[n.id] = lid

    for syn in column.synapses:
        w = syn.weight
        categories['all'].append(w)

        if syn.is_excitatory:
            categories['excitatory'].append(w)
        else:
            categories['inhibitory'].append(w)

        pre_layer = neuron_to_layer.get(syn.pre_id, -1)
        post_layer = neuron_to_layer.get(syn.post_id, -1)

        # 分类特定连接
        if syn.is_excitatory:
            if pre_layer == 4 and post_layer == 23:
                categories['ff_l4_l23'].append(w)
            elif pre_layer == 23 and post_layer == 5:
                categories['ff_l23_l5'].append(w)
            elif pre_layer == 5 and post_layer == 6:
                categories['ff_l5_l6'].append(w)
            elif pre_layer == 6 and post_layer == 23 and \
                    syn.target_compartment == CompartmentType.APICAL:
                categories['fb_l6_l23'].append(w)
            elif pre_layer == 6 and post_layer == 5 and \
                    syn.target_compartment == CompartmentType.APICAL:
                categories['fb_l6_l5'].append(w)
        else:
            if syn.synapse_type == SynapseType.GABA_A and \
                    syn.target_compartment == CompartmentType.SOMA:
                categories['inh_pv'].append(w)

    result = {}
    for name, weights in categories.items():
        if weights:
            arr = np.array(weights)
            result[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'count': len(weights),
            }
        else:
            result[name] = {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0,
            }

    return result


# =============================================================================
# 格式化打印
# =============================================================================

def print_header(title: str) -> None:
    """打印实验标题"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_window_table(windows: List[Dict]) -> None:
    """打印时间窗口统计表格"""
    print(f"\n  {'窗口(ms)':<15} {'L23 reg':>8} {'L23 bst':>8} "
          f"{'burst%':>8} {'L5 spk':>8} {'L6 spk':>8} "
          f"{'L4':>6} {'L23':>6} {'L5':>6} {'L6':>6}")
    print(f"  {'-' * 90}")
    for w in windows:
        ls = w['layer_spikes']
        print(f"  {w['window_start']:>4}-{w['window_end']:<8} "
              f"{w['l23_regular']:>8} {w['l23_burst']:>8} "
              f"{w['l23_burst_ratio']:>7.1%} "
              f"{w['l5_regular'] + w['l5_burst']:>8} "
              f"{w['l6_spikes']:>8} "
              f"{ls[4]:>6} {ls[23]:>6} {ls[5]:>6} {ls[6]:>6}")


def print_weight_table(snapshots: List[Tuple[int, Dict]], categories: List[str] = None) -> None:
    """打印权重演化表

    Args:
        snapshots: [(time_ms, snapshot_dict), ...]
        categories: 要显示的类别列表 (默认显示核心类别)
    """
    if categories is None:
        categories = ['ff_l4_l23', 'ff_l23_l5', 'ff_l5_l6',
                       'fb_l6_l23', 'fb_l6_l5', 'inh_pv']

    for cat in categories:
        print(f"\n  [{cat}]")
        print(f"  {'时间(ms)':<10} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'count':>6}")
        print(f"  {'-' * 52}")
        for t, snap in snapshots:
            s = snap.get(cat, {})
            if s.get('count', 0) > 0:
                print(f"  {t:<10} {s['mean']:>8.4f} {s['std']:>8.4f} "
                      f"{s['min']:>8.4f} {s['max']:>8.4f} {s['count']:>6}")
            else:
                print(f"  {t:<10} {'(无连接)':>40}")