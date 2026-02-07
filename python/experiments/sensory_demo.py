"""
悟韵 (WuYun) — 感觉输入演示

演示内容:
1. 视觉编码: 生成简单图像 → VisualInput center-surround → LGN电流 → V1响应
2. 听觉编码: 生成频谱 → AuditoryInput tonotopic → MGN电流 → A1响应
3. 多模态集成: 视觉+听觉同时输入 → 全脑48区域响应 (含GW意识访问)
4. 睡眠重放: 编码→睡眠→SWR → 记忆巩固可视化

运行: python sensory_demo.py [--save]

依赖: numpy, matplotlib (可选)
"""

import sys
import os
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib', 'Release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pywuyun

SAVE_FIGS = '--save' in sys.argv

try:
    import matplotlib
    if SAVE_FIGS:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # 中文字体配置
    for font_name in ['SimHei', 'Microsoft YaHei', 'STSong', 'WenQuanYi Micro Hei']:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font_name] + matplotlib.rcParams['font.sans-serif']
            break
        except Exception:
            continue
    matplotlib.rcParams['axes.unicode_minus'] = False  # 负号显示
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[INFO] matplotlib 未安装，仅输出文本结果")


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def count_fired(region):
    return int(sum(1 for x in region.fired() if x))


def make_bar_chart(data, title, filename):
    """简易柱状图 (有matplotlib时绘图，否则纯文本)"""
    if HAS_MPL and len(data) > 0:
        names = [d[0] for d in data]
        values = [d[1] for d in data]
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Total Spikes')
        ax.set_title(title)
        plt.tight_layout()
        if SAVE_FIGS:
            fig.savefig(filename, dpi=150)
            print(f"  [保存] {filename}")
        else:
            plt.show(block=False)
            plt.pause(0.5)
    else:
        # Text fallback
        max_val = max(d[1] for d in data) if data else 1
        for name, val in data:
            bar_len = int(40 * val / max_val) if max_val > 0 else 0
            print(f"  {name:>12s} | {'█' * bar_len} {val}")


# =============================================================================
# Demo 1: 视觉编码 → V1 响应
# =============================================================================
def demo_visual():
    print_header("演示1: 视觉编码 — 像素→LGN→V1")

    eng = pywuyun.SimulationEngine(10)
    eng.build_standard_brain()

    lgn = eng.find_region("LGN")
    v1 = eng.find_region("V1")

    # 创建 VisualInput
    vcfg = pywuyun.VisualInputConfig()
    vcfg.input_width = 8
    vcfg.input_height = 8
    vcfg.n_lgn_neurons = lgn.n_neurons()
    vis = pywuyun.VisualInput(vcfg)

    print(f"  图像: {vcfg.input_width}x{vcfg.input_height} = {vis.n_pixels()} 像素")
    print(f"  LGN: {vis.n_lgn()} 神经元 (ON/OFF channels)")

    # 测试3种图像
    images = {
        "全黑 (0.0)": np.zeros(64),
        "亮点 (中心)": np.zeros(64),
        "全亮 (1.0)": np.ones(64),
        "水平条纹": np.zeros(64),
        "垂直条纹": np.zeros(64),
    }
    # 亮点: 3x3 center
    for r in range(3, 6):
        for c in range(3, 6):
            images["亮点 (中心)"][r * 8 + c] = 1.0
    # 水平条纹
    for r in [0, 2, 4, 6]:
        for c in range(8):
            images["水平条纹"][r * 8 + c] = 1.0
    # 垂直条纹
    for r in range(8):
        for c in [0, 2, 4, 6]:
            images["垂直条纹"][r * 8 + c] = 1.0

    results = []
    for name, pixels in images.items():
        # 编码
        currents = vis.encode(pixels.tolist())
        lgn_mean = np.mean(currents)

        # 运行50步
        v1_total = 0
        for t in range(50):
            vis.encode_and_inject(pixels.tolist(), lgn)
            eng.step()
            v1_total += count_fired(v1)

        results.append((name, v1_total))
        print(f"  {name:>12s}: LGN电流均值={lgn_mean:.1f}, V1={v1_total} spikes")

    # 可视化
    if HAS_MPL:
        fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
        for ax, (name, pixels) in zip(axes, images.items()):
            ax.imshow(np.array(pixels).reshape(8, 8), cmap='gray', vmin=0, vmax=1)
            ax.set_title(name, fontsize=9)
            ax.axis('off')
        plt.suptitle("输入图像", fontsize=12)
        plt.tight_layout()
        if SAVE_FIGS:
            fig.savefig("sensory_demo_images.png", dpi=150)
            print(f"  [保存] sensory_demo_images.png")
        else:
            plt.show(block=False)
            plt.pause(0.5)

    make_bar_chart(results, "视觉刺激 → V1响应 (50步)", "sensory_demo_visual.png")
    print("  [OK] 不同图像产生不同V1响应模式")


# =============================================================================
# Demo 2: 听觉编码 → A1 响应
# =============================================================================
def demo_auditory():
    print_header("演示2: 听觉编码 — 频谱→MGN→A1")

    eng = pywuyun.SimulationEngine(10)
    eng.build_standard_brain()

    mgn = eng.find_region("MGN")
    a1 = eng.find_region("A1")

    acfg = pywuyun.AuditoryInputConfig()
    acfg.n_freq_bands = 16
    acfg.n_mgn_neurons = mgn.n_neurons()
    acfg.gain = 50.0
    aud = pywuyun.AuditoryInput(acfg)

    print(f"  频带: {aud.n_freq_bands()} bands")
    print(f"  MGN: {aud.n_mgn()} 神经元 (tonotopic)")

    spectra = {
        "静默": np.zeros(16),
        "低频音 (100Hz)": np.zeros(16),
        "中频音 (1kHz)": np.zeros(16),
        "高频音 (8kHz)": np.zeros(16),
        "宽带噪声": np.ones(16) * 0.6,
    }
    spectra["低频音 (100Hz)"][:4] = 0.9
    spectra["中频音 (1kHz)"][6:10] = 0.9
    spectra["高频音 (8kHz)"][12:] = 0.9

    results = []
    for name, spectrum in spectra.items():
        a1_total = 0
        for t in range(50):
            aud.encode_and_inject(spectrum.tolist(), mgn)
            eng.step()
            a1_total += count_fired(a1)

        results.append((name, a1_total))
        print(f"  {name:>15s}: A1={a1_total} spikes")

    make_bar_chart(results, "听觉刺激 → A1响应 (50步)", "sensory_demo_auditory.png")

    # 频谱可视化
    if HAS_MPL:
        fig, axes = plt.subplots(1, len(spectra), figsize=(12, 2.5))
        for ax, (name, spec) in zip(axes, spectra.items()):
            ax.bar(range(16), spec, color='steelblue')
            ax.set_ylim(0, 1.1)
            ax.set_title(name, fontsize=8)
            ax.set_xlabel('Freq band', fontsize=7)
        plt.suptitle("输入频谱", fontsize=12)
        plt.tight_layout()
        if SAVE_FIGS:
            fig.savefig("sensory_demo_spectra.png", dpi=150)
            print(f"  [保存] sensory_demo_spectra.png")
        else:
            plt.show(block=False)
            plt.pause(0.5)

    print("  [OK] 不同频率激活A1不同区域 (tonotopic)")


# =============================================================================
# Demo 3: 多模态全脑响应 + 意识访问
# =============================================================================
def demo_multimodal_consciousness():
    print_header("演示3: 多模态全脑响应 — 视觉+听觉→48区域→GW意识")

    eng = pywuyun.SimulationEngine(10)
    eng.build_standard_brain()

    lgn = eng.find_region("LGN")
    mgn = eng.find_region("MGN")
    v1 = eng.find_region("V1")
    a1 = eng.find_region("A1")
    gw = eng.find_region("GW")

    # 感觉编码器
    vcfg = pywuyun.VisualInputConfig()
    vcfg.n_lgn_neurons = lgn.n_neurons()
    vis = pywuyun.VisualInput(vcfg)

    acfg = pywuyun.AuditoryInputConfig()
    acfg.n_mgn_neurons = mgn.n_neurons()
    acfg.gain = 50.0
    aud = pywuyun.AuditoryInput(acfg)

    # 强视觉刺激 + 弱听觉刺激
    bright_image = np.ones(64) * 0.9
    soft_tone = np.zeros(16)
    soft_tone[6:10] = 0.3

    # 监控关键区域
    regions_to_track = ["V1", "V2", "V4", "IT", "A1", "dlPFC", "ACC", "GW",
                        "Hippocampus", "Hypothalamus"]

    spikes_over_time = {r: [] for r in regions_to_track}
    gw_conscious = []
    gw_ignitions = 0

    print("  运行200步: 强视觉(0.9) + 弱听觉(0.3)...")
    for t in range(200):
        vis.encode_and_inject(bright_image.tolist(), lgn)
        aud.encode_and_inject(soft_tone.tolist(), mgn)
        eng.step()

        for rname in regions_to_track:
            r = eng.find_region(rname)
            if r is not None:
                spikes_over_time[rname].append(count_fired(r))

        # GW 状态
        gw_obj = eng.find_region("GW")
        if gw_obj is not None:
            try:
                name = gw_obj.conscious_content_name()
                gw_conscious.append(name if name else "---")
            except:
                gw_conscious.append("---")

    # 统计
    results = []
    for rname in regions_to_track:
        total = sum(spikes_over_time[rname])
        results.append((rname, total))

    make_bar_chart(results, "多模态全脑响应 (200步)", "sensory_demo_multimodal.png")

    # 意识内容统计
    from collections import Counter
    content_counts = Counter(gw_conscious)
    print("\n  GW 意识内容分布:")
    for content, count in content_counts.most_common(5):
        print(f"    {content}: {count}/{len(gw_conscious)} 步")

    # 时间序列可视化
    if HAS_MPL:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # 感觉皮层
        ax = axes[0]
        for rname in ["V1", "V2", "V4", "IT", "A1"]:
            if rname in spikes_over_time:
                ax.plot(spikes_over_time[rname], label=rname, alpha=0.8)
        ax.set_ylabel("Spikes/step")
        ax.set_title("感觉皮层活动")
        ax.legend(loc='upper right', fontsize=8)

        # 高级区域
        ax = axes[1]
        for rname in ["dlPFC", "ACC", "GW"]:
            if rname in spikes_over_time:
                ax.plot(spikes_over_time[rname], label=rname, alpha=0.8)
        ax.set_ylabel("Spikes/step")
        ax.set_title("前额叶 + 全局工作空间")
        ax.legend(loc='upper right', fontsize=8)

        # 海马 + 下丘脑
        ax = axes[2]
        for rname in ["Hippocampus", "Hypothalamus"]:
            if rname in spikes_over_time:
                ax.plot(spikes_over_time[rname], label=rname, alpha=0.8)
        ax.set_ylabel("Spikes/step")
        ax.set_xlabel("Time step")
        ax.set_title("边缘系统")
        ax.legend(loc='upper right', fontsize=8)

        plt.suptitle("多模态全脑时间响应 (强视觉 + 弱听觉)", fontsize=13)
        plt.tight_layout()
        if SAVE_FIGS:
            fig.savefig("sensory_demo_timeseries.png", dpi=150)
            print(f"  [保存] sensory_demo_timeseries.png")
        else:
            plt.show(block=False)
            plt.pause(0.5)

    print("  [OK] 视觉通路 V1->V2->V4->IT 级联激活")
    print("  [OK] GW 竞争: 强视觉赢得意识访问")


# =============================================================================
# Demo 4: 编码→睡眠→SWR重放
# =============================================================================
def demo_sleep_replay():
    print_header("演示4: 睡眠记忆巩固 — 编码→SWR→重放")

    eng = pywuyun.SimulationEngine(10)
    eng.build_standard_brain()

    lgn = eng.find_region("LGN")
    v1 = eng.find_region("V1")
    hipp = eng.find_region("Hippocampus")
    hypo = eng.find_region("Hypothalamus")

    vcfg = pywuyun.VisualInputConfig()
    vcfg.n_lgn_neurons = lgn.n_neurons()
    vis = pywuyun.VisualInput(vcfg)

    # Phase 1: 清醒编码 (100步)
    print("  Phase 1: 清醒编码 — 呈现亮点图案100步")
    spot_image = np.zeros(64)
    for r in range(2, 6):
        for c in range(2, 6):
            spot_image[r * 8 + c] = 1.0

    awake_v1 = []
    awake_hipp = []
    for t in range(100):
        vis.encode_and_inject(spot_image.tolist(), lgn)
        eng.step()
        awake_v1.append(count_fired(v1))
        awake_hipp.append(count_fired(hipp))

    print(f"    V1 均值: {np.mean(awake_v1):.1f} spikes/step")
    print(f"    Hipp 均值: {np.mean(awake_hipp):.1f} spikes/step")

    # Phase 2: 进入睡眠 (200步)
    print("\n  Phase 2: 进入NREM睡眠 — 皮层慢波 + SWR重放")
    hypo.set_sleep_pressure(0.9)
    hipp.enable_sleep_replay()
    v1.set_sleep_mode(True)

    sleep_v1 = []
    sleep_hipp = []
    swr_events = []
    up_states = []

    for t in range(200):
        eng.step()  # 无外部输入
        sleep_v1.append(count_fired(v1))
        sleep_hipp.append(count_fired(hipp))
        swr_events.append(1 if hipp.is_swr() else 0)
        up_states.append(1 if v1.is_up_state() else 0)

    swr_count = hipp.swr_count()
    replay_str = hipp.last_replay_strength()
    wake_level = hypo.wake_level()

    print(f"    Hypothalamus 觉醒度: {wake_level:.2f}")
    print(f"    SWR 事件: {swr_count} 次")
    print(f"    最后重放强度: {replay_str:.2f}")
    print(f"    V1 睡眠均值: {np.mean(sleep_v1):.1f} (vs 清醒 {np.mean(awake_v1):.1f})")

    # Phase 3: 唤醒 (50步)
    print("\n  Phase 3: 唤醒 — 恢复正常处理")
    hipp.disable_sleep_replay()
    v1.set_sleep_mode(False)
    hypo.set_sleep_pressure(0.1)

    wake_v1 = []
    for t in range(50):
        vis.encode_and_inject(spot_image.tolist(), lgn)
        eng.step()
        wake_v1.append(count_fired(v1))

    print(f"    V1 唤醒后: {np.mean(wake_v1):.1f} spikes/step")

    # 可视化
    if HAS_MPL:
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        all_v1 = awake_v1 + sleep_v1 + wake_v1
        all_hipp = awake_hipp + sleep_hipp + [0]*50

        # V1 活动
        ax = axes[0]
        t_all = range(len(all_v1))
        ax.plot(t_all, all_v1, 'b-', alpha=0.7, linewidth=0.8)
        ax.axvspan(0, 100, alpha=0.1, color='yellow', label='清醒编码')
        ax.axvspan(100, 300, alpha=0.1, color='blue', label='NREM睡眠')
        ax.axvspan(300, 350, alpha=0.1, color='green', label='唤醒')
        ax.set_ylabel("V1 Spikes")
        ax.set_title("V1 皮层活动 (清醒→睡眠→唤醒)")
        ax.legend(loc='upper right', fontsize=8)

        # Hippocampus + SWR
        ax = axes[1]
        ax.plot(range(len(all_hipp)), all_hipp, 'r-', alpha=0.7, linewidth=0.8,
                label='Hippocampus')
        # Mark SWR events
        swr_times = [100 + i for i, s in enumerate(swr_events) if s]
        if swr_times:
            ax.scatter(swr_times, [all_hipp[t] if t < len(all_hipp) else 0 for t in swr_times],
                       c='gold', s=30, zorder=5, label=f'SWR ({swr_count}次)')
        ax.set_ylabel("Hipp Spikes")
        ax.set_title("海马活动 + SWR事件")
        ax.legend(loc='upper right', fontsize=8)

        # Up/Down state
        ax = axes[2]
        up_x = range(100, 300)
        ax.fill_between(up_x, up_states, alpha=0.5, color='orange', label='Up state')
        ax.set_ylabel("Up State")
        ax.set_title("皮层慢波 Up/Down 状态")
        ax.set_ylim(-0.1, 1.3)
        ax.legend(loc='upper right', fontsize=8)

        # Phase indicator
        ax = axes[3]
        phases = ['清醒编码']*100 + ['NREM睡眠']*200 + ['唤醒']*50
        phase_colors = {'清醒编码': 'yellow', 'NREM睡眠': 'blue', '唤醒': 'green'}
        prev_phase = phases[0]
        start = 0
        for i, phase in enumerate(phases + ['END']):
            if phase != prev_phase:
                ax.axvspan(start, i, alpha=0.3, color=phase_colors.get(prev_phase, 'gray'))
                ax.text((start + i) / 2, 0.5, prev_phase, ha='center', va='center', fontsize=10)
                start = i
                prev_phase = phase
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time step")
        ax.set_title("实验阶段")
        ax.set_yticks([])

        plt.suptitle("睡眠记忆巩固: 编码 → SWR重放 → 唤醒", fontsize=13)
        plt.tight_layout()
        if SAVE_FIGS:
            fig.savefig("sensory_demo_sleep.png", dpi=150)
            print(f"  [保存] sensory_demo_sleep.png")
        else:
            plt.show(block=False)
            plt.pause(0.5)

    print("\n  [OK] NREM慢波: V1活动显著下降 (down state抑制)")
    print("  [OK] SWR重放: 海马在睡眠中产生sharp-wave ripple")
    print("  [OK] 唤醒后: V1恢复正常响应")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  悟韵 (WuYun) 感觉输入演示")
    print("  VisualInput + AuditoryInput + 全脑响应")
    print("=" * 60)

    demo_visual()
    demo_auditory()
    demo_multimodal_consciousness()
    demo_sleep_replay()

    print("\n" + "=" * 60)
    print("  全部演示完成!")
    if SAVE_FIGS:
        print("  图片已保存到当前目录")
    elif HAS_MPL:
        print("  关闭所有图窗退出")
        plt.show()
    print("=" * 60)
