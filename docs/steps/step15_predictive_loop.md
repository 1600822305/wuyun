# Step 15 系列: 预测编码闭环 + 环境扩展 + 皮层巩固尝试

> 日期: 2026-02-08
> 状态: ✅ 完成

---

## Step 15: 预测编码基础设施 — dlPFC→V1 反馈通路

**目标**: dlPFC→V1 顶下反馈, 提升 DA-STDP 信用分配精度

**新增**:
- dlPFC→V1 投射 (delay=3, enable_predictive_coding 配置开关)
- 拓扑反馈映射 (比例映射, 窄 fan-out=3)
- 促进模式 (Bastos et al. 2012, 替代经典抑制性预测误差)
- AgentConfig::enable_predictive_coding (默认 false)

**5 轮调优**:

| 版本 | 模式 | 增益 | improvement |
|------|------|:----:|:-----------:|
| Step 14 基线 | 无PC | — | **+0.120** |
| v1 抑制性 | suppressive | -0.5 | -0.030 |
| v3 拓扑抑制 | suppressive | -0.12 | -0.112 |
| **v4 促进** | **facilitative** | **+0.3** | **+0.022** |
| v5 弱促进 | facilitative | +0.1 | -0.161 |

**结论**: 3×3 视野太小, PC 无冗余可压缩。默认禁用, 保留基础设施。

---

## Step 15-B: 环境扩展 + 大环境 PC 验证

**环境扩展**: GridWorldConfig::vision_radius (1=3×3, 2=5×5, 3=7×7), 自动缩放 LGN/V1/dlPFC

**大环境实验** (15×15 grid, 5×5 vision):

| 配置 | improvement | 5k danger |
|------|:-----------:|:---------:|
| No PC | -0.279 | 37 |
| **PC ON** | **-0.158** | **22** |
| **PC 优势** | **+0.121** | **-15** |

**关键发现**: PC 在大环境提供 +0.121 improvement 优势, 降低 40% danger!
与小环境 (3×3) 完全相反 — 预测编码效果与环境复杂度正相关。

**默认策略**: 小环境禁用, 视野 ≥ 5×5 启用。

---

## Step 15-C: 皮层巩固尝试 (Awake SWR → 皮层 STDP)

**目标**: SWR 重放同时巩固 V1→dlPFC 皮层表征 (学习回路第⑨步)

**实现**:
- SpikeSnapshot::sensory_events — V1 spikes 录制
- CorticalRegion::replay_cortical_step() — 轻量回放步
- capture_dlpfc_spikes() 同时录制 V1

**实验结果**:

| 方案 | improvement | 问题 |
|------|:-----------:|------|
| **BG-only 基线** | **+0.120** | — |
| replay_cortical_step | +0.034 (-72%) | LTD 主导 |
| PSP priming only | +0.053 (-56%) | 残留污染 |

**结论**: Awake SWR 皮层巩固不可行。皮层表征巩固应发生在 NREM 睡眠。
保留基础设施 (V1 录制 + replay_cortical_step), 未来 NREM 巩固直接可用。

29/29 CTest, 基线恢复 improvement +0.120。
