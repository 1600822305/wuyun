# Step 40: Phase 1 三区域扩展 — NAcc + SNc + SC

> 日期: 2026-02-09
> 状态: ✅ 完成
> 核心突破: D1 47→58 (+23%), elig 63.5→114.2 (1.8×), weight range 0.0464→0.1080 (2.3×)

## 设计动机

当前系统瓶颈分析后确定三个 P0 优先级区域，每个直接改善闭环性能:

1. **NAcc**: BG 把运动选择和奖赏动机混在一起 → 分离 ventral/dorsal
2. **SNc**: 只有 VTA 一个 DA 源，phasic 波动导致已学行为退化 → 加 tonic 稳定
3. **SC**: 危险物体要等 14 步皮层处理 → 皮层下快通道

## ① NAcc 伏隔核 (Mogenson 1980: limbic-motor interface)

**架构**: 16 神经元 = 4 Core D1 + 4 Core D2 + 4 Shell + 4 VP

```
VTA → NAcc Core D1/D2 (mesolimbic DA: 奖赏→趋近动机)
Amygdala → NAcc Core D2 (恐惧→回避动机)
Hippocampus → NAcc Shell (空间上下文→情境动机)
IT → NAcc Core (物体身份→"看到食物"→趋近)
NAcc Core D1 → VP (抑制) → 释放运动活力
```

**输出**: `motivation_output()` 调制 BG motor vigor (noise_scale × [0.7, 1.3])
- motivation > 0: D1 dominant → approach → 更多运动
- motivation < 0: D2 dominant → avoidance → 减少运动

**新增文件**: `nucleus_accumbens.h/cpp`

## ② SNc 黑质致密部 (Yin & Knowlton 2006: habit learning)

**架构**: 4 DA 神经元, tonic-dominant

```
VTA: phasic RPE → 新行为学习 (快速波动)
SNc: tonic DA → 已学习惯维持 (缓慢适应)

BG DA = 70% VTA + 30% SNc
```

**核心机制**:
- `tonic_baseline_` 缓慢跟踪 `avg_reward(200)` (habit_lr=0.002)
- D1 反馈: 活跃的 D1 MSN → SNc 维持信号 (正反馈环)
- 当习惯形成: SNc tonic 升高 → VTA phasic 波动占比降低 → 已学权重更稳定
- 抗灾难性遗忘: SNc 的 30% 贡献 buffer 了 VTA 单次 RPE 的冲击

**新增文件**: `snc_da.h/cpp`

## ③ SC 上丘 (Krauzlis 2013: subcortical saliency)

**架构**: 8 神经元 = 4 浅层(视觉地图) + 4 深层(多模态输出)

```
LGN → SC 浅层 (delay=1, 快速视觉输入)
V1 → SC 深层 (delay=2, 皮层反馈)
SC 深层 → BG (delay=1, 显著性驱动, 补充丘脑纹状体通路)
```

**核心机制**:
- 浅层: 低阈值(-48mV), 快膜常数(8ms) → 比皮层(20ms)快 2.5×
- 深层: 汇聚浅层 + 皮层反馈 → 多模态显著性
- `saliency_output()`: 输入变化检测 (onset/offset, 不响应静态场景)

**新增文件**: `superior_colliculus.h/cpp`

## 投射连接

```
新增投射:
  VTA → NAcc (delay=2)     # mesolimbic DA
  Hippocampus → NAcc (d=3) # 空间上下文
  Amygdala → NAcc (d=2)    # 情绪价值
  IT → NAcc (d=2)          # 物体身份
  M1 → SNc (d=2)           # 运动反馈
  LGN → SC (d=1)           # 快速视觉
  V1 → SC (d=2)            # 皮层反馈
  SC → BG (d=1)            # 显著性驱动
```

## 修改文件

新增:
- `src/region/subcortical/nucleus_accumbens.h/cpp`
- `src/region/neuromod/snc_da.h/cpp`
- `src/region/subcortical/superior_colliculus.h/cpp`

修改:
- `src/CMakeLists.txt`: 3 个新 cpp
- `src/engine/closed_loop_agent.h`: 3 个 config flags + 3 个 cached pointers
- `src/engine/closed_loop_agent.cpp`: build_brain() 创建 + 投射 + agent_step() 集成

## 效果对比

| 指标 | Step 39 | Step 40 | 变化 |
|------|---------|---------|------|
| D1 fires (50步) | 47 | **58** | +23% |
| D2 fires (50步) | 51 | **59** | +16% |
| Max eligibility | 63.5 | **114.2** | **1.8×** |
| Weight range | 0.0464 | **0.1080** | **2.3×** |
| Cortical events/10步 | 358 | **616** | **1.7×** |

从 Step 37 起的累计提升: weight range 0.0045 → 0.1080 = **24×**

## 反作弊修复 (00_design_principles.md §6 检查项 #7)

三处违反"信号传递走 SpikeBus"原则的直接函数调用被修复:

| 违规代码 | 问题 | 修复方式 |
|---------|------|---------|
| `snc_->inject_reward_history(avg_reward(200))` | Agent 全知视角直接注入奖励历史 | SNc 从 BG→SNc SpikeBus 接收 D1 脉冲，自行跟踪 spike rate 适应 tonic |
| `snc_->inject_d1_activity(d1_rate)` | 手动计算 D1 发放率，绕过 SpikeBus | BG→SNc 投射 (delay=2)，D1 反馈通过脉冲到达 |
| `nacc_->motivation_output()` → noise_scale | 标量直接调制探索噪声 | NAcc→M1 投射 (delay=2)，VP 脉冲通过 SpikeBus 调制 M1 活动 |

新增 2 条投射: BG→SNc (d=2), NAcc→M1 (d=2)。`set_da_level()` 保留 (ModulationBus 体积传递，非 SpikeBus 违规)。

## 系统状态

```
57区域 · ~188闭环神经元 · ~123投射
学习链路 13/13 (新增: NAcc/SNc/SC)
D1 58/50步, Max elig 114.2, Weight range 0.1080
30/30 CTest 通过
```
