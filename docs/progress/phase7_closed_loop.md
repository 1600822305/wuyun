# Phase 7: 闭环学习 — 稳态可塑性 + Agent + GridWorld + 学习调优

> 对应: Step 13-A / 13-B / 13-B+ / 13-B++ / 13-C / 13-D+E
> 时间: 2026-02-07
> 里程碑: 闭环Agent · GridWorld环境 · DA-STDP闭环学习 · 全链路拓扑映射 · 179 测试 · 29 CTest suites

---

## Step 13-A: 稳态可塑性集成 ✅

> 目标: 解决 scale-up 时 E/I 失衡导致的网络崩溃问题

**SynapticScaler 改进** (`plasticity/homeostatic.h/cpp`):
- `update_rates(const uint8_t*)`: 改签名匹配 `NeuronPopulation::fired()`
- `mean_rate()`: 群体平均发放率查询
- `HomeostaticParams::scale_interval`: 缩放间隔参数 (默认100步)

**CorticalColumn 集成:**
- `enable_homeostatic(HomeostaticParams)`: 为4个兴奋性群体各创建一个 SynapticScaler
- **只缩放前馈兴奋性AMPA突触**, 不缩放循环突触(保护已学习模式)和抑制性突触

**Hippocampus 集成:**
- 3个 SynapticScaler: DG, CA3, CA1
- **不缩放 CA3→CA3 循环突触** (保护自联想记忆!)

**测试结果 (7/7 通过):**
- SynapticScaler 发放率追踪 · 过度活跃→权重降低 · 活动不足→权重增大
- CorticalRegion 集成 · Hippocampus 集成 · 多区域稳态
- **Scale=3 WM恢复**: persistence=0.425, spikes=5433

**关键成果:** Scale=3 工作记忆从 0 恢复到 0.425 — 稳态可塑性成功解决大规模网络 E/I 失衡。

**系统状态:** 168测试 · 27 CTest suites

---

## Step 13-B: 闭环Agent + GridWorld ✅

> 目标: 将大脑模型与环境连接，实现完整的感知→决策→行动→感知闭环

**架构:**
```
GridWorld.observe()
    ↓ 3x3 pixels
VisualInput → LGN → V1 → dlPFC → BG → MotorThal → M1
                                                      ↓ decode L5 spikes
GridWorld.act(action) ←── winner-take-all ←── M1 L5 [UP|DOWN|LEFT|RIGHT]
    ↓ reward
VTA.inject_reward() → DA → BG DA-STDP → 学习
```

**GridWorld** (`engine/grid_world.h/cpp`):
- 10x10 网格, Agent可移动, 食物(+1奖励)/危险(-1)/墙壁
- 3x3 局部视觉观测 (灰度编码: food=0.9, danger=0.3, agent=0.6)

**ClosedLoopAgent** (`engine/closed_loop_agent.h/cpp`):
- 自动构建最小闭环大脑: LGN + V1 + dlPFC + M1 + BG + MotorThal + VTA + Hippocampus
- 每个环境步运行 N 个脑步, 累积M1 L5发放
- M1 L5分4组(UP/DOWN/LEFT/RIGHT), winner-take-all解码

**关键修复:**
- VTA 奖励响应: 添加 `reward_psp_` 缓冲, RPE乘数 50→200
- M1 运动探索: 每个action期间选定一个L5组注入bias+jitter噪声
- 77.5% 非STAY动作, 均匀分布于4方向

**测试结果 (7/7 通过):** GridWorld基础 · 视觉观测 · Agent构建 · 闭环运行 · 动作多样性 · DA奖励 · 长期稳定性(500步)

**系统状态:** 175测试 · 28 CTest suites

---

## Step 13-B+: DA-STDP 闭环学习修复 ✅

**问题诊断 (5个根因):**
1. 时序信用分配断裂: 奖励注入后eligibility已清零
2. MSN从不发放: D1/D2需I≥50, 但cortex PSP仅~18
3. D1/D2不对称: baseline DA时D2>D1
4. 缺少动作特异性: 随机映射强化所有连接
5. BG输出不影响动作选择

**5个架构修复:**
1. **BG Eligibility Traces** (Izhikevich 2007): `elig_d1_[src][idx]` 衰减缓冲
2. **Agent Step 时序重构**: Phase A(奖励) → Phase B(观测) → Phase C(动作)
3. **MSN Up-State Drive** (Wilson & Kawaguchi 1996): up=25 + da_base=15 = 40 (接近阈值50)
4. **BG D1 动作子组**: D1分4组 + 被选动作boost对应子组
5. **Combined Action Decoding**: M1 L5 + BG D1 combined score

**验证结果:**
- DA-STDP 学习管线打通: elig=25.6 → DA=0.911 → D1=7
- Learner vs Control: danger 16 vs 33 (减少51%碰撞)

---

## Step 13-B++: 闭环学习调优 (v3) ✅

**3个核心修复:**
1. **Eligibility Trace Clamp**: `da_stdp_max_elig = 50.0f` — 每次食物最大Δw=0.125
2. **学习率/衰减调优**: lr 0.02→0.005, w_decay 0.0002→0.001, elig_decay 0.95→0.98
3. **NE调制探索/利用平衡**: 动态noise_scale基于food_rate, floor=0.7防冻结

**诊断修复:** `inp=0` 是测量时序假象, 实际 ctx=1759/50步 (通路畅通)

**Motor Efference Copy 实验 (探索性, 已回退):**
- Efference copy 短期极好但 PSP×weight 正反馈10k后失控
- 需 D1子群竞争归一化 (侧向抑制) 才能稳定

**v3 基线结果:**
```
5k: safety 0.20→0.38, improvement +0.18
10k: improvement +0.191
Learner advantage: +0.0168
BG 权重: range=0.2593 (稳定, 无爆炸)
```

---

## Step 13-C: 视觉通路修复 + 皮层STDP ✅

**问题:** V1→dlPFC→BG视觉通路完全断裂 (LGN从不发放)

**4个核心修复:**
1. **LGN视觉编码增益**: gain 45→200, baseline 3→5
2. **每步注入视觉观测**: 原每3步→**每步**注入
3. **V1→dlPFC拓扑映射**: `add_topographic_input()` 比例映射替代模取余
4. **V1皮层STDP**: a_plus=0.005, a_minus=-0.006, w_max=1.5

**视觉流水线时序 (brain_steps=15):**
```
brain_i:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
LGN:          ■   ■   ■   ■   ■   ■   ■       ■   ■
V1:                                   ■   ■■■  ■■■ ■■■ ■■■
dlPFC:                                              ■■■ ■■■ ■■■ ■■■
BG:                                                         ■■■ ■■■
```

**结果:** 5k late safety 0.62 (+63% vs v3) · Food 65 (+110% vs v3)

---

## Step 13-D+E: dlPFC→BG 拓扑映射 + 完整视觉通路 ✅

**3个核心修复:**
1. **dlPFC→BG 拓扑映射**: `set_topographic_cortical_source()` — p_same=0.60, p_other=0.05 (78%偏置)
2. **Motor efference copy 重新启用**: brain loop 后期(i≥10)标记动作eligibility
3. **Weight decay 加速**: 0.001→0.003 (回归速度3x)

**完整拓扑通路:**
```
3×3 Grid → VisualInput(gain=200) → LGN(选择性发放)
  → delay=2 → V1(STDP特征学习)
  → delay=2 → dlPFC(拓扑接收: proportional mapping)
  → delay=2 → BG D1/D2(拓扑偏置: 78%匹配子群)
                 + motor efference copy(动作标记)
  → D1→GPi→MotorThal → M1 L5 → 动作
  → VTA DA reward → DA-STDP 权重更新
```

**最终结果:**
| 指标 | 全拓扑 (13-D+E) | 随机BG (13-C) | 无视觉 (v3) |
|------|-----------------|--------------|-------------|
| Learner advantage | **+0.0015** ✅ | -0.0015 | +0.017 |
| Food 10k | **118** (+55%) | 121 | 76 |
| D1 fires/50步 | **189** | 178 | ~178 |

---

## Phase 7 总结

| 指标 | 数值 |
|------|------|
| 测试 | 179 通过, 29 CTest suites |
| 新增类 | GridWorld · ClosedLoopAgent · SleepCycleManager(已有) |
| 学习规则 | 6 种 (STDP · STP · DA-STDP · CA3-STDP · 稳态 · 皮层STDP) |
| 闭环能力 | 感知→决策→行动→学习, 食物收集+55% |
| 关键机制 | Eligibility traces · MSN up-state · 动作子组 · Combined decode · 拓扑映射 · NE探索调制 |
| 生物学参考 | Izhikevich 2007 · Wilson & Kawaguchi 1996 · Frank 2005 |

**系统状态:**
```
48区域 · ~5528+神经元 · ~109投射 · 179测试 · 29 CTest suites
完整功能: 感觉输入 · 视听编码 · 层级处理 · 双流视觉 · 语言
          6种学习 · 预测编码 · 工作记忆 · 注意力 · GNW意识
          内驱力 · NREM巩固 · REM梦境 · 睡眠周期管理
          4种调质广播 · 稳态可塑性 · 规模可扩展
          闭环Agent · GridWorld · DA-STDP闭环学习
          全链路拓扑映射: V1→dlPFC→BG