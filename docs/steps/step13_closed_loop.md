# Step 13 系列: 闭环 Agent + GridWorld + 学习调优

> 日期: 2026-02-07 ~ 2026-02-08
> 状态: ✅ 完成

此文档涵盖 Step 13-A 到 13-D+E 的全部闭环系统构建和调优过程。

---

## Step 13-A: 稳态可塑性集成

**目标**: 解决 scale-up 时 E/I 失衡导致的网络崩溃问题。

- SynapticScaler: 4个兴奋性群体各一个缩放器, 只缩放前馈 AMPA, 不缩放循环/抑制
- CorticalColumn: L6→L4, L4→L2/3, L2/3→L5, L5→L6
- Hippocampus: DG/CA3/CA1 各一个, 不缩放 CA3→CA3 自联想
- **Scale=3 WM 从 0 恢复到 0.425** — 稳态成功解决 E/I 失衡
- 27/27 CTest, 168 测试

---

## Step 13-B: 闭环 Agent + GridWorld

**目标**: 感知→决策→行动→感知完整闭环。

**GridWorld**: 10x10 网格, food(+1)/danger(-1), 3x3 视觉观测
**ClosedLoopAgent**: LGN+V1+dlPFC+M1+BG+MotorThal+VTA+Hipp, M1 L5 分 4 组 WTA 解码

**关键修复**:
- VTA reward_psp_ 缓冲 (PSP衰减=0.85), RPE→电流 50→200
- M1 探索噪声 bias+jitter, 77.5% 非 STAY
- 28/28 CTest, 7/7 测试

---

## Step 13-B+: DA-STDP 闭环学习修复

**5 个根因**: 时序断裂 / MSN 不发放 / D1/D2 不对称 / 缺动作特异性 / BG 输出无效

**5 个修复**:
1. BG Eligibility Traces (Izhikevich 2007): co-activation 递增, DA 到时乘 trace
2. Agent Step 时序重构: Phase A(奖励)→B(观测)→C(动作)
3. MSN Up-State Drive: +25 接近阈值, DA 对称调制
4. D1 动作子组 + Action-Specific Boost
5. Combined Action Decoding: M1 L5 + BG D1 combined score

**结果**: Learner 减少 51% 危险碰撞, D1 发放 31/50步 (原 0)

---

## Step 13-B++: 闭环学习调优 v3

**3 个问题修复**:
1. Eligibility Trace Clamp (max=50, 防 Δw 爆炸)
2. 学习率/衰减调优 (lr 0.02→0.005, decay 0.0002→0.001, elig_decay 0.95→0.98)
3. NE 调制探索/利用 (动态 noise_scale, floor=0.7 防冻结)

**Motor Efference Copy 实验** (探索性, 已回退): 短期好但正反馈长期失控

**最终**: improvement +0.191, learner advantage +0.017, 29/29 CTest

---

## Step 13-C: 视觉通路修复 + 皮层 STDP + 拓扑映射

**诊断**: V1→dlPFC→BG 视觉通路完全断裂 (LGN 从不发放, 空间信息被打散)

**4 个修复**:
1. LGN 增益 gain 45→200, baseline 3→5
2. 每步注入视觉 (原每 3 步)
3. V1→dlPFC 拓扑映射 (比例映射替代模取余)
4. V1 皮层 STDP, brain_steps 10→15

**结果**: food +110%, safety +63%, 但 learner advantage 消失 (dlPFC→BG 随机连接)

---

## Step 13-D+E: dlPFC→BG 拓扑映射 + 完整通路验证

**修复**:
1. dlPFC→BG 拓扑偏置 (p_same=0.60, p_other=0.05, 78% 匹配)
2. Motor efference copy 重新启用 (i≥10 标记)
3. Weight decay 加速 0.001→0.003

**完整拓扑通路**:
```
Grid → VisualInput → LGN → V1(STDP) → dlPFC(拓扑) → BG D1/D2(拓扑78%) → M1
                                                        + efference copy
```

**结果**: learner advantage 恢复正值 +0.0015, food +55%, 29/29 CTest

### 系统状态

```
48区域 · ~5528+神经元 · ~109投射 · 179测试 · 29 CTest suites
闭环Agent · GridWorld · DA-STDP · Eligibility Traces · 拓扑映射
V1在线STDP · NE探索/利用调制 · Motor Efference Copy
```
