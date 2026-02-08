# Step 32: 皮层 STDP + LHb Bug 修复 + 重新进化

> 日期: 2026-02-08
> 状态: ✅ 完成

## 问题诊断

Step 31 ablation 发现皮层 STDP (+0.74) 和 LHb (+0.58) 是最有害的两个模块。
深入分析发现两个 bug：

**Bug 1: 皮层 STDP LTD/LTP 比例失衡**
- 进化出的 `a_minus=-0.011` 是 `a_plus=0.003` 的 **3.7 倍**
- 生物学正常比例 ~1.1-1.2×
- 27 条突触 × LTD 3.7× → 权重持续下降 → 视觉信号消亡

**Bug 2: LHb 双重计数**
- 同一个负奖励同时通过两条路径抑制 DA：
  1. VTA RPE: `reward=-1.8` → `phasic_negative=-0.45`
  2. LHb: `punishment=1.8×1.5` → `vta_suppression=~0.2-0.4`
- 合计 DA 被压到 0.0，双倍惩罚
- 生物学: LHb 主要编码 frustrative non-reward（期望落空），不处理直接惩罚

## 修复内容

| 文件 | 修改 |
|------|------|
| `closed_loop_agent.h` | `a_plus=0.005, a_minus=-0.006` (1.2× 正常比例) |
| `closed_loop_agent.cpp` | 删除 `lhb_->inject_punishment()` 对直接惩罚的调用 |
| `closed_loop_agent.cpp` | 删除 CeA→LHb 放大 (也是双重计数) |

## 重新进化 (30gen×40pop Baldwin)

修复后代码重新进化，因为旧参数是在有 bug 的代码上优化的：

```
best fitness = 2.41 (gen27)
关键参数变化:
  cortical_a_plus:  0.003 → 0.017  (进化主动增大 LTP)
  cortical_a_minus: 0.011 → 0.010  (ratio: 3.7× → 0.6×, LTD < LTP)
  bg_to_m1_gain:    21.0  → 2.42   (大幅降低 BG→M1 增益)
  exploration_noise: 55   → 70.3   (更多探索)
  ne_food_scale:    1.0   → 6.13   (NE 对 food 响应更强)
  homeostatic_eta:  0.0015 → 0.0068 (稳态更强)
```

## Ablation 验证

```
修复前 (Step 31):
  baseline safety = 0.08
  皮层 STDP: +0.74 (最有害)
  LHb:       +0.58 (有害)

修复后 (Step 32):
  baseline safety = 1.00
  皮层 STDP: +0.00 (中性) ← 不再有害!
  LHb:       +0.00 (中性) ← 不再有害!
  sleep:     -0.20 (有用)
```

## 完整测试结果 (6/6 PASS, 3.2秒)

```
测试1: 1000步时 safety=1.00, 但 1500步后→0.00 (灾难性遗忘)
测试2: 学习组 0.12 < 对照组 0.60 (学习反而有害)
测试4: early=0.51 → late=0.00, improvement=-0.51 (越学越差)
测试5: 15x15 大环境 improvement=+0.265 (唯一正向)
测试6: trained=0.25 < fresh=0.46 (泛化为负, 训练有害)
```

## 新暴露的核心问题: 稳定性-可塑性困境

Agent 在 ~1000 步时确实学到了东西 (safety 短暂到 1.00)，
但随后权重持续漂移导致灾难性遗忘。原因：

1. **无权重保护机制** — 学到的好权重被后续噪声更新冲掉
2. **Homeostatic eta=0.0068 过强** — 比之前高 4.5×，拉平已学到的差异化权重
3. **DA baseline 恒定** — 没有"已经学会就降低学习率"的元学习机制

## 系统状态

```
53区域 · ~120闭环神经元 · ~110投射
所有学习模块重新启用 (STDP/LHb bug 修复后无害)
Bug 修复: STDP LTD/LTP 比例 + LHb 双重计数
待解决: 灾难性遗忘 (stability-plasticity dilemma)
速度: 3.2秒/6测试
```
