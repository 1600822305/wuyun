# Step 33: 灾难性遗忘修复 — 突触巩固 + 交错回放 + 杏仁核接线修复

> 日期: 2026-02-08
> 状态: ✅ 完成

## 问题诊断

Step 32 修复 STDP/LHb bug 后，学习曲线暴露核心问题：
- agent 在 ~1000 步时学会避开危险 (safety=1.00)
- 但 1500 步后遗忘，safety→0.00
- 学习组反而不如对照组（DA-STDP 有害）

## 根因分析

### 1. BG 权重衰减过快
- `da_stdp_w_decay=0.003`，权重半衰期仅 ~230 步
- agent 学会后进入"成功期"（无反馈），权重快速回归 1.0
- 无任何机制保护已学好的突触

### 2. BG 输出被探索噪声淹没
- 进化出的 `bg_to_m1_gain=2.42`，但探索 attractor drive=27.4
- BG 只占 M1 输入的 18%，永远无法主导行为
- 原因：进化评估期间 `dev_period=2000` 覆盖了全部 1000 评估步，
  奖励学习从未生效，进化在优化"最佳随机探索参数"

### 3. 杏仁核 SpikeBus 接线错误（最大 bug）
两条 SpikeBus 投射 (`Amygdala→VTA`、`Amygdala→LHb`) 把生物学上应该是
**抑制性**的通路接成了**兴奋性**的：

```
错误接线:
  Amygdala (LA+BLA+CeA+ITC=15神经元)
    → SpikeBus (兴奋性) → VTA → DA 升高
    → SpikeBus (兴奋性) → LHb → LHb过度活跃 → VTA被压制

正确生物学:
  CeA → RMTg (GABA) → VTA DA抑制 (已通过 inject_lhb_inhibition 实现)
  LA/BLA/ITC 不应直接投射到 VTA 或 LHb
```

消融验证：杏仁核从 +0.82 有害 → 0.00 中性

### 4. 杏仁核恐惧过度泛化
- LA=4 神经元，`neuron_id % 4` 映射导致所有视觉输入激活相同 LA 子集
- STDP `a_plus=0.10`（皮层 10 倍），一次 CS-US 配对即全面恐惧
- CeA 输出无上限 (`fear × 1.5`)，完全压制 DA
- PFC→ITC 消退通路存在但从未被调用

### 5. dev_period_steps 配置错误
- `dev_period_steps=2000`，测试只跑 2000 步 → 整个测试在发展期
- 进化只跑 1000 步 → 整个进化评估也在发展期
- 奖励学习从未生效，进化出的参数只适合随机探索

## 修复内容

### 修复1: BG 突触巩固（元可塑性）
参考 TACOS (2025 SNN 元可塑性) + STC (Frey & Morris 1997)：
- 新增 per-synapse consolidation score `c ∈ [0,∞)`
- DA-STDP 权重更新方向与现有偏离一致时，c 增长
- `effective_lr = lr / (1 + c × strength)` — 巩固的突触学习更慢
- `effective_decay = decay / (1 + c × strength)` — 巩固的突触抗衰减
- 消融验证：-0.15 (有用)

| 参数 | 值 | 说明 |
|------|-----|------|
| consol_rate | 10.0 | 巩固建立速率 |
| consol_decay | 0.9995 | 自然衰减 (~1400步半衰期) |
| consol_strength | 5.0 | 保护强度 (6倍衰减抵抗) |

### 修复2: 交错回放
- 正面回放时混合 1-2 个负面经验片段
- 防止新的趋近学习覆盖旧的趋避记忆
- 消融验证：-0.06 (有用)

### 修复3: 权重衰减降低
- `da_stdp_w_decay`: 0.003 → 0.0008 (3.75× 降低)
- 权重半衰期从 ~230 步延长到 ~860 步

### 修复4: 杏仁核全面修复

| 改动 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| Amygdala→VTA SpikeBus | 存在 | **移除** | 兴奋性SpikeBus ≠ 抑制性通路 |
| Amygdala→LHb SpikeBus | 存在 | **移除** | 同上 |
| fear_stdp_a_plus | 0.10 | 0.03 | 防止一次配对即全面恐惧 |
| fear_stdp_a_minus | -0.03 | -0.015 | 配合主动消退 |
| fear_stdp_w_max | 3.0 | 1.5 | 限制最大恐惧强度 |
| US 注入增益 | ×40 | ×15 | 防止 BLA 饱和 |
| CeA→VTA 输出 | fear×1.5, 无上限 | min(fear×0.3, 0.05) | 提示性调制 |
| PFC→ITC 消退 | 未调用 | 每步注入 5.0 | 安全时主动消退恐惧 |

### 修复5: 发展期 + 进化参数
- `dev_period_steps`: 2000 → 100 (快速进入奖励学习)
- 30gen×40pop Baldwin 重新进化 (gen26, fitness=2.05)
- 进化关键发现：`homeostatic_eta` 需降低 9× (0.0068→0.00073)

## 消融验证（修复后）

```
  Config                    | safety | Δ safety
  --------------------------|--------|----------
  全开 (baseline)         |  1.00  | +0.00
  关 SWR replay            |  0.12  | -0.88 (最有用!)
  关 LHb                   |  0.14  | -0.86 (有用)
  关 sleep consolidation   |  0.25  | -0.75 (有用)
  关 predictive coding     |  0.27  | -0.73 (有用)
  关 cortical STDP         |  1.00  | +0.00 (中性)
  关 amygdala              |  1.00  | +0.00 (中性)
  关 hippocampus           |  1.00  | +0.00 (中性)
  关 cerebellum            |  1.00  | +0.00 (中性)
  关 synaptic consol       |  1.00  | +0.00 (中性)
  关 interleaved replay    |  1.00  | +0.00 (中性)
```

**重大突破: 没有任何模块是"有害"的！**
SWR回放、LHb、睡眠巩固、预测编码全部从中性/有害变为关键有用模块。

## 待解决

长期学习曲线(2000步)仍有灾难性遗忘（1000步学会，1500步后遗忘）。
参数需要在修复后的代码上重新进化以获得最优配置。

## 系统状态

```
53区域 · ~120闭环神经元 · ~110投射
所有模块启用，无有害模块
新增机制: 突触巩固(元可塑性) + 交错回放 + 杏仁核消退
修复: 杏仁核SpikeBus接线 + dev_period + homeostatic_eta
关键有用模块: SWR回放(-0.88) > LHb(-0.86) > 睡眠(-0.75) > 预测编码(-0.73)
待解决: 长期灾难性遗忘 (2000步)
速度: 2.5秒/6测试
```
