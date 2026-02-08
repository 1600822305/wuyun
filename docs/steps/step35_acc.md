# Step 35: ACC 前扣带回皮层 — 冲突监测与动态探索

> 日期: 2026-02-08
> 状态: ✅ 完成

## 目标

替代硬编码 `ne_floor` 和手工 `arousal` 计算，用神经动力学驱动探索/利用平衡。

## 问题诊断

当前系统的探索/利用平衡完全依赖硬编码参数：

```
// 旧机制 (closed_loop_agent.cpp)
float arousal = std::max(0.0f, 0.05f - fr * 0.1f);  // 手工food_rate计算
lc_->inject_arousal(arousal);
// fallback: noise_scale = max(ne_floor=0.67, 1.0 - fr * ne_food_scale)
```

问题：
1. `ne_floor=0.67` 是进化出的魔法数字，没有生物学基础
2. `arousal` 只看 `food_rate`，忽略了动作冲突、环境变化、策略失效等关键信号
3. 没有冲突监测 — 系统不知道自己在"犹豫"
4. 没有惊讶检测 — 意外结果不会改变行为
5. 没有波动性追踪 — 环境变化时学习率不调整

## 文献基础

整合 5 个经典 ACC 计算模型：

| 模型 | 来源 | 功能 |
|------|------|------|
| **冲突监测** | Botvinick et al. 2001 | D1子群竞争 → 探索需求 |
| **PRO预测误差** | Alexander & Brown 2011 | \|actual-predicted\| → 惊讶（不分正负效价） |
| **波动性检测** | Behrens et al. 2007 | fast/slow奖励率差 → 学习率调制 |
| **觅食决策** | Kolling et al. 2012 | local vs global奖励率 → 策略切换 |
| **努力/控制** | Shenhav et al. 2013 EVC | 综合信号 → LC-NE唤醒驱动 |

解剖学连接 (StatPearls, Neuroanatomy Cingulate Cortex)：
- 输入: dlPFC(上下文), BG D1(动作竞争), VTA-DA(RPE), Amygdala-CeA(威胁)
- 输出: LC(唤醒/探索), dlPFC(控制/注意), VTA(惊讶调制)

## 实现

### 神经元群体

| 群体 | 数量 | 类型 | 功能 |
|------|------|------|------|
| **dACC** | 12×s | L2/3 Pyramidal | 冲突监测 + 觅食 + 认知控制 |
| **vACC** | 8×s | L2/3 Pyramidal | 情绪评价 + 惊讶 + 动机 |
| **PV Inh** | 6×s | PV Basket | E/I 平衡 |

内部突触: dACC↔vACC (AMPA), Exc→Inh (AMPA→SOMA), Inh→Exc (GABA_A→SOMA)

### 计算模块

**1. 冲突监测 (Botvinick 2001)**
```
conflict = Σ_{i≠j} rate_i × rate_j / total²  // Hopfield能量
conflict_level = EMA(conflict × gain, decay=0.85)
```

**2. PRO惊讶 (Alexander & Brown 2011)**
```
predicted_reward = EMA(outcome, τ=0.97)
surprise = |actual - predicted|
```

**3. 波动性 (Behrens 2007)**
```
reward_rate_fast = EMA(|outcome|, τ=0.90)
reward_rate_slow = EMA(|outcome|, τ=0.99)
volatility = |fast - slow| × gain
→ learning_rate_modulation ∈ [0.5, 2.0]
```

**4. 觅食决策 (Kolling 2012)**
```
foraging_signal = max(0, global_rate - local_rate) × 5
```

**5. 综合输出 (Shenhav 2013 EVC)**
```
arousal_drive = conflict×0.4 + surprise×0.3 + foraging×0.2 + threat×0.1
→ ACC→LC: inject_arousal(arousal_drive × 0.15)
attention_signal = conflict×0.5 + surprise×0.3 + volatility×0.2
→ ACC→dlPFC: 认知控制增强
```

## 接入 ClosedLoopAgent

### build_brain() 新增
```
ACC (12+8+6=26 neurons × scale)
SpikeBus: dlPFC → ACC (delay=3), ACC → dlPFC (delay=3)
```

### agent_step() 改变

**输入注入**:
- `acc_->inject_d1_rates()`: 读取 BG D1 4个方向子群发放率 → 冲突检测
- `acc_->inject_outcome()`: 注入 `last_reward_` → PRO惊讶计算
- `acc_->inject_threat()`: 注入 `amyg_->cea_vta_drive()` → 情绪唤醒

**输出使用**:
```cpp
// 旧: arousal = max(0, 0.05 - food_rate * 0.1)  // 硬编码
// 新: arousal = acc_->arousal_drive() * 0.15      // 神经动力学涌现
lc_->inject_arousal(arousal);
```

## 测试修复（非ACC引起的已有问题）

| 测试 | 根因 | 修复 |
|------|------|------|
| `bg_learning` 反转学习 | Step 33 新增 `synaptic_consolidation=true` 但测试未更新 | 测试中关闭巩固 |
| `cortical_stdp` 训练增强 | LTD > LTP → 训练反而削弱；时间戳重叠 | LTP>LTD, 修正时间戳 |

## 新增文件

| 文件 | 说明 |
|------|------|
| `src/region/anterior_cingulate.h` | ACCConfig + AnteriorCingulate 类定义 |
| `src/region/anterior_cingulate.cpp` | 完整实现 (5个计算模块, 6组内部突触) |

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/engine/closed_loop_agent.h` | +include, +enable_acc=true, +acc_ pointer |
| `src/engine/closed_loop_agent.cpp` | build_brain() ACC+投射; agent_step() ACC驱动LC |
| `src/CMakeLists.txt` | +anterior_cingulate.cpp |
| `tests/cpp/test_bg_learning.cpp` | 反转学习测试关闭巩固 |
| `tests/cpp/test_cortical_stdp.cpp` | 修正LTP/LTD比例+时间戳 |

## Step 35b: ACC 输出全接线

| ACC 输出 | 接线目标 | 生物学机制 |
|----------|----------|-----------|
| `attention_signal()` | `dlpfc_->set_attention_gain(1.0 + att×0.5)` | Shenhav 2013 |
| `foraging_signal()` | `noise_scale *= (1.0 + forage×0.3)` | Kolling 2012 |
| `learning_rate_modulation()` | DA error scaling | Behrens 2007 |

信号流总结：
```
BG D1竞争 → ACC冲突 ─┬→ LC arousal → NE↑ → 探索噪声
                      └→ dlPFC attention_gain → 决策精度
奖励结果 → ACC惊讶 ──┬→ LC arousal → NE↑
                      └→ dlPFC attention_gain
奖励波动 → ACC波动性 ──→ DA error scaling → 学习率调制
局部vs全局 → ACC觅食 ──→ noise_scale↑ → 策略切换
```

## ACC 消融对比

| 指标 | ACC ON | ACC OFF | 结论 |
|------|--------|---------|------|
| 早期 safety | **0.750** | 0.543 | ✅ +0.207 |
| 早期 danger | **2** | 21 | ✅ 少19个 |
| Learner advantage | **+0.023** | +0.001 | ✅ 好22× |
| 总 danger | **19** | 28 | ✅ 少32% |

## 灾难性遗忘根因分析

```
权重衰减: (1-0.0008)^12 = 0.9904/agent步 → 500步后仅剩 0.8% 偏差
巩固衰减: 0.9995^12 = 0.994/agent步  → 半衰期仅 115 agent步
D1 MSN 发放: 50步内 0 次 → 学习信号极弱
```

30/30 CTest 通过，零回归。
