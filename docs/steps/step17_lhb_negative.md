# Step 17: LHb 负RPE 脑区 + 负经验重放

> 日期: 2026-02-08
> 状态: ✅ 完成

---

## Step 17: LHb 负RPE 脑区 (外侧缰核)

> 目标: 补全奖惩学习闭环 — 惩罚/期望落空 → LHb → VTA DA pause → D2 NoGo 强化

### 生物学基础 (Matsumoto & Hikosaka 2007, Bromberg-Martin 2010)

LHb 是大脑的"反奖励中心":
- **负RPE编码**: 预期奖励未出现 或 遭遇惩罚 → LHb 兴奋
- **VTA抑制**: LHb → RMTg(GABA中间神经元) → VTA DA pause
- **D2 NoGo学习**: DA低于基线 → D2权重增强 → 抑制导致惩罚的动作
- **互补关系**: VTA编码正RPE(食物→DA burst), LHb编码负RPE(危险→DA pause)

### 新增文件

- `src/region/limbic/lateral_habenula.h/cpp` — LateralHabenula 脑区类

### 修改文件

- `src/region/neuromod/vta_da.h/cpp` — 新增 `inject_lhb_inhibition()`, LHb抑制PSP缓冲, DA level计算整合LHb抑制
- `src/engine/closed_loop_agent.h/cpp` — LHb区域创建/投射/接线, Phase A惩罚→LHb驱动, 期望落空检测
- `src/CMakeLists.txt` — 添加 lateral_habenula.cpp

### 信号通路

```
Phase A (奖励处理):
  danger事件 → reward = -1.5
    → VTA inject_reward(-1.5) → 负RPE → DA↓ (已有机制)
    → LHb inject_punishment(1.5) → LHb burst → vta_inhibition=0.8 → VTA DA pause (新增!)
    → DA_level ≈ 0 (远低于baseline 0.3) → da_error = -0.3
    → D1: Δw = +lr × (-0.3) × elig → Go 弱化
    → D2: Δw = -lr × (-0.3) × elig = +0.009 × elig → NoGo 强化

期望落空 (Frustrative Non-Reward):
  agent 学会取食后, 预期有食物但没拿到 → expected_reward_level > 0.05
    → LHb inject_frustration(mild) → 温和DA dip → 微调D2

Phase B (动作处理):
  每个brain step: LHb → VTA抑制广播 (持续效应, 指数衰减)
```

### LHb 配置参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `enable_lhb` | true | 启用LHb负RPE |
| `lhb_punishment_gain` | 1.5 | 惩罚信号→LHb兴奋增益 |
| `lhb_frustration_gain` | 1.0 | 期望落空→LHb兴奋增益 |
| `n_neurons` | 25×scale | LHb神经元数 |
| `vta_inhibition_gain` | 0.8 | LHb输出→VTA抑制强度 |

### VTA 修改详情

- 新增 `lhb_inh_psp_` 持续抑制缓冲 (衰减常数 0.85, 与 reward_psp_ 对称)
- DA neuron drive: `net_drive = 20 + reward_psp + psp - lhb_inh_psp` (LHb抑制扣减)
- DA level: `total_negative = min(phasic_negative - lhb_suppression, 0)` (双重负信号)

### 回归测试: 29/29 CTest 全通过, 零回归

---

## Step 17-B: 负经验重放集成 (LHb-controlled avoidance replay)

> 日期: 2026-02-08

之前 Step 14 明确禁用了负重放: "负重放导致D2过度强化→行为振荡"。
有了 LHb 的受控 DA pause 后，负重放变得安全且有效。

### 机制

```
danger事件 → run_negative_replay()
  1. 收集最近的负奖励 episodes (reward < -0.05)
  2. DA level 设为 baseline 以下: da_replay = 0.3 - |reward|×0.3 ∈ [0.05, 0.25]
  3. 重放旧 danger 的 cortical spikes → BG DA-STDP
  4. D2: Δw = -lr × (0.15-0.3) × elig = +0.0045×elig (NoGo 强化)
  5. D1: Δw = +lr × (0.15-0.3) × elig = -0.0045×elig (Go 弱化)
```

### 安全措施 (防止 D2 过度强化)

- **fewer passes**: 负重放 2 passes vs 正重放 5 passes
- **DA floor**: da_replay 不低于 0.05 (不会完全 DA 清零)
- **延迟启用**: agent_step >= 200 才启动 (避免早期噪声)
- **依赖 LHb**: enable_lhb=false 时自动禁用

### 效果验证

| 指标 | 仅LHb (无负重放) | LHb + 负重放 | 提升 |
|------|----------|--------|------|
| Learner advantage | -0.0035 | **+0.0085** | ✅ 翻正 |
| Learner safety | 0.44 | **0.63** | +43% |
| 10k Improvement | -0.031 | **+0.158** | ✅ 翻正 |
| 10k Late safety | 0.556 | **0.603** | +8% |

**Improvement +0.158 超过 Step 14 基线 +0.120 (+32%)**

### 回归测试: 29/29 CTest 全通过, 零回归

### 系统状态

```
49区域 · 自适应神经元数 · ~110投射 · 29 CTest suites
新增: LHb 负RPE + 负经验重放
完整奖惩回路:
  食物→VTA DA burst→D1 Go 强化 + 正重放巩固 (5 passes)
  危险→LHb→VTA DA pause→D2 NoGo 强化 + 负重放巩固 (2 passes)
学习能力: improvement +0.158 (+32% vs Step 14 基线 +0.120)
```
