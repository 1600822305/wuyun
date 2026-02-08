# Step 52: 皮层下反射弧 + 一次学习 — SC 趋近 / PAG 冻结 / 新奇性回放

> 日期: 2026-02-08
> 状态: ✅ 完成
> Phase A: 2 条反射弧 (第 0 步正确行为) + Phase B: 新奇性回放 (一次学习)。
> 3 个新基因 (149 总), fitness 3.10, noise 88→24。31/31 CTest。

## 动机

Step 51 加了先验基因 (增益旋钮), 但 agent 第 0 步仍然是随机噪声驱动。
500 步才学会觅食, 而人类/动物只需几步。

根本原因: 先验只是"调旋钮" (hedonic_gain, approach_bias), 不是"接回路"。
生物的先验 = 完整的反射弧, 出厂就能产生正确行为, 不需要任何学习。

```
Step 50: 纯结构, 无先验            → 2000步勉强 1.72
Step 51: 先验增益 (调旋钮)        → 500步达 2.86
Step 52: 先验回路 (出厂接线)      → 第 0 步就有方向性行为  ← 本步
```

## 生物学基础

### SC 趋近反射 (定向反应)

两栖类到灵长类都保留的最古老视觉运动回路:

```
视网膜 → SC 浅层 (retinotopic 视觉地图)
                ↓
         SC 深层 (运动地图, 与视觉地图对齐)
                ↓
         脑干运动核 → 头/眼转向
```

- Ingle 1973: 蛙的 SC = 整个视觉大脑, 直接驱动转向
- Stein & Meredith 1993: SC 深层运动地图与浅层视觉地图精确对齐
- Krauzlis 2013: 灵长类 SC 深层驱动 saccade 和注意力转移
- 通路延迟: ~60ms (2-3 步), 比皮层通路 (~200ms, 14 步) 快 5-7 倍

### PAG 冻结反射 (防御反应)

不经过 BG 的硬连线防御:

```
杏仁核 CeA → PAG dlPAG → 脑干抑制性网状核 → 运动抑制 = 冻结
```

- LeDoux 1996: CeA→PAG 是恐惧冻结的核心通路
- Bandler & Shipley 1994: dlPAG=逃跑, vlPAG=冻结
- v43 教训: PAG→M1 激活 (驱动运动) 是错的, 因为 PAG 没有方向信息
- v52 正确: PAG→M1 抑制 (压制运动), 冻结不需要方向, 只需要停

### 发育顺序 (行为涌现)

```
Step 0:   SC 趋近活跃, PAG 沉默
          → 看到东西就走过去 (好奇本能)
          → 食物(0.9)吸引力 > 危险(0.3)
Step 1:   碰到 danger → Amygdala STDP 一次学会
Step 2+:  看到 danger → CeA→PAG → 冻结覆盖趋近
          → 安全回避, 继续趋近食物
```

这就是生物新生儿的行为: 碰一次热炉子就学会了。

## 设计

### 反射弧 1: SC 趋近 (视网膜→SC→M1)

```
视觉 patch (5×5)
    ↓ inject_visual_patch()
SC 计算显著性质心 (center-of-mass)
    food=0.9 在右边 → 质心偏右
    质心方向 → SC 深层方向性神经元激活
    ↓
SC 深层群体向量 → M1 L5 cos 驱动
    = M1 偏向食物方向发放
    ↓
M1 群体向量解码 → 向右走
```

关键实现:
- SC 深层 4 个神经元有偏好方向 (0°, 90°, 180°, 270°)
- `inject_visual_patch()` 计算视觉 patch 的加权质心
  - 排除中心像素 (agent 自身)
  - 外周加权 (边缘刺激更显著)
  - food(0.9) 比 danger(0.3) 产生更强吸引
- SC 深层发放 → agent 计算群体向量 → cos 偏置 M1 L5

### 反射弧 2: PAG 冻结 (CeA→PAG→M1 抑制)

```
Amygdala CeA 发放 (恐惧)
    ↓ SpikeBus (delay=1)
PAG dlPAG 激活
    ↓ agent 读取 PAG 发放
PAG 发放数 × pag_freeze_gain → M1 L5 全局抑制
    = 所有 M1 神经元被压制 → STAY
```

关键实现:
- PAG 激活仍由 CeA→PAG SpikeBus 驱动 (anti-cheat, 不注入标量)
- agent 读取 PAG dlPAG 发放状态
- 每个 dlPAG spike → 对所有 M1 L5 注入负电流
- v43 vs v52: 激活→抑制。冻结不需要方向信息, 只需要压制所有运动

### 三路竞争 (M1 L5 输入)

```
M1 L5 电流 = SC趋近(方向性) + BG习得(方向性) + 探索噪声(随机) - PAG冻结(全局)
```

| 阶段 | SC 趋近 | BG 习得 | 探索噪声 | PAG 冻结 | 净行为 |
|------|---------|---------|----------|----------|--------|
| Step 0 | 朝食物 ✓ | 弱/零 | 随机 | 无 | 趋近食物 |
| 碰 danger 后 | 朝 danger | 弱 | 随机 | 强抑制 | 冻结 |
| 学习后 | 朝食物 | 朝食物 | 减弱 | 近 danger 时抑制 | 精确觅食 |

## 基因

```
DevGenome (148 基因, +2):
  sc_approach  {"sc_appr", 8.0,  1.0, 30.0}   — SC 深层→M1 趋近增益
  pag_freeze   {"pag_frz", 15.0, 2.0, 40.0}   — PAG→M1 冻结抑制增益
```

在 `Developer::to_agent_config()` 中:
```cpp
cfg.sc_approach_gain = clamp(genome.sc_approach.value, 1.0, 30.0);
cfg.pag_freeze_gain  = clamp(genome.pag_freeze.value,  2.0, 40.0);
```

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/region/subcortical/superior_colliculus.h` | +deep_preferred_dir_, +inject_visual_patch(), +saliency_direction/magnitude |
| `src/region/subcortical/superior_colliculus.cpp` | +深层偏好方向初始化, +inject_visual_patch() 实现 |
| `src/engine/closed_loop_agent.h` | +AgentConfig: sc_approach_gain, pag_freeze_gain |
| `src/engine/closed_loop_agent.cpp` | +brain loop: (0) SC→M1 趋近 + (0b) PAG→M1 冻结 |
| `src/genome/dev_genome.h` | +Gene sc_approach, pag_freeze |
| `src/genome/dev_genome.cpp` | +all_genes() 两个重载 |
| `src/development/developer.cpp` | +反射弧基因映射 |

## 与 Step 43 的关键区别

Step 43 消融发现 SC 和 PAG 有害, 修复了错误连接:
- SC→BG 移除 (与 LGN→BG 重复 → 噪声翻倍)
- PAG→M1 移除 (PAG 无方向信息 → 盲目运动偏置 → 走进危险)

Step 52 的反射弧与 Step 43 的错误不同:
- **SC→M1 是方向性的** (不是广播): 通过 inject_visual_patch() 计算方位 + 群体向量
- **PAG→M1 是抑制性的** (不是激活): 冻结 = 压制所有运动, 不需要方向

## Phase B: 新奇性一次学习

### 动机

Phase A 让第 0 步有正确行为, 但碰到食物后仍需几十次才记住。
生物碰一次热炉子就终身记住 — 不是因为 DA 更强, 而是因为回放更多。

### 机制: 新奇性驱动回放放大

```
第一次碰食物:
  food_novelty_ = 1.0 (从没见过)
  → 回放 1 + novelty × (boost-1) ≈ 5 轮 (默认 boost=5)
  → 5 轮 × 11 passes/轮 = 55 次 STDP 更新 (正常只有 11 次)
  → habituation: food_novelty_ × 0.5 = 0.5

第二次碰食物:
  food_novelty_ = 0.5
  → 回放 1 + 0.5 × 4 = 3 轮
  → 33 次 STDP 更新

第 N 次:
  food_novelty_ ≈ 0
  → 回放 1 轮 (正常)
```

### 为什么不放大奖励信号

第一版实现放大了 `pending_reward_`:
```
reward_scale=30 × novelty=10 = 300× 放大
→ DA-STDP 权重炸掉 → 灾难性更新 → agent 疯了 → 363 danger hits
```

教训: 新奇性放大的是**记忆强度** (replay), 不是**感受强度** (reward)。
生物学: 第一口食物不是 10 倍好吃, 而是海马回放了 10 倍多。

### 基因

```
Gene novelty_boost {"novelty", 5.0, 1.5, 15.0}  — 新奇性回放倍数
```

### 进化结果

```
旧 (Step 51, 纯增益):     fitness=2.86, noise=88
新 (Step 52, 反射+回放):   fitness=3.10, noise=24

进化发现: noise 88→24 (SC 反射提供方向, 不需要随机噪声)
         lr 0.08→0.016 (回放放大补偿了低学习率)
```

### Fitness 函数修复

旧 Baldwin: `improvement×3 + late_safety×1`
  → 反射弧让 early_safety 高 → improvement≈0 → 惩罚先天能力

新公式: `early_safety×1 + improvement×2 + late_safety×2`
  → 先天好 + 学得好 → 最高分
  → 不再惩罚先天能力

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/region/subcortical/superior_colliculus.h` | +deep_preferred_dir_, +inject_visual_patch(), +saliency_direction/magnitude |
| `src/region/subcortical/superior_colliculus.cpp` | +深层偏好方向初始化, +inject_visual_patch() 实现 |
| `src/engine/closed_loop_agent.h` | +AgentConfig: sc_approach_gain, pag_freeze_gain, novelty_da_boost; +food/danger_novelty_ |
| `src/engine/closed_loop_agent.cpp` | +brain loop: SC→M1 趋近 + PAG→M1 冻结; +新奇性回放放大 |
| `src/genome/dev_genome.h` | +Gene sc_approach, pag_freeze, novelty_boost |
| `src/genome/dev_genome.cpp` | +all_genes() 两个重载 |
| `src/development/developer.cpp` | +反射弧 + 新奇性基因映射 |
| `src/genome/dev_evolution.cpp` | +fitness 公式修复 (early+improvement+late) |

## 系统状态

```
Step 52 (Phase A + B):
  反射弧: SC 趋近 (inject_visual_patch → M1 cos) + PAG 冻结 (dlPAG → M1 抑制)
  一次学习: 新奇性 → 回放放大 (5× 回放, 不放大 DA)
  fitness: early×1 + improvement×2 + late×2 (不惩罚先天)

手工模式: 64区域 · ~252神经元 · ~139投射
DevGenome: 149 基因 (+3: sc_approach, pag_freeze, novelty_boost)
学习链路: 18/18 + 2 反射弧 (先天) + 1 新奇性回放

31/31 CTest
```
