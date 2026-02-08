# Step 53: 多任务"天才基因"评估

> 日期: 2026-02-08
> 状态: ✅ 完成
> 3 种任务 (开放觅食 + 稀疏奖赏 + 反转学习), 选通用学习能力而非应试专才。31/31 CTest。

## 动机

Step 52 的进化只考一道题 (10×10 觅食), 出来的是觅食专家。
真正的智能不是单科状元, 而是什么都能学会。

```
专才: 10×10 觅食满分 → 换环境就崩
天才: 任何环境都能快速适应 → 通用学习能力
```

解决方法: 多任务评估。同一个基因组在 3 种不同任务上评估, 加权平均。

## 3 种任务

### Task 1: 开放觅食 (Open Field)
- 10×10, 5 food, 3 danger, 400 steps, 3 seeds
- 测试: 基本趋近/回避能力
- 权重: ×1.0

### Task 2: 稀疏奖赏 (Sparse Reward)
- 10×10, **1 food, 0 danger**, 400 steps, 2 seeds
- 测试: 耐心 + 探索效率 + 稀疏信号学习
- 专才习惯密集奖赏, 在这里崩。天才能适应。
- 权重: ×1.0

### Task 3: 反转学习 (Reversal Learning)
- 10×10, 5 food, 3 danger, 400 steps, 2 seed pairs
- Phase 1 (200 步): seed A 正常学习
- Phase 2 (200 步): **换 seed B** (大脑保留, 世界变了)
- 测试: 灵活性 — 旧策略失效时能否快速适应
- 这是区分专才和天才的关键
- 权重: ×1.5 (最重要)

### 计算量

```
Task 1: 400 steps × 3 seeds = 1200 steps
Task 2: 400 steps × 2 seeds =  800 steps
Task 3: 400 steps × 2 seeds =  800 steps
Total: 2800 steps/individual (vs 旧方案 2500, 仅 +12%)
```

## 实现

### 反转学习基础设施

```cpp
// GridWorld: 换种子重置 (保留大脑, 换世界)
void GridWorld::reset_with_seed(uint32_t new_seed);

// Agent: 换世界但保留大脑
void ClosedLoopAgent::reset_world_with_seed(uint32_t seed);
// 清空: 空间价值图 + 回放缓冲 + 奖励历史
// 保留: 所有突触权重 + DA-STDP 资格迹 + 新奇性
```

### 多任务评估器

```cpp
MultitaskFitness evaluate(const DevGenome& genome) {
    // Task 1: Open field (3 seeds, 权重 1.0)
    open = avg(eval_open_field(seed=42,77,123))

    // Task 2: Sparse (2 seeds, 权重 1.0)
    sparse = avg(eval_sparse(seed=256,789))

    // Task 3: Reversal (2 pairs, 权重 1.5)
    reversal = avg(eval_reversal(42→789, 77→256))

    fitness = (open×1 + sparse×1 + reversal×1.5) / 3.5 + conn_bonus
}
```

每个 eval_xxx 内部:
- 构建 fresh ClosedLoopAgent (独立大脑, 互不影响)
- 跑指定步数
- 计算 early_safety×1 + improvement×2 + late_safety×2

反转任务: 跑 200 步 seed_a → reset_world_with_seed(seed_b) → 跑 200 步
只评估 Phase 2 表现 (适应能力, 不是记忆力)

### 输出增强

进化输出显示各任务分数:
```
Gen 5/30 | best=2.80 avg=0.65 | best_ever=2.80 | 12.1s
    fit=2.80 ctx=203n bg=0.9 lr=0.034 noise=65 | open=2.67 sparse=1.75 rev=3.00
```

## 进化结果

```
Gen 5 冠军:
  open     = 2.67  ← 开放觅食 ✅
  sparse   = 1.75  ← 稀疏奖赏 ✅
  reversal = 3.00  ← 反转学习 ✅
  同一个大脑, 三种任务都能做!
```

### 天才 vs 专才参数对比

| 参数 | 专才 (单任务) | 天才 (多任务) | 含义 |
|------|-------------|-------------|------|
| ctx neurons | 172n | 203n | 多任务需要更大的脑 |
| DA-STDP lr | 0.016 | 0.034 | 需要更快适应 |
| noise | 24 | 65 | 需要更多探索 |
| bg_size | 0.7 | 0.9 | 更大 BG 选择网络 |

进化发现: 天才需要比专才更大的脑、更高的学习率、更多的探索。
这和生物学一致 — 人类比其他动物有更大的前额叶和更强的学习灵活性。

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/engine/grid_world.h/cpp` | +reset_with_seed() |
| `src/engine/closed_loop_agent.h/cpp` | +reset_world_with_seed() |
| `src/engine/episode_buffer.h` | +clear() |
| `src/genome/dev_evolution.h` | +MultitaskFitness, 重写评估接口 |
| `src/genome/dev_evolution.cpp` | +eval_open_field/sparse/reversal, +evaluate_multitask, +精英分数保留 |
| `tools/run_dev_evolution.cpp` | 多任务配置 + 输出增强 |

## 系统状态

```
Step 53 (多任务天才评估):
  Task 1: 开放觅食 (10×10, 5food, 3danger, 3seeds)
  Task 2: 稀疏奖赏 (10×10, 1food, 0danger, 2seeds)
  Task 3: 反转学习 (10×10, seed_a→seed_b, 2pairs, ×1.5权重)
  fitness = (open + sparse + reversal×1.5) / 3.5

DevGenome: 149 基因
学习链路: 18/18 + 2 反射弧 + 1 新奇性回放

31/31 CTest
```
