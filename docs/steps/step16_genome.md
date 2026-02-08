# Step 16: 基因层 v1 (遗传算法自动优化参数)

> 日期: 2026-02-08
> 状态: ✅ 完成

## 目标

用遗传算法搜索 ClosedLoopAgent 的最优参数组合。

## 新增文件

- `src/genome/genome.h/cpp` — Genome 数据结构 (23 个基因, 直接编码)
- `src/genome/evolution.h/cpp` — GA 引擎 (锦标赛选择/均匀交叉/高斯变异, 多线程并行评估)
- `tools/run_evolution.cpp` — 进化运行器

## 参数化改造

将 `build_brain()` / `agent_step()` 中 8 个硬编码参数提升为 `AgentConfig` 字段:
- `lgn_gain/baseline/noise_amp` — 视觉编码
- `bg_to_m1_gain, attractor_drive_ratio, background_drive_ratio` — 运动耦合
- `ne_food_scale, ne_floor` — NE 探索调制
- `homeostatic_target_rate, homeostatic_eta` — 稳态可塑性
- `v1_size_factor, dlpfc_size_factor, bg_size_factor` — 脑区大小缩放

## 23 个可进化基因

```
学习: da_stdp_lr, reward_scale, cortical_a_plus/minus, cortical_w_max
探索: exploration_noise, bg_to_m1_gain, attractor_ratio, background_ratio
重放: replay_passes, replay_da_scale
视觉: lgn_gain, lgn_baseline, lgn_noise
稳态: homeostatic_target, homeostatic_eta
大小: v1_size, dlpfc_size, bg_size
时序: brain_steps, reward_steps
NE:   ne_food_scale, ne_floor
```

## 进化实验 (15 代 × 40 个体, 16 线程并行)

首轮 (eval_steps=2000, 2 种子):
- 10.7 分钟完成, best fitness 从 0.97 → 1.16
- 进化发现: reward_scale ↑3×, dlpfc_size ↑2.3×, replay_passes ↑2×

**关键发现: 短评估陷阱**

| 评估方式 | 进化最优 | 10k 标准测试 | 问题 |
|----------|----------|-------------|------|
| 2000 步 × 2 种子 | fitness 1.16 | improvement **-0.120** | 优化了短期随机表现 |
| 手动基线 | — | improvement **+0.120** | 真正的学习能力 |

**根因**: 2000 步太短, 进化找到了"初始表现好"的参数 (高 reward_scale 导致 DA 饱和),
而非"学习能力强"的参数。正确的适应度评估需要 ≥5000 步 + ≥3 种子。

## 进化洞察 (值得关注但需长评估验证)

- dlPFC 可能确实偏小 (进化一致倾向 ↑1.5-2.3×)
- 更多重放 passes (5→8-10) 可能有益
- bg_to_m1_gain 可能需要更强 (8→13)
- 这些需要在 5000 步评估下重新进化验证

## 已修正: 评估配置升级

```
eval_steps: 2000 → 5000 (捕捉完整学习曲线)
eval_seeds: 2 → 3 (泛化性)
默认参数: 恢复手动基线 (improvement +0.120, late safety 0.667)
```

## 回归测试: 29/29 CTest 全通过, 基线完全恢复

## 系统状态

```
48区域 · 自适应神经元数 · ~109投射 · 179测试 · 29 CTest suites
新增: 基因层 v1 (23基因, GA引擎, 多线程并行评估)
学习维持: improvement +0.120, late safety 0.667
```
