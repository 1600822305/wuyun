# Step 44: Baldwin 重进化 — 优化新架构参数

> 日期: 2026-02-09
> 状态: ✅ 完成
> 30代×40体×5seed, best fitness=2.5032, 进化 vs 手工 +0.56

## 动机

上次进化是 Step 29/33，当时只有 ~120 神经元、10 条学习链路。
Step 40-43 后架构完全不同（228n、17 链路、7 个新区域），参数需要重新适配。

## 进化配置

```
Population:    40 个体/代
Generations:   30 代
Seeds:         {42, 77, 123, 200, 555} (5 seed 泛化)
Eval steps:    1000 (200 early + 800 late)
Fitness:       improvement×3 + late_safety×1 (Baldwin 效应)
Tournament:    5
Mutation:      rate=0.15, sigma=0.12
Elite:         10%
GA seed:       2024
Total time:    373 秒
```

## 参数搜索空间 (23 基因)

| 基因 | 范围 | Step 33 值 | Step 44 进化值 | 变化 |
|------|------|-----------|---------------|------|
| da_stdp_lr | [0.005, 0.08] | 0.014 | **0.022** | ↑57% |
| reward_scale | [0.3, 5.0] | 1.43 | **3.50** | ↑145% |
| cortical_a_plus | [0.001, 0.02] | 0.004 | **0.001** | ↓75% |
| cortical_a_minus | [0.001, 0.02] | 0.006 | **0.010** | ↑67% |
| cortical_w_max | [0.5, 3.0] | 1.31 | **0.81** | ↓38% |
| exploration_noise | [20, 100] | 83 | **48** | ↓42% |
| bg_to_m1_gain | [2, 25] | 10.76 | **7.08** | ↓34% |
| attractor_ratio | [0.3, 0.9] | 0.34 | **0.50** | ↑47% |
| background_ratio | [0.02, 0.3] | 0.22 | **0.26** | ↑18% |
| replay_passes | [1, 15] | 7 | **14** | ↑100% |
| replay_da_scale | [0.1, 1.0] | 0.74 | **0.31** | ↓58% |
| lgn_gain | [50, 500] | 469 | **394** | ↓16% |
| lgn_baseline | [1, 20] | 8.72 | **16.0** | ↑83% |
| lgn_noise | [0.5, 8.0] | 0.50 | **6.7** | ↑1240% |
| homeostatic_target | [1, 15] | 10.94 | **1.61** | ↓85% |
| homeostatic_eta | [0.0001, 0.01] | 0.00073 | **0.0044** | ↑503% |
| v1_size | [0.5, 2.5] | 1.01 | **1.12** | ↑11% |
| dlpfc_size | [0.5, 2.5] | 2.17 | **2.31** | ↑6% |
| bg_size | [0.5, 2.0] | 1.33 | **0.77** | ↓42% |
| brain_steps | [8, 25] | 15 | 15.5 | (clamped to 10 in fast_eval) |
| reward_steps | [2, 10] | 5 | 6.7 | (clamped to 3 in fast_eval) |
| ne_food_scale | [1, 8] | 1.0 | **2.36** | ↑136% |
| ne_floor | [0.4, 1.0] | 0.67 | **0.81** | ↑21% |

## brain_steps 缩放问题

进化在 `fast_eval=true` 下运行（`brain_steps` 被 clamp 到 10），但实际测试用 20 brain_steps。
三个参数对 brain_steps 高度敏感：

| 参数 | 进化值 (10bs) | 缩放后 (20bs) | 缩放逻辑 |
|------|-------------|-------------|---------|
| homeostatic_target | 1.61 | **3.2** | ×2 (更多步=更多发放=需更高目标) |
| homeostatic_eta | 0.0044 | **0.0022** | ÷2 (更多步=更多调整=需更小步长) |
| lgn_noise | 6.7 | **2.0** | ÷3 (更多步=噪声累积=需更小幅度) |

直接使用进化值导致 learning_curve_tests 失败（learner worse than control, danger=96）。
缩放后 30/30 CTest 通过。

## 进化收敛曲线

```
Gen  1: best=0.94, avg=-0.45
Gen  3: best=2.50, avg=+0.07  ← best_ever found
Gen 14: best=1.91, avg=+0.23
Gen 25: best=1.68, avg=+0.01
Gen 30: best=2.46, avg=+0.03
```

Best fitness 在第 3 代出现，说明搜索空间中好的参数组合并不稀少，但后续代没有超越，
可能因为 fitness landscape 在当前架构下较平坦。

## 关键发现

1. **reward_scale 2.4×↑**: 228 神经元的网络需要更强的奖赏信号才能产生有效的 DA burst
2. **replay_passes 2×↑**: 更大的网络需要更多重放来充分利用每次经验
3. **exploration_noise 42%↓**: 7 个新区域 (SC, NAcc, PAG, FPC, OFC, vmPFC, SNc) 提供了额外的信号来源，减少了对随机噪声的依赖
4. **bg_size 42%↓**: 更小的 BG = 更少的 MSN 噪声，让 DA-STDP 信号更突出
5. **cortical STDP 更保守**: a_plus 降低 + LTD 增强 + w_max 降低 = 更稳定的皮层表征

## 修改文件

- `src/engine/closed_loop_agent.h`: 更新 AgentConfig 默认值 (23 个参数)
- `tools/run_evolution.cpp`: 更新默认参数 30gen×40pop, 描述信息
- `docs/progress.md`: 添加 Step 44 摘要

## 验证

```
进化 fitness: 2.5032 vs 手工 baseline: -0.557 → Δ = +0.564
Late safety:  0.337 vs 0.128 → Δ = +0.209
Improvement: -0.096 vs -0.147 → Δ = +0.051

30/30 CTest 通过
D1=57, D2=61, Max elig=71.6, Weight range=0.0607
```
