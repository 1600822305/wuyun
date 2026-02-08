# Step 47: Baldwin 重进化 (群体向量 + VTA RPE 架构)

> 日期: 2026-02-09
> 状态: ✅ 完成
> 30代×40体×5seed, 264秒。improvement -0.207→+0.198。30/30 CTest。

## 动机

Step 45 (M1 群体向量) 和 Step 46 (VTA 内部 RPE) 两个架构升级改变了信号流语义。
Step 44 的进化参数为旧架构优化，在新架构上崩溃 (improvement -0.207)。
需要重新进化找到适配新架构的参数。

## 进化结果

| 指标 | v44 参数 (旧架构) | v47 进化 (新架构) | Delta |
|------|-----------------|-----------------|-------|
| fitness | -0.827 | +0.323 | **+1.15** |
| late_safety | 0.000 | 0.200 | **+0.20** |
| improvement | -0.207 | **+0.198** | **+0.405** |

## 关键参数变化

| 参数 | v44 | v47 | 原因 |
|------|-----|-----|------|
| da_stdp_lr | 0.022 | **0.080** (3.6×) | VTA 脉冲 RPE 信号更弱，需要更大 LR 补偿 |
| reward_scale | 3.50 | **2.39** | Hypo→VTA 通路自带增益，外部缩放减小 |
| exploration_noise | 48 | **36** | 群体向量更高效，需要更少噪声 |
| background_ratio | 0.26 | **0.02** | attractor cos 驱动足够，不需要背景噪声 |
| lgn_gain | 394 | **234** (÷1.7) | Hypothalamus 新增信号通路 |
| brain_steps | 20 | **12** | 脉冲 RPE 传播更快 |
| v1_size | 1.12 | **1.75** | 更大 V1 产生更丰富视觉信号 |
| homeostatic_target | 3.2 | **11.0** (3.4×) | 神经元应该更活跃 |
| cortical_w_max | 0.81 | **1.42** | 皮层权重空间更大 |
| replay_da_scale | 0.31 | **0.61** | 脉冲 RPE 下重放需要更强 DA |

**da_stdp_lr 撞上限 (0.08 = 旧 max)**：已扩大基因搜索范围至 [0.005, 0.15]。

## CTest 性能改善

brain_steps 20→12 导致 CTest 总耗时 36s→22s (**39% 加速**)。

## 修改文件

- `src/engine/closed_loop_agent.h`: 全部默认参数更新为 v47 进化值
- `src/genome/genome.h`: da_stdp_lr 范围 [0.005, 0.08] → [0.005, 0.15]

## 系统状态

```
64区域 · ~252闭环神经元 · ~139投射
架构: M1 群体向量 + VTA 内部 RPE + v47 Baldwin 参数
improvement: +0.198, late_safety: 0.200
30/30 CTest (22s, 39% faster)
```
