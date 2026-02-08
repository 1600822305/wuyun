# Step 39: 皮层信号链三修复 — 信号穿透五级层级

> 日期: 2026-02-09
> 状态: ✅ 完成
> 核心突破: 皮层 events 34→358 (10.5×), eligibility 10.9→63.5 (5.8×), weight range 2.8×

## 问题诊断

**皮层信号在 5 级层级中衰减到几乎为零**

定量追踪每一级的神经元数量:
```
层级    总神经元  L4(输入)  L5(输出)
LGN     25        -         -
V1      25        6         5
V2      15        3         3
V4       8        2         1
IT       8        2         1
dlPFC   12        3         2
```

### 瓶颈 ①: brain_steps 不够

```
brain_steps_per_action = 12
层级最小延迟 = 14 步 (6 跳 × delay=2 + 内部处理)
→ 皮层信号在一个 agent step 内走不完全程
```

SpikeBus delay buffer 跨 agent step 保持 (pipeline 效应存在), 但每步只有上一步残留脉冲到达 BG, 信号量极低。

### 瓶颈 ②: 皮层 PSP_DECAY=0.7 太快

与 BG 的 CTX_MSN_PSP_DECAY 问题完全相同:
- PSP_DECAY=0.7 → 半衰期 1.9 步
- V4 L4 只有 2 个神经元, 收到 1-2 个 spike
- PSP 在 ~3 步后衰减到 0, L4 来不及充电到阈值
- 信号在 V2→V4→IT 逐级消亡

### 瓶颈 ③: L4/L5 最小值 2 个神经元

- V4 L5 = 1 个神经元, IT L5 = 1 个神经元
- 单个神经元因随机因素未发放 → 整条信号链断裂
- 无冗余保障

## 解决方案

三个根本修复, 不删除任何层级:

### 修复 A: brain_steps_per_action 12→20

```cpp
// closed_loop_agent.h
size_t brain_steps_per_action = 20;  // v39: 12→20
size_t reward_processing_steps = 10; // v39: 7→10 (按比例)
```

层级延迟 14 步 < brain_steps 20 步, 留出 6 步皮层→BG 有效重叠。
时序: 丘脑 step 1 → MSN up-state; 皮层 step 14-20 → 方向特异 STDP 配对。

### 修复 B: 皮层 PSP_DECAY 0.7→0.85

```cpp
// cortical_region.h
static constexpr float PSP_DECAY = 0.85f;  // v39: 0.7→0.85
```

半衰期 1.9→4.3 步。同 BG CTX_MSN_PSP_DECAY 修复原理:
小网络 (L4=2-3 神经元) 需要更慢的突触后电位积分时间。

### 修复 C: L4/L5 最小值 2→3

```cpp
// closed_loop_agent.cpp add_ctx lambda
c.n_l4_stellate  = std::max<size_t>(3, N * 25 / 100) * s;
c.n_l5_pyramidal = std::max<size_t>(3, N * 20 / 100) * s;
```

3 个神经元提供冗余: 即使 1 个因随机因素未发放, 剩余 2 个仍可传递信号。

## 修改文件

- `src/engine/closed_loop_agent.h`: brain_steps 12→20, reward_steps 7→10
- `src/region/cortical_region.h`: PSP_DECAY 0.7→0.85
- `src/engine/closed_loop_agent.cpp`: L4/L5 min 2→3

## 效果对比

| 指标 | Step 38 | Step 39 | 变化 |
|------|---------|---------|------|
| Cortical events/10步 | 34 | **358** | **10.5×** |
| D1 fires (50步) | 36 | **47** | +31% |
| D2 fires (50步) | 36 | **51** | +42% |
| Max eligibility | 10.9 | **63.5** | **5.8×** |
| Weight range | 0.0165 | **0.0464** | **2.8×** |

## 为什么有效

1. brain_steps=20: 皮层信号在 step 14 到达 BG, 还有 6 步可以 co-activate
2. PSP_DECAY=0.85: L4 有 ~4 步积分时间, V4/IT 的 2-3 个 L4 神经元可以充电到阈值
3. L4/L5 min=3: 每层至少 3 个输入/输出神经元, 信号链不因单点故障断裂
4. 三修复协同: 时间充足 + 信号不衰减 + 冗余传递 = 10.5× 皮层 events 到达 BG

## 系统状态

```
54区域 · ~160闭环神经元 · ~113投射
皮层信号穿透五级层级 (events 10.5× 提升)
D1 发放 47/50步, Max elig 63.5, Weight range 0.0464
30/30 CTest 通过
```
