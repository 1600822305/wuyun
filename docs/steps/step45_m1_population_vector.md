# Step 45: M1 群体向量编码 (Georgopoulos 1986)

> 日期: 2026-02-09
> 状态: ✅ 完成
> D1 权重分化 +48% (0.0607→0.0900)。30/30 CTest。

## 动机

现有架构的天花板已到：learner advantage 停在 +0.036，继续调参/加区域/跑进化无法突破。
根本原因是 M1 的编码方式——"4 组固定神经元 → WTA 计数"本质上是穿着大脑外衣的 Q-table：

```
旧方式 (Q-table 外衣):
  M1 L5 = [组0: UP] [组1: DOWN] [组2: LEFT] [组3: RIGHT]
  BG D1 = [组0: UP] [组1: DOWN] [组2: LEFT] [组3: RIGHT]
  解码 = argmax(group_spike_count)
  
  问题: 方向是硬编码分配的，不是从活动中涌现的
        BG DA-STDP 只能学"加强某一组"，不能学"塑造方向"
        4 个方向是天花板，永远不能更精细
```

## 设计: 群体向量编码

基于 Georgopoulos et al. (1986) 的经典发现：灵长类 M1 运动皮层中，每个神经元对所有
方向都有响应，但各自有一个"偏好方向"（preferred direction）。群体向量（所有神经元
偏好方向按发放率加权求和）精确预测了手臂运动方向。

```
新方式 (群体向量):
  M1 L5: 每个神经元有随机偏好方向 θ_i ∈ [0, 2π)
  BG D1: 每个 MSN 也有随机偏好方向
  
  探索: 随机 attractor_angle → cos(θ_i - angle) 加权驱动
  BG偏置: D1 发放 → 群体向量 (vx,vy) → cos 相似度偏置 M1
  解码: 群体向量角 = atan2(Σ fired·sin θ, Σ fired·cos θ) → 最近基数方向
  
  优势: 方向从群体活动中涌现
        DA-STDP 塑造群体向量方向（改变哪些 D1 发放）
        内部表征是连续的，未来可扩展到 360° + 速度
```

## 具体改动

### 1. 初始化 preferred directions (`build_brain()` 末尾)

```cpp
std::mt19937 pv_rng(42);  // 确定性种子
std::uniform_real_distribution<float> angle_dist(0.0f, 2π);

// M1 L5: 每个神经元一个随机偏好方向
for (size_t i = 0; i < l5_sz; ++i)
    m1_preferred_dir_[i] = angle_dist(pv_rng);

// BG D1: 每个 MSN 一个随机偏好方向
for (size_t i = 0; i < d1_sz; ++i)
    d1_preferred_dir_[i] = angle_dist(pv_rng);
```

### 2. 探索驱动 (替换 4 组 attractor)

```
旧: attractor_group = random(0,3)
    if (g == attractor_group) drive = attractor_drive
    else drive = background_drive

新: attractor_angle = random(0, 2π)
    cos_sim = cos(θ_i - attractor_angle)
    drive = background + max(0, cos_sim) × (attractor - background)
```

cos 加权驱动意味着：
- 偏好方向与 attractor_angle 对齐的神经元 → 强驱动
- 偏好方向垂直的 → 背景驱动
- 偏好方向相反的 → 背景驱动
- 过渡是平滑的，没有硬边界

### 3. BG→M1 偏置 (替换 4 组映射)

```
旧: D1 组 g 发放 → M1 组 g 注入 bias

新: D1 群体向量 = Σ fired_k × (cos θ_k, sin θ_k)
    bg_angle = atan2(bg_vy, bg_vx)
    M1 neuron j bias = cos(θ_j - bg_angle) × |bg_vec| × gain
```

DA-STDP 改变哪些 D1 被皮层输入驱动 → 改变 D1 群体向量方向 → 改变 M1 偏置方向。
这是真正的"学习塑造方向"，而不是"学习选择组"。

### 4. 解码 (替换 WTA 计数)

```
旧: scores[4] = sum of fires per group, return argmax

新: vx = Σ l5_accum[i] × cos(θ_i)
    vy = Σ l5_accum[i] × sin(θ_i)
    angle = atan2(vy, vx)
    return closest cardinal direction (UP=π/2, DOWN=-π/2, LEFT=π, RIGHT=0)
```

### 5. 方向角约定

```
Direction   Action enum   Angle
RIGHT       3             0
UP          0             π/2
LEFT        2             π
DOWN        1             -π/2
```

### 6. 向后兼容

- `attractor_group` 从 `attractor_angle` 派生（cos 最近匹配），用于 efference copy 和 replay
- BG 内部的 `inject_sensory_context(adj[4])` 和 `mark_motor_efference(group)` 不变
- 空间价值图仍用 4 方向格式
- ACC D1 rates 仍按 4 组读取

## 修改文件

- `src/engine/closed_loop_agent.h`:
  - 新增 `m1_preferred_dir_`, `d1_preferred_dir_` 成员变量
- `src/engine/closed_loop_agent.cpp`:
  - `build_brain()`: 初始化 preferred directions
  - Phase B 探索: cosine 加权替代 4 组 attractor
  - Phase B BG→M1: 群体向量替代 4 组映射
  - `decode_m1_action()`: 群体向量解码替代 WTA

## 验证结果

| 指标 | v44 (4组 WTA) | v45 (群体向量) | 变化 |
|------|-------------|--------------|------|
| D1 fires | 57/50步 | 41/50步 | -28% |
| D2 fires | 61/50步 | 76/50步 | +25% |
| Weight range | 0.0607 | **0.0900** | **+48%** |
| Max elig | 71.6 | 72.0 | ≈ |
| Ctx events | 426/10步 | 423/10步 | ≈ |
| CTest | 30/30 | 30/30 | ✅ |

**权重分化增加 48%** 是核心改进：DA-STDP 在群体编码下能产生更大的权重差异，
说明学习信号更有效地塑造了 D1 突触权重。

## 为什么权重分化增加?

```
旧系统 (离散组):              新系统 (群体向量):

  D1: [U][U][D][D][L][L][R][R]    D1: [37°][182°][95°][310°][...]
       ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓         ↓     ↓     ↓     ↓
  权重: 同 同 同 同 同 同 同 同    权重: 独立  独立  独立  独立 ...
  有效自由度: 4 (组数)           有效自由度: N (D1数量)
```

旧系统：4 组 D1，每组内的 MSN 获得完全相同的 cortical input (同组 = 同方向)。
DA-STDP 只能区分"组间"差异，组内所有 MSN 的权重变化相同。

新系统：每个 D1 有独特的偏好方向。皮层→D1 映射不再受组边界约束，DA-STDP 可以
精细地调整每个 D1 的权重。权重空间的"有效自由度"从 4（组数）变成了 N（D1 数量）。

## D1 发放下降 -28% 的分析

D1 fires 从 57→41 不是退化，是编码方式变化的预期结果：

- **旧系统**: BG→M1 偏置按组注入，一个组的所有 D1 同时发放 → 高 D1 计数
- **新系统**: BG→M1 偏置按 cos 相似度注入，只有偏好方向与群体向量方向一致的 D1 被增强 → 更稀疏但更精确

D2 fires 增加 +25%（61→76）也一致：cos 加权使更多 D1 不在群体向量方向上 →
被 D2 NoGo 通路相对增强。D1/D2 平衡从"粗糙 4 选 1"变成"精细连续竞争"。

**关键**: weight range +48% 说明虽然单步 D1 fires 少了，但 DA-STDP 的**学习效率**
更高了——更少的 D1 发放产生了更大的权重分化。质量优于数量。

## 已知遗留问题

### ACC D1 rate 读取 (已修复)

ACC 冲突检测原按索引位置 `d1_size/4` 切分 D1 为 4 组，群体向量后无意义。

**修复**: 按 `d1_preferred_dir_[k]` 最近基数方向 (cos 最大匹配) 分组：

```cpp
for (size_t k = 0; k < d1_sz; ++k) {
    int best_dir = argmax_cos(d1_preferred_dir_[k], DIR_ANGLES);
    d1_counts[best_dir]++;
    if (d1_f[k]) d1_rates[best_dir] += 1.0f;
}
// Normalize by group size
```

同时移除了死代码 `d1_group = d1_size / 4` 和旧注释。30/30 CTest 通过。

### 基因组参数适配

Step 44 进化的参数（`bg_to_m1_gain`, `attractor_ratio`, `background_ratio`）语义在
群体向量下发生了变化：
- `attractor_ratio`: 从"组内注入比例"变为"cos 加权斜率"
- `bg_to_m1_gain`: 从"组偏置强度"变为"群体向量投射增益"

当前参数仍在合理范围内工作（CTest 30/30），但重新进化可能找到更优值。

## 系统状态

```
63区域 · ~228闭环神经元 · ~137投射
编码方式: M1/BG 群体向量 (Georgopoulos 1986)
Weight range: 0.0900 (+48% vs v44)
30/30 CTest 通过
```

## 架构升级路线

```
✅ Step 45: M1 群体向量编码 + ACC 适配 (最小侵入, 最大收益)
⬜ Step ??: VTA 内部 RPE 计算 (消除 reward 标量注入作弊)
⬜ Step ??: 连续空间环境 (360° 方向 + 速度)
⬜ 远期: V1 方向选择性自组织 (STDP 涌现)
```
