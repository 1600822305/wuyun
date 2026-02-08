# Step 22: D1 侧向抑制 — MSN 竞争归一化

> 日期: 2026-02-08
> 状态: ✅ 完成

## 目标

解决"方向权重趋同"问题 — D1 子群间 GABA 侧向抑制实现竞争性动作选择。

## 生物学基础 (Humphries et al. 2009, Wickens et al. 2007)

纹状体 MSN 之间有 ~1-3% 的 GABAergic 侧枝连接 (collateral synapses)。
这些连接在功能上实现了 **动作通道间的竞争**：

```
无侧向抑制:
  食物在左 → D1-LEFT 被强化
  D1-RIGHT/UP/DOWN 不受影响
  → 长期所有方向权重趋同 → 无法形成稳定偏好

有侧向抑制:
  食物在左 → D1-LEFT 活跃 → GABA 抑制 D1-RIGHT/UP/DOWN
  → D1-LEFT 权重上升, 其他方向权重相对下降
  → 方向选择性涌现!
```

## 实现

**BasalGangliaConfig 新增:**
- `lateral_inhibition = true` — 启用 D1/D2 子群间竞争
- `lateral_inh_strength = 8.0f` — GABA 侧枝抑制电流强度

**BasalGanglia::step() 新增 (在 GPi/GPe 注入前):**
1. 统计 D1/D2 每个方向子群的上一步发放数
2. 找到最活跃的子群 (winner)
3. 其他子群按 `(max_fires - group_fires) × strength` 注入负电流
4. D2 同理 (winning NoGo 通道抑制其他 NoGo)

## 结果对比

| 指标 | Step 21 (无侧向抑制) | **Step 22 (有侧向抑制)** | 变化 |
|------|---------------------|------------------------|------|
| Test 1 (5k) late safety | 0.47 | **0.56** | **+19%** |
| Test 1 total food | 89 | **112** | **+26%** |
| Test 4 (10k) improvement | **-0.086** | **+0.005** | **修复退化!** |
| Test 4 late safety | 0.336 | **0.469** | **+40%** |
| Test 4 total food | 187 | **162** | -13% (更谨慎) |
| Test 4 total danger | 301 | **258** | **-14%** |

**关键突破: 10k 训练不再退化!** improvement 从 -0.086 → +0.005

## 为什么有效

1. **方向选择性涌现**: 被奖励的方向 D1 子群在 DA-STDP 后更活跃 → 通过侧向抑制压制其他方向
2. **信用分配聚焦**: 侧向抑制减少了"无关 D1 子群也被强化"的问题
3. **防止权重趋同**: 长时训练中，侧向抑制持续维护方向间的权重差异

## 修改文件

- `src/region/subcortical/basal_ganglia.h` — 新增 `lateral_inhibition`, `lateral_inh_strength` 配置
- `src/region/subcortical/basal_ganglia.cpp` — step() 中新增 D1/D2 子群竞争逻辑

## 回归测试: 29/29 CTest 全通过, 零回归

## 系统状态

```
50区域 · 自适应神经元数 · ~115投射 · 29 CTest suites
默认环境: 10×10 grid, 5×5 vision, 5 food, 3 danger
新增: D1/D2 侧向抑制 (MSN collateral GABA, 方向竞争)
5k learning: improvement +0.16, late safety 0.56 (+19% vs Step 21)
10k learning: improvement +0.005 (修复了 Step 21 的 -0.086 退化)
```
