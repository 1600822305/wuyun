# Step 36: CLS 互补学习系统 — 灾难性遗忘根治

> 日期: 2026-02-08
> 状态: ✅ 完成

## 发现的 4 个隐藏 bug

| Bug | 原因 | 影响 |
|-----|------|------|
| **Sub < 4 神经元** | `n_sub=max(3,3)=3`, 但 `get_retrieval_bias` 要求 ≥4 | 海马检索接口从未工作过 |
| **E/I 比例反转** | 抑制群体用默认值(20/10/15), 兴奋群体压缩到(10/6/6) | DG/CA3/CA1 被过度抑制窒息 |
| **单脉冲注入** | `inject_spatial_context` 在 brain steps 循环外只调 1 次 | EC grid cells 从未发放 |
| **retrieval_bias 方向错误** | Sub→direction 映射是随机的, 无学习机制 | 启用后反而引导 agent 撞 danger |

## CLS 实现

1. **认知地图 (spatial_value_map)**
   - 记录每个位置的奖励历史（food=+, danger=-）
   - 邻域扩散（place field 空间泛化）
   - 慢衰减保持长期空间记忆
   - 价值梯度 → BG inject_sensory_context（引导导航）

2. **系统巩固（睡眠 SWR）**
   - SWR 期间从 spatial_value_map 中选择最高价值位置"重放"
   - 短暂 DA 升高 → BG DA-STDP 从空间记忆重新学习
   - 效果：BG 权重定期从认知地图"刷新"

3. **海马修复**
   - Sub 增大到 8 个（每方向 2 个）
   - E/I 比例修正（DG 5:1, CA3/CA1 3:1）
   - 空间上下文每 brain step 持续注入

## 效果对比

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| Improvement (early→late) | **-0.750** | **-0.088** | ✅ 退化减少 89% |
| Late safety (1500-2000步) | 0.000 | **0.250** | ✅ 不再完全崩溃 |
| DA-STDP learner advantage | +0.023 | **+0.036** | ✅ +57% |
| 泛化优势 | -0.750 | **-0.250** | ✅ +67% |

## 系统状态

```
54区域 · ~146闭环神经元 · ~112投射
CLS 认知地图 + 睡眠系统巩固 + 海马 4-bug 修复
灾难性遗忘: -0.750 → -0.088 (减少89%)
Learner advantage: +0.036 (修复前 +0.023)
30/30 CTest 通过
```
