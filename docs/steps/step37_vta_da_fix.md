# Step 37: VTA DA 信号通路修复 + 皮层→MSN PSP 衰减

> 日期: 2026-02-09
> 状态: ✅ 完成
> 核心修复: DA 从恒定 0.300 → 动态变化, D1 发放从 0 → 2, 权重范围 5.6×

## 问题诊断

**Bug 1: VTA DA 负 phasic 只持续 1 个 engine step**
- 正奖励通路正常: `reward_psp_` > 0 → DA 神经元持续发放 → DA 升高多步
- 负奖励通路断裂: `reward_psp_` < 0 → DA 神经元不发放 → `phasic_positive = 0`
  但 `phasic_negative` 依赖 `last_rpe_`，而 `reward_input_` 在第一步后重置为 0
  → DA 在第 2 步回到 0.3
- 结果: DA dip 只持续 1 个 engine step

**Bug 2: Phase A 中 BG 在 VTA 处理奖励前读取 DA**
```
bg_->set_da_level(vta_->da_output());  // 读取旧值 0.3
engine_.step();                         // VTA 在这里处理奖励 → DA 变化
```

**Bug 3: 皮层→MSN PSP 衰减太快 (0.7, 半衰期 1.9 步)**
- MSN tau_m=20，从 V_rest=-80 充电到 V_thresh=-50 需要 ~15 步
- PSP 衰减 0.7 → 皮层 spike 的 PSP 在 ~3 步后几乎为 0
- D1 来不及充电就失去驱动 → D1 发放 0 次

## 修复方案

**Fix 1: Firing-rate-based DA (Grace 1991, Schultz 1997)**
```
phasic = (firing_rate - tonic_firing_smooth_) * phasic_gain * 3.0
da_level_ = clamp(tonic_rate + phasic - lhb_suppression, 0, 1)
```

**Fix 2: Warmup 期 (前 50 engine steps)**
- DA 保持 tonic_rate (0.3)，避免虚假 D2 强化
- 用 α=0.1 快速收敛 tonic_firing_smooth_

**Fix 3: Phase A 时序修正**
```
engine_.step();
bg_->set_da_level(vta_->da_output());  // 改为步进后读取
```

**Fix 4: 皮层→MSN PSP 衰减 0.7 → 0.9 (半衰期 6.6 步)**

## 修改文件

- `src/region/neuromod/vta_da.h`: 添加 `tonic_firing_smooth_`, `step_count_`, `WARMUP_STEPS`
- `src/region/neuromod/vta_da.cpp`: firing-rate-based DA + warmup
- `src/engine/closed_loop_agent.cpp`: Phase A 时序修正
- `src/region/subcortical/basal_ganglia.h`: 添加 `CTX_MSN_PSP_DECAY=0.9`
- `src/region/subcortical/basal_ganglia.cpp`: D1/D2 用 CTX_MSN_PSP_DECAY

## 效果对比

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| DA level | **0.300 恒定** | **0.273-0.300 (动态)** | ✅ DA 通路激活 |
| D1 fires (50步) | 0 | 2 | ✅ 从零到非零 |
| D2 fires (50步) | 0 | 3 | ✅ |
| Max eligibility | 0.0 | 6.6 | ✅ |
| Weight range | 0.0008 | **0.0045** | ✅ 5.6× |
| Learner advantage | ~0 | **+0.020** | ✅ |

## 遗留问题

- D1 MSN 发放仍然稀疏 (2/950 engine steps)
- 需要更多皮层→BG 通路或更高皮层发放率

## 系统状态

```
54区域 · ~146闭环神经元 · ~112投射
VTA DA: firing-rate-based (修复 3 个 bug)
皮层→MSN PSP: 0.7 → 0.9 (D1 可以发放)
30/30 CTest 通过
```
