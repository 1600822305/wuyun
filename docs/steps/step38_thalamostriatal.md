# Step 38: 丘脑-纹状体直接通路 + ACh 门控巩固

> 日期: 2026-02-09
> 状态: ✅ 完成
> 核心突破: D1 发放 2→36 (18×), 权重范围 3.7×, 2000步首次正向改善 +0.212

## 问题诊断

**视觉层级延迟超过 brain_steps_per_action**
```
LGN → V1 → V2 → V4 → IT → dlPFC → BG
 d=2   d=2   d=2   d=2   d=2    d=2
           最少需要 ~14 步到达 BG
```
- `brain_steps_per_action=12` < 14 步延迟
- 结果: ~2.3 cortical events/step, D1 发放 2 次/950 engine steps

## 解决方案: LGN→BG 直接投射

**人脑解剖学事实** (Smith et al. 2004, Lanciego et al. 2012):
- 板内核群 (CM/Pf) 直接投射到纹状体 MSN
- 编码行为显著性，不编码方向
- MSN up-state 需要丘脑+皮层两条通路协同

```
快通路 (thalamostriatal): LGN → BG (delay=1, 1跳)
  → MSN 获得"有东西出现"的粗糙信号 → 维持 up-state firing
  → 不参与 DA-STDP 学习

慢通路 (corticostriatal): LGN → V1 → ... → dlPFC → BG (delay=14, 6跳)
  → MSN 获得精细方向信号 → DA-STDP 学习的是这条通路的权重
```

## 实现细节

**BG::receive_spikes() 区分丘脑/皮层脉冲:**
- 丘脑脉冲: 广泛投射到 ALL D1/D2 神经元, 弱电流 8.0 pA, 不标记 input_active_
- 皮层脉冲: 通过学习权重的拓扑映射, 30 pA, 标记 input_active_ 参与 DA-STDP

## 修改文件

- `src/engine/closed_loop_agent.cpp`: 添加 `LGN→BG` 投射 (delay=1) + 注册丘脑源
- `src/region/subcortical/basal_ganglia.h`: 添加 `thalamic_source_`, `THAL_MSN_CURRENT`, `set_thalamic_source()`
- `src/region/subcortical/basal_ganglia.cpp`: receive_spikes() 区分丘脑/皮层通路

## 效果对比

| 指标 | Step 37 (VTA修复) | Step 38 (丘脑通路) | 变化 |
|------|--------|--------|------|
| D1 fires (50步) | 2 | **36** | **18×** |
| D2 fires (50步) | 3 | **36** | **12×** |
| Max eligibility | 6.6 | **10.9** | +65% |
| Weight range | 0.0045 | **0.0165** | **3.7×** |
| 2000步 Improvement | -0.350 | **+0.212** | ✅ **正值！** |

## Step 38b: ACh 门控巩固 — 反转学习与遗忘保护兼容

> 解决: 突触巩固(STC)阻止反转学习的矛盾

**生物学方案**: ACh 信号区分"保持"和"更新"模式 (Hasselmo 1999, Yu & Dayan 2005)
- 低 ACh（平时）→ 巩固保护完整 → 防遗忘
- 高 ACh（意外/新环境）→ 降低巩固保护 → 允许反转

**实现**:
1. `BG::set_ach_level()`: ACh 水平接口
2. `ach_gate = clamp(1 - (ach-0.2)×2, 0.1, 1.0)` (ACh=0.7 → 90% 减保护)
3. 反向 Δw 主动侵蚀巩固分数 (2× 建设速率)
4. Phase A 结束后重置 ACh 到 baseline

**结果**: 巩固开启下反转学习 B/A +0.027 (vs 关闭巩固 +0.018, +50%)
30/30 CTest 通过

## 系统状态

```
54区域 · ~146闭环神经元 · ~113投射
丘脑-纹状体直接通路 + ACh 门控巩固
D1 发放 36/50步 (18× 提升)
2000步 Improvement: +0.212 (首次正值!)
30/30 CTest 通过
```
