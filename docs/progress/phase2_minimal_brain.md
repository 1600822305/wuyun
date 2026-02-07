# Phase 2: 最小可工作大脑 — 核心回路

> 对应: Step 3 / Step 3.5
> 时间: 2026-02-07
> 里程碑: 7 区域 · 906 神经元 · 7 投射 · 26 测试全通过

---

## Step 3: 核心回路 — 最小可工作大脑 ✅

> 目标: 感觉→认知→动作的最短通路能跑通

**架构层 (新增):**
- ✅ `BrainRegion` 基类 — 统一接口: 注册SpikeBus + step/receive/submit + 振荡/调质
- ✅ `CorticalRegion` — CorticalColumn 的 BrainRegion 包装 + PSP 输入缓冲
- ✅ `SimulationEngine` — 全局时钟 + SpikeBus 编排 + step循环

**3a. 感觉-认知通路:**
- ✅ `ThalamicRelay` (LGN) — Relay+TRN 双群体, Relay↔TRN 互连突触, Tonic/Burst 切换
- ✅ V1 (CorticalRegion 实例, 270 神经元)
- ✅ dlPFC (CorticalRegion 实例, 202 神经元)

**3b. 动作选择通路:**
- ✅ `BasalGanglia` — D1/D2 MSN + GPi/GPe + STN, Direct/Indirect/Hyperdirect 三条通路
- ✅ MotorThalamus (ThalamicRelay 实例)
- ✅ M1 (CorticalRegion 实例, 169 神经元)

**3c. 奖励信号:**
- ✅ `VTA_DA` — DA 神经元 + RPE 计算 + DA level 输出 + phasic/tonic
- ✅ DA→BG D1/D2 调制: DA↑=D1增强(Go), DA↓=D2增强(NoGo)

**端到端验证 (5 测试全通过):**
- 构造: 7 区域, 906 神经元, 6 投射
- 沉默: 无输入→全系统安静
- **信号传播**: 视觉(35.0)→LGN(124 spikes)→V1(23 spikes) ✓
- **DA 调制**: DA=0.1→D1=50, DA=0.6→D1=150 (3倍增强) ✓
- **TRN 门控**: 正常=60, TRN抑制=3 (95%抑制) ✓
- **26 测试全通过** (9 neuron + 6 column + 6 foundation + 5 minimal_brain)

---

## Step 3.5: 反作弊修复 ✅

> 根据 00_design_principles.md §6 审计

- ✅ BG `receive_spikes` 中 `id%5` hyperdirect 硬编码 → 构造时随机稀疏映射表 (`ctx_to_d1/d2/stn_map_`)
- ✅ DA→BG 调制走 SpikeBus: VTA 脉冲 → BG `receive_spikes` 自动推算 DA 水平 (`da_spike_accum_` + 指数平滑)
- ✅ VTA→BG 投射添加到 SimulationEngine (delay=1)
- ✅ 7 条投射 (原6条 + VTA→BG), 26 测试全通过

---

## Phase 2 总结

| 指标 | 数值 |
|------|------|
| 区域 | 7 (LGN · V1 · dlPFC · BG · MotorThal · M1 · VTA) |
| 神经元 | 906 |
| 投射 | 7 |
| 测试 | 26 通过 |
| 新增类 | BrainRegion · CorticalRegion · SimulationEngine · ThalamicRelay · BasalGanglia · VTA_DA |
| 关键验证 | 信号传播 · DA调制 · TRN门控 · 反作弊修复 |