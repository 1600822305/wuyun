# Step 3 + 3.5: 核心回路 — 最小可工作大脑

> 日期: 2026-02-07
> 状态: ✅ 完成

## 架构层
- `BrainRegion` 基类 — 统一接口: 注册SpikeBus + step/receive/submit
- `CorticalRegion` — CorticalColumn 的 BrainRegion 包装 + PSP 输入缓冲
- `SimulationEngine` — 全局时钟 + SpikeBus 编排 + step循环

## 脑区
- `ThalamicRelay` (LGN) — Relay+TRN 双群体, Tonic/Burst 切换
- V1, dlPFC (CorticalRegion 实例)
- `BasalGanglia` — D1/D2 MSN + GPi/GPe + STN, Go/NoGo/Hyperdirect
- MotorThalamus, M1
- `VTA_DA` — DA 神经元 + RPE + DA level

## Step 3.5 反作弊修复
- BG `id%5` 硬编码 → 随机稀疏映射表
- DA→BG 调制走 SpikeBus

## 验证
- 7 区域, 906 神经元, 7 投射
- 信号传播: LGN→V1 ✓, DA 调制 3× ✓, TRN 门控 95%抑制 ✓
- 26 测试全通过
