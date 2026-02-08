# Step 11: REM 睡眠 + 梦境

> 日期: 2026-02-07
> 状态: ✅ 完成

## 新增类

- `SleepCycleManager` (engine/sleep_cycle.h/cpp) — AWAKE→NREM→REM→NREM 完整睡眠周期管理
  - 可配置 NREM/REM 持续时间、周期增长率
  - REM 周期增长: 模拟后半夜 REM 延长 (rem_growth)
  - PGO 波随机生成 (rem_pgo_prob)
  - Theta 相位追踪

## CorticalRegion REM 扩展

- `set_rem_mode(bool)` — 去同步化噪声 (bias=18+jitter=±12) 注入 L2/3 和 L5
- `inject_pgo_wave(amplitude)` — PGO 波随机激活 L4 (梦境视觉)
- `set_motor_atonia(bool)` — M1 L5 强抑制 (-20) 防止梦境运动输出
- NREM 和 REM 互斥: set_rem_mode 自动关闭 sleep_mode

## Hippocampus REM theta 扩展

- `enable_rem_theta()` — ~6Hz theta 振荡调制 CA3/CA1
- Theta peak → CA3 drive (编码相位), Theta trough → CA1 drive (检索相位)
- 创造性重组: 1%/步概率随机激活 20% CA3 子集 (梦境联想)
- REM theta 和 NREM SWR 互斥

## 生物学

- NREM: 皮层慢波 (1Hz up/down) + 海马 SWR 重放 → 记忆巩固
- REM: 皮层去同步化 + 海马 theta + PGO 波 → 创造性重组
- NREM→REM 交替: VLPO↔PnO flip-flop 模型
- 后半夜 REM 增长: NREM 缩短, REM 延长

## 测试 (test_rem_sleep.cpp, 7测试)

1. SleepCycleManager 基础: AWAKE→NREM→REM→NREM→wake
2. REM 周期增长: Cycle0 REM=50 → Cycle1=80 → Cycle2=110
3. PGO 波: 35/500 步 (rate=0.07)
4. CorticalRegion REM: V1 REM=156, PGO=3, M1 atonia=71
5. Hippocampal theta: phase cycling, 2 recombination events
6. 完整睡眠周期: NREM 250步 + REM 100步, 1 cycle
7. 全脑 NREM→REM: NREM up=600/down=600, REM=351 spikes

## 系统状态

- **48区域** | 可扩展至44k neurons | **~109投射**
- 4调质 | 4学习 | 预测编码 | WM | 注意力 | 内驱力 | GNW | **NREM+REM完整睡眠** | 感觉输入 | 规模可扩展
- **161 测试全通过** (154+7), 零回归
