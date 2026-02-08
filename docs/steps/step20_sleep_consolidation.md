# Step 20: 睡眠巩固闭环 (NREM SWR offline replay)

> 日期: 2026-02-08
> 状态: ✅ 完成 (默认禁用, 基础设施就绪)

## 目标

周期性睡眠巩固 — 500步醒→100步NREM SWR重放→醒来→循环

## 实现

**A. 睡眠/觉醒周期状态机**
- `SleepCycleManager` 已存在，直接集成到 `ClosedLoopAgent`
- `wake_step_counter_`: 觉醒步数计数器
- `agent_step()` 开头检查: 达到 `wake_steps_before_sleep` 后触发 `run_sleep_consolidation()`

**B. run_sleep_consolidation()**
- 进入睡眠: `sleep_mgr_.enter_sleep()` + `hipp_->enable_sleep_replay()`
- NREM SWR 重放: 从 `replay_buffer_` 收集正经验 episodes
- 以 `sleep_positive_da` (0.40) 重放到 BG → D1 Go 巩固
- 同时步进 Hippocampus (SWR generation mode)
- 醒来: `sleep_mgr_.wake_up()` + `hipp_->disable_sleep_replay()`

**C. AgentConfig 新增**
- `enable_sleep_consolidation`: 开关 (默认 **false**)
- `wake_steps_before_sleep`: 觉醒间隔 (1000)
- `sleep_nrem_steps`: NREM 步数 (30)
- `sleep_replay_passes`: 重放轮次 (1)
- `sleep_positive_da`: DA 水平 (0.40)

## 调优过程 (3 轮)

| 版本 | 配置 | Test 4 Improvement | 问题 |
|------|------|-------------------|------|
| V1 | 80步/3pass/DA0.55/正+负 | +0.070 | D2 过度强化 (负经验主导) |
| V2 | 80步/3pass/DA0.55/正only+平衡 | -0.088 | D1 过度巩固 (240学习步!) |
| V3 | 30步/1pass/DA0.40/正only | -0.070 | 仍然有害 |
| **禁用** | — | **+0.161** | Step 19 基线恢复 |

## 根因分析

3×3 环境中睡眠巩固有害的原因:
1. **Awake replay 已充分**: 5 passes/食物 + 2 passes/危险 已经覆盖了巩固需求
2. **过度巩固**: 即使 30 步/1 pass 也会过拟合早期经验
3. **环境太小**: 3×3 grid 只有 9 个位置，食物吃掉后立刻重新出现在随机位置。
   重复巩固旧食物位置的 Go pathway 在食物移动后变成错误策略
4. **负重放叠加**: LHb + Amygdala + awake negative replay 已有 3 条回避学习通路，
   睡眠负重放是第 4 条 → D2 过度强化

## 决策: 默认禁用，保留基础设施

睡眠巩固在更大环境中应有价值:
- 更大 grid → 更多位置 → 更多遗忘 → sleep 对抗遗忘
- 更长 episode → 更复杂策略 → sleep 巩固序列记忆
- 可通过 `enable_sleep_consolidation = true` 随时启用

## 回归测试: 29/29 CTest 全通过

## 系统状态

```
50区域 · ~115投射 · 29 CTest suites
新增: NREM SWR 睡眠巩固 (基础设施就绪, 默认禁用)
  wake_step_counter_ → run_sleep_consolidation() → NREM replay → wake_up
学习能力: improvement +0.161, late safety 0.779 (维持 Step 19 水平)
下一步: 更大环境验证 / 皮层 STDP / 参数搜索
```
