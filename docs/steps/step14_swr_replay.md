# Step 14: Awake SWR Replay — 经验重放记忆巩固

> 日期: 2026-02-08
> 状态: ✅ 完成

## 生物学基础

清醒状态的海马 SWR (awake sharp-wave ripples) 在奖励事件后 100-300ms 内发生，
以压缩时间尺度重放最近的奖赏关联空间序列 (Foster & Wilson 2006, Jadhav et al. 2012)。
这不是简单的"重复当前经验"，而是**巩固旧的成功记忆**——对抗突触权重衰减导致的遗忘。

## 核心设计

```
信号流:
  正常学习: dlPFC spikes → SpikeBus → BG receive_spikes → DA-STDP (1次)
  SWR 重放: 存储的 dlPFC 快照 → BG receive_spikes → replay_learning_step → DA-STDP (×N)

关键决策:
  1. 只重放正奖励 (食物) 事件，不重放负奖励 (危险)
  2. 重放旧成功经验，不重放当前 episode
  3. 轻量级 replay_learning_step: 只步进 D1/D2 + DA-STDP
```

## 新增/修改文件

| 文件 | 变更 |
|------|------|
| `src/engine/episode_buffer.h` | **新增** EpisodeBuffer, SpikeSnapshot, Episode |
| `src/region/subcortical/basal_ganglia.h` | 新增 replay_mode_, set_replay_mode(), replay_learning_step() |
| `src/region/subcortical/basal_ganglia.cpp` | 实现 replay_learning_step(), apply_da_stdp 中 replay_mode 跳过 w_decay |
| `src/engine/closed_loop_agent.h` | 新增 replay 配置参数, replay_buffer_, capture/replay 方法 |
| `src/engine/closed_loop_agent.cpp` | brain loop 中记录 dlPFC spikes, 奖励后触发 run_awake_replay |

## 调优过程

| 尝试 | replay_passes | da_scale | 策略 | improvement |
|------|:---:|:---:|------|:---:|
| v1: 重放当前, full step | 8 | 0.6 | 重放当前 ep + 正负奖励 | -0.004 |
| v2: 只正奖励, full step | 8 | 0.6 | 只正奖励 + bg->step() | -0.019 |
| v3: 轻量 replay_learning_step | 8 | 0.6 | + replay_learning_step | +0.108 |
| v4: 降低强度 | 3 | 0.3 | 减少 passes/scale | -0.034 |
| v5: 重放旧经验 | 3 | 0.3 | 重放旧成功 ep | +0.079 |
| **v6: 最终** | **5** | **0.5** | **重放旧 + 中等强度** | **+0.120** |

## 关键调优教训

1. **不能重放当前 episode** — 与 Phase A 双重学习导致过拟合
2. **不能重放负奖励** — D2 NoGo 通路过度强化导致行为瘫痪
3. **不能用 bg->step() 重放** — GPi/GPe/STN 状态被破坏
4. **replay_learning_step 是关键** — 只步进 D1/D2 + DA-STDP

## 结果

```
improvement: +0.077 → +0.120 (+56%)
late safety: 0.524 → 0.667 (+27%)
```

29/29 CTest 全通过。
