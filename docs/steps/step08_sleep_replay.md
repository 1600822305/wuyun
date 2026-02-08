# Step 8: 睡眠 / 海马重放 / 记忆巩固

> 日期: 2026-02-07
> 状态: ✅ 完成

## 8a. 海马 Sharp-Wave Ripple (SWR) 重放

- `Hippocampus::enable_sleep_replay()` / `disable_sleep_replay()`
- SWR 机制: 睡眠模式→CA3 bias+jitter噪声→自联想补全→SWR burst
- 检测: CA3 firing fraction > threshold → SWR 事件
- 不应期: swr_refractory 步间隔，防止连续SWR
- SWR 期间: 增强活跃CA3神经元(swr_boost)延长重放
- 配置: swr_noise_amp/swr_duration/swr_refractory/swr_ca3_threshold/swr_boost
- 查询: `is_swr()`, `swr_count()`, `last_replay_strength()`
- **关键设计**: 无需显式存储模式 — CA3 STDP 自联想权重即是记忆

## 8b. 皮层 NREM 慢波振荡

- `CorticalRegion::set_sleep_mode(bool)` — 进入/退出慢波模式
- Up/Down 状态交替: ~1Hz (SLOW_WAVE_FREQ=0.001)
- Up state (40%占比): 正常处理，神经元可兴奋
- Down state (60%占比): 注入抑制电流(DOWN_STATE_INH=-8)，抑制发放
- 查询: `is_up_state()`, `slow_wave_phase()`, `is_sleep_mode()`

## 8c. 记忆巩固通路

- SWR → CA3 pattern completion → CA1 burst → SpikeBus → 皮层 L4
- 皮层 up state 期间接收重放 → 已有 STDP 增强连接 = 系统巩固
- 无需额外巩固代码 — 利用现有 SpikeBus + STDP 架构自然实现

## pybind11 新增绑定

- Hippocampus: enable/disable_sleep_replay, is_swr, swr_count, last_replay_strength, dg_sparsity
- CorticalRegion: set_sleep_mode, is_sleep_mode, is_up_state, slow_wave_phase

## 测试 (test_sleep_replay.cpp, 7测试)

1. SWR基础生成: 5次SWR (400步)
2. SWR不应期: 7次 ≤ max possible ~10 (refractory=50)
3. 清醒无SWR: count=0 ✓
4. 皮层慢波: 2次up→down, 1次down→up转换 (~1Hz)
5. Down state抑制: awake=3298 > sleep=3008 (up=2702 > down=306)
6. 编码→重放: 7次SWR, 重放活动263 (SWR期88)
7. 多区域集成: LGN→V1→dlPFC+Hipp+Hypo, V1 awake=1111 > sleep=94

## 生物学对应

- Buzsaki (1989) Two-stage model of memory formation
- Saper et al (2005) Hypothalamic sleep-wake flip-flop
- Steriade et al (1993) Cortical slow oscillation
- Wilson & McNaughton (1994) Hippocampal replay during sleep

## Step 8 剩余 (低优先级)

- ⬜ 注意力: TRN门控 + ACh + 上丘 (MB-01~02)
- ⬜ 发育/关键期: 连接修剪 + PV+成熟
- ⬜ REM睡眠: PnO (MB-11) + 梦境 + theta节律

## 系统状态

- **48区域** | **5528神经元** | **~109投射** | 4调质 | 4学习 | 预测编码 | 工作记忆 | 注意力 | 内驱力 | 意识(GNW) | **睡眠/重放**
- **142 测试全通过** (135+7), 零回归
