# Step 46: VTA 内部 RPE 计算 (Schultz 1997)

> 日期: 2026-02-09
> 状态: ✅ 完成
> 消除系统最后一个"上帝视角"：VTA 不再接收 agent 注入的 reward 标量。30/30 CTest。

## 动机

系统中最大的反作弊违规：agent 直接把 reward 数字塞给 VTA。

```
旧流程 (作弊):
  GridWorld → reward → agent → vta_->inject_reward(scalar) → VTA 计算 RPE
  VTA 直接看到原始 reward 数字 = "上帝视角"

新流程 (合规):
  GridWorld → reward → agent → hypo_->inject_hedonic(reward) → LH 神经元发放
  LH → SpikeBus (d=1) → VTA hedonic_psp_ (实际奖赏)
  OFC → SpikeBus (d=2) → VTA prediction_psp_ (预期价值)
  VTA: RPE = hedonic_psp_ - prediction_psp_ → DA burst/pause
```

奖赏信号通过下丘脑 LH 进入（感觉接口，类比 LGN 之于视觉），VTA 只看到脉冲模式。

## 设计

### 关键洞察: SpikeEvent.region_id

`SpikeEvent` 已有 `region_id` 字段（源区域 ID）。VTA 在 `receive_spikes()` 中可以按
`region_id` 区分来自 Hypothalamus 的脉冲（实际奖赏）和来自 OFC 的脉冲（预期价值），
无需任何核心基础设施改动。

### 生物学基础

- **LH → VTA**: 外侧下丘脑谷氨酸能投射驱动 VTA DA 释放 (Nieh et al. 2015, Nature)
- **OFC → VTA**: 眶额皮层提供预期价值信号 (Takahashi et al. 2011, Neuron)
- **RPE = 兴奋 - 抑制**: DA 神经元接收汇聚的兴奋性（实际奖赏）和抑制性（预期价值）输入，净驱动 = RPE (Schultz 1997)

## 实现

### 1. Hypothalamus: `inject_hedonic(float reward)`

- 正奖赏 → LH 神经元兴奋（食物满足，按 |reward| 比例）
- 负奖赏 → PVN 神经元兴奋（疼痛/应激）
- 类比 `VisualInput::encode_and_inject()` — "身体"将环境信号转为神经信号

### 2. VTA: 脉冲驱动 RPE

- 删除 `inject_reward()` 和 `set_expected_reward()`（反作弊违规 API）
- 新增 `register_hedonic_source(region_id)` 和 `register_prediction_source(region_id)`
- `receive_spikes()` 按 `evt.region_id` 分流:
  - Hypothalamus → `hedonic_psp_`（兴奋性，"实际奖赏到了"）
  - OFC → `prediction_psp_`（抑制性，"预期奖赏"→ 抑制 DA 惊讶）
  - 其他源 → `psp_da_`（通用皮层/纹状体调制）
- `step()` 中 DA 神经元驱动:
  - `net_drive = 20.0 + hedonic_psp_ - prediction_psp_ * 0.7 + psp_input - lhb_inh_psp_`
  - RPE 从脉冲率差值涌现，不从标量计算

### 3. 新投射 + Hypothalamus 加入闭环

- Hypothalamus 首次加入 ClosedLoopAgent（之前只在 build_standard_brain 中）
- Hypothalamus → VTA (delay=1, 快速享乐信号)
- OFC → VTA (delay=2, 价值预测)
- VTA 注册源: `register_hedonic_source(hypo_id)`, `register_prediction_source(ofc_id)`

### 4. Agent 奖赏路由改变

- `inject_reward(reward)` 内部从 `vta_->inject_reward(r)` 改为 `hypo_->inject_hedonic(r)`
- VTA 永远看不到原始 reward 数字

## 修改文件

新增:
- `docs/steps/step46_vta_internal_rpe.md`

修改:
- `src/region/limbic/hypothalamus.h`: 新增 `inject_hedonic()` 方法
- `src/region/limbic/hypothalamus.cpp`: 实现 inject_hedonic (LH/PVN 分流)
- `src/region/neuromod/vta_da.h`: 删除 inject_reward/set_expected_reward, 新增 register_hedonic/prediction_source
- `src/region/neuromod/vta_da.cpp`: RPE 从标量改为脉冲率, receive_spikes 按 region_id 分流
- `src/engine/closed_loop_agent.h`: 加 Hypothalamus include/指针/访问器
- `src/engine/closed_loop_agent.cpp`: 加 Hypothalamus 创建+投射+源注册, inject_reward 改为 hypo_->inject_hedonic
- `tests/cpp/test_closed_loop.cpp`: DA 测试改用 hypo()->inject_hedonic()

## 反作弊审计

| 检查项 | 结果 |
|--------|------|
| `inject_reward()` 直接标量注入 | 已删除 |
| `set_expected_reward()` 直接标量注入 | 已删除 |
| Hypo→VTA 走 SpikeBus | 合规 (delay=1) |
| OFC→VTA 走 SpikeBus | 合规 (delay=2) |
| `inject_lhb_inhibition()` 保留 | 合规 (ModulationBus 体积传递) |
| `set_da_level()` 保留 (NAcc/OFC/BG) | 合规 (ModulationBus 体积传递) |
| `inject_hedonic()` 在 Hypothalamus | 合规 (感觉接口，等同 LGN inject_external) |

## 系统状态

```
64区域 · ~252闭环神经元 · ~139投射
VTA RPE: 脉冲驱动 (Hypothalamus hedonic - OFC prediction)
编码: M1/BG 群体向量 (Georgopoulos 1986)
30/30 CTest 通过
```
