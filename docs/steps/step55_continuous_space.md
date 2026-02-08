# Step 55: 连续空间环境

> 日期: 2026-02-08
> 状态: ✅ 完成
> 31/31 CTest (离散模式零回归 + 连续模式新测试)

## 动机

Step 45 实现了 M1 群体向量编码，内部产生连续角度，但被 GridWorld 的离散输出卡住：

```
现有 (v45-v54):
  群体向量 → atan2(vy, vx) → 最近基数方向 (4选1) → act(Action) → ±1格

问题:
  1. 内部是连续的，输出被强制离散化 — 精度从 360° 压缩到 4 个方向
  2. 速度不可学 — 每步固定移动 1 格，magnitude 信息被丢弃
  3. DA-STDP 实质上仍在学 4 种映射 — 本质是 Q-table
```

真实大脑的运动输出是连续的（Georgopoulos 1988, Todorov 2004）。

## 设计

**最小侵入策略**：保留 GridWorld 离散格子作为底层基板（食物/危险/墙仍在整数格子），
agent 在其中做连续浮点移动。

```
新增 (v55):
  群体向量 → (angle, coherence) → (dx, dy) 浮点位移 → act_continuous(dx, dy)
  
  coherence = |population_vector| / total_fires ∈ [0, 1]
    所有神经元朝同一方向 → coherence=1 → 全速
    神经元均匀发放 → coherence≈0 → 不动 (等价于 STAY)
  
  speed = coherence × continuous_step_size (默认 0.8)
  dx = speed × cos(angle)
  dy = -speed × sin(angle)  // 注意: 数学 +y=UP, 格子 +y=DOWN

  碰撞检测: target_cell = grid[floor(new_fx), floor(new_fy)]
    WALL → 弹回不动
    FOOD → 吃掉 (+1)
    DANGER → 惩罚 (-1)
```

### 向后兼容

- `AgentConfig::continuous_movement = false` 默认关闭
- 所有 31 个现有测试用离散模式 → 零回归
- `decode_m1_action()` 仍然计算（用于 efference copy、replay、callback）

## 具体改动

### 1. GridWorld (`grid_world.h/cpp`)

- 新增 `float agent_fx_, agent_fy_` 浮点坐标（离散模式下 = int + 0.5）
- 新增 `act_continuous(float dx, float dy) → StepResult`
- `StepResult` 新增 `float agent_fx, agent_fy`
- `set_agent_pos()` 和 `reset()` 同步 float 坐标
- 新增 `agent_fx()`, `agent_fy()` 访问器

### 2. ClosedLoopAgent (`closed_loop_agent.h/cpp`)

- `AgentConfig::continuous_movement` (bool, 默认 false)
- `AgentConfig::continuous_step_size` (float, 默认 0.8)
- `decode_m1_continuous(l5_accum) → pair<float, float>` 连续解码
- `agent_step()` Phase C 分支：continuous → `act_continuous(dx, dy)`

### 3. decode_m1_continuous 算法

```
输入: l5_accum (每个 L5 神经元的累计发放次数)

1. 计算群体向量:
   vx = Σ l5_accum[i] × cos(preferred_dir[i])
   vy = Σ l5_accum[i] × sin(preferred_dir[i])
   total_fires = Σ l5_accum[i]

2. 计算 coherence (方向一致性):
   coherence = |v| / total_fires
   coherence < 0.05 → 返回 (0, 0) = STAY

3. 计算位移:
   speed = coherence × step_size
   angle = atan2(vy, vx)
   dx = speed × cos(angle)
   dy = -speed × sin(angle)   // Y轴翻转
```

**Coherence 是关键创新**：它让速度从群体活动的一致性中涌现。
- 强方向选择 → 高 coherence → 快速移动
- 弱/冲突选择 → 低 coherence → 慢/不动
- 生物学：Moran & Schwartz 1999 证明 M1 群体向量幅度编码运动速度

## 修改文件

- `src/engine/grid_world.h`: float 坐标 + `act_continuous()` + 访问器
- `src/engine/grid_world.cpp`: `act_continuous()` 实现 + float 坐标同步
- `src/engine/closed_loop_agent.h`: config 选项 + `decode_m1_continuous()` 声明
- `src/engine/closed_loop_agent.cpp`: `decode_m1_continuous()` 实现 + Phase C 分支
- `tests/cpp/test_closed_loop.cpp`: 新增测试 8 (GridWorld 直接 + Agent 闭环)

## 验证

```
离散模式: 31/31 CTest 全通过 (零回归)
连续模式:
  - GridWorld act_continuous: 浮点移动精确, 小步保持同格
  - Agent 闭环 200 步: 不崩溃, 正常运行
```

## 架构升级路线

```
✅ Step 45: M1 群体向量编码
✅ Step 46: VTA 内部 RPE
✅ Step 48: 迷宫环境
✅ Step 55: 连续空间环境 (本步)
⬜ 下一步: 在连续模式下跑进化, 验证学习效果
```
