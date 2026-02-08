# Step 18: 海马空间记忆闭环

> 日期: 2026-02-08
> 状态: ✅ 完成

## 目标

海马从被动接收变为主动影响行为 — 空间编码→记忆→决策反馈

## 之前的问题

海马在闭环里是"死胡同"：接收 dlPFC/V1 输入，有 CA3 STDP 编码，但无输出投射回决策回路。

## 三个缺口补齐

**A. 空间编码 (EC grid cells)**
- `inject_spatial_context(x, y, w, h)`: 位置 → EC 网格细胞激活模式
- 每个 EC 神经元有预计算的 2D 余弦调谐曲线 (4种空间频率)
- 不同位置产生不同 EC 群体编码 → DG 模式分离 → CA3 place cells
- 文献: Hafting et al. 2005, Moser & Moser 2008

**B. 奖励标记 (DA-modulated LTP)**
- `inject_reward_tag(magnitude)`: 奖励事件时增强 CA3 STDP
- CA3 a_plus 临时提升 (1 + reward×3)× → 更强记忆痕迹
- 同时 boost 已激活 CA3 neurons → 确保 STDP 配对
- 文献: Lisman & Grace 2005 (DA gates hippocampal memory)

**C. 输出投射 (Hippocampus → dlPFC)**
- `engine_.add_projection("Hippocampus", "dlPFC", 3)` — Sub→EC→dlPFC
- 当 CA3 模式补全激活 → CA1→Sub fires → SpikeBus → dlPFC
- dlPFC 获得记忆检索信号 → 自然通过 BG 影响动作选择
- 文献: Preston & Eichenbaum 2013

## 修改文件

- `src/region/limbic/hippocampus.h/cpp` — 新增空间编码、奖励标记、检索接口
- `src/core/synapse_group.h` — 新增 `stdp_params()` 可变访问器
- `src/engine/closed_loop_agent.cpp` — 集成：每步注入位置、奖励时标记、新增投射

## 设计教训

**移除了失败的 BG 方向偏置注入**: Sub 分 4 组映射到方向是假设 — 位置记忆≠导航指令。
正确做法: 依靠自然 SpikeBus 投射路径 (Hippocampus→dlPFC→BG)。

## 回归测试: 29/29 CTest 全通过

## 系统状态

```
49区域 · ~111投射 · 29 CTest suites
新增: EC grid cell 空间编码 + CA3 奖励标记 + Hippocampus→dlPFC 反馈投射
闭环路径: 位置→EC→DG→CA3(STDP+奖励boost)→CA1→Sub→dlPFC→BG
学习能力: improvement +0.094 (3×3环境空间记忆贡献有限)
```
