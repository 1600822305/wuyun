# Step 41: Phase 2 防御+规划扩展 — PAG + FPC

> 日期: 2026-02-09
> 状态: ✅ 完成
> 30/30 CTest, 皮层 events 616→655

## 设计动机

Phase 1 完成了动机/习惯/显著性三个"低挂果实"，Phase 2 补充两个不同维度：

1. **PAG**: 应急防御通路 — 第一次遇到 danger 时 BG 还没学会回避，需要硬连线反射
2. **FPC**: 最高层级前额叶 — 当前 dlPFC 只做即时决策，缺少长期目标维持

## ① PAG 导水管周围灰质 (LeDoux 1996: defense circuit)

**架构**: 8 神经元 = 4 dlPAG (主动防御) + 4 vlPAG (被动防御)

```
CeA → PAG (delay=1, 恐惧驱动)
  ├── dlPAG: 主动防御 (flight/fight)
  │   - 低阈值 -45mV, 快膜常数 8ms
  │   - defense_output() → M1 motor bias (逃跑)
  └── vlPAG: 被动防御 (freeze)
      - 较高阈值 -48mV, 慢膜常数 12ms
      - freeze_output() → suppress motor (冻结)
      - dlPAG 抑制 vlPAG (mutual antagonism)

PAG → LC: arousal_drive() (恐惧→NE↑→警觉)
```

**核心机制**:
- Fear gating: fear_input_ 必须超过 FEAR_THRESHOLD (0.03) 才激活
- dlPAG vs vlPAG: dlPAG 更低阈值 + 更快 → 优先逃跑; 只有 dlPAG 沉默时 vlPAG 才冻结
- 与 BG 互补: PAG = 本能反射 (无需学习), BG = 习得策略 (DA-STDP)

**新增文件**: `periaqueductal_gray.h/cpp`

## ② FPC 前额极皮层 BA10 (Koechlin 2003: highest prefrontal)

**架构**: 12 神经元 (皮层柱, 用 `add_ctx("FPC", n_act*3, false)`)

```
IT → FPC (delay=3)           # 物体身份→目标规划
dlPFC ↔ FPC (delay=3)        # 双向: 目标↔决策
Hippocampus → FPC (delay=3)  # 记忆引导规划

FPC → dlPFC: top-down 目标调制 (长期目标→即时决策)
```

**设计原则**:
- FPC 是 CorticalRegion (不是自定义类) — 它本质上就是一个皮层区域
- 12 个神经元 = dlPFC 同等大小 (4方向×3)
- 无 STDP (stable=false): 表征稳定, 依赖连接学习
- Hippocampus→FPC: "我记得食物在X" → 维持搜索目标

## 投射连接

```
新增投射:
  Amygdala → PAG (delay=1)     # CeA恐惧→应急防御
  IT → FPC (delay=3)           # 物体→目标
  dlPFC → FPC (delay=3)        # 决策→目标更新
  FPC → dlPFC (delay=3)        # 目标→决策调制
  Hippocampus → FPC (delay=3)  # 记忆→规划
```

## 修改文件

新增:
- `src/region/subcortical/periaqueductal_gray.h/cpp`

修改:
- `src/CMakeLists.txt`: 1 个新 cpp
- `src/engine/closed_loop_agent.h`: 2 个 config flags + 2 个 cached pointers
- `src/engine/closed_loop_agent.cpp`: build_brain() 创建 PAG+FPC + 投射 + agent_step() PAG 集成

## 系统状态

```
60区域 · ~208闭环神经元 · ~130投射
学习链路 15/15 (新增: PAG/FPC)
D1 50/50步, ctx events 655/10步
30/30 CTest 通过
```
