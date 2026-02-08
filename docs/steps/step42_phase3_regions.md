# Step 42: Phase 3 价值+情绪扩展 — OFC + vmPFC

> 日期: 2026-02-09
> 状态: ✅ 完成
> D1 50→61, elig 107→120, ctx 655→689。30/30 CTest。

## 设计动机

Phase 1-2 完成了动机/习惯/显著性/防御/规划，但价值评估链缺失：
- BG 知道"往哪走好" (动作-结果)，但不知道"看到的东西值不值得追" (刺激-结果)
- Amygdala 知道"可怕"，但没有"安全了可以放松" 的消退机制

OFC 和 vmPFC 补上这两个缺口。

## ① OFC 眶额皮层 BA11/47 (Rolls 2000: stimulus-outcome)

**架构**: 12 神经元 = 4 value_pos + 4 value_neg + 4 inh

```
IT → OFC (delay=2)           # 物体身份 → 价值关联
Amygdala → OFC (delay=2)     # 情绪价值 → 价值调制
OFC → dlPFC (delay=2)        # 价值信号 → 决策偏置
OFC → NAcc (delay=2)         # 价值 → 动机
```

**核心机制**:
- **value_pos**: 编码正期望价值 (食物相关刺激 → 活跃)
- **value_neg**: 编码负期望价值 (危险相关刺激 → 活跃)
- **inh**: PV 中间神经元，pos/neg 之间竞争抑制 (winner-take-all)
- **DA 调制** (volume transmission, 合规):
  - DA > baseline (0.3): 增强 value_pos (正RPE → "比预期好")
  - DA < baseline: 增强 value_neg (负RPE → "比预期差")
  - 这实现了 reversal learning: 当奖赏条件改变时 OFC 快速更新

**与 BG 的分工**:
| 区域 | 学什么 | 怎么学 | 输出 |
|------|--------|--------|------|
| BG | 动作→结果 | DA-STDP | Go/NoGo 动作选择 |
| OFC | 刺激→结果 | DA 调制 value neurons | 价值信号→dlPFC/NAcc |
| Amygdala | 刺激→情绪 | BLA STDP (快, 难消退) | CeA→PAG/VTA 恐惧 |

**新增文件**: `region/prefrontal/orbitofrontal.h/cpp` (新建 prefrontal 目录)

## ② vmPFC 腹内侧前额叶 BA14/25 (Milad & Quirk 2002: fear extinction)

**架构**: 8 神经元 (CorticalRegion 皮层柱, 用 `add_ctx("vmPFC", n_act*2, false)`)

```
OFC → vmPFC (delay=2)            # 价值信息 → 安全评估
Hippocampus → vmPFC (delay=3)    # 空间上下文 → "这里以前安全"
vmPFC → Amygdala (delay=2)       # 安全信号 → ITC 兴奋 → CeA 抑制
vmPFC → NAcc (delay=2)           # 安全 → 动机调制
```

**设计原则**:
- vmPFC 是 CorticalRegion (与 FPC 一致) — 通过 SpikeBus 连接发挥作用
- 8 个神经元 = dlPFC 的 2/3 (更小, 专注安全/价值综合)
- vmPFC → Amygdala: 安全信号通过 SpikeBus 到达 Amygdala，
  Amygdala 的 ITC 接收后抑制 CeA → 恐惧消退 (生物学完整通路)
- Hippocampus → vmPFC: 上下文依赖的安全记忆 ("这个位置以前没有危险")

## 投射连接

```
新增投射:
  IT → OFC (delay=2)           # 物体→价值
  Amygdala → OFC (delay=2)     # 情绪→价值
  OFC → dlPFC (delay=2)        # 价值→决策
  OFC → NAcc (delay=2)         # 价值→动机
  OFC → vmPFC (delay=2)        # 价值→安全
  Hippocampus → vmPFC (delay=3)# 上下文→安全
  vmPFC → Amygdala (delay=2)   # 安全→消退
  vmPFC → NAcc (delay=2)       # 安全→动机
```

## DA 调制合规性

- `ofc_->set_da_level(vta_->da_output())` — DA 体积传递 (ModulationBus)
  - 与 `bg_->set_da_level()` 和 `nacc_->set_da_level()` 同类
  - 不是 SpikeBus 违规，设计文档明确区分两种传递机制

## 修改文件

新增:
- `src/region/prefrontal/orbitofrontal.h`: OFCConfig + OrbitofrontalCortex class
- `src/region/prefrontal/orbitofrontal.cpp`: 完整实现

修改:
- `src/CMakeLists.txt`: 添加 `region/prefrontal/orbitofrontal.cpp`
- `src/engine/closed_loop_agent.h`: enable_ofc/enable_vmpfc flags + ofc_/vmpfc_ pointers
- `src/engine/closed_loop_agent.cpp`:
  - build_brain(): 创建 OFC + vmPFC + 8 条投射
  - 缓存 ofc_/vmpfc_ 指针
  - reward processing: OFC DA volume transmission

## 效果对比

| 指标 | Step 41 | Step 42 | 变化 |
|------|---------|---------|------|
| D1 fires (50步) | 50 | **61** | +22% |
| Max eligibility | 107.3 | **120.0** | +12% |
| Cortical events/10步 | 655 | **689** | +5% |

## 系统状态

```
63区域 · ~228闭环神经元 · ~140投射
学习链路 17/17 (新增: OFC/vmPFC)
D1 61/50步, Max elig 120.0, ctx 689/10步
30/30 CTest 通过
```
