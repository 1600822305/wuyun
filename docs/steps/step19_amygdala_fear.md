# Step 19: 杏仁核恐惧回避闭环 (one-shot fear conditioning)

> 日期: 2026-02-08
> 状态: ✅ 完成

## 目标

杏仁核从未接入变为恐惧学习核心 — 视觉CS→恐惧记忆→DA pause→回避

## 之前的问题

Amygdala 类存在但完全未接入 ClosedLoopAgent。恐惧回避仅靠 VTA 负DA + LHb，效果微弱。

## 实现

**A. La→BLA STDP (one-shot 恐惧条件化)**
- 启用 La→BLA 突触 STDP: a_plus=0.10 (10× cortical), a_minus=-0.03 (弱LTD)
- w_max=3.0 (高天花板: 强恐惧关联)
- 生物学: BLA LTP 是 NMDA 依赖的, 单次 CS-US 配对即可建立恐惧记忆
- 文献: LeDoux 2000, Maren 2001, Rogan et al. 1997

**B. inject_us() + fear_output() + cea_vta_drive()**
- `inject_us(mag)`: 危险→BLA 强电流 (US), 驱动 BLA burst → STDP
- `fear_output()`: CeA firing rate [0,1] — 恐惧强度
- `cea_vta_drive()`: CeA→VTA/LHb 抑制信号 (×1.5 放大)

**C. 闭环集成**
- `build_brain`: 创建 Amygdala (La=25, BLA=40, CeA=15, ITC=10)
- 投射: V1→Amygdala (视觉CS输入), Amygdala→VTA, Amygdala→LHb
- Phase A: 危险→inject_us (US注入, La→BLA STDP)
- Phase B: 每步 CeA→VTA inhibition + CeA→LHb amplification

## 恐惧学习信号通路

```
第1次碰到危险:
  V1(视觉CS) → Amygdala La → La→BLA(STDP: 未学习, 弱连接)
  同时: inject_us(pain) → BLA burst
  结果: La→BLA STDP 大幅增强 (a_plus=0.10, one-shot)

第2次看到相似视觉:
  V1(视觉CS) → Amygdala La → La→BLA(已增强!) → BLA→CeA burst
  → CeA→VTA: DA pause (直接抑制)
  → CeA→LHb: 放大 DA pause (间接抑制)
  → BG: DA dip → D2 NoGo 强化 → 回避行为!
```

## 效果验证 (历史最佳!)

| 指标 | Step 18 (海马) | **Step 19 (杏仁核)** | 变化 |
|------|---------------|---------------------|------|
| Test 4 Improvement | +0.094 | **+0.161** | +71% |
| Test 4 Late safety | 0.579 | **0.779** | +35% |
| Test 4 Total food | 113 | **126** | +12% |
| Test 4 Early safety | 0.485 | **0.619** | +28% |

## 回归测试: 29/29 CTest 全通过

## 系统状态

```
50区域 · ~115投射 · 29 CTest suites
新增: Amygdala 恐惧条件化 (La→BLA STDP + CeA→VTA/LHb)
完整恐惧回路:
  V1(CS) → La → BLA(STDP one-shot) → CeA → VTA DA pause → D2 NoGo
  (叠加LHb): CeA → LHb → VTA 双重抑制
学习能力: improvement +0.161, late safety 0.779 (历史最佳)
```
