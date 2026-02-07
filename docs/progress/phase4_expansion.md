# Phase 4: 脑区扩展 — 调质 + 视觉层级 + 小脑 + 决策皮层

> 对应: Step 5.0 / 5a / 5b / 5c+5d
> 时间: 2026-02-07
> 里程碑: 21 区域 · 3239 神经元 · 36 投射 · 4 种调质 · 4 种学习 · 80 测试全通过

---

## Step 5.0: 神经调质广播系统 ✅

> 目标: 补全 4 大调质系统的全脑广播

**新增区域 (3个):**
- ✅ `LC_NE` (蓝斑核, 15 NE神经元) — 增益调节/警觉, inject_arousal()
- ✅ `DRN_5HT` (背侧缝核, 20 5-HT神经元) — 折扣/耐心, inject_wellbeing()
- ✅ `NBM_ACh` (基底核, 15 ACh神经元) — 学习模式/注意力, inject_surprise()

**广播机制:**
- ✅ `SimulationEngine::register_neuromod_source()` 注册调质源
- ✅ `collect_and_broadcast_neuromod()` 每步: 收集4源输出 → 设置全局tonic → 广播到所有区域
- ✅ 所有调质区域输出用指数平滑 (0.1率, 避免同步发放振荡)

**效应接入:**
- ✅ `CorticalRegion` NE增益调制: PSP × gain, gain = 0.5 + 1.5×NE
- ✅ **Yerkes-Dodson倒U型涌现**: NE=0.1→213, NE=0.5→361, NE=0.9→333
  (高NE增益也放大PV抑制 → 活动反降, 无任何硬编码!)

**系统状态:** 12区域 | 1641神经元 | 14投射 | 4种调质广播 | **62 测试全通过**

---

## Step 5a: 视觉皮层层级 V2/V4/IT ✅

> 目标: V1→V2→V4→IT 逐级抽象的腹侧视觉通路

**新增区域 (3个, 全部复用 CorticalRegion, 无新代码):**
- ✅ `V2` (214n) — 纹理/轮廓所有权, L4=40
- ✅ `V4` (164n) — 颜色/曲率/中级形状, L4=30
- ✅ `IT` (130n) — 物体/面孔/类别识别, L4=20

**投射 (7条: 4前馈 + 3反馈):**
- 前馈: LGN→V1(d=2) → V2(d=2) → V4(d=2) → IT(d=2)
- 反馈: V2→V1(d=3), V4→V2(d=3), IT→V4(d=3)
- IT→dlPFC(d=3): 物体识别 → 决策

**验证结果:**
- **层级传播**: LGN=124 → V1=8194 → V2=6067 → V4=3849 → IT=2397
- **逐层延迟**: V1=t11 → V2=t13 → V4=t15 → IT=t18 (每层~2ms)
- **STDP习惯化涌现**: 训练后IT=4 vs 未训练IT=697 (LTD导致重复抑制, 无硬编码!)
- **15区域全系统**: IT=2397 → dlPFC=4438 → BG=3272 → M1=975

**系统状态:** 15区域 | 2149神经元 | 19投射 | **68 测试全通过**

---

## Step 5b: 小脑运动学习 Cerebellum ✅

> 目标: 扩展-收敛-纠错架构 + 第4种学习规则 (攀爬纤维LTD)

**新增区域 (1个, 全新 BrainRegion 子类):**
- ✅ `Cerebellum` (cerebellum.h/cpp, 275n):
  - 颗粒细胞 GrC (200n) — 扩展层, 稀疏编码
  - 浦肯野细胞 PC (30n) — GABA抑制输出, CF-LTD目标
  - 深部核团 DCN (20n) — 最终输出, 35f tonic drive
  - 分子层中间神经元 MLI (15n) — 前馈抑制
  - 高尔基细胞 Golgi (10n) — 反馈抑制

**攀爬纤维学习规则 (第4种学习):**
- CF + PF激活 → PF→PC LTD (cf_ltd_rate=0.02, 减弱错误运动)
- PF单独激活 → PF→PC LTP (cf_ltp_rate=0.005, 强化正确运动)
- 4种学习对比: 皮层STDP | 海马快速STDP | BG DA-STDP | **小脑CF-LTD**

**验证结果:**
- **信号传播**: MF→GrC=534 → PC=299 → DCN=280
- **CF-LTD学习**: PC(无误差)=749 → PC(CF-LTD)=496 (-34%)
- **误差校正**: PC逐epoch下降 1010→893→891→767→702

**系统状态:** 16区域 | 2424神经元 | 20投射 | **4种学习规则** | **74 测试全通过**

---

## Step 5c+5d: 决策皮层 + 背侧视觉 ✅

> 目标: 价值决策三角 (OFC/vmPFC/ACC) + 双流视觉 (what+where)

**Step 5c 决策皮层 (3个, 复用 CorticalRegion):**
- ✅ `OFC` (151n) — 眶额皮层, 价值评估 (IT→OFC, Amyg→OFC)
- ✅ `vmPFC` (140n) — 腹内侧前额叶, 情绪决策 (OFC→vmPFC→BG, vmPFC→Amyg)
- ✅ `ACC` (135n) — 前扣带回, 冲突监控 (ACC→dlPFC, ACC→LC_NE)

**Step 5d 背侧视觉通路 (2个, 复用 CorticalRegion):**
- ✅ `MT/V5` (185n) — 中颞区, 运动方向感知 (V1→MT, V2→MT)
- ✅ `PPC` (174n) — 后顶叶, 空间注意/视觉运动整合 (MT→PPC→dlPFC/M1)
- 双流架构: 腹侧(V1→V2→V4→IT, what) + 背侧(V1→V2→MT→PPC, where)
- 跨流: PPC↔IT (空间引导识别 / 物体引导注意)

**新增投射 (16条):**
- 决策: IT→OFC, OFC→vmPFC, vmPFC→BG, vmPFC→Amyg, Amyg→OFC
- 冲突: ACC→dlPFC, ACC→LC, dlPFC→ACC
- 背侧: V1→MT, V2→MT, MT→PPC, PPC→MT(fb)
- 跨流: PPC→IT, IT→PPC
- 空间运动: PPC→dlPFC, PPC→M1

**验证结果:**
- **决策通路**: IT→OFC=1432 → vmPFC=1387 → BG=1307
- **双流视觉**: 腹侧IT=1637, 背侧MT=2164→PPC=2353
- **ACC冲突**: NE基线0.200→冲突0.204 (ACC→LC)
- **21区域全系统**: OFC=3412, vmPFC=2573, ACC=2456, MT=4837, PPC=4130, M1=3921

**系统状态:** 21区域 | 3239神经元 | 36投射 | **80 测试全通过**

---

## Phase 4 总结

| 指标 | 数值 |
|------|------|
| 区域 | 21 (Phase 3 的 9 + LC + DRN + NBM + V2 + V4 + IT + CB + OFC + vmPFC + ACC + MT + PPC) |
| 神经元 | 3239 |
| 投射 | 36 |
| 测试 | 80 通过 |
| 学习规则 | 4 种 (皮层STDP · CA3-STDP · BG DA-STDP · 小脑CF-LTD) |
| 调质 | 4 种广播 (DA · NE · 5-HT · ACh) |
| 新增类 | LC_NE · DRN_5HT · NBM_ACh · Cerebellum |
| 关键涌现 | Yerkes-Dodson倒U · STDP习惯化 · CF-LTD误差校正 |