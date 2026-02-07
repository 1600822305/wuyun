# Phase 6: 完整大脑 — 扩展皮层 + 下丘脑 + GNW意识 + 睡眠 + 感觉输入 + 规模扩展

> 对应: Step 5(扩展皮层) / Step 6(下丘脑) / GNW / Step 8(睡眠/重放) / Step 9(感觉) / Step 10(规模) / Step 11(REM)
> 时间: 2026-02-07
> 里程碑: 48 区域 · 5528 神经元 · ~109 投射 · 意识(GNW) · NREM+REM · 感觉输入 · 可扩展至44k · 161 测试全通过

---

## Step 5: 扩展皮层 + 丘脑高级核群 ✅

> 目标: 完整的感觉层级 + 联合皮层 + 丘脑全核群

**新增13个皮层区:**
- **感觉**: S1 (体感), S2 (二级体感), A1 (听觉), Gustatory (味觉), Piriform (嗅觉)
- **联合**: PCC (后扣带), Insula (岛叶), TPJ (颞顶联合), Broca (语言产出), Wernicke (语言理解)
- **运动**: PMC (前运动), SMA (辅助运动), FEF (额眼区)

**新增9个丘脑核:**
- VPL (体感中继), MGN (听觉中继), MD (背内侧→PFC), VA (腹前→运动计划)
- LP (外侧后→PPC), LD (外侧背→扣带/海马), Pulvinar (视觉注意枢纽)
- CeM (中央内侧→觉醒), ILN (板内核群CL/CM/Pf→意识)

**~90条解剖学投射 (原40→90):**
- 视觉 · 体感 · 听觉 · 化学感觉 · 语言 (弓状束) · 运动 · DMN · 丘脑 · FEF↔Pulvinar

**8项通路测试全部通过:**
1. 全系统构建: 46区域, 5409神经元
2. 体感通路: VPL→S1=2897 → S2=1544 → PPC=2125
3. 听觉→语言: MGN→A1=566 → Wernicke=695 → Broca=1840
4. 运动层级: dlPFC→PMC=2539, SMA=1849, M1=2908
5. DMN: PCC=1628, vmPFC=1973, TPJ=1795
6. Pulvinar: 843→V2=4607, V4=2830
7. MD↔PFC: MD=586→dlPFC=3480, OFC=2530, ACC=1920
8. 全链路: V1=8194→IT=2397→dlPFC=4647→BG=4059→M1=3921

**系统状态:** **46区域** | **5409神经元** | **~90投射** | **121 测试全通过**

---

## Step 6: 下丘脑内驱力系统 ✅

> 目标: 内在动机引擎 — 睡眠/觉醒/应激/摄食

**Hypothalamus 类 (6个核团, 89个神经元):**
- **SCN** (n=20) — 昼夜节律起搏器, 正弦振荡
- **VLPO** (n=15) — 睡眠促进, GABA/galanin→抑制觉醒中枢
- **Orexin** (n=15) — 觉醒稳定, →LC/DRN/NBM
- **PVN** (n=15) — 应激反应, CRH→HPA轴→Amygdala
- **LH** (n=12) — 摄食/饥饿驱力, →VTA
- **VMH** (n=12) — 饱腹/能量平衡

**内部回路:**
- Sleep-wake flip-flop (Saper 2005): VLPO⟷Orexin互相抑制
- SCN→VLPO昼夜门控 + LH⟷VMH摄食平衡

**8条新投射:** Orexin→LC/DRN/NBM, Hypo→VTA, Hypo↔Amyg, Insula→Hypo, Hypo→ACC

**7项测试全通过:** SCN振荡 · Flip-flop · 睡眠压力 · Orexin稳定 · PVN应激 · LH/VMH · 全系统

**系统状态:** **47区域** | **~98投射** | **128 测试全通过**

---

## GNW: 全局工作空间理论 ✅

> 目标: Baars/Dehaene 意识访问模型 — 竞争→点火→广播

**GlobalWorkspace 类 (30个workspace整合神经元):**
- **竞争**: 多个皮层区域L5输出→GW, per-region salience累积 (指数衰减防锁定)
- **点火**: 赢者salience超阈值 → ignition (全局点火事件)
- **广播**: 点火后workspace神经元爆发活动→ILN/CeM→全皮层L2/3
- **间隔控制**: min_ignition_gap防止连续点火 (意识是离散采样)

**9条竞争投射 + 2条广播投射 (→~109条总投射)**

**7项测试全通过:** 基础点火 · 竞争门控 · 广播持续 · 衰减 · 间隔 · 无输入不点火 · 全系统

**系统状态:** **48区域** | **~109投射** | **意识(GNW)** | **135 测试全通过**

---

## Step 8: 睡眠/海马重放 ✅

**8a. 海马 Sharp-Wave Ripple (SWR) 重放:**
- `enable_sleep_replay()` / `disable_sleep_replay()`
- SWR 机制: 睡眠模式→CA3 bias+jitter噪声→自联想补全→SWR burst
- **关键设计**: 无需显式存储模式 — CA3 STDP 自联想权重即是记忆

**8b. 皮层 NREM 慢波振荡:**
- `set_sleep_mode(bool)` — Up/Down 状态交替 ~1Hz
- Up state (40%): 正常 | Down state (60%): 抑制

**8c. 记忆巩固通路:**
- SWR → CA3 → CA1 burst → SpikeBus → 皮层 L4 → STDP = 系统巩固
- 无需额外巩固代码 — 利用现有架构自然实现

**7项测试全通过:** SWR生成 · 不应期 · 清醒无SWR · 慢波 · Down state抑制 · 编码→重放 · 多区域

**系统状态:** **142 测试全通过**

---

## Step 9: 感觉输入接口 ✅

**VisualInput — 视觉编码器:**
- 图像像素 [0,1] → LGN relay 电流向量
- Center-surround 感受野 (Kuffler 1953): ON/OFF 通道

**AuditoryInput — 听觉编码器:**
- 频谱功率 [0,1] → MGN relay 电流向量
- Tonotopic mapping + Onset emphasis

**7项测试全通过:** 视觉基础 · Center-surround · 视觉E2E · 听觉基础 · Onset · 听觉E2E · 多模态

**系统状态:** **149 测试全通过**

---

## Step 10: 规模扩展验证 ✅

**`build_standard_brain(scale)` 参数化放大:**
- `scale=1`: ~5,500 神经元 (默认)
- `scale=3`: ~16,500 神经元
- `scale=8`: ~44,000 神经元

**涌现发现:**
- V1 活动超线性增长 (5x vs 3x neurons) → 更密集网络产生更多协同激活
- CA3 模式补全在大网络中接近完美 (比率1.02 ≈ 100%)
- BG 训练/测试差异显著 (4185 vs 721) → DA-STDP 在大规模下仍有效

**系统状态:** **154 测试全通过**

---

## Step 11: REM睡眠 + 梦境 ✅

**SleepCycleManager (engine/sleep_cycle.h/cpp):**
- AWAKE→NREM→REM→NREM 完整睡眠周期管理
- REM 周期增长: 模拟后半夜 REM 延长 (rem_growth)
- PGO 波随机生成 (rem_pgo_prob)

**CorticalRegion REM 扩展:**
- `set_rem_mode(bool)` — 去同步化噪声注入 L2/3 和 L5
- `inject_pgo_wave(amplitude)` — PGO 波随机激活 L4 (梦境视觉)
- `set_motor_atonia(bool)` — M1 L5 强抑制防止梦境运动输出

**Hippocampus REM theta 扩展:**
- `enable_rem_theta()` — ~6Hz theta 振荡调制 CA3/CA1
- 创造性重组: 1%/步概率随机激活 20% CA3 子集 (梦境联想)

**7项测试全通过:** 周期管理 · REM增长 · PGO波 · 皮层REM · 海马theta · 完整周期 · 全脑NREM→REM

**系统状态:** **161 测试全通过**

---

## 低优先级待办

**调质系统扩展:**
- ⬜ 5-HT细分: DR + MnR
- ⬜ ACh细分: PTg + LDTg + BF

**小脑扩展:**
- ⬜ 运动小脑: 前叶+VIIIa/b+绒球
- ⬜ 认知小脑: Crus I/II + VIIb

**其他:**
- ⬜ TRN门控注意力 + 上丘
- ⬜ 发育/关键期: 连接修剪 + PV成熟

---

## Phase 6 总结

| 指标 | 数值 |
|------|------|
| 区域 | 48 |
| 神经元 | 5528 (scale=1) / 16.5k (scale=3) / 44k (scale=8) |
| 投射 | ~109 |
| 测试 | 161 通过 |
| 新增功能 | 扩展皮层(13区) · 丘脑(9核) · 下丘脑(6核) · GNW意识 · NREM睡眠 · REM梦境 · SWR重放 · 视觉编码 · 听觉编码 · 规模扩展 |
| 新增类 | Hypothalamus · GlobalWorkspace · SleepCycleManager · VisualInput · AuditoryInput |