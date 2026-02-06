# 悟韵 (WuYun) v3 开发进度

> 上次更新: 2026-02-07
> 仓库: https://github.com/1600822305/wuyun (agi3 = main, agi2 = v2 分支)

---

## 已完成

### 设计文档 (v0.3, 2026-02-07)
- ✅ 四份设计文档校对完毕 (00设计原则 / 01脑区计划 / 02神经元系统 / 03项目结构)
- ✅ 01脑区文档按真实解剖分区重组 (前脑→端脑/间脑 | 中脑 | 后脑)
- ✅ 01脑区文档升级至 **NextBrain 混合方案**: 皮层(功能分区) + 皮层下(NextBrain, Nature 2025)
  - 丘脑 16核团 | 杏仁核 8核团 | 海马 7亚区 | 中脑 11核团 | 小脑 8小叶区
  - 总分区: **~97区** (皮层25 + 皮层下72)，每个皮层下核团标注 FreeSurfer 编号

### P0 代码 (2026-02-07)
- ✅ `spike/` — SpikeType、Spike、SpikeBus、OscillationClock
- ✅ `synapse/` — SynapseBase、STP、STDP、DA-STDP、抑制性STDP、稳态可塑性
- ✅ `neuron/` — NeuronBase(16种参数预设)、双区室Compartment
- ✅ `core/` — 向量化 NeuronPopulation、SynapseGroup
- ✅ 全模块导入测试通过

---

## 待开始

> 神经元+突触 → 皮层柱 → 核心回路 → 扩展脑区 → 布线 → 涌现

### Step 1: 验证基础
- ⬜ 迁移 test_phase0_neuron.py 并验证通过

### Step 2: 皮层柱模板
- ⬜ circuit/ 目录 (cortical_column, layer, column_factory)
- ⬜ 6层微环路 (L2/3↔L4↔L5↔L6，含E/I平衡)
- ⬜ 预测编码验证 (前馈→regular, 前馈+反馈→burst)

### Step 3: 核心回路 — 最小可工作大脑
> 目标: 感觉→认知→动作的最短通路能跑通
**3a. 感觉-认知通路 (皮层+丘脑核心):**
- ⬜ 丘脑感觉中继 (T-01 LGN, T-02 MGN, T-03 VPL, T-16 TRN)
- ⬜ V1 (S-01) + A1 (S-02) — 皮层柱实例 + 丘脑输入
- ⬜ dlPFC (A-01) — 工作记忆参数
**3b. 动作选择通路 (基底节):**
- ⬜ 纹状体 D1/D2 (BG-01~02) + GPi/GPe (BG-03~04) + STN
- ⬜ 丘脑运动中继 (T-07 VLa, T-08 VLp)
- ⬜ M1 (M-01) — 运动输出
**3c. 奖励信号 (DA系统):**
- ⬜ VTA (MB-03) + SNc (MB-04) → 纹状体/PFC DA投射
- ⬜ LHb (ET-01) → VTA 负RPE

### Step 4: 记忆与情感回路
> 目标: 能学习、能记住、能赋予情感价值
**4a. 海马记忆系统:**
- ⬜ EC→DG→CA3→CA1→Subiculum (H-01~05)
- ⬜ 前下托 + HATA (H-06~07)
- ⬜ 隔核 theta 起搏 (SP-01~02)
**4b. 杏仁核情感系统 (NextBrain 8核):**
- ⬜ La→BLA→CeA 核心通路 (AM-01, AM-02, AM-05)
- ⬜ ITC门控 + MeA/CoA (AM-04, AM-06~08) — 恐惧消退
**4c. Papez记忆回路:**
- ⬜ 乳头体 (HY-06) → 丘脑前核 (T-10 AV) → ACC (A-05)

### Step 5: 扩展皮层 + 丘脑高级核群
> 目标: 完整的感觉层级 + 联合皮层 + 丘脑全部16核
**5a. 感觉皮层层级:**
- ⬜ V2/V4/IT/S1/S2/味觉/嗅觉 (S-03~09)
**5b. 联合皮层:**
- ⬜ OFC/vmPFC/ACC/PCC/PPC/TPJ/Broca/Wernicke/FEF/岛叶 (A-02~12)
- ⬜ PMC/SMA/FEF (M-02~04)
**5c. 丘脑联合/高级核群:**
- ⬜ 运动中继: VA/VAmc (T-05~06), MD (T-09)
- ⬜ 联合核群: LD/LP/Pulvinar (T-11~13)
- ⬜ 板内核群: CeM + CL/CM/Pf (T-14~15) — 觉醒/意识

### Step 6: 调质系统 + 内驱力 + 小脑
> 目标: 全局调制 + 内部状态 + 运动/认知预测
**6a. 完整调质系统 (NextBrain脑干核团):**
- ⬜ 5-HT: DR (MB-05) + MnR (MB-08)
- ⬜ NE: LC (HB-01)
- ⬜ ACh: PTg (MB-09) + LDTg (MB-10) + BF (BF-01)
**6b. 下丘脑内驱力:**
- ⬜ 睡眠开关: VLPO (HY-07) ⟷ orexin (HY-02)
- ⬜ 应激: PVN (HY-04) → HPA轴
- ⬜ 节律: SCN (HY-01)
- ⬜ 摄食/饱腹: LH (HY-02) / VMH (HY-03)
**6c. 小脑 (NextBrain 8小叶区):**
- ⬜ 运动小脑: 前叶+VIIIa/b+绒球 (CB-01, CB-06~07)
- ⬜ 认知小脑: Crus I/II + VIIb (CB-03~05) — 工作记忆/推理
- ⬜ 蚓部 (CB-08) + 过渡区 (CB-02)

### Step 7: 连接组学布线
- ⬜ 脑区间连接矩阵 (configs/connectome/) — 按01文档§3
- ⬜ 感觉输入接口 (外界→丘脑→皮层)
- ⬜ 运动输出接口 (皮层→BG→丘脑→运动)

### Step 8: 全脑功能 (涌现)
- ⬜ 睡眠/巩固: NREM慢波 + 海马重放 + PnO (MB-11) REM
- ⬜ 注意力: TRN门控 + ACh + 上丘 (MB-01~02)
- ⬜ 发育/关键期: 连接修剪 + PV+成熟

---

## 架构备忘

```
神经元 + 突触             ← 已完成 (spike/ synapse/ neuron/ core/)
     ↓
皮层柱模板 (6层)          ← Step 2
     ↓
核心回路 (最小大脑)       ← Step 3: V1+PFC+BG+丘脑4核+DA → 感觉→决策→动作
     ↓
记忆+情感                 ← Step 4: 海马7区+杏仁核8核+Papez
     ↓
扩展皮层+丘脑             ← Step 5: 25皮层区+丘脑全16核
     ↓
调质+内驱力+小脑          ← Step 6: 5-HT/NE/ACh+下丘脑+小脑8叶
     ↓
连接组学布线              ← Step 7
     ↓
全脑功能涌现              ← Step 8: 睡眠/注意力/发育

OOP版 (neuron/synapse/) → 调试    向量化版 (core/) → 生产
```
