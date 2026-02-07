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

### P0 Python 原型 (2026-02-07) → 已归档为算法参考
- ✅ `wuyun/spike/` — SpikeType、Spike、SpikeBus、OscillationClock
- ✅ `wuyun/synapse/` — SynapseBase、STP、STDP、DA-STDP、抑制性STDP、稳态可塑性
- ✅ `wuyun/neuron/` — NeuronBase(16种参数预设)、双区室Compartment
- ✅ `wuyun/core/` — 向量化 NeuronPopulation、SynapseGroup
- → 归档至 `_archived/python_prototype/`，数学方程不变，C++ 重新实现

### 架构决策: C++ 核心引擎 (2026-02-07)
- ✅ 决定: 仿真核心用 **C++17**，Python 只做配置/实验/可视化
- ✅ 理由: Python 100万神经元 ~10秒/step 不可用; C++ ~10-50ms/step 接近实时; 未来迁移 CUDA 容易
- ✅ 03项目结构文档重写为 C++ core + pybind11 + Python 实验层 (v0.4)
- ✅ 技术栈: C++17 / CMake / pybind11 / Google Test / SoA布局 / CSR稀疏 / 事件驱动

---

## 待开始

> 神经元+突触 → 皮层柱 → 核心回路 → 扩展脑区 → 布线 → 涌现

### Step 1: C++ 工程骨架 + 基础验证 ✅ (2026-02-07)
- ✅ CMake 工程搭建 (MSVC 17, Visual Studio 2022, C++17)
- ✅ src/core/types.h — 枚举、参数结构体、4种预设神经元
- ✅ src/core/neuron.h/cpp — 单神经元 step (调试用)
- ✅ src/core/population.h/cpp — SoA 向量化双区室 AdLIF+ 群体
- ✅ src/core/synapse_group.h/cpp — CSR 稀疏突触组 (电导型)
- ✅ src/core/spike_queue.h/cpp — 环形缓冲延迟队列
- ✅ src/plasticity/stdp.h/cpp — 经典 STDP
- ✅ src/plasticity/stp.h/cpp — 短时程可塑性 (Tsodyks-Markram)
- ✅ tests/cpp/test_neuron.cpp — 9 测试全通过 (silence/regular/burst/refrac/adapt/pop×4)
- ⏸ pybind11 绑定 — 延后到 Step 3 再做 (先验证回路)
- **Benchmark (Release, 单线程, CPU):**
  | 规模 | 每步耗时 | 吞吐量 |
  |------|---------|--------|
  | 10K 神经元 | 146 μs | 68M/s |
  | 100K 神经元 | 1.5 ms | 67M/s |
  | 1M 神经元 | 26 ms | 38M/s |
  | 100K 突触 | 75 μs | — |
  | 1M 突触 | 1.2 ms | — |

### Step 2: 皮层柱模板 (C++) ✅ (2026-02-07)
- ✅ src/circuit/cortical_column.h/cpp — 6层通用模板, **18组突触** (AMPA+NMDA+GABA)
- ✅ SST→L2/3 **AND** L5 apical (GABA_B), PV→L4/L5/L6 **全层** soma (GABA_A)
- ✅ NMDA 并行慢通道 (L4→L23, L23→L5, L23 recurrent)
- ✅ L2/3 层内 recurrent 连接 (AMPA+NMDA)
- ✅ burst 加权传递: burst spike ×2 增益
- ✅ 6 测试全通过 (540 神经元, 40203 突触)

### Step 2.5: 地基补全 ✅ (2026-02-07)
> 目标: 在 Step 3 (向上建) 之前, 确保 Layer 0-1 基础设施完整
- ✅ **NMDA Mg²⁺ 电压阻断** B(V) = 1/(1+[Mg²⁺]/3.57·exp(-0.062V))
  - 巧合检测: 需突触前谷氨酸 + 突触后去极化 → 赫布学习硬件基础
  - 测试验证: B(-65)=0.06(阻断), B(-40)=0.23(部分), B(0)=0.78(开放)
- ✅ **STP 集成到 SynapseGroup** — per-pre Tsodyks-Markram, deliver_spikes 中 w_eff = burst_gain × stp_gain
- ✅ **SpikeBus 全局脉冲路由** — 跨区域延迟投递 (环形缓冲), 支持注册区域+投射
- ✅ **DA-STDP 三因子学习** — 资格痕迹 + DA 调制, 解决信用分配/奖励延迟
- ✅ **神经调质系统** — DA/NE/5-HT/ACh tonic+phasic, compute_effect() 计算增益/学习率/折扣/basal权重
- ✅ **特化神经元参数集** (8种):
  - 丘脑 Tonic (κ=0.3) / Burst (κ=0.5, T-type Ca²⁺ threshold=-50mV)
  - TRN (纯抑制门控), MSN D1/D2 (超极化 v_rest=-80mV)
  - 颗粒细胞 (高阈值稀疏), 浦肯野 (高频), DA神经元 (慢适应 burst)
- ✅ **21 测试全通过** (9 neuron + 6 column + 6 foundation)

### Step 3: 核心回路 — 最小可工作大脑 ✅ (2026-02-07)
> 目标: 感觉→认知→动作的最短通路能跑通

**架构层 (新增):**
- ✅ `BrainRegion` 基类 — 统一接口: 注册SpikeBus + step/receive/submit + 振荡/调质
- ✅ `CorticalRegion` — CorticalColumn 的 BrainRegion 包装 + PSP 输入缓冲
- ✅ `SimulationEngine` — 全局时钟 + SpikeBus 编排 + step循环

**3a. 感觉-认知通路:**
- ✅ `ThalamicRelay` (LGN) — Relay+TRN 双群体, Relay↔TRN 互连突触, Tonic/Burst 切换
- ✅ V1 (CorticalRegion 实例, 270 神经元)
- ✅ dlPFC (CorticalRegion 实例, 202 神经元)

**3b. 动作选择通路:**
- ✅ `BasalGanglia` — D1/D2 MSN + GPi/GPe + STN, Direct/Indirect/Hyperdirect 三条通路
- ✅ MotorThalamus (ThalamicRelay 实例)
- ✅ M1 (CorticalRegion 实例, 169 神经元)

**3c. 奖励信号:**
- ✅ `VTA_DA` — DA 神经元 + RPE 计算 + DA level 输出 + phasic/tonic
- ✅ DA→BG D1/D2 调制: DA↑=D1增强(Go), DA↓=D2增强(NoGo)

**端到端验证 (5 测试全通过):**
- 构造: 7 区域, 906 神经元, 6 投射
- 沉默: 无输入→全系统安静
- **信号传播**: 视觉(35.0)→LGN(124 spikes)→V1(23 spikes) ✓
- **DA 调制**: DA=0.1→D1=50, DA=0.6→D1=150 (3倍增强) ✓
- **TRN 门控**: 正常=60, TRN抑制=3 (95%抑制) ✓
- **26 测试全通过** (9 neuron + 6 column + 6 foundation + 5 minimal_brain)

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
C++ 工程骨架 + 基础验证  ← Step 1: CMake + core/ + pybind11
     ↓
皮层柱模板 (6层, C++)    ← Step 2: circuit/cortical_column
     ↓
核心回路 (最小大脑)       ← Step 3: V1+PFC+BG+丘脑4核+DA
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

技术栈: C++17 核心引擎 + pybind11 → Python 实验/可视化
Python 原型 (wuyun/) → _archived/ 算法参考
```
