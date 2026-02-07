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

### Step 3.5: 反作弊修复 ✅ (2026-02-07)
> 根据 00_design_principles.md §6 审计

- ✅ BG `receive_spikes` 中 `id%5` hyperdirect 硬编码 → 构造时随机稀疏映射表 (`ctx_to_d1/d2/stn_map_`)
- ✅ DA→BG 调制走 SpikeBus: VTA 脉冲 → BG `receive_spikes` 自动推算 DA 水平 (`da_spike_accum_` + 指数平滑)
- ✅ VTA→BG 投射添加到 SimulationEngine (delay=1)
- ✅ 7 条投射 (原6条 + VTA→BG), 26 测试全通过

### Step 4: 海马记忆 + 杏仁核情感 ✅ (2026-02-07)
> 目标: 情景记忆编码/回忆 + 恐惧条件化/消退

**4a. 海马体 (Hippocampus):**
- ✅ `Hippocampus` 类 — 5 兴奋性群体 (EC/DG/CA3/CA1/Sub) + 3 抑制性群体 (DG_inh/CA3_inh/CA1_inh)
- ✅ 三突触通路: EC→DG(perforant) → CA3(mossy fiber) → CA1(Schaffer) → Sub
- ✅ CA3 自联想循环连接 (~2% 概率, 模式补全基底)
- ✅ EC→CA1 直接通路 (绕过 DG/CA3, 投射到 apical)
- ✅ DG 稀疏编码: 高阈值颗粒细胞 (v_rest=-75, threshold=-45) + 前馈+反馈抑制
- ✅ EC→DG_inh 前馈抑制 (feedforward inhibition, 与 EC→DG 同步)
- ✅ 8 组兴奋性突触 + 6 组抑制性突触 (含 GABA_A 分流抑制)
- 神经元类型: GRID_CELL, GRANULE_CELL, PLACE_CELL, PV_BASKET

**4b. 杏仁核 (Amygdala):**
- ✅ `Amygdala` 类 — 4 群体 (La/BLA/CeA/ITC)
- ✅ 恐惧条件化通路: La(输入) → BLA(学习) → CeA(输出)
- ✅ La→CeA 快速直接通路
- ✅ ITC 恐惧消退门控: PFC→ITC → ITC抑制CeA (GABA_A)
- ✅ BLA 自联想循环 (维持价值表征)

**关键bug修复:**
- ✅ **GABA 权重符号**: 发现所有 GABA 突触不应用负权重 (公式 `I = g_max * w * g * (e_rev - v)` 中反转电位已处理符号方向; 负权重造成双重否定=兴奋)
- ✅ DG 颗粒细胞 v_rest=-75 < GABA_A e_rev=-70: GABA_A 在 DG 上是分流抑制 (shunting), 仅在 v>-70 时有效

**端到端验证 (7 测试全通过):**
- 海马构造: 505 神经元 (EC=80, DG=200, CA3=60, CA1=80, Sub=40, inh=45)
- 海马沉默: 无输入=0 发放 ✓
- **三突触传播**: EC=271→DG=5904→CA3=1396→CA1=331→Sub=23 ✓
- **DG 稀疏**: 稳态平均 18.6% (前馈+反馈抑制 E/I 平衡) ✓
- 杏仁核构造+沉默: 180 神经元 ✓
- **恐惧通路**: La=50→BLA=28→CeA=17 ✓
- **ITC 消退**: CeA无消退=27, CeA有消退=1 (96%抑制) ✓
- **33 测试全通过** (9 neuron + 6 column + 6 foundation + 5 minimal_brain + 7 memory_emotion)

### Step 4.5: 整合大脑 — 9区域闭环 ✅ (2026-02-07)
> 目标: 海马+杏仁核接入主回路，形成感觉→情感→记忆→决策→动作闭环

**新增 6 条跨区域投射:**
- ✅ V1 → Amygdala(La): 视觉威胁快速评估 (delay=2)
- ✅ dlPFC → Amygdala(ITC): 恐惧消退/情绪调控 (delay=2)
- ✅ dlPFC → Hippocampus(EC): 认知驱动记忆编码 (delay=3)
- ✅ Hippocampus(Sub) → dlPFC: 回忆影响决策 (delay=3)
- ✅ Amygdala(CeA) → VTA: 情绪调制奖励信号 (delay=2)
- ✅ Amygdala(BLA) → Hippocampus(EC): 情绪标记增强记忆 (delay=2)

**架构改进:**
- ✅ Amygdala `receive_spikes` 来源路由: PFC→ITC, 其他→La (`pfc_source_region_`)
- ✅ VTA 添加 PSP 缓冲: 跨区域脉冲持续累积 (DA神经元平衡态低于阈值，需 PSP 驱动)

**端到端验证 (7 测试全通过):**
- 构造: 9 区域, 1591 神经元, 13 投射
- 沉默: 无输入=0 发放 ✓
- **视觉→杏仁核**: V1=4896 → Amyg=3477 (CeA=119) ✓
- **情绪标记记忆增强**: 中性Hipp=10633, 情绪Hipp=12617 (+19%) ✓
- **杏仁核→VTA**: VTA基线=0, VTA+情绪=284 ✓
- **PFC→ITC路由**: dlPFC=4533 → ITC=1628 (SpikeBus 正确路由) ✓
- **40 测试全通过** (9+6+6+5+7+7)

### Step 4.6: 开机学习 — 记忆/强化学习验证 ✅ (2026-02-07)
> 目标: 突触可塑性接入运行脑区，验证真正的学习、记忆、泛化能力

**架构新增:**
- ✅ SynapseGroup STDP 集成: `enable_stdp()` + `apply_stdp()` (类似 STP 集成模式)
  - per-neuron 最后发放时间跟踪 (`last_spike_pre_/post_`)
  - CSR 遍历，仅对本步发放的 pre/post 突触计算 Δw
- ✅ CA3 循环突触启用 fast STDP (A+=0.05, 5x cortical, one-shot learning)
  - `HippocampusConfig::ca3_stdp_*` 参数组
  - step() 末尾调用 `syn_ca3_to_ca3_.apply_stdp()`

**学习能力验证 (5 测试全通过):**
- **CA3 STDP 权重变化**: 学习后 CA3=1215 > 无学习 CA3=1142 (+6.4%) ✓
- **记忆编码/回忆**: 编码 60 neurons → 部分线索(30%)回忆 60 neurons, **100% 重叠** (模式补全) ✓
- **模式分离**: 不同EC模式→不同CA3子集 (重叠仅10%, DG稀疏化有效) ✓
- **DA-STDP 强化学习**: 奖励 w=0.5063 > 无奖励 w=0.5000 (三因子学习) ✓
- **记忆容量**: 3个模式各编码到不同CA3子集 ✓
- **45 测试全通过** (9+6+6+5+7+7+5)

**关键里程碑:** 悟韵从"通电的硬件"变为"能学习的系统"
- 记忆不是字典查找，而是CA3自联想网络的STDP权重变化
- 回忆是模式补全（部分线索→完整重建），不是精确匹配
- 强化学习通过DA调制资格痕迹实现，不是IF-ELSE规则

### Step 4 剩余 (低优先级):
- ⬜ 前下托 + HATA (H-06~07)
- ⬜ 隔核 theta 起搏 (SP-01~02)
- ⬜ 杏仁核扩展: MeA/CoA/AB (AM-04, AM-06~08)
- ⬜ Papez回路: 乳头体 (HY-06) → 丘脑前核 (T-10 AV) → ACC (A-05)

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
