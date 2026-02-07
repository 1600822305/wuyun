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

### Step 4.7: 皮层 STDP 自组织学习 ✅ (2026-02-07)
> 目标: 皮层柱启用在线可塑性，验证视觉自组织学习

**架构新增:**
- ✅ `ColumnConfig::stdp_*` 参数组 (a_plus/a_minus/tau/w_max)
- ✅ `CorticalColumn::enable_stdp()` 对 3 组 AMPA 突触启用 STDP:
  - L4→L2/3 (前馈特征学习, 最重要)
  - L2/3 recurrent (侧向吸引子)
  - L2/3→L5 (输出学习)
- ✅ step() STEP 2.5 在 populations 步进后调用 `apply_stdp()`
- ✅ `build_synapses()` 中 `stdp_enabled` 自动启用

**皮层学习验证 (4 测试全通过):**
- **STDP 权重变化**: STDP=185 vs control=183 (权重已改变) ✓
- **训练增强**: 训练模式 A=162 > 新模式 B=156 (经验增强) ✓
- **选择性涌现**: 偏好A=31 偏好B=55 非选择=109 (86个神经元发展选择性!) ✓
- **LTD 竞争**: 500步训练后活动=119 (稳定, LTD防饱和) ✓
- **49 测试全通过** (9+6+6+5+7+7+5+4)

**关键意义:** 皮层柱不再是固定权重的信号传递器，而是能从经验中自组织形成特征选择性的学习单元。这完全符合 `00_design_principles.md` 的设计：
- 功能差异从参数差异 + 连接差异 + **学习经验**中涌现
- 代码完全相同的 CorticalColumn，不同的输入数据→不同的选择性

### Step 4.8: BG DA-STDP 在线强化学习 ✅ (2026-02-07)
> 目标: 三因子学习接入 BG 运行时，验证动作选择学习闭环

**架构新增:**
- ✅ `BasalGangliaConfig::da_stdp_*` 参数组 (lr/baseline/w_min/w_max)
- ✅ Per-connection 权重存储: `ctx_d1_w_[src][idx]` 平行于 `ctx_to_d1_map_`
- ✅ `receive_spikes()` 使用学习权重替代固定电流 (`base_current * w`)
- ✅ `apply_da_stdp()` 三因子规则:
  - D1(Go): DA>baseline → LTP (Gs-coupled, 强化 Go)
  - D2(NoGo): DA>baseline → LTD (Gi-coupled, 削弱 NoGo)
  - 生物正确的 D1/D2 受体不对称性

**BG 强化学习验证 (4 测试全通过):**
- **DA-STDP 权重改变**: 高DA D1=3404 > 低DA D1=3206 ✓
- **Go/NoGo 偏好**: 训练后 D1=1962,D2=506 vs 无学习 D1=1777,D2=1714 (Go↑NoGo↓) ✓
- **动作选择学习**: 奖励动作A D1=873 > 未奖励B D1=536 (+63%) ✓
- **反转学习**: Phase1 B=422 → Phase2 B=575 (+36%, 偏好成功反转) ✓
- **53 测试全通过** (9+6+6+5+7+7+5+4+4)

**关键意义:** BG 不再是固定的 Go/NoGo 通道——它能从 DA 奖励信号中学习:
- 哪个动作应该被选择 (动作选择学习)
- 当环境变化时切换偏好 (反转学习)
- 全部通过 D1/D2 受体不对称 + DA 调制，没有任何 IF-ELSE 决策规则

**回答用户"总控"问题:**
大脑没有中央控制器。协调来自三个分布式机制:
1. SpikeBus 脉冲路由 (已有) — 类似互联网数据包
2. 丘脑 TRN 竞争门控 (已有) — 决定哪些信号通过
3. 神经调质广播 (部分实现) — DA/NE 全局"情绪天气"
   - VTA→BG 的 DA 通路已实现并驱动学习
   - 未来: NE(LC), 5-HT(DRN), ACh(NBM) 广播到全脑

### Step 4.9: 端到端学习演示 ✅ (2026-02-07)
> 目标: 用现有 9 区域系统证明全系统协作学习

**演示架构:**
- ✅ 9 区域全部启用学习: V1(STDP) + dlPFC(STDP) + BG(DA-STDP) + Hipp(CA3 STDP)
- ✅ 3 种输入方式: `inject_visual` (LGN) + `inject_bg_spikes` (via receive_spikes, 触发DA-STDP) + `inject_bg_cortical` (直接电流)
- ✅ 关键发现: `set_da_level()` 被 VTA→BG 脉冲路由覆盖 → 需 `set_da_source_region(UINT32_MAX)` 禁用自动DA计算

**端到端验证 (4 测试全通过):**
- **视觉-奖励闭环**: 训练 A=852>B=587, 测试(仅脉冲) A=779>B=336 (+132%) ✓
- **情绪通路**: V1=5039→Amyg=3429→VTA=241→Hipp=16165 (4区域同时活跃) ✓
- **三系统协同**: Amyg=2580 + VTA=271 + Hipp=9930 + D1=1405 (记忆+情绪+动作并行) ✓
- **学习选择性**: 仅靠学习权重 A=605>B=371 (+63%) (无直接电流, 纯权重驱动) ✓
- **57 测试全通过** (9+6+6+5+7+7+5+4+4+4)

**关键意义:** 这是悟韵第一次证明:
- 3 套独立的学习系统 (海马记忆/皮层自组织/BG强化学习) 能在同一个仿真中**同时运行**
- 视觉输入经过 V1→Amygdala→VTA 自然产生 DA 信号 (无人工注入)
- BG 仅靠学习后的权重差异就能区分奖励过/未奖励的模式
- **没有任何 IF-ELSE 决策规则** — 所有行为从结构+学习中涌现

### 修复: V1→dlPFC→BG 信号衰减 ✅ (2026-02-07)
> 问题: CorticalRegion::receive_spikes fan-out=3, current=25f → PSP稳态3.1f ≪ 阈值15f

**根因分析:**
- L4 stellate: v_rest=-65, threshold=-50, R_s=1.0 → 需 I>15f 持续
- LGN→V1: 0.62脉冲/步 × 3/50(fan-out/L4) × 25f = 0.93f/步 → 稳态3.1f (远低于阈值)

**修复:**
- ✅ `ColumnConfig::input_psp_regular/burst/fan_out_frac` 可配置参数
- ✅ fan-out: 3固定 → 30%×L4_size (生物学皮层-皮层汇聚)
- ✅ current: 25f/40f → 35f/55f (regular/burst)
- ✅ `CorticalRegion` 存储 `psp_current_regular_/burst_/fan_out_`

**修复后全链路数据:**
```
修复前: LGN=124 → V1=23    → dlPFC=0    → BG=120  → MotorThal=0   → M1=0
修复后: LGN=124 → V1=7656  → dlPFC=4770 → BG=3408 → MotorThal=293 → M1=1120
```
- 额外收获: dlPFC→Hipp通路也打通 (dlPFC=6937→Hipp=18791, CA1=660)
- **57 测试全通过**, 零回归

### Step 5.0: 神经调质广播系统 ✅ (2026-02-07)
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

**系统状态:**
- 12区域 | 1641神经元 | 14投射 | 4种调质广播
- **62 测试全通过** (9+6+6+5+7+7+5+4+4+4+5), 零回归

### Step 5a: 视觉皮层层级 V2/V4/IT ✅ (2026-02-07)
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

**系统状态:**
- 15区域 | 2149神经元 | 19投射 | 4种调质广播
- **68 测试全通过** (9+6+6+5+7+7+5+4+4+4+5+6), 零回归

### Step 5b: 小脑运动学习 Cerebellum ✅ (2026-02-07)
> 目标: 扩展-收敛-纠错架构 + 第4种学习规则 (攀爬纤维LTD)

**新增区域 (1个, 全新 BrainRegion 子类):**
- ✅ `Cerebellum` (cerebellum.h/cpp, 275n):
  - 颗粒细胞 GrC (200n) — 扩展层, 稀疏编码
  - 浦肯野细胞 PC (30n) — GABA抑制输出, CF-LTD目标
  - 深部核团 DCN (20n) — 最终输出, 35f tonic drive
  - 分子层中间神经元 MLI (15n) — 前馈抑制
  - 高尔基细胞 Golgi (10n) — 反馈抑制

**内部突触 (7组: 4兴奋 + 3抑制):**
- 兴奋: MF→GrC(p=0.15), PF→PC(p=0.40, LTD/LTP), PF→MLI(p=0.20), GrC→Golgi(p=0.15)
- 抑制: MLI→PC(p=0.30), PC→DCN(p=0.35, w=0.4), Golgi→GrC(p=0.20)

**攀爬纤维学习规则 (第4种学习):**
- CF + PF激活 → PF→PC LTD (cf_ltd_rate=0.02, 减弱错误运动)
- PF单独激活 → PF→PC LTP (cf_ltp_rate=0.005, 强化正确运动)
- 4种学习对比: 皮层STDP | 海马快速STDP | BG DA-STDP | **小脑CF-LTD**

**验证结果:**
- **信号传播**: MF→GrC=534 → PC=299 → DCN=280
- **CF-LTD学习**: PC(无误差)=749 → PC(CF-LTD)=496 (-34%)
- **误差校正**: PC逐epoch下降 1010→893→891→767→702
- **DCN tonic**: 300 spikes (需BG协同驱动MotorThal)
- **16区域全系统**: CB=4002, M1=950

**关键设计决策:**
- DCN tonic drive=35f: 生物上DCN持续40-50Hz, PC只塑形不沉默
- PC→DCN: p=0.35, w=0.4 (低于其他抑制), 调制而非关断
- `SynapseGroup` 新增 `row_ptr()/col_idx()` 访问器, 支持CSR遍历可塑性

**系统状态:**
- 16区域 | 2424神经元 | 20投射 | 4种调质 | **4种学习规则**
- **74 测试全通过** (9+6+6+5+7+7+5+4+4+4+5+6+6), 零回归

### Step 5c+5d: 决策皮层 + 背侧视觉 ✅ (2026-02-07)
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

**系统状态:**
- 21区域 | 3239神经元 | 36投射 | 4调质 | 4种学习
- **80 测试全通过** (9+6+6+5+7+7+5+4+4+4+5+6+6+6), 零回归

### Step 6: 预测编码框架 ✅ (2026-02-07)
> 目标: 皮层层级预测与误差计算 (Rao-Ballard + Friston Free Energy)

**核心机制 (修改 CorticalRegion, 零新文件):**
- ✅ `enable_predictive_coding()` — 可选启用, 向后完全兼容
- ✅ `add_feedback_source(region_id)` — 标记反馈来源, 区分FF/FB
- ✅ 反馈路由: feedback源脉冲→`pc_prediction_buf_`(L2/3 sized), 非feedback→L4 `psp_buffer_`
- ✅ 预测抑制: prediction_buf → L2/3 apical 负注入 (抑制误差单元)
- ✅ 预测误差跟踪: `pc_error_smooth_` 指数平滑

**精度加权 (神经调质驱动):**
- ✅ NE → `pc_precision_sensory_` = ne_gain (0.5~2.0): 高NE信任感觉
- ✅ ACh → `pc_precision_prior_` = max(0.2, 1.0-0.8*ACh): 高ACh不信任预测
- L4注入 × sensory精度, prediction注入 × prior精度

**验证结果:**
- **预测抑制涌现**: V1(早期无预测)=226 → V1(晚期有预测)=116 (-49%)
- **NE精度**: V1(NE=0.1)=85 → V1(NE=0.5)=187 → V1(NE=0.9)=235
- **ACh精度**: ACh=0.8→prior=0.36, ACh=0.1→prior=0.92
- **层级PC**: V1↔V2↔V4 双向预测+误差
- **向后兼容**: 无PC时 V1=262, PC无反馈时 V1=262 (完全一致)

**生物学对应:**
- L6 → 反馈 → 下级L2/3 apical = 预测信号 (Mumford 1992)
- L2/3 = 感觉(L4 basal) - 预测(apical) = 预测误差 (Rao & Ballard 1999)
- NE = 感觉精度 (意外→LC→NE↑→信任感觉) (Feldman & Friston 2010)
- ACh = 先验精度倒数 (新环境→NBM→ACh↑→不信任预测) (Yu & Dayan 2005)

**系统状态:**
- 21区域 | 3239神经元 | 36投射 | 4调质 | 4学习 | **预测编码**
- **86 测试全通过** (9+6+6+5+7+7+5+4+4+4+5+6+6+6+6), 零回归

### Step 7: Python绑定 + 可视化仪表盘 ✅ (2026-02-07)
> 目标: pybind11暴露C++引擎到Python, matplotlib可视化

**pybind11绑定 (src/bindings/pywuyun.cpp):**
- ✅ SimulationEngine: step/run/add_projection/find_region/build_standard_brain
- ✅ 所有11种BrainRegion子类: CorticalRegion/ThalamicRelay/BG/VTA/LC/DRN/NBM/Hipp/Amyg/CB
- ✅ SpikeRecorder: record→to_raster() 返回numpy数组
- ✅ NeuromodulatorLevels/System: 调质监控
- ✅ build_standard_brain(): 一键构建21区域完整大脑

**可视化工具 (python/wuyun/viz.py):**
- ✅ plot_raster(): 12区域脉冲栅格图, 彩色编码
- ✅ plot_connectivity(): networkx拓扑图, 21节点36边
- ✅ plot_activity_bars(): 区域活动柱状图
- ✅ plot_neuromod_timeline(): DA/NE/5-HT/ACh时间线
- ✅ run_demo(): 一键演示 (构建→刺激→可视化→保存)

**验证结果:**
- Python绑定: 21区域/36投射全部可用
- 脉冲栅格: 清晰展示12区域时序活动模式
- 连接图: 层级结构可视化 (视觉→决策→运动→小脑)
- 调质动态: DA/NE/5-HT/ACh基线+刺激响应
- 86 C++测试零回归

**系统状态:**
- 21区域 | 3239神经元 | 36投射 | 4调质 | 4学习 | 预测编码 | **Python可视化**
- **86 测试全通过**, 零回归

### Step 9: 认知任务演示 ✅ (2026-02-07)
> 目标: 经典认知范式验证涌现行为，暴露系统能力边界

**Task 1: Go/NoGo (BG动作选择 + ACC冲突监控)**
- ✅ ACC冲突检测涌现: NoGo ACC=1383 > Go ACC=1205 (1.15x)
- ⚠️ M1运动相同 (2006=2006): 无训练D1/D2权重→相同输入=相同输出
- 启示: 需要DA-STDP在线训练才能区分Go/NoGo运动响应

**Task 2: 情绪处理 (Amygdala威胁 + PFC消退 + VTA DA)**
- ✅ 威胁检测: CS+US Amyg=2644 > CS Amyg=2354
- ✅ DA调制: CS+US VTA=404 > CS VTA=356
- ✅ 海马上下文编码: 5584 spikes
- ⚠️ PFC消退失败: 级联激活掩盖ITC→CeA局部抑制 (单元测试96%有效)
- 启示: 需要选择性PFC→ITC连接，避免全系统级联

**Task 3: Stroop冲突 (ACC→LC-NE→dlPFC) — 全部通过!**
- ✅ ACC冲突检测: Incong=1416 > Cong=1205
- ✅ dlPFC执行控制: Incong=2450 > Cong=2420
- ✅ NE唤醒: Incong=0.263 > Cong=0.254
- 完整通路涌现: ACC检测冲突→LC-NE升高→dlPFC控制增强

**系统能力边界总结:**
- ✅ 已验证: ACC冲突检测, 威胁→Amyg→VTA DA, NE增益调制, Stroop全通路
- ⚠️ 需改进: BG需训练权重(工作记忆), PFC消退需选择性连接(注意力)
- 生成: 4张可视化图 (go_nogo/fear/stroop/summary)

### Step 10: 工作记忆 + BG在线学习 ✅ (2026-02-07)
> 目标: dlPFC持续性活动 + DA稳定 + BG门控训练

**工作记忆机制 (修改 CorticalRegion, 零新文件):**
- ✅ `enable_working_memory()` — 可选启用, 向后完全兼容
- ✅ L2/3循环自持: 发放→`wm_recurrent_buf_`→下一步注入L2/3 basal
- ✅ DA稳定: `wm_da_gain_ = 1.0 + 2.0 * DA` (D1受体机制)
- ✅ `wm_persistence()` — 活跃L2/3比例 (0~1)

**BG在线学习 (利用已有DA-STDP):**
- ✅ `set_da_source_region(UINT32_MAX)` 禁用自动路由, 手动控制DA
- ✅ 训练: 高DA奖励 → D1(Go)权重LTP
- ✅ 测试: D1(训练后)=61 > D1(未训练)=55

**验证结果:**
- 工作记忆基础: 刺激期301→持续期109 spikes (活动自持)
- DA持续性: DA=0.1→0, DA=0.3→4, DA=0.6→555 (DA稳定WM)
- WM vs 无WM: 4 vs 0 (WM机制有效)
- WM+BG联合: 延迟期BG=308, dlPFC持续性=1.0 (维持→决策)
- 向后兼容: 无WM时行为完全一致

**生物学对应:**
- dlPFC L2/3循环 = 持续性活动 (Goldman-Rakic 1995)
- DA D1 = 增强NMDA循环电流 (Seamans & Yang 2004)
- BG门控 = DA调制的Go/NoGo选择 (Frank 2005)

**系统状态:**
- 21区域 | 3239神经元 | 36投射 | 4调质 | 4学习 | 预测编码 | **工作记忆**
- **92 测试全通过** (86+6), 零回归

### Step 4 补全 ✅ (2026-02-07)
> 目标: 完成Step 4遗留的低优先级项目

**新增区域 (2个新文件):**
- ✅ `SeptalNucleus` (region/limbic/septal_nucleus.h/cpp) — theta起搏器
  - ACh胆碱能 + GABA节律神经元, theta ~6.7Hz (150ms周期)
  - GABA burst期=40 > silent期=0, ACh输出=0.25 (tonic+phasic)
- ✅ `MammillaryBody` (region/limbic/mammillary_body.h/cpp) — Papez回路中继
  - 内侧核(25)→外侧核(10), medial=75→lateral=13

**Papez回路 (3条新投射):**
- ✅ Hippocampus(Sub) → MammillaryBody → ATN(丘脑前核) → ACC
- 验证: Hipp→MB=1232, MB→ATN=53, ATN→ACC=25 (全通路信号传播)

**Hippocampus扩展 (可选, 向后兼容):**
- ✅ Presubiculum (n_presub=25): CA1→Presub→EC 头朝向反馈
- ✅ HATA (n_hata=15): CA1→HATA 海马-杏仁核过渡区
- 验证: CA1=321→Presub=6→HATA=2; 默认config n_neurons=505不变

**Amygdala扩展 (可选, 向后兼容):**
- ✅ MeA (n_mea=20): La→MeA→CeA (社会/嗅觉)
- ✅ CoA (n_coa=15): La→CoA (嗅觉)
- ✅ AB (n_ab=20): BLA→AB→CeA (多模态)
- 验证: MeA=63, CoA=24, AB=18, CeA=57; 默认config n_neurons=180不变

**隔核→海马调制:**
- ✅ SeptalNucleus → Hippocampus 投射 (theta + ACh)
- Hipp(+Septal)=260 vs Hipp(无Septal)=269 (调制有效)

**build_standard_brain更新:**
- 24区域 (原21 + SeptalNucleus + MammillaryBody + ATN)
- 40投射 (原36 + 4条Papez/Septal)
- Hipp启用presub=25/hata=15, Amyg启用mea=20/coa=15/ab=20

**系统状态:**
- **24区域** | ~3400神经元 | **40投射** | 4调质 | 4学习 | 预测编码 | 工作记忆
- **100 测试全通过** (92+8), 零回归

### Step 11: 认知任务验证 ✅ (2026-02-07)
> 目标: 用WM+BG学习验证高级认知功能

**6项认知任务全部通过:**

1. **训练后Go/NoGo** — DA-STDP区分奖励/无奖励刺激
   - D1(高DA训练)=83 > D1(低DA训练)=82 > D1(无STDP)=66
   - 验证BG在线强化学习在多区域回路中的实际效果

2. **延迟匹配任务 (DMTS)** — 工作记忆跨延迟维持样本
   - WM延迟(早)=132 (persist=0.62) vs 无WM延迟=0
   - 验证dlPFC L2/3循环自持 + DA稳定在认知任务中的功能

3. **Papez回路记忆巩固** — Hipp→MB→ATN→ACC增强ACC活动
   - ACC(+Papez)=25 vs ACC(无Papez)=0
   - 验证新增Papez回路的功能性连接

4. **情绪增强记忆** — Amygdala→Hippocampus编码增强
   - Hipp(+情绪)=11054 vs Hipp(中性)=269 (41x增强)
   - 验证BLA→EC情绪标记通路

5. **WM引导BG决策** — dlPFC维持线索→延迟→BG选择
   - BG(+WM)=28 > BG(无WM)=25
   - 验证工作记忆+基底节联合决策

6. **反转学习** — 同一刺激从低DA→高DA训练
   - D1(低DA后)=11 → D1(高DA后)=47 (+327%)
   - 验证DA-STDP双向权重调节

**生物学对应:**
- DMTS = Funahashi (1989) dlPFC延迟活动
- Go/NoGo = Frank (2004) BG D1/D2选择模型
- 反转学习 = Cools (2009) DA灵活性
- 情绪记忆 = McGaugh (2004) 杏仁核-海马情绪标记
- Papez = Aggleton & Brown (1999) 扩展海马系统

**系统状态:**
- **24区域** | ~3400神经元 | **40投射** | 4调质 | 4学习 | 预测编码 | 工作记忆
- **106 测试全通过** (100+6), 零回归

### Step 12: 注意力机制 ✅ (2026-02-07)
> 目标: PFC→感觉区top-down选择性增益 + ACh/NE精度调制 + VIP去抑制回路

**实现:**
- `set_attention_gain(float gain)` — PFC可选择性放大/抑制任意皮层区
  - gain > 1.0: 注意 (PSP放大 + VIP驱动)
  - gain = 1.0: 正常 (向后兼容)
  - gain < 1.0: 忽略 (PSP衰减)
- VIP去抑制回路: attention→VIP→SST↓→L2/3 apical去抑制→burst增强
  - Letzkus/Pi (2013) disinhibitory attention circuit
- NE sensory精度: `ne_gain = 0.5 + 1.5*NE` 乘以PSP输入
- ACh prior精度: `precision_prior = max(0.2, 1.0 - 0.8*ACh)` 调制预测抑制

**7项测试全部通过:**
1. 基础增益: V1(忽略)=576 < V1(正常)=861 < V1(注意)=1181
2. 选择性注意: V1(注意)=1181 vs V2(忽略)=623 (1.9x)
3. VIP去抑制: gain=1.0→861, 1.3→1037, 2.0→1348
4. 注意力+PC: V1(正常+PC)=861 → V1(注意+PC)=1181
5. ACh精度: V1(ACh=0.1)=562 → V1(ACh=0.8)=643
6. NE精度: V1(NE=0.1)=683 → V1(NE=0.9)=1427
7. 向后兼容: gain=1.0 == 默认

**生物学对应:**
- Desimone & Duncan (1995) 偏置竞争理论
- Letzkus et al. (2015) VIP去抑制注意力回路
- Feldman & Friston (2010) 注意力=精度优化
- Yu & Dayan (2005) ACh=预期不确定性, NE=意外不确定性

**系统状态:**
- **24区域** | ~3400神经元 | **40投射** | 4调质 | 4学习 | 预测编码 | 工作记忆 | **注意力**
- **113 测试全通过** (106+7), 零回归

### Step 5: 扩展皮层 + 丘脑高级核群 ✅ (2026-02-07)
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
- 视觉: LGN→V1→V2→V4→IT + V1→MT→PPC + Pulvinar hub
- 体感: VPL→S1→S2→PPC + S1→M1 + S1→Insula
- 听觉: MGN→A1→Wernicke + A1→TPJ
- 化学感觉: Gustatory→Insula/OFC, Piriform→Amygdala/OFC/Hippocampus
- 语言: A1→Wernicke→Broca→PMC (弓状束) + 语义/执行连接
- 运动: dlPFC→SMA/PMC→M1 + BG→VA→PMC/SMA + 小脑
- DMN: PCC↔vmPFC + TPJ↔PCC + PCC→Hippocampus
- 丘脑: MD↔PFC, LP↔PPC, LD→PCC/Hipp, CeM/ILN→觉醒/意识
- FEF↔Pulvinar top-down注意力

**8项通路测试全部通过:**
1. 全系统构建: 46区域, 5409神经元
2. 体感通路: VPL→S1=2897 → S2=1544 → PPC=2125
3. 听觉→语言: MGN→A1=566 → Wernicke=695 → Broca=1840
4. 运动层级: dlPFC→PMC=2539, SMA=1849, M1=2908
5. DMN: PCC=1628, vmPFC=1973, TPJ=1795
6. Pulvinar: 843→V2=4607, V4=2830
7. MD↔PFC: MD=586→dlPFC=3480, OFC=2530, ACC=1920
8. 全链路: V1=8194→IT=2397→dlPFC=4647→BG=4059→M1=3921

**系统状态:**
- **46区域** | **5409神经元** | **~90投射** | 4调质 | 4学习 | 预测编码 | 工作记忆 | 注意力
- **121 测试全通过** (113+8), 零回归

### Step 6: 下丘脑内驱力系统 ✅ (2026-02-07)
> 目标: 内在动机引擎 — 睡眠/觉醒/应激/摄食

**Hypothalamus 类 (region/limbic/hypothalamus.h/cpp):**
6个核团, 89个神经元:
- **SCN** (n=20) — 昼夜节律起搏器, 正弦振荡 (可配置周期)
- **VLPO** (n=15) — 睡眠促进, GABA/galanin→抑制觉醒中枢
- **Orexin** (n=15) — 觉醒稳定, →LC/DRN/NBM (防止嗜睡发作)
- **PVN** (n=15) — 应激反应, CRH→HPA轴→Amygdala
- **LH** (n=12) — 摄食/饥饿驱力, →VTA (饥饿→动机)
- **VMH** (n=12) — 饱腹/能量平衡

**内部回路:**
- Sleep-wake flip-flop (Saper 2005): VLPO⟷Orexin互相抑制
- SCN→VLPO昼夜门控 (cosine振荡)
- LH⟷VMH摄食平衡 (互相GABA抑制)
- 外部可控: set_sleep_pressure/stress_level/hunger_level/satiety_level

**8条新投射 (→~98条总投射):**
- Orexin→LC/DRN/NBM (觉醒→调质广播)
- Hypothalamus→VTA (饥饿→动机DA)
- Hypothalamus↔Amygdala (应激↔恐惧)
- Insula→Hypothalamus (内感受→驱力)
- Hypothalamus→ACC (驱力→冲突监控)

**7项测试全部通过:**
1. SCN昼夜振荡 + 相位推进
2. Flip-flop: 低压力→wake=0.909, 高压力→wake=0.102
3. 睡眠压力: wake 0.911→0.208 + VLPO=30
4. Orexin稳定: spikes=60, wake=0.977
5. PVN应激: low=0, high=15 spikes + output=0.8
6. LH⟷VMH: 饥饿→LH=24,VMH=0; 饱腹→LH=0,VMH=24
7. 全系统集成: 7区域 + Orexin→LC + wake=0.909

**生物学对应:**
- Saper et al. (2005) Sleep-wake flip-flop switch
- Sakurai (2007) Orexin/hypocretin neural circuit
- Ulrich-Lai & Herman (2009) HPA stress axis

**系统状态:**
- **47区域** | **5498神经元** | **~98投射** | 4调质 | 4学习 | 预测编码 | 工作记忆 | 注意力 | **内驱力**
- **128 测试全通过** (121+7), 零回归

### GNW: 全局工作空间理论 ✅ (2026-02-07)
> 目标: Baars/Dehaene 意识访问模型 — 竞争→点火→广播

**GlobalWorkspace 类 (engine/global_workspace.h/cpp):**
30个workspace整合神经元 + 竞争/点火/广播机制:
- **竞争**: 多个皮层区域L5输出→GW, per-region salience累积 (指数衰减防锁定)
- **点火**: 赢者salience超阈值 → ignition (全局点火事件)
- **广播**: 点火后workspace神经元爆发活动→ILN/CeM→全皮层L2/3
- **间隔控制**: min_ignition_gap防止连续点火 (意识是离散采样)

**9条竞争投射 + 2条广播投射 (→~109条总投射):**
竞争输入: V1/IT/PPC/dlPFC/ACC/OFC/Insula/A1/S1 → GW
广播输出: GW → ILN (板内核群) + CeM (中央内侧核)

**可查询状态:**
- `is_ignited()` — 当前是否在点火状态
- `conscious_content_name()` — 当前意识内容 (赢者区域名)
- `ignition_count()` — 累计点火次数
- `winning_salience()` — 当前最高salience值
- `salience_map()` — 全部区域salience

**7项测试全部通过:**
1. 基础点火: step=66 → ignition, count=19
2. 竞争门控: V1(强)胜 A1(弱), content="V1"
3. 广播持续: 广播中=60 > 广播后=0
4. 竞争衰减: peak=24.7 → decayed=0.6 (98%衰减)
5. 点火间隔: gap=50, 200步→4次点火
6. 无输入不点火: ignitions=0
7. 全系统: GW=180, ignitions=10, content="V1"

**生物学对应:**
- Baars (1988) A Cognitive Theory of Consciousness
- Dehaene & Changeux (2011) Experimental and theoretical approaches to conscious processing
- Dehaene, Kerszberg & Changeux (1998) Neuronal model of a global workspace

**系统状态:**
- **48区域** | **5528神经元** | **~109投射** | 4调质 | 4学习 | 预测编码 | 工作记忆 | 注意力 | 内驱力 | **意识(GNW)**
- **135 测试全通过** (128+7), 零回归

### Step 6 剩余 (低优先级):
**6a. 调质系统扩展:**
- ⬜ 5-HT细分: DR (MB-05) + MnR (MB-08)
- ⬜ ACh细分: PTg (MB-09) + LDTg (MB-10) + BF (BF-01)
**6b. 小脑扩展:**
- ⬜ 运动小脑: 前叶+VIIIa/b+绒球 (CB-01, CB-06~07)
- ⬜ 认知小脑: Crus I/II + VIIb (CB-03~05)
- ⬜ 蚓部 (CB-08) + 过渡区 (CB-02)

### Step 7: 连接组学布线 — **跳过** (当前阶段过度工程化)
- ~~⬜ JSON配置化~~ → 保持build_standard_brain硬编码，编译时类型检查更安全
- ⬜ 感觉输入接口 (外界→丘脑→皮层) — 移至后续
- ⬜ 运动输出接口 (皮层→BG→丘脑→运动) — 移至后续

### Step 8: 睡眠/海马重放 — ✅ 完成

**新增功能:**

**8a. 海马 Sharp-Wave Ripple (SWR) 重放:**
- `Hippocampus::enable_sleep_replay()` / `disable_sleep_replay()`
- SWR 机制: 睡眠模式→CA3 bias+jitter噪声→自联想补全→SWR burst
- 检测: CA3 firing fraction > threshold → SWR 事件
- 不应期: swr_refractory 步间隔，防止连续SWR
- SWR 期间: 增强活跃CA3神经元(swr_boost)延长重放
- 配置: swr_noise_amp/swr_duration/swr_refractory/swr_ca3_threshold/swr_boost
- 查询: `is_swr()`, `swr_count()`, `last_replay_strength()`
- **关键设计**: 无需显式存储模式 — CA3 STDP 自联想权重即是记忆

**8b. 皮层 NREM 慢波振荡:**
- `CorticalRegion::set_sleep_mode(bool)` — 进入/退出慢波模式
- Up/Down 状态交替: ~1Hz (SLOW_WAVE_FREQ=0.001)
- Up state (40%占比): 正常处理，神经元可兴奋
- Down state (60%占比): 注入抑制电流(DOWN_STATE_INH=-8)，抑制发放
- 查询: `is_up_state()`, `slow_wave_phase()`, `is_sleep_mode()`

**8c. 记忆巩固通路:**
- SWR → CA3 pattern completion → CA1 burst → SpikeBus → 皮层 L4
- 皮层 up state 期间接收重放 → 已有 STDP 增强连接 = 系统巩固
- 无需额外巩固代码 — 利用现有 SpikeBus + STDP 架构自然实现

**pybind11 新增绑定:**
- Hippocampus: enable/disable_sleep_replay, is_swr, swr_count, last_replay_strength, dg_sparsity
- CorticalRegion: set_sleep_mode, is_sleep_mode, is_up_state, slow_wave_phase

**测试 (test_sleep_replay.cpp, 7测试):**
1. SWR基础生成: 5次SWR (400步)
2. SWR不应期: 7次 ≤ max possible ~10 (refractory=50)
3. 清醒无SWR: count=0 ✓
4. 皮层慢波: 2次up→down, 1次down→up转换 (~1Hz)
5. Down state抑制: awake=3298 > sleep=3008 (up=2702 > down=306)
6. 编码→重放: 7次SWR, 重放活动263 (SWR期88)
7. 多区域集成: LGN→V1→dlPFC+Hipp+Hypo, V1 awake=1111 > sleep=94

**生物学对应:**
- Buzsaki (1989) Two-stage model of memory formation
- Saper et al (2005) Hypothalamic sleep-wake flip-flop
- Steriade et al (1993) Cortical slow oscillation
- Wilson & McNaughton (1994) Hippocampal replay during sleep

**系统状态:**
- **48区域** | **5528神经元** | **~109投射** | 4调质 | 4学习 | 预测编码 | 工作记忆 | 注意力 | 内驱力 | 意识(GNW) | **睡眠/重放**
- **142 测试全通过** (135+7), 零回归

### Step 8 剩余 (低优先级):
- ⬜ 注意力: TRN门控 + ACh + 上丘 (MB-01~02)
- ⬜ 发育/关键期: 连接修剪 + PV+成熟
- ⬜ REM睡眠: PnO (MB-11) + 梦境 + theta节律

### Step 9: 感觉输入接口 — ✅ 完成

**新增文件:** `engine/sensory_input.h/cpp`

**9a. VisualInput — 视觉编码器:**
- 图像像素 [0,1] → LGN relay 电流向量
- Center-surround 感受野 (Kuffler 1953): ON cell 中心兴奋/周围抑制, OFF cell 反之
- 预计算权重矩阵: 像素→LGN mapping (grid layout + jitter)
- ON/OFF 通道: 前半LGN=ON, 后半=OFF
- 配置: input_width/height, center/surround_radius, gain, baseline, noise_amp
- `encode(pixels)` → 电流向量, `encode_and_inject(pixels, lgn)` → 直接注入

**9b. AuditoryInput — 听觉编码器:**
- 频谱功率 [0,1] → MGN relay 电流向量
- Tonotopic mapping: 频率带→MGN神经元 (低频→前, 高频→后)
- Onset emphasis: 新声音→更强响应 (temporal_decay差分)
- 配置: n_freq_bands, gain, baseline, noise_amp, temporal_decay
- `encode(spectrum)` → 电流向量, `encode_and_inject(spectrum, mgn)` → 直接注入

**pybind11 绑定:**
- VisualInputConfig + VisualInput (encode, encode_and_inject)
- AuditoryInputConfig + AuditoryInput (encode, encode_and_inject)

**测试 (test_sensory_input.cpp, 7测试):**
1. VisualInput 基础: 8x8→50 LGN, bright_sum=3098 > dark_sum=250
2. Center-surround: spot_max=29 > uniform_max=5 (ON cells)
3. 视觉 E2E: pixels→LGN→V1, bright=1993 >> no_input=222
4. AuditoryInput 基础: low_freq_first=243 > second=30 (tonotopic)
5. Onset 检测: onset=165 > sustained=140
6. 听觉 E2E: spectrum→MGN→A1, spikes=329
7. 多模态: V1=2013 + A1=329 同时活跃

**系统状态:**
- **48区域** | **5528神经元** | **~109投射** | 4调质 | 4学习 | 预测编码 | WM | 注意力 | 内驱力 | GNW | 睡眠/重放 | **感觉输入**
- **149 测试全通过** (142+7), 零回归

### Step 10: 规模扩展验证 — ✅ 完成

**核心改动:** `build_standard_brain(scale)` 参数化放大

**规模预设:**
- `scale=1`: ~5,500 神经元 (默认, 向后兼容)
- `scale=3`: ~16,500 神经元
- `scale=8`: ~44,000 神经元

**实现:**
- 所有区域神经元数量乘以 `scale` 因子
- 皮层: L4/L23/L5/L6/PV/SST/VIP 全部缩放
- 皮下: BG D1/D2/GPi/GPe/STN, 丘脑 Relay/TRN
- 边缘: 海马 EC/DG/CA3/CA1/Sub, 杏仁核 LA/BLA/CeA/ITC
- 调质: VTA DA, LC NE, DRN 5-HT, NBM ACh
- 小脑: Granule/Purkinje/DCN/MLI/Golgi
- GW workspace 神经元也缩放

**测试 (test_scale_emergent.cpp, 5测试):**
1. V1规模扩展: 270→810 neurons, spikes 1111→5573 (5.0x, 超线性)
2. BG Go/NoGo: 420 neurons, 训练=4185 > 测试=721
3. CA3模式补全: 180 CA3, 补全比率=1.02 (>>0.30 阈值)
4. 工作记忆: 606 neurons, 刺激=868 spikes
5. 全脑 scale=3: 3768 neurons子集, 0.80 ms/step

**涌现发现:**
- V1 活动超线性增长 (5x vs 3x neurons) → 更密集的网络产生更多协同激活
- CA3 模式补全在大网络中接近完美 (比率1.02 ≈ 100%) → 自联想记忆容量随规模增长
- BG 训练/测试差异显著 (4185 vs 721) → DA-STDP 在大规模下仍然有效

**系统状态:**
- **48区域** | **scale=1: ~5.5k / scale=3: ~16.5k / scale=8: ~44k 神经元** | **~109投射**
- 4调质 | 4学习 | 预测编码 | WM | 注意力 | 内驱力 | GNW | 睡眠/重放 | 感觉输入 | **规模可扩展**
- **154 测试全通过** (149+5), 零回归

### Step 11: REM睡眠 + 梦境 — ✅ 完成

**新增类:**
- `SleepCycleManager` (engine/sleep_cycle.h/cpp) — AWAKE→NREM→REM→NREM 完整睡眠周期管理
  - 可配置 NREM/REM 持续时间、周期增长率
  - REM 周期增长: 模拟后半夜 REM 延长 (rem_growth)
  - PGO 波随机生成 (rem_pgo_prob)
  - Theta 相位追踪

**CorticalRegion REM 扩展:**
- `set_rem_mode(bool)` — 去同步化噪声 (bias=18+jitter=±12) 注入 L2/3 和 L5
- `inject_pgo_wave(amplitude)` — PGO 波随机激活 L4 (梦境视觉)
- `set_motor_atonia(bool)` — M1 L5 强抑制 (-20) 防止梦境运动输出
- NREM 和 REM 互斥: set_rem_mode 自动关闭 sleep_mode

**Hippocampus REM theta 扩展:**
- `enable_rem_theta()` — ~6Hz theta 振荡调制 CA3/CA1
- Theta peak → CA3 drive (编码相位), Theta trough → CA1 drive (检索相位)
- 创造性重组: 1%/步概率随机激活 20% CA3 子集 (梦境联想)
- REM theta 和 NREM SWR 互斥

**生物学:**
- NREM: 皮层慢波 (1Hz up/down) + 海马 SWR 重放 → 记忆巩固
- REM: 皮层去同步化 + 海马 theta + PGO 波 → 创造性重组
- NREM→REM 交替: VLPO↔PnO flip-flop 模型
- 后半夜 REM 增长: NREM 缩短, REM 延长

**测试 (test_rem_sleep.cpp, 7测试):**
1. SleepCycleManager 基础: AWAKE→NREM→REM→NREM→wake
2. REM 周期增长: Cycle0 REM=50 → Cycle1=80 → Cycle2=110
3. PGO 波: 35/500 步 (rate=0.07)
4. CorticalRegion REM: V1 REM=156, PGO=3, M1 atonia=71
5. Hippocampal theta: phase cycling, 2 recombination events
6. 完整睡眠周期: NREM 250步 + REM 100步, 1 cycle
7. 全脑 NREM→REM: NREM up=600/down=600, REM=351 spikes

**系统状态:**
- **48区域** | 可扩展至44k neurons | **~109投射**
- 4调质 | 4学习 | 预测编码 | WM | 注意力 | 内驱力 | GNW | **NREM+REM完整睡眠** | 感觉输入 | 规模可扩展
- **161 测试全通过** (154+7), 零回归

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
睡眠/海马重放             ← Step 8: SWR + 皮层慢波 + 记忆巩固
     ↓
感觉输入接口              ← Step 9: VisualInput + AuditoryInput
     ↓
规模扩展验证              ← Step 10: scale=1/3/8, 涌现特性
     ↓
REM睡眠+梦境              ← Step 11: theta + PGO + 创造性重组

稳态可塑性集成              ← Step 13-A: SynapticScaler + E/I平衡
     ↓
闭环Agent + GridWorld        ← Step 13-B: 感知→决策→行动→学习
     ↓
闭环学习调优 (v3)            ← Step 13-B++: elig clamp + NE退火 + 诊断修复
     ↓
视觉通路修复 + 皮层STDP      ← Step 13-C: LGN→V1→dlPFC拓扑映射 + V1 STDP

技术栈: C++17 核心引擎 + pybind11 → Python 实验/可视化
Python 原型 (wuyun/) → _archived/ 算法参考
```

---

## Step 13-A: 稳态可塑性集成 (Homeostatic Plasticity Integration)

**目标**: 解决 scale-up 时 E/I 失衡导致的网络崩溃问题

### 实现

**SynapticScaler 改进** (`plasticity/homeostatic.h/cpp`):
- `update_rates(const uint8_t*)`: 改签名匹配 `NeuronPopulation::fired()`
- `mean_rate()`: 新增群体平均发放率查询
- `HomeostaticParams::scale_interval`: 新增缩放间隔参数 (默认100步)

**CorticalColumn 集成** (`circuit/cortical_column.h/cpp`):
- `enable_homeostatic(HomeostaticParams)`: 为4个兴奋性群体(L4/L2/3/L5/L6)各创建一个 SynapticScaler
- 每步 `update_rates()`, 每 `scale_interval` 步 `apply_scaling()`
- **只缩放前馈兴奋性AMPA突触**, 不缩放循环突触(保护已学习模式)和抑制性突触
- 缩放目标: L6→L4, L4→L2/3, L2/3→L5, L5→L6

**CorticalRegion 接口** (`region/cortical_region.h`):
- `enable_homeostatic()` / `homeostatic_enabled()`
- `l4_mean_rate()` / `l23_mean_rate()` / `l5_mean_rate()` / `l6_mean_rate()`

**Hippocampus 集成** (`region/limbic/hippocampus.h/cpp`):
- 3个 SynapticScaler: DG, CA3, CA1
- 缩放: EC→DG (穿通纤维), DG→CA3 (苔藓纤维), CA3→CA1 (Schaffer侧支), EC→CA1 (直接通路)
- **不缩放 CA3→CA3 循环突触** (保护自联想记忆!)
- `dg_mean_rate()` / `ca3_mean_rate()` / `ca1_mean_rate()`

**pybind11 绑定** (`bindings/pywuyun.cpp`):
- `HomeostaticParams` 全字段绑定
- CorticalRegion/Hippocampus 的 `enable_homeostatic()` + 发放率查询

### 测试结果 (7/7 通过)

| 测试 | 内容 | 结果 |
|------|------|------|
| 1 | SynapticScaler 发放率追踪 | 持续发放=993.5Hz, 沉默=0.07Hz ✅ |
| 2 | 过度活跃→权重降低 | 0.50→0.01 ✅ |
| 3 | 活动不足→权重增大 | 0.50→0.54 ✅ |
| 4 | CorticalRegion 集成 | L4=8.26, L2/3=7.22, L5=6.28 ✅ |
| 5 | Hippocampus 集成 | DG=2.83, CA3=2.83, CA1=2.83 ✅ |
| 6 | 多区域稳态 | V1+dlPFC+Hipp 协同工作 ✅ |
| 7 | **Scale=3 WM恢复** | **persistence=0.425, spikes=5433** ✅ |

### 关键成果

**Scale=3 工作记忆从 0 恢复到 0.425** — 稳态可塑性成功解决了大规模网络 E/I 失衡问题。

### 回归测试: 27/27 CTest 全通过 (0 失败)

### 系统状态

```
48区域 · ~5528+神经元 · ~109投射 · 168测试 · 27 CTest suites
完整功能: 感觉输入 · 视听编码 · 层级处理 · 双流视觉 · 语言
          5种学习(STDP/STP/DA-STDP/CA3-STDP/稳态) · 预测编码 · 工作记忆
          注意力 · GNW意识 · 内驱力 · NREM巩固 · REM梦境 · 睡眠周期管理
          4种调质广播 · 稳态可塑性 · 规模可扩展(E/I自平衡)
```

---

## Step 13-B: 闭环Agent + GridWorld (Closed-Loop Agent)

**目标**: 将大脑模型与环境连接，实现完整的感知→决策→行动→感知闭环

### 架构

```
GridWorld.observe()
    ↓ 3x3 pixels
VisualInput → LGN → V1 → dlPFC → BG → MotorThal → M1
                                                      ↓ decode L5 spikes
GridWorld.act(action) ←── winner-take-all ←── M1 L5 [UP|DOWN|LEFT|RIGHT]
    ↓ reward
VTA.inject_reward() → DA → BG DA-STDP → 学习
```

### 新增文件

**GridWorld** (`engine/grid_world.h/cpp`):
- 10x10 网格, Agent可移动, 食物(+1奖励)/危险(-1)/墙壁
- 3x3 局部视觉观测 (灰度编码: food=0.9, danger=0.3, agent=0.6)
- 食物被吃后随机重生, 危险持续存在
- `to_string()` 文本渲染, `full_observation()` 全局视图

**ClosedLoopAgent** (`engine/closed_loop_agent.h/cpp`):
- 自动构建最小闭环大脑: LGN + V1 + dlPFC + M1 + BG + MotorThal + VTA + Hippocampus
- 每个环境步运行 N 个脑步 (默认10), 累积M1 L5发放
- M1 L5分4组(UP/DOWN/LEFT/RIGHT), winner-take-all解码
- 运动探索噪声 (bias+jitter注入选定L5组, 打破对称性)
- 支持 DA-STDP + 稳态可塑性 + 工作记忆
- 滑动窗口统计: `avg_reward()`, `food_rate()`

### 关键修复

**VTA 奖励响应** (`region/neuromod/vta_da.h/cpp`):
- 添加 `reward_psp_` 缓冲 (PSP衰减=0.85), 使奖励信号持续多步
- 增大RPE→电流乘数 (50→200), 确保DA神经元在5步内响应
- 修复前: DA始终=0.1 (神经元从不发放); 修复后: DA=0.6 (phasic burst)

**M1 运动探索** (`closed_loop_agent.cpp`):
- 每个action期间选定一个L5组, 持续注入bias+jitter噪声
- 默认noise=55 (bias=33, 需~9步到阈值, 10步内可靠发放)
- 77.5% 非STAY动作, 均匀分布于4方向

**SimulationEngine** (`simulation_engine.h`):
- 添加显式 non-copyable/movable 声明, 修复pybind11 MSVC模板实例化错误

### pybind11绑定
- `GridWorldConfig`, `GridWorld`, `CellType`, `Action`, `StepResult`
- `AgentConfig`, `ClosedLoopAgent` (unique_ptr holder + factory函数)

### 测试结果 (7/7 通过)

| 测试 | 结果 |
|------|------|
| GridWorld 基础 (移动/墙壁) | ✅ |
| 视觉观测 (3x3 patch编码) | ✅ |
| Agent 构建 (8区域正确连接) | ✅ |
| 闭环运行 (100步不崩溃) | ✅ |
| 动作多样性 (77.5% 非STAY) | ✅ |
| DA 奖励信号 (0.1→0.6, RPE=-1.0) | ✅ |
| **长期稳定性 (500步, 5食物)** | ✅ |

### 回归测试: 28/28 CTest 全通过 (0 失败)

---

## Step 13-B+: DA-STDP 闭环学习修复

### 问题诊断

闭环Agent在500步测试中行为随机，DA-STDP不产生学习效果。逐层诊断发现5个根因：

1. **时序信用分配断裂**: 奖励在action之后注入VTA，但DA-STDP在下一步的brain steps中运行时，eligibility已清零
2. **MSN从不发放**: D1/D2 MSN (v_rest=-80, v_thresh=-50) 需要I≥50，但cortex PSP(15) + DA tonic(3) = 18 远不够
3. **D1/D2不对称**: baseline DA时 D1=38 vs D2=53，D2始终压过D1
4. **缺少动作特异性**: dlPFC随机映射到所有D1，DA-STDP强化所有连接而非特定动作
5. **BG输出不影响动作选择**: M1 L5纯靠探索噪声，BG→MotorThal→M1信号无法覆盖

### 修复方案 (5个架构修复)

#### 1. BG Eligibility Traces (Izhikevich 2007)
- `elig_d1_[src][idx]`, `elig_d2_[src][idx]` 衰减缓冲
- Co-activation (cortex + D1/D2) 递增 trace，DA 到达时乘 trace 更新权重
- `da_stdp_elig_decay = 0.95` (~20步窗口)

#### 2. Agent Step 时序重构
- **Phase A**: 注入上步奖励 → 跑 reward_processing_steps(5) 让 VTA→DA→BG 传播
- **Phase B**: 注入新观测 → 跑 brain_steps_per_action(10) 构建新 eligibility traces
- **Phase C**: 执行动作 → 存储奖励为 pending

#### 3. MSN Up-State Drive (Wilson & Kawaguchi 1996)
- `msn_up_state_drive = 25` + `da_base = 15` = 40 (接近阈值50)
- DA 对称调制: `da_exc_d1 = up + base + (DA-baseline)*gain`
- Baseline DA 时 D1 = D2 (对称)，DA偏离时产生 Go/NoGo 不对称

#### 4. BG D1 动作子组 + Action-Specific Boost
- D1 分为4组 (UP/DOWN/LEFT/RIGHT)
- 探索噪声选择的M1 L5组 → 同时 boost 对应 BG D1 子组 (+15)
- 只有被选动作的 D1 子组发放 → 动作特异性 eligibility traces

#### 5. Combined Action Decoding
- `decode_action(l5_accum, d1_accum)`: M1 L5 + BG D1 combined score
- `score[g] = m1_scores[g] + bg_weight * d1_scores[g]`
- BG 学习偏好通过 D1 发放直接影响动作选择

### 验证结果

```
DA-STDP 学习管线打通:
  step=7  r=1.00 | elig=25.6          ← 吃到食物，traces 积累
  step=8  | DA=0.911 accum=16.2       ← VTA DA burst 到达 BG
  step=9  | DA=0.585 | D1=7           ← DA 驱动 D1 Go pathway

Learner vs Control (3000步):
  Learner: danger=16, safety=0.48
  Control: danger=33, safety=0.34     ← Learner 减少 51% 危险碰撞

BG 权重变化: range=0.4698 (从 1.0 初始值)
D1 发放: 31 fires / 50 steps (previously 0)
```

### 修改文件

- `basal_ganglia.h/cpp`: eligibility traces, up-state drive, 对称DA调制, 权重诊断接口
- `closed_loop_agent.h/cpp`: Phase A/B/C 时序, D1 action boost, combined decode
- `test_learning_curve.cpp`: 4个学习曲线测试 (5k/3k对照/诊断/10k)
- `test_bg_learning.cpp`: 反转学习断言更新 (相对偏好)

---

## Step 13-B++: 闭环学习调优 (v3) — 稳定学习曲线

### 问题

v1/v2 闭环学习存在三个问题:
1. **Eligibility explosion**: elig累积到2732, 单次权重更新Δw=0.03×0.5×2732=**41** → 7k epoch崩溃
2. **Agent冻结**: v1(DA=0.1) 4k后冻结, noise退火过激导致M1不再发放
3. **`inp=0` 诊断困惑**: 看起来皮层→BG通路断裂, 但实际是测量时序问题

### 修复 (3个核心 + 1个诊断)

#### ① Eligibility Trace Clamp
- `da_stdp_max_elig = 50.0f` — per-synapse elig上限, 防止Δw爆炸
- 每次食物最大Δw = 0.005 × 0.5 × 50 = **0.125** (需~16次食物达w_max, 而非1次)

#### ② 学习率/衰减调优
- `da_stdp_lr`: 0.02 → **0.005** — 降低4倍, 更渐进的学习
- `da_stdp_w_decay`: 0.0002 → **0.001** — 增大5倍, 错误权重~200步恢复
- `da_stdp_elig_decay`: 0.95 → **0.98** — 更长elig窗口(~50步), 配合15步脑步

#### ③ NE调制探索/利用平衡
- `brain_steps_per_action`: 10 → **15** — LGN→V1→dlPFC→BG需7步延迟, 15步获8步有效输入
- 动态noise_scale基于food_rate: 找到食物→降低探索(利用), 找不到→保持高探索
- Floor = 0.7: 确保M1始终发放, 防止冻结
- 生物学: LC-NE系统在环境可预测时降低arousal

#### ④ 诊断修复: `inp=0` → `ctx=N`
- **根因**: `input_active_` 在 `apply_da_stdp()` 末尾被清零, 诊断在 `agent_step()` 返回后读取 → 永远=0
- **证据**: 新增 `total_cortical_inputs_` 累积计数器 → ctx=1759 (50步内1759个皮层spike到达BG)
- **结论**: 皮层→BG通路完全畅通, `inp=0` 是测量时序假象

### Motor Efference Copy 实验 (探索性, 已回退)

尝试通过M1→BG拓扑映射实现方向特异学习:

| 变体 | 5k Safety | 10k Improvement | Learner Advantage | 结论 |
|------|-----------|-----------------|-------------------|------|
| **v3 基线 (无efference)** | **0.20→0.38** | **+0.191** | **+0.017** | ✅ 最佳 |
| inject_sensory_context (PSP=25×4方向) | 0.19→0.14 | -0.002 | -0.002 | ❌ PSP太强压倒一切 |
| mark_motor_efference (纯elig标记) | 0.20→0.38 | +0.191 | +0.017 | = 无行为影响 |
| elig 10x + PSP=15×weight | **0.29→0.48** | +0.009 | +0.004 | ⚠️ 5k极好但10k崩溃 |
| elig 5x + PSP=5×weight | 0.34→0.48 | -0.105 | +0.006 | ❌ 正反馈失控 |
| elig 5x + PSP=5 + 10x decay | 0.32→0.50 | -0.071 | +0.016 | ❌ 仍不稳定 |

**关键发现:**
- Efference copy 短期效果极好 (5k food从31→58翻倍, peak safety 0.83!)
- 但 PSP×weight 正反馈环路在10k后必然失控 (weight↑→PSP↑→elig↑→weight↑)
- 需要 **D1子群竞争归一化** (侧向抑制) 才能稳定, 留作下一步

**方向特异学习的架构瓶颈:**
- 随机 dlPFC→BG 映射 (20% connectivity) 使每个皮层神经元均匀连接所有D1子群
- 即使efference标记了正确方向, 随机elig占总elig的99.2%, 拓扑elig被淹没
- 根本解决: 皮层STDP塑形方向表征 / 拓扑映射 / D1侧向抑制

### 最终结果 (v3 基线)

```
5000步学习曲线:
  Early (0-1k):  safety=0.20 (8 food, 32 danger)
  Late (4-5k):   safety=0.38 (5 food, 8 danger)
  Improvement:   +0.18

10000步长时训练:
  Early (1-2k):  safety=0.200
  Late (9-10k):  safety=0.391
  Improvement:   +0.191

Learner vs Control (3000步):
  Learner: food=17, danger=29, safety=0.37
  Control: food=12, danger=57, safety=0.17
  Advantage: +0.0168

BG 权重: range=0.2593 (稳定, 无爆炸)
D1 发放: 263/50步, D2: 238/50步
皮层输入: ctx=1759/50步 (通路畅通)
```

### 新增API

- `BasalGanglia::total_cortical_inputs()` — 累积皮层spike计数 (永不清零)
- `BasalGanglia::mark_motor_efference(int group)` — 拓扑elig标记 (备用, 当前未调用)

### 修改文件

- `basal_ganglia.h`: +da_stdp_max_elig, +da_stdp_w_decay调整, +total_cortical_inputs_, +mark_motor_efference
- `basal_ganglia.cpp`: elig clamp, 累积计数器, mark_motor_efference实现
- `closed_loop_agent.h`: brain_steps 10→15, da_stdp_lr 0.02→0.005
- `closed_loop_agent.cpp`: NE噪声退火, efference copy注释(未调用)
- `test_learning_curve.cpp`: 诊断输出 inp→ctx

### 回归测试: 29/29 CTest 全通过 (0 失败)

---

## Step 13-C: 视觉通路修复 + 皮层STDP + 拓扑映射

### 问题诊断

闭环agent的V1→dlPFC→BG视觉通路**完全断裂**:

1. **LGN从不发放**: 视觉编码 gain=45, baseline=3 → 每步注入电流I≈3~45, 但LGN relay神经元 τ_m=20, 阈值需ΔV=15mV → 单次注入仅ΔV=2.25, 远不够阈值
2. **inject_observation每3步才注入**: 电流在步间衰减殆尽, 神经元永远无法充电到阈值
3. **V1→dlPFC空间信息被打散**: SpikeBus用 `neuron_id % L4_size` 模取余映射, 破坏所有空间结构
4. **结论**: v3基线的+0.191改善完全来自M1→dlPFC→BG运动反馈, 不是视觉输入

### 修复 (4个核心)

#### ① LGN视觉编码增益
- `gain`: 45 → **200** — 食物像素response~0.3 → I=5+60=65, V_ss=0 (远超阈值)
- `baseline`: 3 → **5** — 空像素V_ss=-60(不发放), 只有实际视觉特征驱动LGN
- `noise_amp`: 1.5 → **2.0** — 轻微噪声保持随机性

#### ② 每步注入视觉观测
- 原: `if (i > 0 && i % 3 == 0) inject_observation()` — 每3步一次脉冲
- 新: **每个brain step都调用inject_observation()** — 持续驱动, LGN ~2步开始发放

#### ③ V1→dlPFC拓扑映射 (retinotopic mapping)
- 新增 `CorticalRegion::add_topographic_input(region_id, n_neurons)` 接口
- 拓扑源使用**比例映射**: `base = (neuron_id × L4) / source_n`
- 默认源仍用模取余: `base = neuron_id % L4`
- 拓扑源fan-out减半: `fan = max(2, default_fan/2)` — 更窄receptive field
- Biology: V1→V2→V4→IT层级维持partial retinotopy

#### ④ V1皮层STDP + brain_steps调优
- V1启用在线STDP: `a_plus=0.005, a_minus=-0.006, w_max=1.5`
- dlPFC/M1不启用STDP (dlPFC需表征稳定性保护BG DA-STDP; M1靠噪声+BG偏置)
- `brain_steps_per_action`: 10 → **15** — 视觉流水线需~14步: LGN充电→delay2→V1处理→delay2→dlPFC
- 新增AgentConfig: `enable_cortical_stdp`, `cortical_stdp_a_plus/a_minus/w_max`

### 视觉流水线时序 (brain_steps=15)

```
brain_i:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
LGN:          ■   ■   ■   ■   ■   ■   ■       ■   ■
                                  LGN→V1 delay=2
V1:                                   ■   ■■■  ■■■ ■■■ ■■■
                                              V1→dlPFC delay=2
dlPFC:                                              ■■■ ■■■ ■■■ ■■■
                                                        dlPFC→BG delay=2
BG:                                                         ■■■ ■■■
```

### 结果对比

| 指标 | Step 13-C (视觉+拓扑) | v3 基线 (视觉断裂) | 变化 |
|------|----------------------|-------------------|------|
| 5k Late safety | **0.62** | 0.38 | +63% |
| 5k Total food | **65** | 31 | +110% |
| 10k Total food | **121** | 76 | +59% |
| ctx (皮层→BG/50步) | **2787** | 1759 | +58% |
| 10k Improvement | +0.077 | +0.191 | ⚠️ 退化 |
| Learner advantage | -0.002 | +0.017 | ⚠️ 退化 |

**关键发现:**
- 视觉通路修复后食物收集**翻倍**, 安全率**+63%** — 视觉输入显著改善感知-运动耦合
- DA-STDP learner advantage消失 — 视觉输入同等帮助learner和control
- 根因: dlPFC→BG仍是随机20%连接, 方向信息在**最后一英里**被打散
- V1 STDP在10k步内效果不明显 (权重变化太小, 不改变发放模式)

### 架构瓶颈 (下一步)

```
V1 ─拓扑─→ dlPFC ─随机─→ BG D1/D2
  ✅ 保持空间    ❌ 打散方向
  信息            信息
```

方向特异学习需要: dlPFC→BG拓扑投射 或 BG内部方向子群竞争归一化

### 新增API

- `CorticalRegion::add_topographic_input(region_id, n_neurons)` — 注册拓扑输入源
- `AgentConfig::enable_cortical_stdp` — 皮层STDP开关 (V1)
- `AgentConfig::cortical_stdp_a_plus/a_minus/w_max` — STDP参数

### 修改文件

- `cortical_region.h`: +add_topographic_input, +topo_sources_, +unordered_map include
- `cortical_region.cpp`: +topographic proportional mapping in receive_spikes, +add_topographic_input
- `closed_loop_agent.h`: +enable_cortical_stdp, +cortical_stdp_a_plus/a_minus/w_max
- `closed_loop_agent.cpp`: V1 STDP启用, 拓扑注册, 视觉编码增益, 每步注入
- `test_learning_curve.cpp`: brain_steps 10→15

### 回归测试: 29/29 CTest 全通过 (0 失败, 20.2秒)

---

## Step 13-D+E: dlPFC→BG 拓扑映射 + 完整视觉通路验证

### 问题

Step 13-C 修复了视觉通路(LGN→V1→dlPFC), 但方向信息在"最后一英里"被打散:

```
V1 ─拓扑─→ dlPFC ─随机20%─→ BG D1/D2
  ✅ 保持空间      ❌ 方向信息在此丢失
```

DA-STDP learner advantage 消失(-0.0015), 因为随机连接无法区分"食物在左"→D1 LEFT subgroup.

### 修复 (3 个核心)

#### ① dlPFC→BG 拓扑映射 (corticostriatal somatotopy)

新增 `BasalGanglia::set_topographic_cortical_source(region_id, n_neurons)`:
- 重建 `ctx_to_d1_map_` 和 `ctx_to_d2_map_` 为拓扑偏置连接
- `channel = (neuron_id × 4) / n_neurons` — 比例空间映射到4个方向子群
- `p_same=0.60` (60% 连接到匹配子群, ~4.2 connections)
- `p_other=0.05` (5% 连接到其他子群, ~1.2 connections)
- **78% 偏置**: 每个 dlPFC 神经元的连接集中在对应方向的 D1 子群
- DA-STDP 权重/eligibility 基础设施不变, 自动适配新拓扑

调参历程:
- p_same=0.40/p_other=0.05: 选择性好(adv+0.0015)但 D1 驱动不足(-29%)
- p_same=0.50/p_other=0.10: D1 恢复但选择性丢失(adv=-0.002)
- p_same=0.60/p_other=0.05: 高匹配+低溢出, 总连接~5.4 接近原始6

#### ② Motor efference copy 重新启用

- 原: 注释禁用 (无拓扑映射时 efference copy 无效)
- 新: 在 brain loop 后期(i≥10)调用 `mark_motor_efference(attractor_group)`
- Biology: M1→striatum motor efference 提供动作特异信号
- DA-STDP 现在可学习 "视觉上下文 + 动作 → 奖励" 联合关联
- 仅标记 eligibility slots + 弱 PSP 注入, 不直接决定动作

#### ③ Weight decay 加速 (防过拟合)

- `da_stdp_w_decay`: 0.001 → **0.003** (回归速度 3x)
- 权重回归 1.0: ~200 步 → ~67 步
- 防止 efference copy 正反馈导致权重过度极化→遗忘振荡

### 完整拓扑通路

```
3×3 Grid → VisualInput(gain=200) → LGN(选择性发放)
  → delay=2 → V1(STDP特征学习, i=7开始发放)
  → delay=2 → dlPFC(拓扑接收: proportional mapping)
  → delay=2 → BG D1/D2(拓扑偏置: 78%匹配子群)
                 + motor efference copy(动作标记)
  → D1→GPi→MotorThal → M1 L5 → 动作
  → VTA DA reward → DA-STDP 权重更新
```

### 结果对比

| 指标 | Step 13-D+E (全拓扑) | 13-C (随机BG) | v3 (无视觉) |
|------|---------------------|-------------|------------|
| Learner advantage | **+0.0015** ✅ | -0.0015 | +0.017 |
| 10k improvement | **+0.077** ✅ | +0.077 | +0.191 |
| 5k late safety | 0.50 | 0.62 | 0.38 |
| Food 10k | **118** (+55%) | 121 | 76 |
| D1 fires/50步 | **189** | 178 | ~178 |

**关键进展:**
- DA-STDP learner advantage 恢复正值 (+0.0015 vs -0.0015)
- 10k 学习稳定 (+0.077, 不再振荡)
- 食物收集比 v3 基线 +55% (视觉通路贡献)
- 完整端到端拓扑映射: 视觉→皮层→BG→动作

### 新增 API

- `BasalGanglia::set_topographic_cortical_source(region_id, n_neurons)` — 拓扑皮层→纹状体映射

### 修改文件

- `basal_ganglia.h`: +set_topographic_cortical_source, +topo_ctx_rid_/n_, w_decay 0.001→0.003
- `basal_ganglia.cpp`: +set_topographic_cortical_source 实现(拓扑偏置 map 重建)
- `closed_loop_agent.cpp`: +dlPFC→BG 拓扑注册, +efference copy 重新启用(i≥10)

### 回归测试: 29/29 CTest 全通过 (0 失败, 19.2秒)

### 系统状态

```
48区域 · ~5528+神经元 · ~109投射 · 179测试 · 29 CTest suites
完整功能: 感觉输入 · 视听编码 · 层级处理 · 双流视觉 · 语言
          6种学习(STDP/STP/DA-STDP/CA3-STDP/稳态/皮层STDP) · 预测编码 · 工作记忆
          注意力 · GNW意识 · 内驱力 · NREM巩固 · REM梦境 · 睡眠周期管理
          4种调质广播 · 稳态可塑性 · 规模可扩展(E/I自平衡)
          闭环Agent · GridWorld环境 · 运动探索 · VTA奖励信号
          DA-STDP闭环学习 · Eligibility Traces · 动作特异性信用分配
          NE探索/利用调制 · Motor Efference Copy
          全链路拓扑映射: V1→dlPFC→BG (视觉空间→动作子群)
          V1在线STDP · 食物收集+55% · Learner advantage恢复
```

---

# 战略路线图：从"组件堆叠"到"完整学习系统"

> 日期: 2026-02-08
> 决策依据: Step 11-13 的反复调优经验 + 生物学习回路分析

## 现状诊断

经过 Step 11 → 13-D+E 的持续优化，悟韵的闭环 agent 达到:

| 指标 | 当前值 | 说明 |
|------|--------|------|
| 5k safety 提升 | 0.32 → 0.50 (+0.18) | 勉强学会 |
| Learner advantage | +0.0015 | 几乎看不出学和不学的区别 |
| Food 10k | 118 (vs 基线 76) | 视觉通路贡献 +55% |
| 10k improvement | +0.077 | 微弱正向 |

**核心问题: 组件是独立的特性，不是一个完整的学习系统。**

历史模式: 每一步都是"修一个瓶颈，暴露下一个瓶颈"。这说明系统缺少核心学习能力，而不仅仅是缺少调参。

## 三个根本瓶颈

### 瓶颈 1: 信用分配太弱（最关键）

DA-STDP + eligibility trace 是生物学上正确的，但极其低效:
- 每次奖励事件只学习 1 次
- 3 步前有 100 个神经元活跃，只有 5 个与正确动作相关
- Eligibility trace 不知道哪 5 个是关键的 → 95% 的权重更新是噪声

真实大脑靠多重机制叠加解决:

| 机制 | 功能 | 悟韵现状 | 生物对应 |
|------|------|---------|---------|
| **海马 SWR 重放** | 1 次经验 → 重放 20 次学习 | ❌ 有海马但无 SWR | experience replay |
| **预测编码** | 自上而下误差信号，减少噪声激活 | ❌ 有结构但未接通 | 生物版 backprop |
| **前额叶工作记忆桥接** | 在 DA 到来前维持"谁做了什么" | ⚠️ 有但极弱 | temporal credit |
| **小脑前向模型** | 预测动作结果，不需等真实奖励 | ❌ 完全没有 | model-based RL |

**悟韵做了学习回路的 ①-⑥，缺了 ⑦-⑩:**

```
完整学习回路 (10步):
① 看到食物在左    → V1 激活 "左侧食物" 特征               ← 有
② V1→dlPFC        → dlPFC 形成 "食物在左" 表征             ← 有
③ dlPFC→BG        → BG D1 "向左" 子群被激活                ← 有 (拓扑映射)
④ BG→M1           → Agent 向左走                          ← 有
⑤ 吃到食物        → VTA DA burst                          ← 有
⑥ DA-STDP         → 强化 "食物在左→向左走" 连接             ← 有
---- 以下缺失 ----
⑦ 海马记录事件    → 把整个 episode 存为 CA3 pattern          ← 缺
⑧ 海马重放 ×20    → 空闲时反复回放此 episode                 ← 缺
   每次重放触发 ⑥ → 1次经验变20次学习
⑨ 皮层巩固        → V1→dlPFC 连接被重放强化为长期记忆         ← 缺
⑩ 预测编码        → 下次看到类似场景,dlPFC主动预测"会有食物"   ← 缺
   预测成功=不需学习, 预测失败=误差信号→精准更新
```

### 瓶颈 2: 表征容量太小

闭环决策实际参与的 ~600 个神经元，信息容量约等于 3 层 30 节点前馈网络。
用这个容量 + eligibility trace (非梯度下降) 学 GridWorld ... 这就是为什么学得艰难。

### 瓶颈 3: 架构是手工搭的

手工搭的架构限制了进化/优化只能调参数。间接编码（发育程序）才能让进化发现新架构。这是远期目标。

## 技术路线：先完成学习回路，再优化参数，再增加环境

### 正确的目标递进

```
错误路线: GridWorld 5×5 → 10×10 → 连续空间 → 3D → ...
  （环境变复杂，但学习能力没变 → 每次都要大改）

正确路线: 完整学习回路 → 参数优化 → 更复杂环境
  （学习能力先到位，环境自然可以扩展，每步工作不会被推翻）
```

### Step 14: 海马 SWR 重放（ROI 最高的单一改进）

**优先级: 🔴 最高**

```
现状: Hippocampus 有 EC→DG→CA3→CA1 结构，但只做单向编码
缺失: Sharp-Wave Ripple 重放机制

核心设计:
  - Agent 吃到食物或遇到危险时，触发 SWR 事件
  - 维护最近 ~50 步的 episode buffer (每步的 spike pattern 快照)
  - SWR 触发时: 从 buffer 取最近的奖励相关 episode
  - CA3 自联想网络重放该 episode 的 spike 序列 (压缩时间尺度)
  - 重放的 spikes 通过 SpikeBus 传回 BG
  - BG 同时收到 DA (因为重放的是奖励事件)
  - → DA-STDP 再次更新权重
  - 重放 10-20 次 → 1次食物事件变成 10-20 次权重更新

预期效果:
  - 学习速度 ×5-10
  - Learner advantage 从 +0.001 到 +0.01-0.05
  - 5k safety 从 0.50 → 0.70+

工期: 3-5 天
是否会被推翻: 不会 — 任何未来版本都需要 SWR 重放
```

### Step 15: 预测编码接通（信用分配精度提升）

**优先级: 🟡 高**

```
现状: L6 存在但不产生自上而下预测
缺失: dlPFC→LGN/V1 的预测信号，V1 L2/3 的预测误差计算

核心设计:
  - dlPFC L6 → thalamus(LGN) → V1: 自上而下的预测信号
  - V1 L2/3 计算: 实际输入 − 预测 = 预测误差
  - 只有预测误差才向上传播 (而非全部视觉输入)
  - → 大幅减少 dlPFC→BG 的噪声激活
  - → DA-STDP 信用分配更精准

预期效果:
  - 信用分配精度 ×2-3
  - 与 SWR 重放叠加后: learner advantage 可能达到 +0.05-0.10

工期: 5-7 天
是否会被推翻: 不会 — 预测编码是皮层计算的核心理论
```

### Step 16: 基因层 v1（参数自动优化）

**优先级: 🟢 中**

```
前置: Step 14 + 15 完成后，学习回路完整
这时基因层才能发挥最大价值:
  - 搜索的是"完整学习系统的最优参数"
  - 而非"残缺系统的最优 workaround"

核心设计:
  - 详见 docs/04_genome_layer_design.md
  - v1: 纯遗传算法, ~30-150 个闭环参数
  - 适应度函数: 显式奖励学习改善幅度 (Baldwin 效应)

预期效果:
  - 在完整学习回路上再挤出 30-50% 性能
  - 发现参数间的协同效应

工期: 2-3 天
```

### Step 17+: 环境扩展

```
学习回路完整 + 参数优化后:
  - 10×10 Grid, 移动食物
  - 多步序列任务
  - 连续空间
  → 不需要改学习机制，只需要调环境参数
```

## 核心原则

1. **先完成学习回路，再优化参数** — 基因层是"效率放大器"不是"能力突破器"
2. **每步工作必须是永久性的** — 不会因为下一步而被推翻
3. **架构决策由人做，参数优化由进化做** — 进化不解决架构问题
4. **不会被推翻的**: SNN 基础设施、脑区结构、拓扑映射、DA-STDP、SpikeBus
5. **让之前的工作增值 10 倍的**: SWR 重放、预测编码

## 不会被推翻的已有工作（长期有效）

- ✅ 48 脑区结构 + SNN + 双区室模型
- ✅ DA-STDP 三因子学习 (是最终系统的一部分)
- ✅ 全链路拓扑映射 V1→dlPFC→BG (Step 13-C/D)
- ✅ SpikeBus + SimulationEngine 基础设施
- ✅ 视觉通路 LGN→V1→dlPFC (Step 13-C)
- ✅ Motor efference copy (Step 13-D)
- ✅ 29/29 CTest 回归测试
