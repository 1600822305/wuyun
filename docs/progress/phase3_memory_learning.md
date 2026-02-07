# Phase 3: 记忆 + 情感 + 学习系统

> 对应: Step 4 / 4.5 / 4.6 / 4.7 / 4.8 / 4.9 / V1修复
> 时间: 2026-02-07
> 里程碑: 9 区域 · 1591 神经元 · 13 投射 · 3 种学习同时运行 · 57 测试全通过

---

## Step 4: 海马记忆 + 杏仁核情感 ✅

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
- **33 测试全通过**

---

## Step 4.5: 整合大脑 — 9区域闭环 ✅

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
- **视觉→杏仁核**: V1=4896 → Amyg=3477 (CeA=119) ✓
- **情绪标记记忆增强**: 中性Hipp=10633, 情绪Hipp=12617 (+19%) ✓
- **杏仁核→VTA**: VTA基线=0, VTA+情绪=284 ✓
- **PFC→ITC路由**: dlPFC=4533 → ITC=1628 (SpikeBus 正确路由) ✓
- **40 测试全通过**

---

## Step 4.6: 开机学习 — 记忆/强化学习验证 ✅

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
- **45 测试全通过**

**关键里程碑:** 悟韵从"通电的硬件"变为"能学习的系统"
- 记忆不是字典查找，而是CA3自联想网络的STDP权重变化
- 回忆是模式补全（部分线索→完整重建），不是精确匹配
- 强化学习通过DA调制资格痕迹实现，不是IF-ELSE规则

---

## Step 4.7: 皮层 STDP 自组织学习 ✅

> 目标: 皮层柱启用在线可塑性，验证视觉自组织学习

**架构新增:**
- ✅ `ColumnConfig::stdp_*` 参数组 (a_plus/a_minus/tau/w_max)
- ✅ `CorticalColumn::enable_stdp()` 对 3 组 AMPA 突触启用 STDP:
  - L4→L2/3 (前馈特征学习, 最重要)
  - L2/3 recurrent (侧向吸引子)
  - L2/3→L5 (输出学习)

**皮层学习验证 (4 测试全通过):**
- **STDP 权重变化**: STDP=185 vs control=183 (权重已改变) ✓
- **训练增强**: 训练模式 A=162 > 新模式 B=156 (经验增强) ✓
- **选择性涌现**: 偏好A=31 偏好B=55 非选择=109 (86个神经元发展选择性!) ✓
- **LTD 竞争**: 500步训练后活动=119 (稳定, LTD防饱和) ✓
- **49 测试全通过**

**关键意义:** 功能差异从参数差异 + 连接差异 + **学习经验**中涌现。代码完全相同的 CorticalColumn，不同的输入数据→不同的选择性。

---

## Step 4.8: BG DA-STDP 在线强化学习 ✅

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
- **53 测试全通过**

**关键意义:** BG 不再是固定的 Go/NoGo 通道——它能从 DA 奖励信号中学习，全部通过 D1/D2 受体不对称 + DA 调制，没有任何 IF-ELSE 决策规则。

---

## Step 4.9: 端到端学习演示 ✅

> 目标: 用现有 9 区域系统证明全系统协作学习

**演示架构:**
- ✅ 9 区域全部启用学习: V1(STDP) + dlPFC(STDP) + BG(DA-STDP) + Hipp(CA3 STDP)
- ✅ 3 种输入方式: `inject_visual` (LGN) + `inject_bg_spikes` (via receive_spikes, 触发DA-STDP) + `inject_bg_cortical` (直接电流)

**端到端验证 (4 测试全通过):**
- **视觉-奖励闭环**: 训练 A=852>B=587, 测试(仅脉冲) A=779>B=336 (+132%) ✓
- **情绪通路**: V1=5039→Amyg=3429→VTA=241→Hipp=16165 (4区域同时活跃) ✓
- **三系统协同**: Amyg=2580 + VTA=271 + Hipp=9930 + D1=1405 (记忆+情绪+动作并行) ✓
- **学习选择性**: 仅靠学习权重 A=605>B=371 (+63%) (无直接电流, 纯权重驱动) ✓
- **57 测试全通过**

**关键意义:** 3 套独立的学习系统 (海马记忆/皮层自组织/BG强化学习) 能在同一个仿真中**同时运行**。**没有任何 IF-ELSE 决策规则** — 所有行为从结构+学习中涌现。

---

## 修复: V1→dlPFC→BG 信号衰减 ✅

> 问题: CorticalRegion::receive_spikes fan-out=3, current=25f → PSP稳态3.1f ≪ 阈值15f

**修复:**
- ✅ fan-out: 3固定 → 30%×L4_size (生物学皮层-皮层汇聚)
- ✅ current: 25f/40f → 35f/55f (regular/burst)
- ✅ `CorticalRegion` 存储 `psp_current_regular_/burst_/fan_out_`

**修复后全链路数据:**
```
修复前: LGN=124 → V1=23    → dlPFC=0    → BG=120  → MotorThal=0   → M1=0
修复后: LGN=124 → V1=7656  → dlPFC=4770 → BG=3408 → MotorThal=293 → M1=1120
```

---

## Phase 3 总结

| 指标 | 数值 |
|------|------|
| 区域 | 9 (Phase 2 的 7 + Hippocampus + Amygdala) |
| 神经元 | 1591 |
| 投射 | 13 |
| 测试 | 57 通过 |
| 学习规则 | 3 种同时运行 (皮层STDP · CA3快速STDP · BG DA-STDP) |
| 新增类 | Hippocampus · Amygdala |
| 关键验证 | 模式补全100% · 模式分离10% · 反转学习 · 三系统协同 |