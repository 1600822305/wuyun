# 悟韵 (WuYun) 神经元系统与连接架构设计

> 版本: 0.2.0 | 日期: 2026-02-06
> 前置文档: 01_brain_region_plan.md
> 目标: 定义统一的神经元计算原语、突触可塑性规则、脉冲编码方案和跨脑区通信协议
> 重要修正: v0.2 引入双区室神经元模型、修正底层架构层级、完善预测编码的树突基础

---

## 零、悟韵 (WuYun) 总体层级架构

人脑的完整层级从分子到全脑共 8 层，悟韵 (WuYun) 不模拟分子/离子通道层，而是从 **突触+信号原语** 开始，建立 6 层计算抽象：

```
人脑完整层级:                              悟韵 (WuYun) 抽象层级:

层级 0  分子层  离子/受体/离子通道          │  (不模拟, 吸收到参数中)
层级 1  亚细胞  树突棘/突触囊泡/轴突         │  (不模拟, 吸收到参数中)
───────────────────────────────────────┼────────────────────────
层级 2  突触+通道  突触传递/短时程动力学       │  Layer 0: Spike+Signal
层级 3  细胞层    ★ 神经元 (双区室: 胞体+树突)   │  Layer 1: Synapse+Channel
层级 4  微环路  局部回路/微柱/E-I motif        │  Layer 2: Neuron (双区室)
层级 5  柱/环路  皮层柱/海马环路/基底节通路     │  Layer 3: Circuit
层级 6  脑区层    V1/PFC/海马体/丘脑/小脑       │  Layer 4: Region
层级 7  系统层    视觉系统/记忆系统/决策系统      │  Layer 5: BrainSystem
```

```
┌─────────────────────────────────────────────────────────┐
│ Layer 5: BrainSystem       全脑系统整合                   │
│          (视觉系统/记忆系统/决策系统/...)                  │
├─────────────────────────────────────────────────────────┤
│ Layer 4: Region            脑区                          │
│          (V1/PFC/海马体/丘脑/基底节/小脑/...)             │
├─────────────────────────────────────────────────────────┤
│ Layer 3: Circuit           环路/皮层柱                    │
│          (6层柱/DG-CA3-CA1环路/Go-NoGo通路/...)          │
├─────────────────────────────────────────────────────────┤
│ Layer 2: Neuron            神经元 (★双区室模型)            │
│          (锥体/篮状/中棘/浦肯野 + 顶端树突+基底树突) │
├─────────────────────────────────────────────────────────┤
│ Layer 1: Synapse+Channel   突触 + 离子通道抽象              │
│          (AMPA/NMDA/GABA + 可塑性规则 + 短时程动力学)  │
├─────────────────────────────────────────────────────────┤
│ Layer 0: Spike+Signal      脉冲与信号原语                  │
│          (脉冲事件/突触电流/调质浓度/振荡相位)       │
└─────────────────────────────────────────────────────────┘
```

### 各层职责与人脑底层吸收关系

| 人脑底层机制 | 在 悟韵 (WuYun) 中的抽象方式 | 吸收到哪一层 |
|---|---|---|
| Na⁺/K⁺ 离子通道动力学 | → AdLIF+ 的 τ_m 和阈值参数 | Layer 2 (Neuron) |
| Ca²⁺ 内流 | → NMDA 突触电压门控 + 树突 Ca²⁺ 脉冲 | Layer 1 (Synapse) + Layer 2 |
| 突触囊泡释放概率 | → 突触权重 + 短时程可塑性(STP) | Layer 1 (Synapse) |
| 树突棘形态/非线性 | → **双区室模型**(顶端+基底树突) | Layer 2 (Neuron) ★ |
| 神经胶质细胞 | → 稳态可塑性(突触缩放) | Layer 1 |
| 基因表达/蛋白合成 | → 长期可塑性的慢速变化 | Layer 1 |

> **设计理念**: 神经元不是绝对底层，但是正确的 **“计算底层”抽象**。分子/离子通道层的效应被吸收到神经元和突触的参数中，
> 但 **树突计算** 不能被简单吸收——它是预测编码的硬件基础，必须显式建模。

---

## 一、神经元模型设计

### 1.1 设计原则

悟韵 (WuYun) 采用 **分层神经元模型策略**：

- **核心层**: 所有神经元共享统一的脉冲神经元基类，采用 **双区室模型**（顶端树突 + 胞体/基底树突）
- **特化层**: 不同脑区的神经元通过参数和子类实现功能分化
- **树突优先**: 顶端树突区室显式建模，这是预测编码的硬件基础——burst vs regular spike 的来源
- **效率平衡**: 不追求精确到离子通道，而是在 **计算效率** 和 **生物合理性** 间取最优平衡

### 1.2 基础神经元模型：自适应LIF+ (Adaptive Leaky Integrate-and-Fire Plus)

> **v0.2 重大修正**: 从单点神经元升级为双区室模型。
> 单点模型无法区分“前馈输入”和“反馈预测”，而这正是预测编码的核心。
> 双区室模型通过分离顶端树突(反馈)和基底树突(前馈)解决这个问题。

```
双区室神经元结构:

        反馈/预测输入 (来自高层 L6/L1)
              │
              ▼
        ┌──────────┐
        │  apical   │ ← 顶端树突区室 (Apical Compartment)
        │ dendrite  │    独立 Ca²⁺ 动力学, 可产生树突脉冲
        └────┬─────┘
             │  耦合系数 κ (0.0 ~ 1.0)
             ▼
        ┌──────────┐
        │   soma    │ ← 胞体区室 (Somatic Compartment)
        │  + basal  │    标准 AdLIF+ 动力学
        └────┬─────┘
             │
             ▼
        轴突输出 (spike / burst)
             ↑
        前馈输入 (来自丘脑/低层 L2/3)


=== 胞体区室 (Somatic/Basal Compartment) ===

膜电位动力学:
  τ_m · dV_s/dt = -(V_s - V_rest) + R_s · I_basal - w + κ · (V_a - V_s)

适应变量动力学:
  τ_w · dw/dt = a · (V_s - V_rest) - w

其中:
  V_s       : 胞体膜电位 (mV)
  I_basal   : 基底树突突触输入 (前馈信号: 丘脑, 低层柱 L2/3)
  κ         : 胞体-顶端树突耦合系数 (0.0-1.0, 类型依赖)
  V_a       : 顶端树突膜电位
  w         : 适应变量
  其余参数同标准 AdLIF+


=== 顶端树突区室 (Apical Compartment) ===

膜电位动力学:
  τ_a · dV_a/dt = -(V_a - V_rest) + R_a · I_apical + κ · (V_s - V_a)

Ca²⁺ 树突脉冲机制:
  当 V_a ≥ V_ca_threshold 时:
    触发树突 Ca²⁺ 脉冲: ca_spike = 1
    V_a → V_a + Ca_boost    (Ca²⁺ 反冲 → 增强胞体去极化)
    ca_spike 持续时间: ~20-50 ms (比 Na⁺ 脉冲慢得多)

其中:
  V_a             : 顶端树突膜电位 (mV)
  τ_a             : 顶端树突时间常数 (20-50 ms, 比胞体慢)
  R_a             : 顶端树突膜电阻
  I_apical        : 顶端树突突触输入 (反馈信号: 高层皮层 L6 预测)
  V_ca_threshold  : Ca²⁺ 脉冲阈值 (-30 mV, 高于 Na⁺ 脉冲阈值)
  Ca_boost        : Ca²⁺ 反冲强度 (+20 mV)


=== 发放逻辑 (核心创新: burst vs regular) ===

胞体发放条件:
  当 V_s ≥ V_threshold 时:
    spike = 1
    V_s → V_reset
    w → w + b

发放模式判定:
  IF ca_spike == 0 (apical 未激活) THEN:
    spike_type = REGULAR        # 普通脉冲: 有前馈输入但未被预测
                                 # → 传递预测误差

  IF ca_spike == 1 (apical 激活) AND V_s ≥ V_threshold THEN:
    spike_type = BURST           # 爆发发放: 前馈和反馈同时激活
                                 # → 预测匹配信号 (注意力/学习)
                                 # burst = 2-5个脉冲 @ >100Hz

  IF ca_spike == 1 AND V_s < V_threshold THEN:
    spike_type = NONE            # 只有预测没有输入 → 沉默

  IF ca_spike == 0 AND V_s < V_threshold THEN:
    spike_type = NONE            # 无任何输入 → 沉默


=== 参数总表 ===

  V_s, V_a    : 胞体/顶端树突膜电位 (mV)
  V_rest      : 静息电位 (-70 mV)
  V_threshold : Na⁺ 发放阈值 (-50 mV, 可动态调整)
  V_ca_threshold: Ca²⁺ 树突脉冲阈值 (-30 mV)
  V_reset     : 重置电位 (-65 mV)
  τ_m        : 胞体时间常数 (10-30 ms)
  τ_a        : 顶端树突时间常数 (20-50 ms)
  τ_w        : 适应时间常数 (50-500 ms)
  R_s, R_a    : 胞体/顶端膜电阻
  κ           : 区室耦合系数 (0.0-1.0)
  a           : 亚阈值适应耦合
  b           : 脉冲后适应增量
  w           : 适应变量
  I_basal     : 基底树突突触输入 (前馈)
  I_apical    : 顶端树突突触输入 (反馈)


#### 发放模式与预测编码的对应关系 (★核心创新)

| 基底树突(前馈) | 顶端树突(反馈) | 发放模式 | 含义 | 信号流向 |
|---|---|---|---|---|
| ✔ 激活 | ✖ 未激活 | **Regular Spike** | 有输入但未被预测 = **预测误差** | ↑ 向高层传递 |
| ✔ 激活 | ✔ 激活 | **Burst** | 输入与预测匹配 = **确认信号** | ↓ 驱动学习+注意力 |
| ✖ 未激活 | ✔ 激活 | **沉默** | 只有预测没有输入 = **无事发生** | — |
| ✖ 未激活 | ✖ 未激活 | **沉默** | 无任何活动 | — |

> **这为什么重要?**
> 在单点神经元模型中，无法区分“输入产生的脉冲”和“输入+预测匹配产生的脉冲”。
> 双区室模型通过 burst vs regular spike 自然编码了这个区分。
> 这是预测编码的 **硬件基础**。

#### 参数集 → 不同发放模式

| 发放模式 | a | b | τ_w | κ | 对应神经元类型 | 用途 |
|---------|---|---|-----|---|--------------|------|
| Regular Spiking (RS) | 低 | 中 | 长 | 0.3 | 皮层锥体细胞 | 最常见的兴奋性神经元 |
| Fast Spiking (FS) | 0 | 0 | - | 0 | 篮状细胞(抑制性) | 快速精确抑制 (无树突区室) |
| Intrinsic Bursting (IB) | 低 | 高 | 短 | 0.6 | L5 大锥体细胞 | 注意力标记/驱动输出 (强耦合) |
| Chattering (CH) | 中 | 中 | 中 | 0.3 | L2/3 锥体细胞 | 快速节律性发放 |
| Low-Threshold Spiking (LTS) | 高 | 低 | 长 | 0 | Martinotti 细胞 | 树突靶向抑制 (无树突区室) |
| Tonic Firing | 0 | 0 | - | 0.2 | 丘脑中继神经元 | 稳定中继模式 |
| Burst/Tonic 切换 | 高 | 高 | 短 | 0.5 | 丘脑中继(静息态) | 丘脑门控状态切换 |
| Tonically Active | 低 | 低 | 长 | 0 | 纹状体胆碱能中间神经元 | 持续监控信号 |

> 注意: κ=0 表示无顶端树突区室(退化为单点模型，用于抑制性中间神经元等)。
> κ 值越高，顶端树突对胞体的影响越大，越容易产生 burst。
> L5 锥体细胞的 κ 最高 (0.6)，因为它们有最长的顶端树突，负责驱动输出。

### 1.3 神经元类型体系

```
NeuronBase (双区室 AdLIF+, 含 apical + somatic/basal 区室)
│
├── ExcitatoryNeuron (兴奋性, 双区室 ★)
│   ├── PyramidalCell          # 锥体细胞 - 皮层主力 [双区室: κ>0]
│   │   ├── L23Pyramidal       # 浅层锥体 - 跨柱通信     κ=0.3, 产生 regular/burst
│   │   ├── L5Pyramidal        # 深层锥体 - 驱动输出     κ=0.6, 最强 burst 能力
│   │   └── L6Pyramidal        # 多形层 - 丘脑反馈       κ=0.2, 预测信号生成
│   ├── StellateCell           # 星形细胞 - L4 输入处理   κ=0.1, 弱顶端耦合
│   ├── GranuleCell            # 颗粒细胞 - 海马DG/小脑   κ=0, 单区室(无顶端)
│   └── MossyCell              # 苔藓细胞 - 海马DG        κ=0, 单区室
│
├── InhibitoryNeuron (抑制性, 单区室 — κ=0)
│   ├── BasketCell             # 篮状细胞 - 胞体靶向抑制(PV+)   [FS模式]
│   ├── MartinottiCell         # Martinotti - 树突靶向抑制(SST+) [LTS模式]
│   │                          # SST+ 抑制锥体细胞的顶端树突 → 调控 burst!
│   ├── ChandlierCell          # 枝形烛台 - 轴突起始段抑制(PV+)
│   ├── VIPInterneuron         # VIP中间神经元 - 去抑制(抑制SST→释放burst)
│   └── NGFCell                # 神经胶质形态 - 慢抑制(GABA_B)
│
├── ModulatoryNeuron (调制性, 单区室 — κ=0)
│   ├── DopamineNeuron         # VTA/SNc 多巴胺神经元
│   ├── SerotoninNeuron        # 中缝核 5-HT 神经元
│   ├── NorepinephrineNeuron   # 蓝斑核 NE 神经元
│   └── CholinergicNeuron      # 基底前脑 ACh 神经元
│
└── SpecializedNeuron (特化型, κ 值各异)
    ├── ThalamicRelay          # 丘脑中继 - Tonic/Burst切换  κ=0.2~0.5(状态依赖)
    ├── TRNNeuron              # 丘脑网状核 - 纯抑制门控     κ=0
    ├── MediumSpinyNeuron      # 纹状体中棘神经元(D1/D2)    κ=0
    ├── PurkinjeCell           # 小脑浦肯野细胞              κ=0(特殊树突结构)
    ├── PlaceCell              # 海马位置细胞                κ=0.3
    ├── GridCell               # 内嗅皮层网格细胞            κ=0.2
    └── HeadDirectionCell      # 头朝向细胞                  κ=0.1
```

> **关键拓扑发现: 抑制性中间神经元如何调控 burst**
> - **SST+ (Martinotti)** 细胞靶向锥体细胞的 **顶端树突** → 抑制 Ca²⁺ 脉冲 → 阻止 burst
> - **VIP** 细胞抑制 SST+ → 去抑制顶端树突 → **释放 burst** (注意力门控!)
> - **PV+ (Basket)** 细胞靶向锥体细胞的 **胞体** → 直接抑制发放
> - 这形成了一个精妙的注意力环路: PFC → VIP → 抑制SST → 释放burst → 注意力聚焦

---

## 二、突触模型设计

### 2.1 突触类型体系

```
SynapseBase
├── ChemicalSynapse (化学突触 - 主要类型)
│   ├── GlutamateSynapse (谷氨酸 - 兴奋性)
│   │   ├── AMPA_Synapse       # 快速兴奋 (τ ~2ms)
│   │   ├── NMDA_Synapse       # 慢速兴奋 + 可塑性门控 (τ ~50-150ms)
│   │   └── Kainate_Synapse    # 调制性兴奋
│   │
│   ├── GABASynapse (GABA - 抑制性)
│   │   ├── GABA_A_Synapse     # 快速抑制 (τ ~5-10ms)
│   │   └── GABA_B_Synapse     # 慢速抑制 (τ ~100-300ms)
│   │
│   └── ModulatorySynapse (调制性)
│       ├── DopamineSynapse    # D1(兴奋) / D2(抑制)
│       ├── SerotoninSynapse   # 多种受体亚型
│       ├── AcetylcholineSynapse # M(慢调制) / N(快兴奋)
│       └── NorepinephrineSynapse # α/β 受体
│
└── ElectricalSynapse (电突触/缝隙连接)
    └── GapJunction            # 快速同步 (抑制性中间神经元间常见)
```

### 2.2 突触电流模型

```
AMPA 突触电流:
  I_AMPA = g_AMPA · s_AMPA · (V - E_exc)
  ds_AMPA/dt = -s_AMPA / τ_AMPA + Σ δ(t - t_spike)
  参数: τ_AMPA = 2 ms, E_exc = 0 mV

NMDA 突触电流 (电压依赖性 Mg²⁺ 阻断):
  I_NMDA = g_NMDA · s_NMDA · B(V) · (V - E_exc)
  B(V) = 1 / (1 + [Mg²⁺]/3.57 · exp(-0.062·V))
  ds_NMDA/dt = -s_NMDA / τ_NMDA + α · x · (1 - s_NMDA)
  dx/dt = -x / τ_rise + Σ δ(t - t_spike)
  参数: τ_NMDA = 100 ms, τ_rise = 2 ms

GABA_A 突触电流:
  I_GABA_A = g_GABA_A · s_GABA · (V - E_inh)
  参数: τ_GABA_A = 6 ms, E_inh = -75 mV

GABA_B 突触电流:
  I_GABA_B = g_GABA_B · [G^n / (G^n + K_d)] · (V - E_K)
  参数: τ_GABA_B = 200 ms, E_K = -95 mV (更强的超极化)
```

### 2.3 突触传导延迟模型

```
总延迟 = 轴突传导延迟 + 突触延迟

轴突传导延迟:
  delay_axon = distance / conduction_velocity
  - 有髓鞘: velocity = 10-100 m/s → 短距离 ~0.1-1 ms
  - 无髓鞘: velocity = 0.5-2 m/s  → 长距离可达 ~10-50 ms

突触延迟 (化学突触):
  delay_synapse = 0.5-2 ms (囊泡释放 + 扩散)

悟韵 (WuYun) 简化方案:
  - 柱内连接: 1 时间步延迟 (代表 ~1 ms)
  - 相邻柱间: 1-2 时间步
  - 跨区域(皮层-皮层): 2-5 时间步
  - 皮层-皮层下: 1-3 时间步
  - 神经调质效应: 10-50 时间步 (慢速体积传递)
```

---

## 三、突触可塑性规则

### 3.1 可塑性规则体系

```
PlasticityRule
├── HebbianRules (赫布学习系列)
│   ├── STDP                   # 脉冲时序依赖可塑性 (皮层主要规则)
│   ├── VoltageDependentSTDP   # 电压依赖STDP (更精确的三因子)
│   ├── BCM                    # BCM理论 (动态阈值, 防止饱和)
│   └── Oja                    # Oja规则 (自归一化赫布)
│
├── ReinforcementRules (强化学习系列)
│   ├── DA_ModulatedSTDP       # 多巴胺调制STDP (三因子规则)
│   ├── RewardModulatedSTDP    # 奖励调制STDP
│   └── TD_Learning            # 时序差分学习 (基底神经节)
│
├── HomeostaticRules (稳态可塑性)
│   ├── SynapticScaling        # 突触缩放 (保持总输入稳定)
│   ├── IntrinsicPlasticity    # 内在可塑性 (调整兴奋性)
│   └── MetaPlasticity         # 元可塑性 (可塑性的可塑性)
│
└── StructuralPlasticity (结构可塑性)
    ├── SynapseFormation       # 突触新生
    ├── SynapsePruning         # 突触修剪
    └── AxonalSprouting        # 轴突出芽
```

### 3.2 核心 STDP 规则

```
经典 STDP (兴奋性突触):

  Δw = {  A+ · exp(-Δt / τ+)   当 Δt > 0 (突触前先于突触后 → LTP)
       { -A- · exp(+Δt / τ-)   当 Δt < 0 (突触后先于突触前 → LTD)

  参数:
    A+ = 0.005 (LTP 幅度)
    A- = 0.00525 (LTD 幅度, 略大于A+以维持稀疏性)
    τ+ = 20 ms (LTP 时间窗口)
    τ- = 20 ms (LTD 时间窗口)

三因子 STDP (多巴胺调制):

  Δw = DA_signal · eligibility_trace

  eligibility_trace 更新:
    de/dt = -e / τ_e + STDP(Δt)

  解释:
    - STDP 产生的权重变化不立即生效
    - 而是存储在"资格痕迹"(eligibility trace)中
    - 当多巴胺信号到达时,资格痕迹才转化为实际权重变化
    - 这解决了信用分配问题(奖励延迟)
    - τ_e ≈ 1000 ms (资格痕迹衰减时间)

抑制性 STDP (对称型):

  Δw = { A · exp(-|Δt| / τ)   当 |Δt| < τ_window (相关 → 增强抑制)
       { -B                    当 |Δt| ≥ τ_window (不相关 → 减弱抑制)

  效果: 维持兴奋-抑制平衡 (E/I balance)
```

### 3.3 各脑区可塑性规则分配

| 脑区 | 主要可塑性规则 | 辅助规则 | 特殊说明 |
|------|--------------|---------|---------|
| 感觉皮层 (S-*) | STDP | SynapticScaling, BCM | 经典赫布学习, 自组织特征映射 |
| 联合皮层 (A-*) | STDP + VoltageDep | MetaPlasticity | 较高可塑性阈值, 需要更强证据 |
| PFC (A-01/03) | DA_Modulated_STDP | SynapticScaling | 多巴胺调制的三因子学习 |
| 运动皮层 (M-*) | RewardModulated_STDP | IntrinsicPlasticity | 强化学习驱动 |
| 纹状体 (BG-01/02) | DA_Modulated_STDP | - | D1:DA增强LTP; D2:DA增强LTD |
| 海马 DG (H-02) | STDP (高阈值) | StructuralPlasticity | 成年神经发生 + 极高稀疏度 |
| 海马 CA3 (H-03) | STDP (标准) | - | 快速单次学习(one-shot) |
| 海马 CA1 (H-04) | STDP + ACh调制 | - | 乙酰胆碱开启学习模式 |
| 小脑 浦肯野 (CB-02) | 监督LTD | - | 攀缘纤维信号 → 平行纤维突触LTD |
| 小脑 DCN (CB-03) | LTP (补偿性) | - | 与浦肯野互补 |

---

## 四、脉冲编码方案

### 4.1 编码方式

悟韵 (WuYun) 同时支持多种生物真实的脉冲编码方式：

```
编码方式                   用途                          信息载体

1. 频率编码 (Rate Coding)
   发放率 ∝ 刺激强度         感觉强度编码                  平均发放率
   示例: 30 Hz = 中等刺激

2. 时序编码 (Temporal Coding)
   首脉冲延迟 ∝ 1/刺激强度   快速识别(前馈)               脉冲精确时间
   示例: 强刺激 → 5ms, 弱刺激 → 50ms

3. 群体编码 (Population Coding)
   N个神经元的联合活动       位置/方向/运动等连续量编码      群体活动向量
   示例: 位置细胞群体 → 位置估计

4. 相位编码 (Phase Coding)
   脉冲相对于LFP振荡的相位   序列位置编码(海马体)           相对相位
   示例: θ相位进动 → 序列位置

5. Burst编码 (Burst Coding)
   单脉冲 vs 爆发发放         注意力标记/可靠传输            脉冲模式
   示例: burst = 高显著性/注意信号

6. 沉默编码 (Silence Coding)
   持续活跃中的暂停           预测编码中的"无意外"信号        发放暂停
   示例: 预测准确 → 暂停发放
```

### 4.2 各脑区优选编码方式

| 脑区 | 主要编码 | 次要编码 | 说明 |
|------|---------|---------|------|
| 初级感觉皮层 | 频率 + 时序 | 群体 | 强度用频率, 速度用首脉冲 |
| 联合皮层 | 群体 + 时序 | 频率 | 高维分布式表示 |
| 海马体 | 相位 + 群体 | - | θ相位进动编码序列 |
| 前额叶 | 持续发放(频率) | burst | 工作记忆=持续活动 |
| 基底神经节 | 频率(抑制性) | 暂停 | 持续抑制+选择性暂停=去抑制 |
| 小脑 | 频率 + burst | - | 浦肯野=高频抑制, 攀缘=burst教学 |
| 丘脑 | tonic/burst切换 | 频率 | tonic=忠实传输, burst=唤醒信号 |
| 神经调质核团 | tonic(背景) + phasic(事件) | - | tonic=基线水平, phasic=事件信号 |

---

## 五、跨脑区通信协议

### 5.1 信号总线架构

悟韵 (WuYun) 定义三类通信总线：

```
┌─────────────────────────────────────────────────────────────┐
│                   悟韵 (WuYun) 通信架构                        │
│                                                             │
│  1. SpikeBus (脉冲总线) ─── 快速、精确、点对点                │
│     用途: 神经元间直接脉冲传递                                │
│     延迟: 1-5 时间步                                        │
│     带宽: 每连接 1 bit/时间步 (脉冲=1, 无脉冲=0)            │
│     拓扑: 稀疏连接矩阵 (邻接表/CSR格式)                     │
│                                                             │
│  2. FieldBus (场总线) ─── 中速、局域、群体广播                │
│     用途: 局部场电位(LFP)振荡同步                            │
│     延迟: 实时 (同一柱/区域内)                               │
│     信息: 振荡相位 (θ/γ/β/α 频段)                           │
│     拓扑: 区域内广播                                         │
│                                                             │
│  3. ModulationBus (调制总线) ─── 慢速、全局、弥散广播         │
│     用途: 神经调质(DA/NE/5-HT/ACh)水平传递                  │
│     延迟: 10-50 时间步 (体积传递)                            │
│     信息: 连续浓度值 [0, 1]                                 │
│     拓扑: 一对多弥散投射                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 消息格式

```
SpikeBus 消息:
┌──────────────┬───────────┬──────────┬──────────┐
│ source_id    │ target_id │ timestamp│ spike_type│
│ (32-bit)     │ (32-bit)  │ (32-bit) │ (8-bit)  │
└──────────────┴───────────┴──────────┴──────────┘
spike_type: 0=regular, 1=burst_start, 2=burst_continue, 3=burst_end

FieldBus 消息:
┌──────────────┬───────────┬──────────┬──────────┬──────────┬───────────┬───────────┐
│ region_id    │ timestamp │ delta    │ theta    │ alpha    │ beta      │ gamma     │
│ (16-bit)     │ (32-bit)  │ (float)  │ (float)  │ (float)  │ (float)   │ (float)   │
└──────────────┴───────────┴──────────┴──────────┴──────────┴───────────┴───────────┘

ModulationBus 消息:
┌──────────────┬───────────┬──────┬──────┬──────┬──────┐
│ source_nucleus│ timestamp│  DA  │  NE  │ 5-HT │ ACh  │
│ (16-bit)     │ (32-bit)  │(f32) │(f32) │(f32) │(f32) │
└──────────────┴───────────┴──────┴──────┴──────┴──────┘
```

### 5.3 路由机制

> **⚠️ 反作弊警告 (参见 00_design_principles.md)**
> 丘脑路由 **不能** 用 IF-ELSE 规则表实现。路由偏好必须来自学习的突触权重。
> 下文描述的是路由的 **物理机制**（硬件），不是路由的 **具体决策**（软件/学习）。

```
丘脑路由的物理机制 (可硬编码 — 解剖结构):

1. 连接拓扑 (先天布线):
   - 感觉器官 → 特定丘脑核团 的轴突投射是解剖学事实 (可硬编码)
     视网膜→LGN, 耳蜗核→MGN, 脊髓→VPL (基因决定的粗粒度布线)
   - 丘脑核团 → 皮层区域 的初始投射概率也是先天的 (可硬编码为初始连接概率)
     LGN→V1 概率高, LGN→A1 概率≈0

2. 门控机制 (先天硬件):
   - TRN (网状核) 对所有丘脑核团施加竞争抑制 (解剖结构, 可硬编码)
   - 皮层→丘脑的反馈投射 (~10:1 比例) 调制门控 (连接拓扑可硬编码)
   - 最终门控状态 = TRN抑制 + 皮层反馈调制 + 调质水平 的综合

3. 信号送达目标层的规则 (先天布线):
   - 丘脑→皮层 前馈投射主要终止于 L4 (解剖事实, 可硬编码)
   - 皮层→皮层 反馈投射主要终止于 L1 (解剖事实, 可硬编码)
   - 皮层→皮层 前馈投射终止于 L4 (解剖事实, 可硬编码)


丘脑路由的学习部分 (禁止硬编码 — 必须学习):

1. 路由偏好/权重:
   - 哪些信号被放行、哪些被过滤 → 由皮层→丘脑反馈突触的权重决定
   - 这些权重通过 STDP 学习形成 (不是路由表!)
   - 初始状态: 近似随机的稀疏路由 → 学习后: 功能特异性路由

2. 注意力选择:
   - TRN 的抑制模式不是固定的 → 由 PFC→TRN 的学习权重动态调整
   - 注意力聚焦 = TRN 选择性抑制无关核团 (竞争结果, 非规则)

3. 新奇检测:
   - 不是 IF novelty > threshold 的规则
   - 而是预测编码产生的 regular spike 频率 → 高 regular spike = 高意外
   - 这自然通过 spike_bus 传播, 海马体的 EC 接收到高误差信号时自动增强编码

4. 威胁/情感快通道:
   - 杏仁核直接接收丘脑的粗糙感觉信号 (解剖连接可硬编码)
   - 但什么是"威胁"不能硬编码 → 杏仁核BLA的价值学习决定什么触发应激
```

### 5.4 振荡同步协议

```
神经振荡频段及其通信功能:

┌──────────┬───────────┬──────────────────────────────────────┐
│ 频段      │ 频率范围   │ 通信功能                              │
├──────────┼───────────┼──────────────────────────────────────┤
│ δ (Delta)│ 0.5-4 Hz  │ 深睡眠; 记忆巩固(海马→皮层)           │
│ θ (Theta)│ 4-8 Hz    │ 导航; 记忆编码; 海马序列压缩           │
│ α (Alpha)│ 8-13 Hz   │ 空闲抑制; 注意力选择(未被选中区域)     │
│ β (Beta) │ 13-30 Hz  │ 运动规划; 状态维持; 长距离反馈通信     │
│ γ (Gamma)│ 30-100 Hz │ 特征绑定; 局部处理; 注意力增强         │
└──────────┴───────────┴──────────────────────────────────────┘

跨频段耦合 (Cross-Frequency Coupling):
  - θ-γ 耦合: γ burst 嵌套在 θ 周期内 → 工作记忆中的多项目编码
    每个 θ 周期(~125-250ms, 典型 ~150ms 即 6-7Hz) 内嵌套 ~4-7 个 γ 周期(~25ms)
    → 这是工作记忆容量约 4-7 项的神经基础假说 (Lisman & Jensen 2013; Cowan 2001 建议核心容量约 4±1)

  - α-γ 耦合: α 抑制非注意区域的 γ 活动 → 注意力选择机制

悟韵 (WuYun) 实现:
  - 每个 Region 维护一个 OscillationState
  - 振荡状态影响该区域内神经元的发放概率
  - 只有在"兴奋相位窗口"内的脉冲才能有效传递
  - 这自然实现了通信通过相干性(Communication Through Coherence)
```

---

## 六、预测编码通信框架

### 6.1 核心思想 (v0.2 修正: 基于双区室树突计算)

预测编码是 悟韵 (WuYun) 的**核心计算范式**，替代 Transformer 的前馈+注意力机制。

> **v0.2 关键修正**: 预测编码不是抽象的“比较器”，而是有确切的硬件基础——
> 双区室神经元的 **顶端树突** 接收自上而下预测，**基底树突** 接收自下而上输入。
> 两者是否同时激活决定了 burst vs regular spike，这就是预测编码的物理实现。

```
传统前馈网络:    输入 → 层1 → 层2 → ... → 输出 (完整信号逐层传递)
Transformer:    输入 → 编码器(自注意力) → 解码器 → 输出
悟韵 (WuYun):     每层同时做预测和误差传递 (双向流动, 基于双区室神经元)

高层 (抽象/预测)
  │ ↓ prediction → 低层锥体细胞的 apical dendrite (顶端树突)
  │
  │      ┌────────────────────────────────────┐
  │      │ 双区室锥体细胞 = “生物比较器”    │
  │      │                                    │
  │      │  apical (feedback) ──┐              │
  │      │                      ├─→ 同时激活?  │
  │      │  basal (feedforward) ┘              │
  │      │                                    │
  │      │  YES → BURST (预测匹配✓)          │
  │      │  NO  → REGULAR (预测误差✗)        │
  │      └────────────────────────────────────┘
  │
  │ ↑ regular spike = prediction_error → 向高层传递
  │ ↑ burst = match_signal → 驱动学习/注意力
低层 (具体/感觉)
     │ ↑ feedforward → 锥体细胞的 basal dendrite (基底树突)

关键: 只有预测误差 (regular spike) 向上传递
     → 大幅减少计算量 (预测准确时只有 burst, 不产生向上误差)
     → burst 信号驱动下游学习和注意力分配
     → 自然实现“注意力” (意外事件 = regular spike 飞涌 = 高注意)
```

### 6.2 层间预测编码接口 (v0.2: 基于双区室树突路由)

```
每个皮层柱对外暴露的标准接口:

class CorticalColumnInterface:

    # ===== 前馈通道 (自下而上) → 进入锥体细胞的 basal dendrite =====
    def receive_feedforward(prediction_error):
        """接收来自低层的预测误差信号
        → 进入 L4 星形细胞
        → L4 输出到 L2/3 锥体细胞的 basal dendrite"""
        pass

    # ===== 反馈通道 (自上而下) → 进入锥体细胞的 apical dendrite =====
    def receive_feedback(prediction):
        """接收来自高层的预测信号
        → 进入 L1 (顶端树突层)
        → 直接连接到 L2/3 和 L5 锥体细胞的 apical dendrite
        → 影响是否产生 Ca²⁺ spike → 决定 burst vs regular"""
        pass

    # ===== 侧向通道 =====
    def receive_lateral(context):
        """接收来自同层其他柱的上下文信号 → 进入 L2/3 basal"""
        pass

    # ===== 输出 (基于双区室 burst/regular 编码) =====
    def emit_prediction_error():
        """L2/3 regular spikes: 预测误差
        basal激活 + apical未激活 → regular spike
        → 上一层柱的 L4 (basal 输入)"""
        return self.L23.regular_spikes

    def emit_match_signal():
        """L2/3 burst spikes: 预测匹配确认
        basal激活 + apical激活 → burst
        → 驱动学习 + 增强注意力"""
        return self.L23.burst_spikes

    def emit_prediction():
        """L6 输出: 预测信号 → 低层柱的 L1 (apical) / 丘脑"""
        return self.L6.output

    def emit_drive():
        """L5 burst 输出: 驱动信号 (只有 burst 才驱动下游)
        → 基底神经节/脑干/高层(驱动性前馈)
        L5 的 κ=0.6 (最强耦合) 使其最容易产生 burst"""
        return self.L5.burst_output

    # ===== 调制接口 =====
    def receive_modulation(neuromodulator_levels):
        """接收神经调质水平 → 调节内部增益/可塑性"""
        self.gain = f(neuromodulator_levels.NE)
        self.learning_rate = f(neuromodulator_levels.ACh)
        self.discount_factor = f(neuromodulator_levels.serotonin)

    # ===== 注意力门控 (新增) =====
    def receive_attention_gate(vip_signal):
        """接收 VIP 中间神经元信号 → 控制 SST 抑制 → 门控 burst
        VIP 激活 → 抑制 SST → 释放 apical dendrite → 允许 burst
        VIP 沉默 → SST 活跃 → 抑制 apical → 只能 regular spike"""
        self.sst_inhibition = 1.0 - vip_signal
```

### 6.3 预测编码计算流程 (v0.2: 基于双区室机制)

```
每个时间步, 每个皮层柱内部执行:

STEP 1: 接收输入 (分流到不同树突区室)
  L4.input ← 丘脑前馈 OR 低层柱的 regular_spikes (prediction_error)
  L1.input ← 高层柱的 L6 输出 (prediction)

  树突路由:
    L4 → L2/3 锥体细胞的 basal dendrite   (前馈路径 → I_basal)
    L1 → L2/3 锥体细胞的 apical dendrite  (反馈路径 → I_apical)

STEP 2: 双区室并行计算
  胞体/基底区室: 更新 V_s (basal 突触输入 + 顶端耦合电流)
  顶端树突区室: 更新 V_a (apical 突触输入 + 胞体耦合电流)

STEP 3: Ca²⁺ 树突脉冲检测
  IF V_a ≥ V_ca_threshold THEN:
    ca_spike = 1  (顶端树突被反馈预测充分激活)
    V_a += Ca_boost (反冲增强胞体去极化)

STEP 4: 胞体发放判定 + 类型分类
  IF V_s ≥ V_threshold THEN:
    spike = 1
    IF ca_spike == 1:
      spike_type = BURST           → 前馈+反馈同时激活 = 预测匹配!
    ELSE:
      spike_type = REGULAR         → 只有前馈 = 预测误差!

STEP 5: 输出分发 (基于 spike_type)
  IF spike_type == REGULAR:
    L2/3 → 向高层柱的 L4 (basal) 传递预测误差
    (意外事件! 要求高层更新其内部模型)

  IF spike_type == BURST:
    L5 burst → 基底神经节/脑干 (驱动行为输出)
    L5 burst → 高层柱的 L4 (驱动性前馈, 与 L2/3 regular 不同)
    burst 还增强局部 STDP 可塑性 (确认的连接被强化)

  IF spike_type == NONE:
    沉默 (无事发生, 无需通信 → 节能!)

STEP 6: 生成下一步预测
  L6 ← 整合 L4 + L5 + 高层反馈
  L6 → 丘脑 (feedback) + 低层柱 L1 (→ apical dendrite)
  (更新预测, 使下一时间步的 regular spike 减少)

STEP 7: 注意力门控 (通过抑制性中间神经元)
  SST+(抑制顶端树突): 减弱 apical → 阻止 burst → 只允许 regular
  VIP(抑制SST): 释放 apical → 允许 burst → 注意力聚焦
  PV+(抑制胞体): 直接抑制发放 → 竞争性选择

STEP 8: 可塑性 (三因子规则)
  可塑性调制:
    burst spike → 强化局部 STDP (确认的连接被巩固)
    regular spike → 触发资格痕迹 (等待 DA 信号确认)
    ACh HIGH → 增强自下而上学习 (basal 主导)
    ACh LOW  → 增强自上而下预测 (apical 主导)
```

---

## 七、全局时钟与仿真参数

### 7.1 时间系统

```
基础时间步: dt = 1 ms (毫秒精度, 足够捕获STDP时序)

仿真时钟层级:
  - 脉冲时钟: 1 ms    (SpikeBus 更新)
  - 振荡时钟: 10 ms   (FieldBus 更新, γ周期~25ms)
  - 调制时钟: 100 ms  (ModulationBus 更新)
  - 巩固时钟: 10000 ms(记忆巩固检查, 模拟"mini-sleep")

实时比:
  - 目标: 1秒仿真时间 ≤ 10秒实际计算时间 (初版)
  - GPU加速后: 争取接近实时
```

### 7.2 仿真循环

```python
def simulation_loop():
    t = 0
    while running:
        # === 每 1ms 时间步 ===

        # 1. 收集所有脉冲 (区分 regular / burst)
        spikes = spike_bus.collect()  # {neuron_id: spike_type}

        # 2. 更新突触电流 (分流到不同树突区室)
        for synapse in active_synapses:
            if synapse.target_compartment == BASAL:
                synapse.update_current(spikes, dt)  # → I_basal
            elif synapse.target_compartment == APICAL:
                synapse.update_current(spikes, dt)  # → I_apical
            else:
                synapse.update_current(spikes, dt)  # → I_soma

        # 3. 双区室神经元更新 (v0.2 核心修正)
        for neuron in all_neurons:
            # 3a. 更新顶端树突区室
            if neuron.kappa > 0:  # 有双区室
                neuron.update_apical(dt)       # V_a 动力学
                neuron.check_ca_spike()        # Ca²⁺ 树突脉冲检测

            # 3b. 更新胞体区室
            neuron.update_soma(dt)             # V_s 动力学 (+顶端耦合)

            # 3c. 发放判定 + 类型分类
            if neuron.should_fire():
                if neuron.ca_spike:
                    spike_bus.emit(neuron.id, t, BURST)
                else:
                    spike_bus.emit(neuron.id, t, REGULAR)

        # 4. 每 10ms: 更新振荡状态
        if t % 10 == 0:
            field_bus.update_oscillations()

        # 5. 每 100ms: 更新神经调质
        if t % 100 == 0:
            modulation_bus.update_neuromodulators()
            for region in all_regions:
                region.apply_modulation(modulation_bus.levels)

        # 6. 每 10s: 记忆巩固
        if t % 10000 == 0:
            hippocampus.consolidation_sweep()

        # 7. 应用可塑性 (三因子: spike_type + STDP + 调质)
        for synapse in plastic_synapses:
            spike_type = spikes.get(synapse.post_id, NONE)
            synapse.apply_plasticity(
                spikes, spike_type, modulation_bus.levels
            )
            # burst → 立即强化; regular → 资格痕迹; none → 衰减

        t += 1
```

---

## 八、数据结构规格

### 8.1 核心数据结构

```
Neuron (v0.2: 双区室模型):
  id:          uint32          # 全局唯一神经元ID
  type:        NeuronType      # 类型枚举
  region_id:   uint16          # 所属脑区
  column_id:   uint16          # 所属柱
  layer:       uint8           # 所属层 (1-6)

  # === 胞体区室 (Somatic Compartment) ===
  V_s:         float32         # 胞体膜电位
  w:           float32         # 适应变量
  threshold:   float32         # Na⁺ 发放阈值 (可动态调整)
  refractory:  uint8           # 不应期倒计时

  # === 顶端树突区室 (Apical Compartment, 可选) ===
  V_a:         float32         # 顶端树突膜电位 (κ=0时不使用)
  ca_spike:    bool            # 当前 Ca²⁺ 树突脉冲状态
  ca_timer:    uint8           # Ca²⁺ 脉冲持续时间倒计时 (20-50ms)
  kappa:       float32         # 区室耦合系数 κ (0.0=单区室, >0=双区室)

  # === 参数包 ===
  params:      NeuronParams    # 类型特异参数 (τ_m, τ_a, a, b, R_s, R_a 等)

Synapse (v0.2: 新增目标区室字段):
  pre_id:      uint32          # 突触前神经元
  post_id:     uint32          # 突触后神经元
  target_comp: CompartmentType # 目标区室: BASAL / APICAL / SOMA
  weight:      float32         # 突触权重
  delay:       uint8           # 传导延迟 (时间步)
  type:        SynapseType     # AMPA/NMDA/GABA_A/GABA_B/...
  plasticity:  PlasticityType  # 可塑性规则
  eligibility: float32         # 资格痕迹 (三因子学习用)
  s:           float32         # 突触门控变量

CompartmentType:
  SOMA   = 0                   # 胞体 (直接影响 V_s)
  BASAL  = 1                   # 基底树突 (前馈输入 → V_s)
  APICAL = 2                   # 顶端树突 (反馈输入 → V_a)

Region:
  id:          uint16          # 脑区ID
  type:        RegionType      # 脑区类型
  columns:     Column[]        # 包含的柱
  oscillation: OscState        # 振荡状态
  modulation:  ModLevels       # 当前神经调质水平

Column:
  id:          uint16          # 柱ID
  region_id:   uint16          # 所属脑区
  layers:      Layer[6]        # 6层结构
  prediction:  float32[]       # 当前预测向量
  pred_error:  float32[]       # 当前预测误差向量

SpikeTrain:
  neuron_id:   uint32          # 发放神经元
  timestamps:  uint32[]        # 发放时间序列 (用于STDP)
```

---

## 九、模块间依赖关系与开发顺序

```
开发依赖图:

[NeuronBase + SynapseBase]  ← 最底层, 无依赖
         │
         ▼
[SpikeBus + FieldBus + ModulationBus]  ← 依赖 Neuron/Synapse
         │
         ▼
[CorticalColumn (6层预测编码)]  ← 依赖 Neuron + SpikeBus
         │
    ┌────┴─────┐
    ▼          ▼
[ThalamicRouter]  [NeuromodulatorSystem]  ← 依赖 Column + Bus
    │                    │
    ▼                    ▼
[MultiColumn System]  ← 依赖 Router + Modulator
    │
    ├──→ [HippocampalMemory]    ← 依赖 MultiColumn
    ├──→ [BasalGangliaEngine]   ← 依赖 MultiColumn + Modulator
    ├──→ [CerebellumPredictor]  ← 依赖 MultiColumn
    └──→ [AmygdalaValence]      ← 依赖 MultiColumn
              │
              ▼
    [PrefrontalExecutive]  ← 依赖以上所有
              │
              ▼
    [悟韵 (WuYun) 全系统整合]
```

---

## 十、与主流架构的本质差异总结

| 维度 | Transformer | 悟韵 (WuYun) |
|------|------------|------------|
| **信息流** | 单向前馈 + 残差 | 双向预测编码 (预测↓ 误差↑) |
| **注意力** | 全局 softmax O(n²) | 丘脑门控稀疏路由 O(k) |
| **记忆** | 无(上下文窗口) | 海马体情景记忆 + 皮层长期记忆 |
| **学习** | 离线反向传播 | 在线 STDP + 三因子强化 |
| **通信** | 浮点激活值 | 二值脉冲 + 时序编码 |
| **能效** | ~100W GPU (推理) | 目标: 稀疏激活 < 20W |
| **可解释性** | 黑盒 | 每个模块功能明确, 可独立分析 |
| **适应性** | 需要微调 | 在线持续学习(神经调质控制) |
| **结构** | 同质化堆叠 | 功能特化异质模块 |
