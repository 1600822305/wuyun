# 悟韵 (WuYun) 项目文件夹架构设计

> 版本: 0.1.0 | 日期: 2026-02-06
> 设计原则: 文件夹结构 1:1 映射到大脑架构层级

---

## 一、核心设计原则

1. **层级映射**: 文件夹层级 ↔ 悟韵 (WuYun) Layer 0-5
2. **脑区模块化**: 每个脑区 = 一个独立模块，可单独测试
3. **接口分离**: 每个模块对外只暴露接口（interface），内部实现自由
4. **依赖单向**: 高层可依赖低层，低层不可依赖高层
5. **配置外置**: 脑区参数（神经元数、κ值、连接权重）全部配置化，不硬编码

---

## 二、顶层目录

```
agi2/
├── docs/                          # 设计文档 (已有)
├── wuyun/                    # ★ 核心源码
├── configs/                       # 脑区/模块参数配置
├── experiments/                   # 实验脚本与结果
├── tools/                         # 辅助工具 (可视化、分析、调试)
└── tests/                         # 测试代码
```

---

## 三、核心源码详细结构

```
wuyun/
│
├── __init__.py
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  Layer 0: Spike + Signal — 脉冲与信号原语             ║
│  ╚═══════════════════════════════════════════════════════╝
│
├── spike/                         # 脉冲与信号原语
│   ├── __init__.py
│   ├── spike.py                   # Spike 事件定义 (REGULAR/BURST/NONE)
│   ├── spike_bus.py               # SpikeBus 脉冲总线 (收发/路由)
│   ├── field_bus.py               # FieldBus 场总线 (振荡同步)
│   ├── modulation_bus.py          # ModulationBus 调制总线 (DA/NE/5-HT/ACh)
│   └── signal_types.py            # 信号类型枚举与消息格式
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  Layer 1: Synapse + Channel — 突触与通道抽象          ║
│  ╚═══════════════════════════════════════════════════════╝
│
├── synapse/                       # 突触模型
│   ├── __init__.py
│   ├── synapse_base.py            # SynapseBase 基类
│   ├── chemical/                  # 化学突触
│   │   ├── __init__.py
│   │   ├── ampa.py                # AMPA 快速兴奋突触
│   │   ├── nmda.py                # NMDA 电压门控突触 (可塑性关键)
│   │   ├── gaba_a.py              # GABA_A 快速抑制
│   │   ├── gaba_b.py              # GABA_B 慢速抑制
│   │   └── modulatory.py          # DA/5-HT/ACh/NE 调制性突触
│   ├── electrical/                # 电突触
│   │   ├── __init__.py
│   │   └── gap_junction.py        # 缝隙连接 (中间神经元间同步)
│   └── plasticity/                # 突触可塑性规则
│       ├── __init__.py
│       ├── stdp.py                # 经典 STDP
│       ├── voltage_stdp.py        # 电压依赖 STDP
│       ├── da_modulated_stdp.py   # 多巴胺调制三因子 STDP
│       ├── bcm.py                 # BCM 理论
│       ├── homeostatic.py         # 稳态可塑性 (缩放/内在/元)
│       └── structural.py          # 结构可塑性 (新生/修剪)
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  Layer 2: Neuron — 双区室神经元模型                   ║
│  ╚═══════════════════════════════════════════════════════╝
│
├── neuron/                        # 神经元模型
│   ├── __init__.py
│   ├── neuron_base.py             # NeuronBase 双区室 AdLIF+ 基类
│   ├── compartment.py             # 区室模型 (SomaticCompartment, ApicalCompartment)
│   ├── excitatory/                # 兴奋性神经元 (双区室, κ>0)
│   │   ├── __init__.py
│   │   ├── pyramidal.py           # PyramidalCell 基类
│   │   ├── l23_pyramidal.py       # L2/3 浅层锥体 κ=0.3
│   │   ├── l5_pyramidal.py        # L5 深层锥体 κ=0.6
│   │   ├── l6_pyramidal.py        # L6 多形层锥体 κ=0.2
│   │   ├── stellate.py            # L4 星形细胞 κ=0.1
│   │   ├── granule.py             # 颗粒细胞 κ=0 (海马DG/小脑)
│   │   └── mossy_cell.py          # 苔藓细胞 - 海马DG
│   ├── inhibitory/                # 抑制性神经元 (单区室, κ=0)
│   │   ├── __init__.py
│   │   ├── basket_cell.py         # PV+ 篮状细胞 → 靶向胞体
│   │   ├── martinotti.py          # SST+ Martinotti → 靶向顶端树突
│   │   ├── vip_interneuron.py     # VIP → 去抑制 (抑制SST)
│   │   ├── chandelier.py          # PV+ 枝形烛台 → 轴突起始段
│   │   └── ngf_cell.py            # 慢抑制 GABA_B
│   ├── modulatory/                # 调制性神经元
│   │   ├── __init__.py
│   │   ├── dopamine.py            # VTA/SNc DA 神经元
│   │   ├── serotonin.py           # 中缝核 5-HT 神经元
│   │   ├── norepinephrine.py      # 蓝斑核 NE 神经元
│   │   └── cholinergic.py         # 基底前脑 ACh 神经元
│   └── specialized/               # 特化型神经元
│       ├── __init__.py
│       ├── thalamic_relay.py      # 丘脑中继 (Tonic/Burst切换)
│       ├── trn_neuron.py          # 丘脑网状核 (纯抑制门控)
│       ├── medium_spiny.py        # 纹状体中棘神经元 (D1/D2)
│       ├── purkinje.py            # 小脑浦肯野细胞
│       ├── place_cell.py          # 海马位置细胞
│       ├── grid_cell.py           # 内嗅皮层网格细胞
│       └── head_direction.py      # 头朝向细胞
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  Layer 3: Circuit — 环路与皮层柱 (★核心层)           ║
│  ╚═══════════════════════════════════════════════════════╝
│
├── circuit/                       # ★ 环路模块 (最先设计!)
│   ├── __init__.py
│   ├── cortical_column/           # ★★★ 皮层柱 (核心中的核心)
│   │   ├── __init__.py
│   │   ├── column.py              # CorticalColumn 主类
│   │   ├── column_interface.py    # 柱对外标准接口 (feedforward/feedback/lateral)
│   │   ├── layers/                # 6 层内部实现
│   │   │   ├── __init__.py
│   │   │   ├── layer1_molecular.py    # L1 分子层 (反馈调制)
│   │   │   ├── layer23_superficial.py # L2/3 浅层 (预测误差 regular)
│   │   │   ├── layer4_granular.py     # L4 颗粒层 (前馈输入)
│   │   │   ├── layer5_deep.py         # L5 深层 (burst 驱动输出)
│   │   │   └── layer6_polymorphic.py  # L6 多形层 (预测生成)
│   │   ├── microcircuit.py        # 柱内微环路 (PV+/SST+/VIP 注意力门控)
│   │   ├── predictive_coding.py   # 预测编码计算引擎 (STEP 1-8)
│   │   └── column_factory.py      # 柱工厂 (根据脑区配置生成不同参数的柱)
│   │
│   ├── hippocampal_loop/          # 海马环路 (DG→CA3→CA1)
│   │   ├── __init__.py
│   │   ├── dentate_gyrus.py       # DG 齿状回 (模式分离)
│   │   ├── ca3.py                 # CA3 (自联想/模式补全)
│   │   ├── ca1.py                 # CA1 (比较/输出)
│   │   ├── entorhinal_cortex.py   # 内嗅皮层 (输入/输出接口)
│   │   └── consolidation.py       # 记忆巩固机制 (Sharp Wave Ripple)
│   │
│   ├── basal_ganglia_pathway/     # 基底神经节通路 (Go/NoGo/Stop)
│   │   ├── __init__.py
│   │   ├── striatum.py            # 纹状体 (D1壳核 + D2尾状核 + NAcc)
│   │   ├── globus_pallidus.py     # 苍白球 (GPe + GPi)
│   │   ├── stn.py                 # 丘脑底核 (超直接通路)
│   │   ├── snr.py                 # 黑质网状部
│   │   └── pathway_selector.py    # Go/NoGo/Stop 通路选择器
│   │
│   └── cerebellar_circuit/        # 小脑环路
│       ├── __init__.py
│       ├── granular_layer.py      # 颗粒层 (高维稀疏展开)
│       ├── purkinje_layer.py      # 浦肯野层 (学习/抑制)
│       ├── deep_nuclei.py         # 深部核团 (输出)
│       └── inferior_olive.py      # 下橄榄核 (误差教学信号)
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  Layer 4: Region — 脑区                              ║
│  ╚═══════════════════════════════════════════════════════╝
│
├── region/                        # 脑区模块
│   ├── __init__.py
│   ├── region_base.py             # Region 基类 (含振荡状态、调制水平)
│   ├── neocortex/                 # 新皮层脑区 (由皮层柱组成)
│   │   ├── __init__.py
│   │   ├── sensory/               # 感觉皮层区 (S-01 ~ S-09)
│   │   │   ├── __init__.py
│   │   │   ├── v1.py              # S-01 初级视觉皮层
│   │   │   ├── v2.py              # S-02 次级视觉皮层
│   │   │   ├── ventral_stream.py  # S-03 腹侧流 (V4/IT)
│   │   │   ├── dorsal_stream.py   # S-04 背侧流 (V5/MT)
│   │   │   ├── a1.py              # S-05 初级听觉皮层
│   │   │   ├── a2.py              # S-06 次级听觉皮层
│   │   │   ├── somatosensory.py   # S-07 躯体感觉皮层
│   │   │   ├── olfactory.py       # S-08 嗅觉皮层
│   │   │   └── gustatory.py       # S-09 味觉皮层
│   │   ├── motor/                 # 运动皮层区 (M-01 ~ M-04)
│   │   │   ├── __init__.py
│   │   │   ├── m1.py              # M-01 初级运动皮层
│   │   │   ├── premotor.py        # M-02 前运动皮层
│   │   │   ├── sma.py             # M-03 辅助运动区
│   │   │   └── fef.py             # M-04 前额叶眼动区
│   │   └── association/           # 联合皮层区 (A-01 ~ A-11)
│   │       ├── __init__.py
│   │       ├── fpc.py             # A-01 前额极皮层
│   │       ├── ofc.py             # A-02 眶额皮层
│   │       ├── dlpfc.py           # A-03 背外侧前额叶
│   │       ├── vmpfc.py           # A-04 腹内侧前额叶
│   │       ├── acc.py             # A-05 前扣带回
│   │       ├── ppc.py             # A-06 后顶叶皮层
│   │       ├── tpj.py             # A-07 颞顶联合区
│   │       ├── broca.py           # A-08 布洛卡区
│   │       ├── wernicke.py        # A-09 韦尼克区
│   │       ├── angular_gyrus.py   # A-10 角回
│   │       └── insula.py          # A-11 岛叶
│   │
│   ├── thalamus/                  # 丘脑 (T-01 ~ T-08)
│   │   ├── __init__.py
│   │   ├── thalamic_router.py     # ★ 丘脑路由器主控
│   │   ├── lgn.py                 # T-01 外侧膝状体 (视觉)
│   │   ├── mgn.py                 # T-02 内侧膝状体 (听觉)
│   │   ├── vpl.py                 # T-03 腹后外侧核 (躯体感觉)
│   │   ├── pulvinar.py            # T-04 丘脑枕 (视觉注意力)
│   │   ├── md.py                  # T-05 背内侧核 (前额叶)
│   │   ├── trn.py                 # T-06 网状核 (全局门控)
│   │   ├── anterior.py            # T-07 前核群 (记忆/导航)
│   │   └── vpm.py                 # T-08 腹后内侧核 (面部/味觉)
│   │
│   ├── hippocampus/               # 海马体 (H-01 ~ H-05)
│   │   ├── __init__.py
│   │   └── hippocampal_memory.py  # 海马记忆系统 (调用 circuit/hippocampal_loop)
│   │
│   ├── basal_ganglia/             # 基底神经节 (BG-01 ~ BG-07)
│   │   ├── __init__.py
│   │   └── decision_engine.py     # 决策引擎 (调用 circuit/basal_ganglia_pathway)
│   │
│   ├── amygdala/                  # 杏仁核 (AM-01 ~ AM-03)
│   │   ├── __init__.py
│   │   └── valence_system.py      # 情感价值标记系统
│   │
│   ├── cerebellum/                # 小脑 (CB-01 ~ CB-04)
│   │   ├── __init__.py
│   │   └── predictor.py           # 前向预测器 (调用 circuit/cerebellar_circuit)
│   │
│   └── neuromodulator/            # 神经调质系统 (NM-01 ~ NM-05)
│       ├── __init__.py
│       ├── vta.py                 # NM-01 VTA (DA - 奖励预测误差)
│       ├── snc.py                 # NM-02 SNc (DA - 运动强化)
│       ├── locus_coeruleus.py     # NM-03 蓝斑核 (NE - 探索/警觉)
│       ├── raphe.py               # NM-04 中缝核 (5-HT - 耐心/折扣)
│       └── basal_forebrain.py     # NM-05 基底前脑 (ACh - 注意力/学习)
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  Layer 5: BrainSystem — 全脑系统                     ║
│  ╚═══════════════════════════════════════════════════════╝
│
├── system/                        # 全脑系统整合
│   ├── __init__.py
│   ├── brain.py                   # ★ Brain 主类 (顶层入口)
│   ├── visual_system.py           # 视觉系统 (V1→V2→V4/IT + 丘脑LGN)
│   ├── auditory_system.py         # 听觉系统 (A1→A2→Wernicke + 丘脑MGN)
│   ├── language_system.py         # 语言系统 (Wernicke→Broca→M1)
│   ├── memory_system.py           # 记忆系统 (海马↔皮层 + 巩固)
│   ├── decision_system.py         # 决策系统 (PFC→BG→丘脑→运动)
│   ├── attention_system.py        # 注意力系统 (PFC→丘脑TRN + VIP门控)
│   └── executive_system.py        # 执行控制 (dlPFC 工作记忆+规划)
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  核心引擎                                             ║
│  ╚═══════════════════════════════════════════════════════╝
│
└── engine/                        # 仿真引擎
    ├── __init__.py
    ├── simulator.py               # 主仿真循环 (simulation_loop)
    ├── clock.py                   # 多层级时钟 (1ms/10ms/100ms/10s)
    ├── scheduler.py               # 事件调度器
    └── state.py                   # 全局状态管理
```

---

## 四、配置目录

```
configs/
├── regions/                       # 脑区参数配置
│   ├── sensory/
│   │   ├── v1.yaml                # V1 皮层柱参数: 柱数量、神经元分布、连接概率
│   │   ├── v2.yaml
│   │   └── ...
│   ├── association/
│   │   ├── dlpfc.yaml             # dlPFC: 更大的工作记忆缓存、DA调制更强
│   │   └── ...
│   ├── thalamus/
│   │   └── routing_rules.yaml     # 丘脑路由表
│   └── neuromodulator/
│       └── modulation_params.yaml # DA/NE/5-HT/ACh 参数
│
├── neurons/                       # 神经元类型参数
│   ├── pyramidal.yaml             # 锥体细胞默认参数 (τ_m, τ_a, κ, a, b...)
│   ├── basket_cell.yaml
│   ├── martinotti.yaml
│   └── ...
│
├── synapses/                      # 突触类型参数
│   ├── ampa.yaml
│   ├── nmda.yaml
│   └── plasticity/
│       ├── stdp.yaml              # A+, A-, τ+, τ- 等
│       └── da_stdp.yaml           # 三因子参数
│
└── connectome/                    # 连接矩阵配置
    ├── intra_column.yaml          # 柱内连接拓扑 (6层间连接概率矩阵)
    ├── inter_column.yaml          # 柱间连接规则
    └── inter_region.yaml          # 脑区间连接 (01文档中的连接矩阵)
```

---

## 五、实验与工具

```
experiments/
├── phase0_single_neuron/          # 单神经元测试
├── phase1_single_column/          # 单柱预测编码测试
├── phase2_multi_column/           # 多柱 + 丘脑路由
├── phase3_memory/                 # 海马记忆测试
└── benchmarks/                    # 性能基准测试

tools/
├── visualizer/                    # 可视化工具
│   ├── spike_raster.py            # 脉冲栅格图
│   ├── membrane_plot.py           # 膜电位波形
│   ├── connectivity_map.py        # 连接矩阵热图
│   └── brain_viewer.py            # 3D 脑区活动可视化
├── analysis/                      # 分析工具
│   ├── firing_rate.py             # 发放率统计
│   ├── burst_ratio.py             # burst/regular 比率分析
│   ├── oscillation_power.py       # 振荡功率谱
│   └── information_flow.py        # 信息流分析 (预测误差追踪)
└── debug/                         # 调试工具
    ├── region_inspector.py        # 脑区状态检查器
    └── spike_tracer.py            # 脉冲路径追踪
```

---

## 六、依赖关系与导入规则

```
依赖方向 (只允许从上到下):

system/    → 可以导入 region/, circuit/, neuron/, synapse/, spike/, engine/
region/    → 可以导入 circuit/, neuron/, synapse/, spike/
circuit/   → 可以导入 neuron/, synapse/, spike/
neuron/    → 可以导入 synapse/, spike/
synapse/   → 可以导入 spike/
spike/     → 无外部依赖 (最底层)
engine/    → 可以导入所有层 (仿真引擎需要驱动所有组件)
configs/   → 被所有层读取 (只读数据)

禁止:
  spike/ 导入 neuron/ (底层不依赖高层)
  neuron/ 导入 circuit/ (低层不依赖高层)
  region/A 导入 region/B (脑区间通过 spike_bus 通信, 不直接导入)
```

> **⚠️ 反作弊约束 (参见 00_design_principles.md)**
> - `region/` 下所有新皮层脑区不允许包含功能特异的 `process()` 逻辑，只能通过配置差异化
> - `region/` 下的脑区模块之间 **禁止** 直接函数调用，必须通过 `spike_bus` 通信
> - 任何 IF-ELSE 分支不得基于“语义含义”，只能基于“物理机制”

---

## 七、文件命名约定

| 类别 | 约定 | 示例 |
|------|------|------|
| Python 模块 | snake_case | `l5_pyramidal.py` |
| 类名 | PascalCase | `L5Pyramidal` |
| 配置文件 | snake_case.yaml | `dlpfc.yaml` |
| 常量 | UPPER_SNAKE_CASE | `SPIKE_TYPE_BURST` |
| 脑区编号 | 文档中的编号系统 | S-01, A-03, T-06, BG-02 |

---

## 八、与设计文档的映射关系

| 设计文档 | 对应代码目录 | 关系 |
|---------|------------|------|
| `00_design_principles.md` | 全部代码 | **第一约束** — 反作弊公约+设计审查清单 |
| `01_brain_region_plan.md` | `region/` + `configs/regions/` | 脑区清单→模块清单 |
| `02_neuron_system_design.md` | `neuron/` + `synapse/` + `spike/` | Layer 0-2 实现规格 |
| `03_project_structure.md` | 本文档 | 文件夹架构设计 |
| (待创建) 皮层柱详细设计 | `circuit/cortical_column/` | ★ 最核心的实现规格 |
