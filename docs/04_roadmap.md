# 悟韵 (WuYun) 开发路线图

> 创建: 2026-02-06
> 状态: ✅ 完成 | 🔄 进行中 | ⏳ 待开始

---

## Phase 0 — 地基层 ✅

> 单个双区室神经元能发放 regular/burst/silence

- ✅ `wuyun/spike/signal_types.py` — 信号类型枚举
- ✅ `wuyun/spike/spike.py` — Spike 事件 + SpikeTrain
- ✅ `wuyun/synapse/synapse_base.py` — 突触基类
- ✅ `wuyun/neuron/compartment.py` — 胞体 + 顶端树突区室
- ✅ `wuyun/neuron/neuron_base.py` — 双区室神经元
- ✅ `tests/test_phase0_neuron.py` — 7/7 测试通过 (2026-02-06)

---

## Phase 1 — 通信 + 学习 + 皮层柱 ✅

> 一个皮层柱能执行预测编码

- ✅ **1-A**: SpikeBus 脉冲总线 — `wuyun/spike/spike_bus.py` — 5/5 测试
- ✅ **1-B**: STDP 可塑性规则 — `wuyun/synapse/plasticity/` — 8/8 测试
  - ClassicalSTDP, DAModulatedSTDP, InhibitorySTDP + 软边界
- ✅ **1-C**: 皮层柱 6 层组装 (★ 核心里程碑) — `wuyun/circuit/` — 5/5 测试
  - Layer, CorticalColumn, create_sensory_column
  - L4→L23→L5→L6 前馈 + L6→L23/L5 apical 反馈 + PV+/SST+ 抑制
- ✅ **1-D**: 预测编码验证实验 — `experiments/` + `tests/test_pred_coding.py` — 4/4 测试
  - 全层级联 | L6 反馈→burst 增加 | 新奇检测 | 权重稳定性
- 📊 **累计 29/29 测试通过** (2026-02-06)

---

## Phase 1.8 — P2 前置补充 ✅

> Phase 2 硬性前置依赖补充

- ✅ **1.8-A**: 稳态可塑性 — `wuyun/synapse/plasticity/homeostatic.py`
  - HomeostaticPlasticity + HomeostaticParams (突触缩放)
  - CorticalColumn.apply_homeostatic_scaling() 集成接口
- ✅ **1.8-B**: 丘脑神经元参数预设 — `wuyun/neuron/neuron_base.py`
  - THALAMIC_RELAY_PARAMS (κ=0.3, ca_duration=40, burst_spike_count=4)
  - TRN_PARAMS (κ=0.0, v_threshold=-45.0, 快速响应)
- ✅ **1.8-C**: CorticalColumn 接口扩展 — `wuyun/circuit/cortical_column.py`
  - receive_lateral() / inject_lateral_current() 侧向输入
  - get_output_summary() 统一输出汇总
  - get_neuron_ids() 柱间连接用
- 📊 **累计 35/35 测试通过** (2026-02-06)

---

## Phase 2 — 多柱 + 丘脑路由 ✅

> 多个柱通过丘脑协同工作

- ✅ **2-A**: 丘脑核团 — `wuyun/thalamus/thalamic_nucleus.py`
  - ThalamicNucleus (TC 中继 + TRN 门控 + 内部 SpikeBus)
  - TC→TRN (AMPA, p=0.3) + TRN→TC (GABA_A, p=0.5) 内部连接
  - inject_sensory_current / inject_cortical_feedback_current / inject_trn_drive_current
  - create_thalamic_nucleus() 工厂函数
- ✅ **2-B**: 丘脑路由器 — `wuyun/thalamus/thalamic_router.py`
  - ThalamicRouter: 多核团管理 + 路由表
  - apply_trn_competition(): 跨核团 TRN 竞争抑制 (winner-take-all 注意力)
  - get_routed_outputs(): TC 输出按路由表分发到目标柱
- ✅ **2-C**: 多柱网络 — `wuyun/circuit/multi_column.py`
  - MultiColumnNetwork: 多柱 + ThalamicRouter + 层级/侧向连接
  - GainParams: 6 种跨模块增益参数 (已调优确保闭环稳定)
  - 电流注入通信: 丘脑↔皮层, 低柱→高柱, 高柱→低柱, 侧向
  - create_hierarchical_network(ff_connection_strength=1.5) 工厂函数
- ✅ **2-D**: 测试验证 — `tests/test_phase2_thalamus.py` — 7/7 测试
  - TC 中继基础 | TRN 门控效应 | Tonic/Burst 双模式
  - 丘脑-皮层环路 (★L6 闭环验证) | 双柱层级预测编码 (★Col1 活跃验证)
  - 注意力切换 (TRN 竞争) | 长期稳定性 + 稳态可塑性
- ✅ **2-E**: 审查修复 (2026-02-06)
  - 修复 L6 不发放问题: ff_connection_strength=1.5 确保深层激活
  - 修复 TC 无限增长: L6→TRN 负反馈 + 调优 GainParams 默认值
  - 修复 Col1 死亡: 增大 error_forward_gain 确保层级传递
  - 加强测试断言: L6 必须发放、Col1 必须活跃、TC < 200Hz
- 📊 **累计 42/42 测试通过** (35 旧 + 7 新, 2026-02-06)

---

## Phase 2.8 — P3 前置补充 ✅

> Phase 3 海马系统硬性前置依赖补充

- ✅ **2.8-A**: 海马神经元参数预设 — `wuyun/neuron/neuron_base.py`
  - GRANULE_PARAMS (DG 颗粒细胞, κ=0, v_threshold=-40mV 高阈值稀疏激活)
  - PLACE_CELL_PARAMS (CA3/CA1 锥体, κ=0.3, burst_spike_count=3)
  - GRID_CELL_PARAMS (EC 网格细胞, κ=0.2, a=0.03 振荡倾向)
- ✅ **2.8-B**: 短时程可塑性 STP — `wuyun/synapse/short_term_plasticity.py`
  - ShortTermPlasticity (Tsodyks-Markram 模型: 囊泡耗竭 + 释放概率易化)
  - MOSSY_FIBER_STP (去极化器突触: p0=0.05, a_f=0.15, PPF=3.6x)
  - SCHAFFER_COLLATERAL_STP (抑制主导: p0=0.5)
  - DEPRESSING_STP / FACILITATING_STP (通用预设)
- ✅ **2.8-C**: Theta 振荡时钟 — `wuyun/spike/oscillation_clock.py`
  - OscillationClock (多频段相位振荡器, 1ms 精度)
  - is_encoding_phase() / is_retrieval_phase() (theta 相位门控)
  - get_encoding_strength() / get_retrieval_strength() (平滑调制)
  - get_modulation() (CTC 发放概率调制)
  - THETA/GAMMA/ALPHA/BETA/DELTA_PARAMS 预设
- 📊 **累计 49/49 测试通过** (42 旧 + 7 新, 2026-02-07)

---

## Phase 3 — 海马记忆系统 ✅

> DG→CA3→CA1 环路，能编码和回忆

- ✅ **3-A**: 齿状回 DG — `wuyun/circuit/hippocampus/dentate_gyrus.py`
  - DentateGyrus (模式分离, 稀疏编码)
  - 100 颗粒细胞 (GRANULE_PARAMS) + 20 PV 抑制性中间神经元
  - EC→GC (AMPA, p=0.15) + EC→PV (AMPA, p=0.2) + PV→GC (GABA_A, 全连接)
  - 稀疏度 < 20%, 相似输入→正交输出
- ✅ **3-B**: CA3 自联想网络 — `wuyun/circuit/hippocampus/ca3_network.py`
  - CA3Network (自联想记忆, 循环连接, 模式补全)
  - 50 锥体细胞 (PLACE_CELL_PARAMS) + 8 PV 中间神经元
  - 循环连接 (ClassicalSTDP, a_plus=0.02 > a_minus=0.01, LTP 偏向)
  - 苔藓纤维 STP (MOSSY_FIBER_STP) + EC 直接通路 (ec_direct_gain=30)
  - **相位依赖路由** (文献: Cutsuridis 2010; PLOS CB 2025):
    - 编码期: 循环沉默 + PV 被 ACh 抑制 → STDP 异突触学习
    - 检索期: 循环放大 + PV 活跃 → 模式补全 + E/I 平衡
  - PV→CA3 weight=0.3 (弱于皮层柱的 0.6, 允许循环兴奋胜出分流抑制)
- ✅ **3-C**: CA1 比较/输出层 — `wuyun/circuit/hippocampus/ca1_network.py`
  - CA1Network (匹配/新奇检测)
  - Schaffer collateral 输入 (CA3→CA1, AMPA+STP)
  - EC-III 穿通纤维→apical (双通路比较)
  - burst=匹配, regular=新奇
- ✅ **3-D**: 海马全环路 — `wuyun/circuit/hippocampus/hippocampal_loop.py`
  - HippocampalLoop (DG + CA3 + CA1 + OscillationClock)
  - Theta 相位门控: 编码相 DG→CA3 + STDP, 检索相 EC→CA3 + 循环放大
  - encode() / recall(force_retrieval=True) 接口
  - STDP 每 5 步更新 (降低 O(n²) 计算负载)
- ✅ **3-E**: 测试验证 — `tests/test_phase3_hippocampus.py` — 7/7 测试
  - DG 模式分离 | DG 稀疏激活 | CA3 模式存储 (STDP 10x 权重比)
  - CA3 模式补全 (100% 回忆率) | CA1 匹配/新奇检测
  - Theta 相位门控 | 全环路编码-回忆 (CA3 活跃细胞确认)
- 📊 **累计 56/56 测试通过** (49 旧 + 7 新, 2026-02-07)

---

## Phase 4 — 基底节 + 强化学习 ⏳

> Go/NoGo/Stop 通路 + DA 调制决策

---

## Phase 5 — 小脑 + 杏仁核 ⏳

> 前向预测 + 情感价值标记

---

## Phase 6 — 神经调质系统 ⏳

> DA/NE/5-HT/ACh 全局状态调制

---

## Phase 7 — 全系统整合 ⏳

> 端到端仿生智能体
