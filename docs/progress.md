# 悟韵 (WuYun) v3 开发进度

> 上次更新: 2026-02-07

---

## 已完成

### 文档修正与补全 (2026-02-07)
- [x] `00_design_principles.md` — D2 受体代码修正 (clamp + 注释)
- [x] `01_brain_region_plan.md` — CA3连接比例、海马容量、巩固周期、小脑数量、PrefrontalExecutive矛盾、皮层统计、微柱/超柱、抑制性比例说明
- [x] `02_neuron_system_design.md` — GABA_A反转电位、NMDA τ_rise、L2/3 burst标注、不应期参数、STP模型、bAP说明、睡眠巩固机制(§11)、能量约束(§12)、发育关键期(§13)
- [x] `03_project_structure.md` — agi2→agi3、sleep_system、sensory/motor接口、STP文件、配置目录补全、文档映射表

### P0 代码迁移 (2026-02-07)
- [x] `wuyun/spike/` — Layer 0: SpikeType、Spike、SpikeBus、OscillationClock (5文件)
- [x] `wuyun/synapse/` — Layer 1: SynapseBase、STP、STDP、DA-STDP、抑制性STDP、稳态可塑性 (8文件)
- [x] `wuyun/neuron/` — Layer 2: NeuronBase(16种参数预设)、双区室Compartment (3文件)
- [x] `wuyun/core/` — 向量化: NeuronPopulation、SynapseGroup (3文件)
- [x] 修正: NMDA tau_rise 2→5, GABA_A e_rev -75→-70, 文档引用路径更新
- [x] 验证: 全模块导入测试通过

---

## 进行中

### Git 仓库整理
- [ ] 将 agi2 历史保存为 `v2` 分支
- [ ] agi3 作为新的 `main` 分支
- [ ] 推送到 https://github.com/1600822305/wuyun

---

## 待开始

### Phase 0: 单神经元验证
- [ ] 从 agi2 迁移 test_phase0_neuron.py 并验证通过

### Phase 1: 皮层柱 (Layer 3)
- [ ] 迁移/重构 circuit/ 目录 (cortical_column, layer, column_factory)
- [ ] 6层微环路实现
- [ ] 预测编码回路验证 (L2/3↔L5↔L6)

### Phase 2: 丘脑路由 (Layer 3-4)
- [ ] 迁移/重构 thalamus/ 目录
- [ ] TRN 门控 + 注意力路由

### Phase 3: 海马记忆 (Layer 4)
- [ ] 迁移/重构 hippocampus/ 目录 (DG→CA3→CA1)
- [ ] one-shot 编码 + SWR 重放

### Phase 4: 基底节 (Layer 4)
- [ ] 迁移/重构 basal_ganglia/ 目录
- [ ] Go/NoGo 通路 + DA 调制

### Phase 5: 全脑整合 (Layer 5)
- [ ] system/ 目录: 视觉/听觉/语言/记忆/决策/注意力系统
- [ ] sensory_interface/ + motor_interface/
- [ ] sleep_system (NREM/REM/巩固)

---

## 架构备忘

```
代码两套实现:
  OOP 版 (neuron/NeuronBase + synapse/SynapseBase) → 调试/测试用
  向量化版 (core/NeuronPopulation + core/SynapseGroup) → 生产仿真用
  数学方程完全等价

层级依赖 (只允许从上到下):
  Layer 5: system/    → 全脑系统
  Layer 4: region/    → 脑区模块
  Layer 3: circuit/   → 皮层柱/微环路
  Layer 2: neuron/    → 双区室神经元
  Layer 1: synapse/   → 突触模型
  Layer 0: spike/     → 脉冲原语 (零依赖)
```
