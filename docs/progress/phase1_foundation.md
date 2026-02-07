# Phase 1: 基础设施 — 设计 + 原型 + C++ 核心

> 对应: 设计文档 / P0 Python 原型 / 架构决策 / Step 1 / Step 2 / Step 2.5
> 时间: 2026-02-07
> 里程碑: 21 测试全通过 | 1M 神经元 26ms/step

---

## 设计文档 (v0.3)

- ✅ 四份设计文档校对完毕 (00设计原则 / 01脑区计划 / 02神经元系统 / 03项目结构)
- ✅ 01脑区文档按真实解剖分区重组 (前脑→端脑/间脑 | 中脑 | 后脑)
- ✅ 01脑区文档升级至 **NextBrain 混合方案**: 皮层(功能分区) + 皮层下(NextBrain, Nature 2025)
  - 丘脑 16核团 | 杏仁核 8核团 | 海马 7亚区 | 中脑 11核团 | 小脑 8小叶区
  - 总分区: **~97区** (皮层25 + 皮层下72)，每个皮层下核团标注 FreeSurfer 编号

---

## P0 Python 原型 → 已归档为算法参考

- ✅ `wuyun/spike/` — SpikeType、Spike、SpikeBus、OscillationClock
- ✅ `wuyun/synapse/` — SynapseBase、STP、STDP、DA-STDP、抑制性STDP、稳态可塑性
- ✅ `wuyun/neuron/` — NeuronBase(16种参数预设)、双区室Compartment
- ✅ `wuyun/core/` — 向量化 NeuronPopulation、SynapseGroup
- → 归档至 `_archived/python_prototype/`，数学方程不变，C++ 重新实现

---

## 架构决策: C++ 核心引擎

- ✅ 决定: 仿真核心用 **C++17**，Python 只做配置/实验/可视化
- ✅ 理由: Python 100万神经元 ~10秒/step 不可用; C++ ~10-50ms/step 接近实时; 未来迁移 CUDA 容易
- ✅ 03项目结构文档重写为 C++ core + pybind11 + Python 实验层 (v0.4)
- ✅ 技术栈: C++17 / CMake / pybind11 / Google Test / SoA布局 / CSR稀疏 / 事件驱动

---

## Step 1: C++ 工程骨架 + 基础验证 ✅

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

**Benchmark (Release, 单线程, CPU):**

| 规模 | 每步耗时 | 吞吐量 |
|------|---------|--------|
| 10K 神经元 | 146 μs | 68M/s |
| 100K 神经元 | 1.5 ms | 67M/s |
| 1M 神经元 | 26 ms | 38M/s |
| 100K 突触 | 75 μs | — |
| 1M 突触 | 1.2 ms | — |

---

## Step 2: 皮层柱模板 (C++) ✅

- ✅ src/circuit/cortical_column.h/cpp — 6层通用模板, **18组突触** (AMPA+NMDA+GABA)
- ✅ SST→L2/3 **AND** L5 apical (GABA_B), PV→L4/L5/L6 **全层** soma (GABA_A)
- ✅ NMDA 并行慢通道 (L4→L23, L23→L5, L23 recurrent)
- ✅ L2/3 层内 recurrent 连接 (AMPA+NMDA)
- ✅ burst 加权传递: burst spike ×2 增益
- ✅ 6 测试全通过 (540 神经元, 40203 突触)

---

## Step 2.5: 地基补全 ✅

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

---

## Phase 1 总结

| 指标 | 数值 |
|------|------|
| 测试 | 21 通过 |
| 神经元类型 | 12种参数预设 |
| 突触类型 | AMPA/NMDA/GABA_A/GABA_B |
| 可塑性 | STDP / STP / DA-STDP |
| 性能 | 1M neurons @ 26ms/step |
| 关键文件 | core/ + plasticity/ + circuit/ |