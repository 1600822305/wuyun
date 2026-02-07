# Step 1: C++ 工程骨架 + 基础验证

> 日期: 2026-02-07
> 状态: ✅ 完成

## 目标

搭建 C++17 核心引擎, 实现双区室 AdLIF+ 神经元群体和 CSR 稀疏突触组。

## 完成内容

### 核心模块
- CMake 工程搭建 (MSVC 17, Visual Studio 2022, C++17)
- `src/core/types.h` — 枚举、参数结构体、4种预设神经元 (RS/FS/IB/LTS)
- `src/core/neuron.h/cpp` — 单神经元 step (调试用)
- `src/core/population.h/cpp` — SoA 向量化双区室 AdLIF+ 群体
- `src/core/synapse_group.h/cpp` — CSR 稀疏突触组 (电导型 AMPA/NMDA/GABA)
- `src/core/spike_queue.h/cpp` — 环形缓冲延迟队列

### 可塑性
- `src/plasticity/stdp.h/cpp` — 经典 STDP (A+/A-/τ+/τ-)
- `src/plasticity/stp.h/cpp` — 短时程可塑性 (Tsodyks-Markram STD/STF)

### 测试
- `tests/cpp/test_neuron.cpp` — 9 测试全通过
  - silence / regular / burst / refractory / adaptation
  - population ×4 (不同参数预设)

### pybind11
- 延后到 Step 3 (先验证回路)

## 性能基准 (Release, 单线程, CPU)

| 规模 | 每步耗时 | 吞吐量 |
|------|---------|--------|
| 10K 神经元 | 146 μs | 68M/s |
| 100K 神经元 | 1.5 ms | 67M/s |
| 1M 神经元 | 26 ms | 38M/s |
| 100K 突触 | 75 μs | — |
| 1M 突触 | 1.2 ms | — |

## 关键设计决策

- **SoA 布局** (Struct of Arrays): 神经元状态分开存储, SIMD 友好
- **CSR 稀疏格式**: 突触按突触前神经元分组, 高效遍历
- **双区室模型**: 胞体 (basal) + 顶端树突 (apical), κ 耦合系数控制 burst vs regular
- **事件驱动**: 只处理发放的神经元, 稀疏活动时省 90%+ 计算
