# 悟韵 (WuYun) 项目文件夹架构设计

> 版本: 0.4.0 | 日期: 2026-02-07
> 设计原则: **C++ 核心引擎 + pybind11 + Python 实验层**
> 文件夹结构 1:1 映射到大脑架构层级

---

## 一、核心设计原则

1. **C++/Python 分层**: 仿真核心全部 C++ 实现，Python 只做配置/实验/可视化
2. **层级映射**: 文件夹层级 ↔ 悟韵 (WuYun) Layer 0-5
3. **脑区模块化**: 每个脑区 = 一个独立模块，可单独测试
4. **接口分离**: 每个模块对外只暴露接口，内部实现自由
5. **依赖单向**: 高层可依赖低层，低层不可依赖高层
6. **配置外置**: 脑区参数（神经元数、κ值、连接权重）全部 YAML 配置化
7. **零拷贝原则**: C++↔Python 边界尽量避免数据拷贝，用 buffer protocol

### 1.1 为什么 C++ 核心

| 问题 | Python | C++ |
|------|--------|-----|
| 100万神经元 1ms step | ~10秒 (不可用) | ~10-50ms (接近实时) |
| 内存/对象 | 每对象 56B 开销 | struct 无开销 |
| 突触遍历 | dict/list 循环 | CSR 稀疏 + SIMD |
| 延迟队列 | Python deque | 无锁环形缓冲 |
| 未来 GPU | 需全部重写 | 加 __global__ 即可 |

### 1.2 C++/Python 职责边界

```
C++ 负责 (性能关键，在仿真循环内):
  ├── 神经元膜电位更新、发放判定
  ├── 突触电流计算、门控变量衰减
  ├── STDP/STP 权重更新
  ├── 事件驱动调度
  └── 延迟队列管理

Python 负责 (仿真循环外):
  ├── YAML 配置加载 → 传给 C++ 构造函数
  ├── 实验脚本: 创建仿真、跑、取结果
  ├── 可视化: matplotlib/plotly 画图
  ├── 分析: 发放率、振荡功率谱、信息流
  └── 高层编排: 训练循环、参数扫描
```

---

## 二、顶层目录

```
agi3/
├── docs/                          # 设计文档
├── src/                           # ★ C++ 核心引擎源码
├── python/                        # Python 包 (pybind11 封装 + 工具)
├── configs/                       # YAML 脑区/模块参数配置
├── experiments/                   # Python 实验脚本
├── tests/                         # 测试 (C++ 单元测试 + Python 集成测试)
├── CMakeLists.txt                 # ★ CMake 构建系统
├── pyproject.toml                 # Python 包构建配置
└── README.md
```

---

## 三、C++ 核心引擎 (`src/`)

```
src/
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  Layer 0-1: core/ — 神经元、突触、脉冲 (最底层)       ║
│  ╚═══════════════════════════════════════════════════════╝
│
├── core/
│   ├── types.h                    # 基础类型: SpikeType, NeuronType, SynapseType 枚举
│   ├── neuron.h / neuron.cpp      # 双区室 AdLIF+ 神经元 (胞体+顶端树突)
│   ├── synapse.h                  # 突触参数 struct (AMPA/NMDA/GABA_A/GABA_B)
│   ├── population.h / population.cpp  # ★ NeuronPopulation: 向量化神经元群体 (SoA 布局)
│   ├── synapse_group.h / synapse_group.cpp  # ★ SynapseGroup: CSR 稀疏突触组
│   ├── spike_queue.h / spike_queue.cpp      # 延迟队列 (环形缓冲, 支持可变延迟)
│   └── modulation.h               # 调制信号 struct (DA/NE/5-HT/ACh 浓度)
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  Layer 1b: plasticity/ — 突触可塑性规则              ║
│  ╚═══════════════════════════════════════════════════════╝
│
├── plasticity/
│   ├── stdp.h / stdp.cpp          # 经典 spike-timing STDP
│   ├── da_stdp.h / da_stdp.cpp    # 多巴胺调制三因子 STDP
│   ├── stp.h / stp.cpp            # 短时程可塑性 (Tsodyks-Markram STD/STF)
│   ├── homeostatic.h / homeostatic.cpp  # 稳态可塑性 (突触缩放)
│   └── structural.h / structural.cpp    # 结构可塑性 (新生/修剪)
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  Layer 2-3: circuit/ — 环路模板                      ║
│  ╚═══════════════════════════════════════════════════════╝
│
├── circuit/
│   ├── cortical_column.h / cortical_column.cpp    # ★ 6层皮层柱 (通用模板)
│   ├── microcircuit.h / microcircuit.cpp          # PV+/SST+/VIP 微环路
│   ├── hippocampal_loop.h / hippocampal_loop.cpp  # EC→DG→CA3→CA1→Sub
│   ├── basal_ganglia.h / basal_ganglia.cpp        # D1/D2 Go/NoGo/Stop
│   └── cerebellar_circuit.h / cerebellar_circuit.cpp  # 颗粒→浦肯野→深核
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  Layer 4: region/ — 脑区 (由环路+配置实例化)         ║
│  ╚═══════════════════════════════════════════════════════╝
│
├── region/
│   ├── region.h / region.cpp       # Region 基类 (含振荡状态、调制水平)
│   └── region_factory.h / region_factory.cpp  # ★ 从 YAML 配置生成 97 个脑区实例
│   # 注意: 不再有每个脑区一个文件!
│   # 皮层区 = CorticalColumn × N (参数不同)
│   # 特殊区 = 对应 circuit/ 模板 (参数不同)
│   # 所有差异在 configs/regions/*.yaml 中
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  Layer 5: engine/ — 仿真引擎                        ║
│  ╚═══════════════════════════════════════════════════════╝
│
├── engine/
│   ├── simulator.h / simulator.cpp    # ★ 主仿真循环
│   ├── scheduler.h / scheduler.cpp    # 事件驱动调度 (优先队列)
│   ├── clock.h / clock.cpp            # 多层级时钟 (1ms/10ms/100ms)
│   └── recorder.h / recorder.cpp      # 数据记录器 (脉冲/膜电位/权重 → 导出)
│
│  ╔═══════════════════════════════════════════════════════╗
│  ║  pybind11 绑定层                                     ║
│  ╚═══════════════════════════════════════════════════════╝
│
├── bindings/
│   └── pywuyun.cpp                # pybind11 绑定 (暴露给 Python 的 API)
│
└── CMakeLists.txt                 # src/ 子目录的 CMake
```

### 3.1 C++ 设计要点

**数据布局: Struct of Arrays (SoA)**
```cpp
// ❌ 错误: Array of Structs (缓存不友好)
struct Neuron { float v_soma, v_apical, w_adapt; };
std::vector<Neuron> neurons;  // v_soma 和 w_adapt 间隔存放

// ✅ 正确: Struct of Arrays (SIMD 友好)
struct NeuronPopulation {
    std::vector<float> v_soma;     // 连续内存, 可 SIMD
    std::vector<float> v_apical;
    std::vector<float> w_adapt;
    std::vector<int8_t> spike_type;
    size_t n;
};
```

**突触存储: CSR 稀疏格式**
```cpp
struct SynapseGroup {
    std::vector<int32_t> row_ptr;   // 每个 pre 神经元的突触起始索引
    std::vector<int32_t> col_idx;   // post 神经元 ID
    std::vector<float> weights;     // 突触权重
    std::vector<float> delays;      // 传导延迟
    std::vector<float> g;           // 门控变量
};
// 遍历 pre=i 的所有突触: for j in [row_ptr[i], row_ptr[i+1])
```

**事件驱动: 只算活跃神经元**
```cpp
// 每步只处理收到脉冲的神经元 → 稀疏活动时省 90%+ 计算
struct Scheduler {
    std::priority_queue<SpikeEvent> event_queue;
    void deliver_spikes(float t);   // 到时间了就投递
};
```

---

## 四、Python 层 (`python/`)

```
python/
└── wuyun/                         # Python 包
    ├── __init__.py                # import wuyun; sim = wuyun.Simulation(...)
    ├── config.py                  # YAML 配置加载/验证 → 传给 C++ 构造函数
    ├── experiment.py              # 实验基类 (setup → run → analyze)
    ├── viz/                       # 可视化工具
    │   ├── spike_raster.py        # 脉冲栅格图
    │   ├── membrane_plot.py       # 膜电位波形
    │   ├── connectivity_map.py    # 连接矩阵热图
    │   └── brain_viewer.py        # 脑区活动可视化
    ├── analysis/                  # 分析工具
    │   ├── firing_rate.py         # 发放率统计
    │   ├── oscillation.py         # 振荡功率谱
    │   └── information_flow.py    # 信息流分析
    └── io/                        # I/O 接口
        ├── sensory_encoder.py     # 外部信号 → 脉冲 (图像/音频/文本)
        └── motor_decoder.py       # 脉冲 → 外部动作/文本
```

### 4.1 Python API 使用示例

```python
import wuyun

# 1. 加载配置
config = wuyun.Config.from_yaml("configs/experiments/minimal_brain.yaml")

# 2. 创建仿真 (C++ 对象, 在 C++ 内存中)
sim = wuyun.Simulation(config)

# 3. 注入输入脉冲
sim.inject_spikes(region="LGN", neuron_ids=[0, 5, 12], times=[10.0, 10.5, 11.0])

# 4. 运行 (纯 C++ 循环, Python 不参与)
sim.run(duration_ms=1000.0)

# 5. 取结果 (零拷贝 numpy array)
spikes = sim.get_spikes("V1")          # → numpy structured array
v_soma = sim.get_membrane("dlPFC")     # → numpy 2D array (time × neurons)
weights = sim.get_weights("V1→V2")     # → scipy.sparse CSR
```

---

## 五、配置目录 (`configs/`)

> 配置文件被 Python 加载后传给 C++ 构造函数。C++ 不直接读 YAML。

```
configs/
├── regions/                       # 脑区参数 (97区, 对应01文档)
│   ├── cortex/                    # 皮层区 (25区)
│   │   ├── v1.yaml               # S-01: 柱数量、L4厚度、丘脑输入权重
│   │   ├── dlpfc.yaml            # A-01: 强循环连接、高DA受体密度
│   │   ├── m1.yaml               # M-01: L5厚、运动输出连接
│   │   └── ...
│   ├── thalamus/                  # 丘脑 (T-01~T-16, NextBrain)
│   │   └── thalamic_nuclei.yaml  # 16核团参数 + TRN抑制参数
│   ├── hippocampus/               # 海马 (H-01~H-07)
│   ├── basal_ganglia/             # 基底节 (BG-01~07 + ST-01)
│   ├── amygdala/                  # 杏仁核 (AM-01~AM-08, NextBrain)
│   ├── hypothalamus/              # 下丘脑 (HY-01~HY-07)
│   ├── brainstem/                 # 中脑+脑桥 (MB-01~11, HB-01~02)
│   └── cerebellum/                # 小脑 (CB-01~CB-08, NextBrain小叶)
│
├── neurons/                       # 神经元类型参数
│   ├── pyramidal.yaml             # τ_m, τ_a, κ, a, b, v_th...
│   ├── basket_cell.yaml
│   ├── medium_spiny.yaml          # D1/D2 MSN
│   └── ...
│
├── synapses/                      # 突触类型参数
│   ├── ampa.yaml                  # τ_rise, τ_fall, g_max, e_rev
│   ├── nmda.yaml
│   ├── gaba_a.yaml
│   └── plasticity/
│       ├── stdp.yaml              # A+, A-, τ+, τ-
│       ├── da_stdp.yaml           # 三因子参数
│       └── stp.yaml               # U, τ_D, τ_F (按突触类型)
│
├── connectome/                    # 连接矩阵
│   ├── intra_column.yaml          # 柱内6层间连接概率
│   ├── inter_region.yaml          # 脑区间连接 (01文档§3)
│   └── modulation_topology.yaml   # 调质投射拓扑 (DA/NE/5-HT/ACh)
│
└── experiments/                   # 实验配置 (组合上述配置)
    ├── minimal_brain.yaml         # Step 3: V1+PFC+BG+丘脑4核+DA
    ├── memory_circuit.yaml        # Step 4: + 海马+杏仁核
    └── full_brain.yaml            # Step 7: 全部97区
```

---

## 六、测试与实验

```
tests/
├── cpp/                           # C++ 单元测试 (Google Test)
│   ├── test_neuron.cpp            # 神经元膜电位/发放
│   ├── test_synapse_group.cpp     # CSR 稀疏突触
│   ├── test_stdp.cpp              # STDP 权重更新
│   ├── test_cortical_column.cpp   # 单柱预测编码
│   └── test_benchmark.cpp         # 性能基准 (N神经元/step时间)
└── python/                        # Python 集成测试
    ├── test_binding.py            # pybind11 绑定验证
    ├── test_single_neuron.py      # 单神经元波形验证
    └── test_minimal_brain.py      # Step 3 最小回路功能验证

experiments/                       # Python 实验脚本
├── step1_neuron_validation/       # 单神经元 regular/burst/silence 模式
├── step2_column_test/             # 单柱预测编码验证
├── step3_minimal_brain/           # LGN→V1→PFC→BG→M1 最短通路
├── step4_memory_emotion/          # 海马记忆 + 杏仁核情感
└── benchmarks/                    # 性能基准 (神经元数 vs 速度)
```

---

## 七、构建系统

### 7.1 CMake 结构

```
agi3/
├── CMakeLists.txt                 # 顶层 CMake
├── src/
│   └── CMakeLists.txt             # C++ 库 + pybind11 模块
└── tests/
    └── cpp/
        └── CMakeLists.txt         # Google Test
```

### 7.2 依赖

| 依赖 | 用途 | 获取方式 |
|------|------|---------|
| **C++17** | 语言标准 | 编译器自带 |
| **pybind11** | C++↔Python 绑定 | CMake FetchContent / pip |
| **yaml-cpp** | YAML 配置解析 (C++ 侧备用) | CMake FetchContent |
| **Google Test** | C++ 单元测试 | CMake FetchContent |
| **Python 3.10+** | 实验/可视化层 | 系统安装 |
| **numpy** | 零拷贝数组交换 | pip |
| **matplotlib** | 可视化 | pip |

### 7.3 构建命令

```bash
# 构建 C++ 库 + pybind11 模块
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# 安装 Python 包 (开发模式)
pip install -e .

# 运行 C++ 测试
cd build && ctest

# 运行 Python 测试
pytest tests/python/
```

---

## 八、依赖关系 (`#include` 规则)

```
C++ 头文件依赖方向 (只允许从上到下):

engine/     → 可以 include region/, circuit/, core/, plasticity/
region/     → 可以 include circuit/, core/
circuit/    → 可以 include core/, plasticity/
plasticity/ → 可以 include core/
core/       → 无外部依赖 (最底层, 只依赖 STL)
bindings/   → 可以 include 所有 (绑定层)

禁止:
  core/ include circuit/ (底层不依赖高层)
  circuit/ include region/ (低层不依赖高层)
  region/A 直接调用 region/B (脑区间通过 SpikeQueue 通信)
```

> **⚠️ 反作弊约束 (参见 00_design_principles.md)**
> - 皮层区全部通过 `RegionFactory` + YAML 配置实例化，不允许功能特异代码
> - 脑区间 **禁止** 直接函数调用，必须通过 `SpikeQueue` 传递脉冲事件
> - C++ 代码中任何 if/switch 不得基于"语义含义"，只能基于"物理机制"

---

## 九、文件命名约定

| 类别 | 约定 | 示例 |
|------|------|------|
| C++ 头文件 | snake_case.h | `neuron.h`, `cortical_column.h` |
| C++ 源文件 | snake_case.cpp | `population.cpp` |
| C++ 类名 | PascalCase | `NeuronPopulation`, `SynapseGroup` |
| C++ 命名空间 | `wuyun::` | `wuyun::core::`, `wuyun::circuit::` |
| Python 模块 | snake_case.py | `config.py`, `spike_raster.py` |
| 配置文件 | snake_case.yaml | `dlpfc.yaml` |
| 常量/枚举 | UPPER_SNAKE_CASE | `SPIKE_REGULAR`, `SPIKE_BURST` |
| 脑区编号 | 文档系统 | S-01, T-16, AM-08, CB-03 |

---

## 十、与设计文档的映射关系

| 设计文档 | 对应代码 | 关系 |
|---------|---------|------|
| `00_design_principles.md` | 全部代码 | **第一约束** — 反作弊公约 |
| `01_brain_region_plan.md` | `src/region/` + `configs/regions/` | 97脑区 → RegionFactory 配置 |
| `02_neuron_system_design.md` | `src/core/` + `src/plasticity/` | 神经元/突触方程 → C++ 实现 |
| `03_project_structure.md` | 本文档 | 文件夹架构设计 |
| (待创建) 皮层柱详细设计 | `src/circuit/cortical_column.*` | ★ 最核心的实现规格 |

---

## 十一、旧 Python 代码处置

> `wuyun/` 目录下的 Python 代码 (spike/, synapse/, neuron/, core/) 是 v0.1-v0.3 的原型实现。
> 作为 **算法参考** 保留在 `_archived/python_prototype/` 中，不再维护。
> 所有数学方程相同，仅实现语言从 Python 迁移到 C++。
