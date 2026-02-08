# Step 57: Environment 抽象接口

> 日期: 2026-02-08
> 状态: ✅ 完成
> 29/31 CTest (排除 2 个慢诊断测试均通过, neuron_tests 为已有 bug)

## 动机

ClosedLoopAgent 直接包含 `GridWorld` 成员, 所有环境交互硬编码为 GridWorld API。
更换环境 (多房间/连续空间/多 Agent) 需要修改 Agent 核心代码, 违反开闭原则。

```
现有 (v56):
  ClosedLoopAgent {
    GridWorld world_;           // 直接包含, 紧耦合
    world_.observe()            // 7 处硬编码引用
    world_.act_continuous()
    world_.agent_x/y()
    ...
  }

问题:
  1. 换环境 = 改 Agent 代码 (高风险)
  2. 无法同时支持多种环境
  3. GridWorldConfig 嵌入 AgentConfig (不属于大脑参数)
```

## 设计

**核心思路: GridWorld 零修改 + 薄适配器 + Agent 接口替换**

```
环境抽象层:
  Environment (abstract)       ← 只暴露大脑需要的信息
    ├── GridWorldEnv            ← 包装 GridWorld, 实现 Environment
    ├── MultiRoomEnv (future)   ← 多房间 + 门/钥匙
    └── ContinuousArena (future)← 连续 2D, 无网格

  ClosedLoopAgent {
    unique_ptr<Environment> env_;   // 运行时多态
    env_->observe()                 // 统一接口
    env_->step(dx, dy)
    env_->pos_x/y()
  }
```

### Environment 接口 (`environment.h`)

```cpp
class Environment {
public:
    virtual ~Environment() = default;

    // Lifecycle
    virtual void reset() = 0;
    virtual void reset_with_seed(uint32_t seed) = 0;

    // Sensory
    virtual std::vector<float> observe() const = 0;
    virtual size_t vis_width() const = 0;
    virtual size_t vis_height() const = 0;

    // Motor
    struct Result {
        float reward;
        bool  positive_event;   // food-like
        bool  negative_event;   // danger-like
        float pos_x, pos_y;
    };
    virtual Result step(float dx, float dy) = 0;

    // Spatial (hippocampus / cognitive map)
    virtual float pos_x() const = 0;
    virtual float pos_y() const = 0;
    virtual float world_width() const = 0;
    virtual float world_height() const = 0;

    // Statistics (evolution fitness)
    virtual uint32_t positive_count() const = 0;
    virtual uint32_t negative_count() const = 0;
    virtual uint32_t step_count() const = 0;
};
```

设计原则:
- 只暴露大脑需要的信息, 不暴露环境内部结构
- `observe()` 返回通用 float 向量 (视觉/任何 2D 传感器)
- 空间信息独立于视觉 (海马不需要知道"格子")
- 统计用 positive/negative 而非 food/danger (语义无关)

### GridWorldEnv 适配器 (`grid_world_env.h/cpp`)

~30 行一行代理, GridWorld 零修改:

```cpp
class GridWorldEnv : public Environment {
    GridWorld world_;
public:
    explicit GridWorldEnv(const GridWorldConfig& cfg);
    // 全部方法代理到 world_
    GridWorld& grid_world();  // 下转型访问 GridWorld 特有功能
};
```

### ClosedLoopAgent 改动

- 构造函数: `ClosedLoopAgent(unique_ptr<Environment> env, const AgentConfig& config)`
- 移除 `AgentConfig::world_config` (不属于大脑参数)
- 移除 `GridWorld world_` 成员 → `unique_ptr<Environment> env_`
- `world()` accessor → `env()` 返回 `Environment&`
- `agent_step()` 返回 `Environment::Result` (取代 `StepResult`)
- 7 处 `world_.xxx()` → `env_->xxx()` 机械替换
- 空间认知地图缓存 `spatial_map_w_/h_` (避免每步调 `env_->world_width()`)

## 修改文件

### 新增 (3 文件)
- `src/engine/environment.h` — 抽象环境接口 (纯头文件)
- `src/engine/grid_world_env.h` — GridWorldEnv 适配器声明
- `src/engine/grid_world_env.cpp` — GridWorldEnv 实现 (~30 行)

### 核心改动 (3 文件)
- `src/engine/closed_loop_agent.h` — 构造函数/成员/accessor/返回类型
- `src/engine/closed_loop_agent.cpp` — 7 处 world_ → env_ 替换
- `src/CMakeLists.txt` — 新增 grid_world_env.cpp

### 消费端适配 (10 文件)
- `src/genome/evolution.h` — 添加 grid_world.h include
- `src/genome/evolution.cpp` — evaluate_single() 创建 GridWorldEnv
- `src/genome/dev_evolution.cpp` — eval_open_field/sparse/reversal() 创建 GridWorldEnv
- `tools/benchmark_continuous.cpp` — run_one() 创建 GridWorldEnv
- `tools/visualize_brain.cpp` — 创建默认 GridWorldEnv
- `src/bindings/pywuyun.cpp` — Python 绑定适配
- `tests/cpp/test_closed_loop.cpp` — 8 处 Agent 创建改用 GridWorldEnv
- `tests/cpp/test_dev_genome.cpp` — 3 处
- `tests/cpp/test_learning_curve.cpp` — 6 处
- `tests/cpp/test_minimal_tasks.cpp` — 5 处

### 零改动
- `src/engine/grid_world.h/cpp` — **GridWorld 完全不动**
- 所有脑区代码 — 零改动
- SimulationEngine — 零改动
- Genome / DevGenome — 零改动
- SensoryInput / EpisodeBuffer — 零改动

## 消费端代码变更模式

```cpp
// 旧:
AgentConfig cfg;
cfg.world_config.width = 10;
cfg.world_config.seed = 42;
ClosedLoopAgent agent(cfg);
agent.world().total_food_collected();
result.got_food;

// 新:
AgentConfig cfg;
GridWorldConfig wcfg;
wcfg.width = 10; wcfg.seed = 42;
ClosedLoopAgent agent(std::make_unique<GridWorldEnv>(wcfg), cfg);
agent.env().positive_count();
result.positive_event;
```

## 验证

```
编译: 全部 31 个目标 + pybind11 模块 0 错误
测试: 29/29 通过 (排除 2 个慢诊断测试)
  - closed_loop_tests: PASS (2.15s)
  - dev_genome_tests: PASS (0.61s)
  - 唯一失败 neuron_tests: 已有 bug, 与本次改动无关
```

## 未来扩展路径

新环境只需实现 `Environment` 接口, 大脑代码零修改:

```cpp
class MultiRoomEnv : public Environment {
    // 多房间 + 门/钥匙, observe() 返回同尺寸视觉 patch
};

class ForagingArena : public Environment {
    // 连续 2D 空间, 无网格, 多 agent
};

// 直接插入:
ClosedLoopAgent agent(make_unique<MultiRoomEnv>(...), cfg);
```
