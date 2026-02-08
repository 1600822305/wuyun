# Step 54: 大脑拓扑可视化

> 日期: 2026-02-08
> 状态: ✅ 完成
> visualize_brain 工具: 动态提取脑区+投射, 输出文本摘要 + Graphviz DOT。31/31 CTest。

## 动机

进化出了 149 基因的天才大脑, 但看不到它长什么样。
需要一个工具把 25 个脑区和 47 条投射画出来。

## 实现

### SpikeBus 投射访问器

```cpp
// src/core/spike_bus.h — 一行改动
const std::vector<Projection>& projections() const { return projections_; }
```

之前 `projections_` 是 private, 外部只能看 `num_projections()` 计数。
现在可以枚举每条投射的 src/dst/delay/name。

### SimulationEngine 拓扑导出

```cpp
// src/engine/simulation_engine.h
std::string export_dot() const;              // Graphviz DOT 格式
std::string export_topology_summary() const;  // 文本表格
```

`export_dot()` 生成:
- 脑区按子系统分组 (皮层/皮层下/边缘/调质)
- 节点大小反映神经元数量
- 边标注传导延迟
- 深色背景 + 彩色分组

`export_topology_summary()` 生成:
- 区域列表 (编号/名称/神经元数/分类)
- 投射列表 (编号/源→目标/延迟)
- 总计统计

### visualize_brain 工具

```
用法:
  visualize_brain                     # 文本摘要
  visualize_brain --dot brain.dot     # 输出 DOT 到文件
```

工作流: DevGenome → Developer → AgentConfig → ClosedLoopAgent → 提取拓扑

## 输出示例

```
=== Brain Topology (25 regions, 47 projections) ===

  #   Name                Neurons  Type
  --- ------------------- -------- -----------
    0 LGN                     33    cortical
    1 V1                      26    cortical
    ...
   24 Hypothalamus            24    limbic

  Total: 388 neurons, 47 projections
```

DOT 文件可粘贴到 https://dreampuf.github.io/GraphvizOnline/ 渲染。

## 脑区分组

| 分组 | 脑区 | 颜色 |
|------|------|------|
| 皮层 | LGN, V1, V2, V4, IT, dlPFC, M1, FPC, ACC, OFC, vmPFC | 蓝 |
| 皮层下 | BG, MotorThal, SC, NAcc | 绿 |
| 边缘 | Hippocampus, Amygdala, Hypothalamus, PAG, LHb | 橙 |
| 调质 | VTA, SNc, LC, DRN, NBM | 红 (菱形) |

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/core/spike_bus.h` | +projections() 公共访问器 |
| `src/engine/simulation_engine.h/cpp` | +export_dot(), +export_topology_summary() |
| `src/bindings/pywuyun.cpp` | 修复 bus() 重载歧义 |
| `tools/visualize_brain.cpp` | 新工具 |
| `CMakeLists.txt` | +visualize_brain target |
