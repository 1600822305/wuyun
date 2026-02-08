# Step 7: Python 绑定 + 可视化仪表盘

> 日期: 2026-02-07
> 状态: ✅ 完成

## 目标

pybind11 暴露 C++ 引擎到 Python, matplotlib 可视化。

## pybind11 绑定 (src/bindings/pywuyun.cpp)

- ✅ SimulationEngine: step/run/add_projection/find_region/build_standard_brain
- ✅ 所有11种BrainRegion子类: CorticalRegion/ThalamicRelay/BG/VTA/LC/DRN/NBM/Hipp/Amyg/CB
- ✅ SpikeRecorder: record→to_raster() 返回numpy数组
- ✅ NeuromodulatorLevels/System: 调质监控
- ✅ build_standard_brain(): 一键构建21区域完整大脑

## 可视化工具 (python/wuyun/viz.py)

- ✅ plot_raster(): 12区域脉冲栅格图, 彩色编码
- ✅ plot_connectivity(): networkx拓扑图, 21节点36边
- ✅ plot_activity_bars(): 区域活动柱状图
- ✅ plot_neuromod_timeline(): DA/NE/5-HT/ACh时间线
- ✅ run_demo(): 一键演示 (构建→刺激→可视化→保存)

## 验证结果

- Python绑定: 21区域/36投射全部可用
- 脉冲栅格: 清晰展示12区域时序活动模式
- 连接图: 层级结构可视化 (视觉→决策→运动→小脑)
- 调质动态: DA/NE/5-HT/ACh基线+刺激响应
- 86 C++测试零回归

## 连接组学布线 — 跳过

> 当前阶段过度工程化

- ~~⬜ JSON配置化~~ → 保持build_standard_brain硬编码，编译时类型检查更安全
- ⬜ 感觉输入接口 (外界→丘脑→皮层) — 移至 Step 9
- ⬜ 运动输出接口 (皮层→BG→丘脑→运动) — 移至后续

## 系统状态

- 21区域 | 3239神经元 | 36投射 | 4调质 | 4学习 | 预测编码 | **Python可视化**
- **86 测试全通过**, 零回归
