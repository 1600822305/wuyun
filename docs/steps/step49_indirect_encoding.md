# Step 49: 间接编码 — 神经发育基因组 (DevGenome)

> 日期: 2026-02-09
> 状态: ✅ 完成 (Phase A+B+C 全部实现)
> 124 个发育规则基因, 轴突导向连接涌现, 31/31 CTest。

## 动机

直接编码的根本缺陷:
- 23 个浮点参数 → AgentConfig → 手工 build_brain() = 参数网格搜索, 不是进化
- 大脑结构是人手动设计的 (64 区域, 139 投射), 不是"长出来的"
- 基因只控制旋钮, 不控制结构 — 违反 00 设计原则的"涌现"精神

间接编码: 基因编码"大脑怎么长出来"的发育规则, 大脑结构从发育过程涌现。

## 生物学基础

- **基因组瓶颈** (Zador 2019): 20,000 基因 → 100 万亿突触, 压缩比 50 亿:1
- **NDP** (Nature 2023): 从单细胞通过局部通信生长功能网络
- **MorphoNAS** (2025): 形态发生素 + 反应扩散 → 自组织网络
- **BCNNM 框架**: 增殖/分化/迁移/轴突寻路/突触形成, 可达 100 万细胞

## 实现: 三阶段发育模拟

```
DevGenome (124基因) → 增殖 → 空间分配 → 轴突导向 → 组装 → 关键期修剪 → SimulationEngine
```

### Phase A: 增殖 + 连接 (~36 基因)

**增殖**: 5 种区域类型 (感觉/运动/前额叶/皮层下/调质), 每种有分裂轮数基因:
```
division_rounds[SENSORY] = 5  →  2^5 = 32 个感觉神经元
division_rounds[MOTOR]   = 4  →  2^4 = 16 个运动神经元
...
总计: 涌现出 88 个神经元 (不是手动指定)
```

**空间分配**: 祖细胞按高斯分布散布在 2D 空间:
- 感觉区: 后部 (y≈0.85)
- 运动区: 前上部 (y≈0.30)
- 前额叶: 最前部 (y≈0.10)
- 皮层下: 中部 (y≈0.50)
- 调质核: 中心 (y≈0.60)

**连接**: 25 个跨类型连接概率基因 (5×5 矩阵) + 距离衰减。

### Phase B: 导向分子 + 分化 (+82 基因)

**导向分子场**: 8 种化学梯度, 每种 5 个参数 (cx, cy, σ, amplitude, attract/repel):
- 生物学: Netrin(吸引), Slit(排斥), Ephrin(拓扑), Semaphorin(层特异)
- 浓度 = amplitude × exp(-d²/2σ²)
- 梯度指向浓度增加方向

**轴突导向**: 每个细胞伸出轴突, 沿导向分子梯度生长 (10步 × 0.05 步长):
```cpp
for (step = 0; step < 10; step++) {
    field.compute_guidance_force(ax, ay, cell.receptors, fx, fy);
    ax += (fx + noise) * step_size;
    ay += (fy + noise) * step_size;
    // 检查附近细胞, 形成突触
}
```

**受体表达**: 40 基因 (5 区域类型 × 8 分子), 决定每种细胞对哪些导向分子敏感。

**分化梯度**: DA/NMDA 受体密度随前后轴变化 (da_gradient, nmda_gradient)。

### Phase C: 修剪 + 关键期 (+6 基因)

**关键期**: 发育完成后运行自发活动 (critical_period 步), 让稳态可塑性调整权重。
- pruning_threshold: 低活动突触修剪阈值
- spontaneous_rate: 自发活动强度

## 验证结果

| 测试 | 结果 |
|------|------|
| 增殖: 基因→神经元数量 | 88 个 (32+16+16+16+8), 精确匹配 |
| 导向连接: 化学梯度→突触 | 820 个突触, 19 条跨区域连接涌现 |
| 运行: 发育大脑步进 | 100 步无崩溃 |
| 变异: 不同基因→不同大脑 | 132 vs 160 神经元 |
| 交叉: 有性繁殖 | 子代正确继承父母基因 |
| 进化: 5 代选择 | 适应度 +0.66, 最佳 300 神经元 |

## 与直接编码对比

| | 直接编码 (v1) | 间接编码 (v49) |
|---|---|---|
| 基因数 | 23 | **124** |
| 编码内容 | AgentConfig 浮点参数 | 发育规则 (增殖/导向/分化/修剪) |
| 大脑结构 | 手工 build_brain() | **develop() 涌现** |
| 连接方式 | 手动 add_projection() | **导向分子梯度驱动** |
| 神经元数 | 固定 ~252 | **基因控制 (32~300+)** |
| 扩展性 | 参数随区域线性增长 | **固定 124 基因, 任意规模** |
| 进化搜索 | 参数网格搜索 | **搜索发育规则** |

## 新增文件

- `src/genome/dev_genome.h` — 发育基因组结构 (124 基因)
- `src/genome/dev_genome.cpp` — 基因操作 (随机/变异/交叉/序列化)
- `src/development/guidance.h` — 导向分子场 (8 种化学梯度)
- `src/development/developer.h` — 发育模拟器接口
- `src/development/developer.cpp` — 5 阶段发育过程
- `tests/cpp/test_dev_genome.cpp` — 6 个验证测试

修改:
- `src/CMakeLists.txt` — 加入 dev_genome.cpp + developer.cpp
- `tests/cpp/CMakeLists.txt` — 加入 test_dev_genome

## 关键代码路径

```
DevGenome::all_genes() → 124 个 Gene 指针
Developer::develop(genome, vision_pixels, seed)
  → proliferate()    — 祖细胞分裂, 按位置+类型分配
  → assign_regions() — 按 region_type 分组
  → form_connections() — 构建 GuidanceField, 模拟轴突生长, 形成突触
  → assemble()       — NeuralCell → BrainRegion → SimulationEngine
  → 关键期           — 自发活动 critical_period 步
  → return engine
```

## 下一步

间接编码基础设施已就绪。下一步是将 DevGenome 集成到 ClosedLoopAgent,
用进化搜索发育规则来"进化出"能解迷宫的大脑——不再手工设计脑区,
让进化自己发现最优的大脑结构。

## 系统状态

```
间接编码: 124 基因 → 发育模拟 → 大脑涌现
发育管线: 增殖 → 空间分配 → 轴突导向(8种化学梯度) → 组装 → 关键期修剪
验证: 6/6 测试通过, 31/31 CTest
进化验证: 5 代后适应度 +0.66, 最佳 300 神经元
```
