# Step 50: 基因连接组模型 — 骨架固定 + 皮层涌现

> 日期: 2026-02-09
> 状态: ✅ 完成
> 141 基因, 条形码兼容性决定皮层连接, 31/31 CTest。

## 动机

Step 49 的间接编码有两个致命问题:
1. 只控制参数不控制拓扑 (实质是直接编码换皮)
2. 纯涌现方案让 30 代进化重新发明 3 亿年的 BG/丘脑/杏仁核

正确做法: **骨架固定 (3 亿年进化产物), 皮层涌现 (条形码兼容性)**。

## 生物学基础

- **Barabasi 2019 (Neuron)**: 基因表达谱决定连接概率, 产生 biclique 模体
- **线虫**: 19 种 innexin 蛋白组合 → 7000 突触精确图谱
- **Protocadherin/Neurexin**: 分子条形码 → 兼容的细胞形成突触
- **皮层区域化**: Pax6/Emx2 前后轴梯度决定区域边界

## 设计: 三部分基因组

### 第一部分: 固定回路参数 (~18 基因)

BG/VTA/丘脑/杏仁核/海马/LGN/M1/Hypothalamus 的内部拓扑写死,
基因只控制大小/增益/学习率。49 步成果全保留。

```
bg_size, da_stdp_lr, bg_gain      — BG 乘法增益/侧向抑制/DA-STDP
vta_size, da_phasic_gain          — VTA 内部 RPE
thal_size, thal_gate              — 丘脑 TRN 门控
lgn_gain, lgn_baseline            — LGN 感觉中继
motor_noise                       — M1 群体向量探索
reward_scale                      — Hypothalamus 奖赏缩放
amyg_size, hipp_size              — 杏仁核/海马
homeo_target, homeo_eta           — 稳态可塑性
ne_floor, replay_passes, dev_period — NE/重放/发育期
```

### 第二部分: 可进化皮层 (~50 基因)

5 种皮层类型, 每种有 8 维分子条形码 (基因决定):

```
cortical_barcode[5][8] = 40 基因   — 每种类型的分子身份
cortical_division[5]   = 5 基因    — 每种类型的神经元数量
cortical_inh_frac[5]   = 5 基因    — 每种类型的抑制比
```

功能不预设——从条形码兼容性涌现:
- 与 LGN 高兼容 → 自动成为感觉处理区
- 与 BG 高兼容 → 自动成为决策区
- 与两者都兼容 → 自动成为中间处理层

### 第三部分: 连接兼容性 (~73 基因)

```
w_connect[8][8] = 64 基因          — 哪些分子维度倾向连接
connect_threshold = 1 基因         — 连接稀疏度
cortical_to_bg[8] = 8 基因         — 皮层→BG 接口条形码
```

连接概率 = sigmoid(barcode_i * W * barcode_j - threshold)

## 评估早停

```
阶段 0: 连通性检查 (0 步)
  BFS: LGN→皮层→BG 有连通路径吗?
  不连通 → fitness=-2.0, 直接淘汰

阶段 1: 运动检查 (100 步)
  100 步内有 food/danger 事件吗?
  零事件 → fitness=-1.5, 淘汰 (agent 不动)

阶段 2: 正式评估 (剩余步数)
  Baldwin 适应度: improvement×3 + late_safety + 连通性奖励
```

效果: 垃圾基因组秒淘汰, 只有通过两关的才值得跑完整评估。

## 测试结果

| 测试 | 结果 |
|------|------|
| 基因组结构 | 141 基因 |
| 条形码兼容性 | LGN→均匀皮层 37.5% 连接概率 |
| 不同基因组→不同参数 | V1/dlPFC/lr/noise 全部不同 |
| 发育→完整人脑→运行 | ClosedLoopAgent 50 步无崩溃 |
| 随机鲁棒性 | 5/5 随机基因组成功运行 |
| 连通性 | 默认 5/5, 随机平均 1.3/5 |

## 修改文件

重写:
- `src/genome/dev_genome.h` — v3: 141 基因 (骨架+条形码+兼容性)
- `src/genome/dev_genome.cpp` — 条形码兼容性计算 + 基因操作
- `src/development/developer.h` — to_agent_config + check_connectivity
- `src/development/developer.cpp` — 固定回路参数 + 条形码→区域映射
- `src/genome/dev_evolution.cpp` — 早停评估 (连通性+运动+正式)
- `tests/cpp/test_dev_genome.cpp` — 6 个 v3 测试

## 与 Step 49 的区别

| | Step 49 | Step 50 |
|---|---|---|
| 固定回路 | 5 种通用区域 (玩具) | BG/VTA/丘脑等完整写死 |
| 皮层 | 也是固定的 (只改参数) | 条形码涌现 (Barabasi) |
| 连接拓扑 | build_brain 手工投射 | 条形码兼容性涌现 |
| 49 步成果 | 丢弃 | 全部保留 |
| 每步效率 | ~5% | ~80-90% (固定回路保证) |
| 评估 | 全部跑完 | 早停淘汰垃圾 |

## 系统状态

```
基因连接组 v3: 141 基因
  骨架: BG/VTA/丘脑/杏仁核/海马/LGN/M1/Hypo (固定, 18 基因控制大小/增益)
  皮层: 5 种可进化类型 × 8 维条形码 (50 基因)
  连接: 兼容性矩阵 8×8 + 阈值 + 接口 (73 基因)
  评估: 连通性→运动→正式 三阶段早停
31/31 CTest
```
