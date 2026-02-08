# Step 15 系列: 预测编码闭环 + 环境扩展 + 皮层巩固尝试

> 日期: 2026-02-08
> 状态: ✅ 完成

---

## Step 15: 预测编码基础设施 — dlPFC→V1 反馈通路

> 目标: 实现皮层预测编码 (dlPFC→V1 顶下反馈), 提升 DA-STDP 信用分配精度

### 已实现的基础设施

CorticalRegion 预测编码机制 (Step 4 时已存在):
- `enable_predictive_coding()` — PC 模式
- `add_feedback_source(region_id)` — 标记反馈源
- `receive_spikes()` 将反馈 spikes 路由到 `pc_prediction_buf_`
- 精度加权: NE→sensory precision, ACh→prior precision

Step 15 新增:
- **dlPFC→V1 投射** (delay=3, 可通过 `enable_predictive_coding` 配置开关)
- **拓扑反馈映射**: dlPFC→V1 使用比例映射 (不是 neuron_id % buf_size)
- **窄 fan-out** (=3): 空间特异性抑制/促进
- **促进模式**: 从经典抑制性预测误差改为促进性注意放大 (Bastos et al. 2012)
- **AgentConfig::enable_predictive_coding** 配置标志 (默认 false)

### 修改文件

| 文件 | 变更 |
|------|------|
| `src/region/cortical_region.cpp` | PC 反馈路径: 拓扑映射+窄fan-out, 促进模式 (0.1f gain) |
| `src/engine/closed_loop_agent.h` | 新增 `enable_predictive_coding` 配置标志 |
| `src/engine/closed_loop_agent.cpp` | dlPFC→V1 投射 + PC 启用 + 拓扑注册, 均受配置控制 |

### 5 轮调优实验

| 版本 | 模式 | 增益 | 映射 | fan | improvement | D1范围 |
|------|------|:----:|------|:---:|:-----------:|:------:|
| Step 14 基线 | 无PC | — | — | — | **+0.120** | 0.069 |
| v1 抑制性 | suppressive | -0.5 | modular | 12 | -0.030 | — |
| v2 弱抑制 | suppressive | -0.12 | modular | 12 | -0.019 | — |
| v3 拓扑抑制 | suppressive | -0.12 | topo | 3 | -0.112 | 0.070 |
| **v4 促进** | **facilitative** | **+0.3** | **topo** | **3** | **+0.022** | **0.106** |
| v5 弱促进 | facilitative | +0.1 | topo | 3 | -0.161 | 0.070 |

### 关键发现

1. **经典预测编码 (抑制性) 在小视觉场景无效**
   - 3×3 视野、3 种像素值 = 太少冗余可压缩
   - dlPFC 反馈不是"预测"而是当前感知的延迟回声
   - 抑制性回声压制 V1 L2/3 → 削弱 V1→dlPFC→BG 信号链

2. **促进性注意 (Bastos 2012) 更有前途**
   - 0.3f 增益: D1 权重范围 0.069→0.106 (+54%)
   - 但 improvement 仅 +0.022 (不及无 PC 的 +0.120)
   - 原因: 放大所有刺激 (含无关信号), 稀释信用分配

3. **反馈环路 V1→dlPFC→V1 的固有问题**
   - 正反馈: 促进→更多V1输出→更多dlPFC→更多促进→过驱动
   - 负反馈: 抑制→更少V1输出→更少dlPFC→更少抑制→恢复→振荡
   - 两种模式都增加系统方差, 不利于稳定学习

4. **正确的启用时机**
   - 环境扩大后 (更大视野, 更多刺激种类) PC 将变得有用
   - 需要学习预测机制 (dlPFC L6 学习预测 V1 模式)
   - 当前: 基础设施就绪, 一个配置标志即可启用

### 决策: 默认禁用, 保留基础设施

```
enable_predictive_coding = false  // 默认不启用
// 启用: config.enable_predictive_coding = true
// 投射: dlPFC → V1 (delay=3)
// 模式: 促进性注意 (0.1f gain, 拓扑映射, fan=3)
// 待启用条件: 视野 > 3×3, 刺激种类 > 3, 或有学习预测机制
```

### 回归测试: 29/29 CTest 全通过 (性能恢复到 Step 14 水平)

---

## Step 15-B: 环境扩展 + 大环境 PC 验证

> 目标: 扩大 GridWorld 环境, 验证预测编码在更丰富视觉场景中的效果

### 环境扩展实现

| 特性 | 修改 |
|------|------|
| `GridWorldConfig::vision_radius` | 新增 (默认1=3×3, 2=5×5, 3=7×7) |
| `GridWorld::observe()` | 从硬编码 3×3 改为参数化 (2r+1)×(2r+1) |
| `ClosedLoopAgent` 构造 | 自动从 world_config 推算 vision_width/height |
| LGN 缩放 | ~3 LGN neurons/pixel (9 pixels→30 LGN, 25 pixels→75 LGN) |
| V1 缩放 | vis_scale = n_pixels/9 (线性) |
| dlPFC 缩放 | sqrt(vis_scale) (平方根, 防止过度膨胀) |

### 大环境 PC 对比实验

```
环境: 15×15 grid, 5 food, 4 danger, 5×5 视野 (25 pixels)
脑: V1=447, dlPFC=223, LGN=100 neurons (自动缩放)
训练: 1000 warmup + 4×1000 epochs
```

| 配置 | early safety | late safety | improvement | 5k food | 5k danger |
|------|:-----------:|:-----------:|:-----------:|:-------:|:---------:|
| No PC | 0.401 | 0.122 | -0.279 | 36 | 37 |
| **PC ON** | 0.283 | 0.125 | **-0.158** | 28 | **22** |
| **PC 优势** | | | **+0.121** | -8 | **-15** |

### 关键发现

1. **PC 在大环境中提供 +0.121 improvement 优势** — 与小环境 (3×3) 相反!
2. **PC 显著减少后期 danger**: 22 vs 37 (降低 40%)
3. **大环境本身太难**: 两个 agent 都退化 (15×15 太稀疏, 随机游走效率低)
4. **PC 的价值在于减缓退化**: 通过注意力反馈维持对视觉特征的敏感性

### Step 15 完整结论

```
预测编码效果与环境复杂度正相关:
  - 3×3 视野 (9 pixels): PC 有害 (反馈=噪声)
  - 5×5 视野 (25 pixels): PC 有益 (+0.121 improvement, -40% danger)
  - 预测: 更大视野 (7×7+) PC 优势会更明显

默认策略: enable_predictive_coding = false (小环境)
大环境:   enable_predictive_coding = true  (视野 ≥ 5×5)
```

### 回归测试: 29/29 CTest 全通过 (5/5 learning_curve tests)

---

## Step 15-C: 皮层巩固尝试 (Awake SWR → 皮层 STDP)

> 目标: 让 SWR 重放同时巩固 V1→dlPFC 皮层表征 (学习回路第⑨步)

### 实现

- `SpikeSnapshot::sensory_events` — 录制 V1 spikes
- `CorticalRegion::replay_cortical_step()` — 轻量回放步 (PSP→L4 + column step + STDP, 不提交 spikes)
- `capture_dlpfc_spikes()` 同时录制 V1 fired patterns
- `run_awake_replay()` 回放时 V1 spikes → dlPFC receive_spikes → replay_cortical_step

### 实验结果

| 方案 | improvement | late safety | 问题 |
|------|:-----------:|:-----------:|------|
| **BG-only (基线)** | **+0.120** | **0.667** | — |
| replay_cortical_step | +0.034 (-72%) | 0.527 | L4 fires, L23 无 WM 支撑 → LTD 主导 |
| PSP priming only | +0.053 (-56%) | 0.600 | PSP 残留污染下一步真实视觉输入 |

### 结论

**Awake SWR 期间的皮层巩固不可行**:
1. 回放时 L4 被 V1 spikes 驱动但 L23 缺乏 WM/attention 辅助 → STDP LTD 主导 → 削弱已学表征
2. 即使仅注入 PSP (不步进), 残留电流也污染下一步的在线视觉处理

**生物学解释**: awake SWR 主要巩固纹状体动作值 (Jadhav 2012)。
皮层表征巩固发生在 **NREM 睡眠** 期间: 慢波 up/down 状态控制全脑同步重激活,
不干扰在线处理。未来实现 NREM 睡眠巩固时可直接使用已建基础设施。

### 保留的基础设施 (NREM 巩固就绪)

```
SpikeSnapshot::sensory_events      — V1 spikes 录制 ✅
CorticalRegion::replay_cortical_step() — 轻量回放方法 ✅
capture_dlpfc_spikes() 同时录制 V1 — 双通道录制 ✅
→ 未来 NREM 睡眠巩固可直接调用, 无需额外开发
```

### 回归测试: 29/29 CTest 全通过, 基线完全恢复

### 系统状态

```
48区域 · 自适应神经元数 · ~109投射 · 179测试 · 29 CTest suites
新增: V1 spike 录制, replay_cortical_step 基础设施 (deferred to NREM)
学习维持: improvement +0.120, late safety 0.667 (与 Step 14 一致)
学习回路: ①-⑧ 完整, ⑨皮层巩固需NREM, ⑩PC就绪
```
