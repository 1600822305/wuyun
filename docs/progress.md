# 悟韵 (WuYun) v3 开发进度

> 上次更新: 2026-02-07
> 仓库: https://github.com/1600822305/wuyun (agi3 = main, agi2 = v2 分支)

---

## 已完成

### 设计文档 (v0.3, 2026-02-07)
- ✅ 四份设计文档校对完毕 (00设计原则 / 01脑区计划 / 02神经元系统 / 03项目结构)
- ✅ 01脑区文档按真实解剖分区重组 (前脑→端脑/间脑 | 中脑 | 后脑)
- ✅ 01脑区文档升级至 **NextBrain 混合方案**: 皮层(功能分区) + 皮层下(NextBrain, Nature 2025)
  - 丘脑 16核团 | 杏仁核 8核团 | 海马 7亚区 | 中脑 11核团 | 小脑 8小叶区
  - 总分区: **~97区** (皮层25 + 皮层下72)，每个皮层下核团标注 FreeSurfer 编号

### P0 Python 原型 (2026-02-07) → 已归档为算法参考
- ✅ `wuyun/spike/` — SpikeType、Spike、SpikeBus、OscillationClock
- ✅ `wuyun/synapse/` — SynapseBase、STP、STDP、DA-STDP、抑制性STDP、稳态可塑性
- ✅ `wuyun/neuron/` — NeuronBase(16种参数预设)、双区室Compartment
- ✅ `wuyun/core/` — 向量化 NeuronPopulation、SynapseGroup
- → 归档至 `_archived/python_prototype/`，数学方程不变，C++ 重新实现

### 架构决策: C++ 核心引擎 (2026-02-07)
- ✅ 决定: 仿真核心用 **C++17**，Python 只做配置/实验/可视化
- ✅ 理由: Python 100万神经元 ~10秒/step 不可用; C++ ~10-50ms/step 接近实时; 未来迁移 CUDA 容易
- ✅ 03项目结构文档重写为 C++ core + pybind11 + Python 实验层 (v0.4)
- ✅ 技术栈: C++17 / CMake / pybind11 / Google Test / SoA布局 / CSR稀疏 / 事件驱动

---

## 待开始

> 神经元+突触 → 皮层柱 → 核心回路 → 扩展脑区 → 布线 → 涌现

### Step 1: C++ 工程骨架 + 基础验证 ✅ (2026-02-07)
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
- **Benchmark (Release, 单线程, CPU):**
  | 规模 | 每步耗时 | 吞吐量 |
  |------|---------|--------|
  | 10K 神经元 | 146 μs | 68M/s |
  | 100K 神经元 | 1.5 ms | 67M/s |
  | 1M 神经元 | 26 ms | 38M/s |
  | 100K 突触 | 75 μs | — |
  | 1M 突触 | 1.2 ms | — |

### Step 2: 皮层柱模板 (C++) ✅ (2026-02-07)
- ✅ src/circuit/cortical_column.h/cpp — 6层通用模板 (L4/L2-3/L5/L6 + PV/SST/VIP)
- ✅ PV+/SST+/VIP 抑制性微环路内嵌于 CorticalColumn (无需独立文件)
- ✅ 10 组柱内突触: L4→L23, L23→L5, L5→L6, L6→L4, Exc→PV/SST/VIP, PV→Soma, SST→Apical, VIP→SST
- ✅ 修复突触电流符号 (V-E → E-V), AMPA 产生正确去极化电流
- ✅ 6 测试全通过 (中文详细输出):
  - 构造验证: 540 神经元, 19352 突触
  - 沉默: 无输入零发放
  - **纯前馈→REGULAR**: L4→L2/3, regular=20, burst=0 ✓ (预测误差)
  - **前馈+反馈→BURST**: regular=92, burst=343, drive=628 ✓ (预测匹配)
  - **注意力门控**: VIP→SST去抑制 (架构正确, 效果待调参)
  - **L5驱动**: 459发放, 452 burst驱动 ✓ (皮层下输出)

### Step 3: 核心回路 — 最小可工作大脑
> 目标: 感觉→认知→动作的最短通路能跑通
**3a. 感觉-认知通路 (皮层+丘脑核心):**
- ⬜ 丘脑感觉中继 (T-01 LGN, T-02 MGN, T-03 VPL, T-16 TRN)
- ⬜ V1 (S-01) + A1 (S-02) — 皮层柱实例 + 丘脑输入
- ⬜ dlPFC (A-01) — 工作记忆参数
**3b. 动作选择通路 (基底节):**
- ⬜ 纹状体 D1/D2 (BG-01~02) + GPi/GPe (BG-03~04) + STN
- ⬜ 丘脑运动中继 (T-07 VLa, T-08 VLp)
- ⬜ M1 (M-01) — 运动输出
**3c. 奖励信号 (DA系统):**
- ⬜ VTA (MB-03) + SNc (MB-04) → 纹状体/PFC DA投射
- ⬜ LHb (ET-01) → VTA 负RPE

### Step 4: 记忆与情感回路
> 目标: 能学习、能记住、能赋予情感价值
**4a. 海马记忆系统:**
- ⬜ EC→DG→CA3→CA1→Subiculum (H-01~05)
- ⬜ 前下托 + HATA (H-06~07)
- ⬜ 隔核 theta 起搏 (SP-01~02)
**4b. 杏仁核情感系统 (NextBrain 8核):**
- ⬜ La→BLA→CeA 核心通路 (AM-01, AM-02, AM-05)
- ⬜ ITC门控 + MeA/CoA (AM-04, AM-06~08) — 恐惧消退
**4c. Papez记忆回路:**
- ⬜ 乳头体 (HY-06) → 丘脑前核 (T-10 AV) → ACC (A-05)

### Step 5: 扩展皮层 + 丘脑高级核群
> 目标: 完整的感觉层级 + 联合皮层 + 丘脑全部16核
**5a. 感觉皮层层级:**
- ⬜ V2/V4/IT/S1/S2/味觉/嗅觉 (S-03~09)
**5b. 联合皮层:**
- ⬜ OFC/vmPFC/ACC/PCC/PPC/TPJ/Broca/Wernicke/FEF/岛叶 (A-02~12)
- ⬜ PMC/SMA/FEF (M-02~04)
**5c. 丘脑联合/高级核群:**
- ⬜ 运动中继: VA/VAmc (T-05~06), MD (T-09)
- ⬜ 联合核群: LD/LP/Pulvinar (T-11~13)
- ⬜ 板内核群: CeM + CL/CM/Pf (T-14~15) — 觉醒/意识

### Step 6: 调质系统 + 内驱力 + 小脑
> 目标: 全局调制 + 内部状态 + 运动/认知预测
**6a. 完整调质系统 (NextBrain脑干核团):**
- ⬜ 5-HT: DR (MB-05) + MnR (MB-08)
- ⬜ NE: LC (HB-01)
- ⬜ ACh: PTg (MB-09) + LDTg (MB-10) + BF (BF-01)
**6b. 下丘脑内驱力:**
- ⬜ 睡眠开关: VLPO (HY-07) ⟷ orexin (HY-02)
- ⬜ 应激: PVN (HY-04) → HPA轴
- ⬜ 节律: SCN (HY-01)
- ⬜ 摄食/饱腹: LH (HY-02) / VMH (HY-03)
**6c. 小脑 (NextBrain 8小叶区):**
- ⬜ 运动小脑: 前叶+VIIIa/b+绒球 (CB-01, CB-06~07)
- ⬜ 认知小脑: Crus I/II + VIIb (CB-03~05) — 工作记忆/推理
- ⬜ 蚓部 (CB-08) + 过渡区 (CB-02)

### Step 7: 连接组学布线
- ⬜ 脑区间连接矩阵 (configs/connectome/) — 按01文档§3
- ⬜ 感觉输入接口 (外界→丘脑→皮层)
- ⬜ 运动输出接口 (皮层→BG→丘脑→运动)

### Step 8: 全脑功能 (涌现)
- ⬜ 睡眠/巩固: NREM慢波 + 海马重放 + PnO (MB-11) REM
- ⬜ 注意力: TRN门控 + ACh + 上丘 (MB-01~02)
- ⬜ 发育/关键期: 连接修剪 + PV+成熟

---

## 架构备忘

```
C++ 工程骨架 + 基础验证  ← Step 1: CMake + core/ + pybind11
     ↓
皮层柱模板 (6层, C++)    ← Step 2: circuit/cortical_column
     ↓
核心回路 (最小大脑)       ← Step 3: V1+PFC+BG+丘脑4核+DA
     ↓
记忆+情感                 ← Step 4: 海马7区+杏仁核8核+Papez
     ↓
扩展皮层+丘脑             ← Step 5: 25皮层区+丘脑全16核
     ↓
调质+内驱力+小脑          ← Step 6: 5-HT/NE/ACh+下丘脑+小脑8叶
     ↓
连接组学布线              ← Step 7
     ↓
全脑功能涌现              ← Step 8: 睡眠/注意力/发育

技术栈: C++17 核心引擎 + pybind11 → Python 实验/可视化
Python 原型 (wuyun/) → _archived/ 算法参考
```
