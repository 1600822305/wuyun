# Step 10 系列: 工作记忆 + 认知验证 + 注意力 + 规模扩展

> 日期: 2026-02-07
> 状态: ✅ 完成

## Step 10: 工作记忆 + BG在线学习

> 目标: dlPFC持续性活动 + DA稳定 + BG门控训练

**工作记忆机制 (修改 CorticalRegion, 零新文件):**
- ✅ `enable_working_memory()` — 可选启用, 向后完全兼容
- ✅ L2/3循环自持: 发放→`wm_recurrent_buf_`→下一步注入L2/3 basal
- ✅ DA稳定: `wm_da_gain_ = 1.0 + 2.0 * DA` (D1受体机制)
- ✅ `wm_persistence()` — 活跃L2/3比例 (0~1)

**BG在线学习 (利用已有DA-STDP):**
- ✅ `set_da_source_region(UINT32_MAX)` 禁用自动路由, 手动控制DA
- ✅ 训练: 高DA奖励 → D1(Go)权重LTP
- ✅ 测试: D1(训练后)=61 > D1(未训练)=55

**验证结果:**
- 工作记忆基础: 刺激期301→持续期109 spikes (活动自持)
- DA持续性: DA=0.1→0, DA=0.3→4, DA=0.6→555 (DA稳定WM)
- WM vs 无WM: 4 vs 0 (WM机制有效)
- WM+BG联合: 延迟期BG=308, dlPFC持续性=1.0 (维持→决策)
- 向后兼容: 无WM时行为完全一致

**生物学对应:**
- dlPFC L2/3循环 = 持续性活动 (Goldman-Rakic 1995)
- DA D1 = 增强NMDA循环电流 (Seamans & Yang 2004)
- BG门控 = DA调制的Go/NoGo选择 (Frank 2005)

**系统状态:**
- 21区域 | 3239神经元 | 36投射 | 4调质 | 4学习 | 预测编码 | **工作记忆**
- **92 测试全通过** (86+6), 零回归

---

## Step 11: 认知任务验证

> 目标: 用WM+BG学习验证高级认知功能

**6项认知任务全部通过:**

1. **训练后Go/NoGo** — DA-STDP区分奖励/无奖励刺激
   - D1(高DA训练)=83 > D1(低DA训练)=82 > D1(无STDP)=66
   - 验证BG在线强化学习在多区域回路中的实际效果

2. **延迟匹配任务 (DMTS)** — 工作记忆跨延迟维持样本
   - WM延迟(早)=132 (persist=0.62) vs 无WM延迟=0
   - 验证dlPFC L2/3循环自持 + DA稳定在认知任务中的功能

3. **Papez回路记忆巩固** — Hipp→MB→ATN→ACC增强ACC活动
   - ACC(+Papez)=25 vs ACC(无Papez)=0
   - 验证新增Papez回路的功能性连接

4. **情绪增强记忆** — Amygdala→Hippocampus编码增强
   - Hipp(+情绪)=11054 vs Hipp(中性)=269 (41x增强)
   - 验证BLA→EC情绪标记通路

5. **WM引导BG决策** — dlPFC维持线索→延迟→BG选择
   - BG(+WM)=28 > BG(无WM)=25
   - 验证工作记忆+基底节联合决策

6. **反转学习** — 同一刺激从低DA→高DA训练
   - D1(低DA后)=11 → D1(高DA后)=47 (+327%)
   - 验证DA-STDP双向权重调节

**生物学对应:**
- DMTS = Funahashi (1989) dlPFC延迟活动
- Go/NoGo = Frank (2004) BG D1/D2选择模型
- 反转学习 = Cools (2009) DA灵活性
- 情绪记忆 = McGaugh (2004) 杏仁核-海马情绪标记
- Papez = Aggleton & Brown (1999) 扩展海马系统

**系统状态:**
- **24区域** | ~3400神经元 | **40投射** | 4调质 | 4学习 | 预测编码 | 工作记忆
- **106 测试全通过** (100+6), 零回归

---

## Step 12: 注意力机制

> 目标: PFC→感觉区top-down选择性增益 + ACh/NE精度调制 + VIP去抑制回路

**实现:**
- `set_attention_gain(float gain)` — PFC可选择性放大/抑制任意皮层区
  - gain > 1.0: 注意 (PSP放大 + VIP驱动)
  - gain = 1.0: 正常 (向后兼容)
  - gain < 1.0: 忽略 (PSP衰减)
- VIP去抑制回路: attention→VIP→SST↓→L2/3 apical去抑制→burst增强
  - Letzkus/Pi (2013) disinhibitory attention circuit
- NE sensory精度: `ne_gain = 0.5 + 1.5*NE` 乘以PSP输入
- ACh prior精度: `precision_prior = max(0.2, 1.0 - 0.8*ACh)` 调制预测抑制

**7项测试全部通过:**
1. 基础增益: V1(忽略)=576 < V1(正常)=861 < V1(注意)=1181
2. 选择性注意: V1(注意)=1181 vs V2(忽略)=623 (1.9x)
3. VIP去抑制: gain=1.0→861, 1.3→1037, 2.0→1348
4. 注意力+PC: V1(正常+PC)=861 → V1(注意+PC)=1181
5. ACh精度: V1(ACh=0.1)=562 → V1(ACh=0.8)=643
6. NE精度: V1(NE=0.1)=683 → V1(NE=0.9)=1427
7. 向后兼容: gain=1.0 == 默认

**生物学对应:**
- Desimone & Duncan (1995) 偏置竞争理论
- Letzkus et al. (2015) VIP去抑制注意力回路
- Feldman & Friston (2010) 注意力=精度优化
- Yu & Dayan (2005) ACh=预期不确定性, NE=意外不确定性

**系统状态:**
- **24区域** | ~3400神经元 | **40投射** | 4调质 | 4学习 | 预测编码 | 工作记忆 | **注意力**
- **113 测试全通过** (106+7), 零回归

---

## Step 10: 规模扩展验证

> 目标: 验证系统在大规模下的涌现特性

**核心改动:** `build_standard_brain(scale)` 参数化放大

**规模预设:**
- `scale=1`: ~5,500 神经元 (默认, 向后兼容)
- `scale=3`: ~16,500 神经元
- `scale=8`: ~44,000 神经元

**实现:**
- 所有区域神经元数量乘以 `scale` 因子
- 皮层: L4/L23/L5/L6/PV/SST/VIP 全部缩放
- 皮下: BG D1/D2/GPi/GPe/STN, 丘脑 Relay/TRN
- 边缘: 海马 EC/DG/CA3/CA1/Sub, 杏仁核 LA/BLA/CeA/ITC
- 调质: VTA DA, LC NE, DRN 5-HT, NBM ACh
- 小脑: Granule/Purkinje/DCN/MLI/Golgi
- GW workspace 神经元也缩放

**测试 (test_scale_emergent.cpp, 5测试):**
1. V1规模扩展: 270→810 neurons, spikes 1111→5573 (5.0x, 超线性)
2. BG Go/NoGo: 420 neurons, 训练=4185 > 测试=721
3. CA3模式补全: 180 CA3, 补全比率=1.02 (>>0.30 阈值)
4. 工作记忆: 606 neurons, 刺激=868 spikes
5. 全脑 scale=3: 3768 neurons子集, 0.80 ms/step

**涌现发现:**
- V1 活动超线性增长 (5x vs 3x neurons) → 更密集的网络产生更多协同激活
- CA3 模式补全在大网络中接近完美 (比率1.02 ≈ 100%) → 自联想记忆容量随规模增长
- BG 训练/测试差异显著 (4185 vs 721) → DA-STDP 在大规模下仍然有效

**系统状态:**
- **48区域** | **scale=1: ~5.5k / scale=3: ~16.5k / scale=8: ~44k 神经元** | **~109投射**
- 4调质 | 4学习 | 预测编码 | WM | 注意力 | 内驱力 | GNW | 睡眠/重放 | 感觉输入 | **规模可扩展**
- **154 测试全通过** (149+5), 零回归
