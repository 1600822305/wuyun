# Step 9: 认知任务演示 + 感觉输入接口

> 日期: 2026-02-07
> 状态: ✅ 完成

## 认知任务演示

> 目标: 经典认知范式验证涌现行为，暴露系统能力边界

### Task 1: Go/NoGo (BG动作选择 + ACC冲突监控)

- ✅ ACC冲突检测涌现: NoGo ACC=1383 > Go ACC=1205 (1.15x)
- ⚠️ M1运动相同 (2006=2006): 无训练D1/D2权重→相同输入=相同输出
- 启示: 需要DA-STDP在线训练才能区分Go/NoGo运动响应

### Task 2: 情绪处理 (Amygdala威胁 + PFC消退 + VTA DA)

- ✅ 威胁检测: CS+US Amyg=2644 > CS Amyg=2354
- ✅ DA调制: CS+US VTA=404 > CS VTA=356
- ✅ 海马上下文编码: 5584 spikes
- ⚠️ PFC消退失败: 级联激活掩盖ITC→CeA局部抑制 (单元测试96%有效)
- 启示: 需要选择性PFC→ITC连接，避免全系统级联

### Task 3: Stroop冲突 (ACC→LC-NE→dlPFC) — 全部通过!

- ✅ ACC冲突检测: Incong=1416 > Cong=1205
- ✅ dlPFC执行控制: Incong=2450 > Cong=2420
- ✅ NE唤醒: Incong=0.263 > Cong=0.254
- 完整通路涌现: ACC检测冲突→LC-NE升高→dlPFC控制增强

### 系统能力边界总结

- ✅ 已验证: ACC冲突检测, 威胁→Amyg→VTA DA, NE增益调制, Stroop全通路
- ⚠️ 需改进: BG需训练权重(工作记忆), PFC消退需选择性连接(注意力)
- 生成: 4张可视化图 (go_nogo/fear/stroop/summary)

---

## 感觉输入接口

> 目标: 外界信号→丘脑→皮层的编码通路

**新增文件:** `engine/sensory_input.h/cpp`

### 9a. VisualInput — 视觉编码器

- 图像像素 [0,1] → LGN relay 电流向量
- Center-surround 感受野 (Kuffler 1953): ON cell 中心兴奋/周围抑制, OFF cell 反之
- 预计算权重矩阵: 像素→LGN mapping (grid layout + jitter)
- ON/OFF 通道: 前半LGN=ON, 后半=OFF
- 配置: input_width/height, center/surround_radius, gain, baseline, noise_amp
- `encode(pixels)` → 电流向量, `encode_and_inject(pixels, lgn)` → 直接注入

### 9b. AuditoryInput — 听觉编码器

- 频谱功率 [0,1] → MGN relay 电流向量
- Tonotopic mapping: 频率带→MGN神经元 (低频→前, 高频→后)
- Onset emphasis: 新声音→更强响应 (temporal_decay差分)
- 配置: n_freq_bands, gain, baseline, noise_amp, temporal_decay
- `encode(spectrum)` → 电流向量, `encode_and_inject(spectrum, mgn)` → 直接注入

### pybind11 绑定

- VisualInputConfig + VisualInput (encode, encode_and_inject)
- AuditoryInputConfig + AuditoryInput (encode, encode_and_inject)

### 测试 (test_sensory_input.cpp, 7测试)

1. VisualInput 基础: 8x8→50 LGN, bright_sum=3098 > dark_sum=250
2. Center-surround: spot_max=29 > uniform_max=5 (ON cells)
3. 视觉 E2E: pixels→LGN→V1, bright=1993 >> no_input=222
4. AuditoryInput 基础: low_freq_first=243 > second=30 (tonotopic)
5. Onset 检测: onset=165 > sustained=140
6. 听觉 E2E: spectrum→MGN→A1, spikes=329
7. 多模态: V1=2013 + A1=329 同时活跃

## 系统状态

- **48区域** | **5528神经元** | **~109投射** | 4调质 | 4学习 | 预测编码 | WM | 注意力 | 内驱力 | GNW | 睡眠/重放 | **感觉输入**
- **149 测试全通过** (142+7), 零回归
