# Phase 5: 认知功能 — 预测编码 + 可视化 + 认知任务 + 工作记忆 + 注意力 + Step4补全

> 对应: Step 6(预测编码) / Step 7(Python) / Step 9(认知任务) / Step 10(WM) / Step 4补全 / Step 11(认知验证) / Step 12(注意力)
> 时间: 2026-02-07
> 里程碑: 24 区域 · ~3400 神经元 · 40 投射 · 预测编码 + 工作记忆 + 注意力 · 113 测试全通过

---

## Step 6: 预测编码框架 ✅

> 目标: 皮层层级预测与误差计算 (Rao-Ballard + Friston Free Energy)

**核心机制 (修改 CorticalRegion, 零新文件):**
- ✅ `enable_predictive_coding()` — 可选启用, 向后完全兼容
- ✅ `add_feedback_source(region_id)` — 标记反馈来源, 区分FF/FB
- ✅ 反馈路由: feedback源脉冲→`pc_prediction_buf_`(L2/3 sized), 非feedback→L4 `psp_buffer_`
- ✅ 预测抑制: prediction_buf → L2/3 apical 负注入 (抑制误差单元)
- ✅ 预测误差跟踪: `pc_error_smooth_` 指数平滑

**精度加权 (神经调质驱动):**
- ✅ NE → `pc_precision_sensory_` = ne_gain (0.5~2.0): 高NE信任感觉
- ✅ ACh → `pc_precision_prior_` = max(0.2, 1.0-0.8*ACh): 高ACh不信任预测

**验证结果:**
- **预测抑制涌现**: V1(早期无预测)=226 → V1(晚期有预测)=116 (-49%)
- **NE精度**: V1(NE=0.1)=85 → V1(NE=0.5)=187 → V1(NE=0.9)=235
- **ACh精度**: ACh=0.8→prior=0.36, ACh=0.1→prior=0.92
- **向后兼容**: 无PC时 V1=262, PC无反馈时 V1=262 (完全一致)

**生物学对应:** Mumford 1992 / Rao & Ballard 1999 / Feldman & Friston 2010 / Yu & Dayan 2005

**系统状态:** 21区域 | 36投射 | **86 测试全通过**

---

## Step 7: Python绑定 + 可视化仪表盘 ✅

> 目标: pybind11暴露C++引擎到Python, matplotlib可视化

**pybind11绑定 (src/bindings/pywuyun.cpp):**
- ✅ SimulationEngine: step/run/add_projection/find_region/build_standard_brain
- ✅ 所有11种BrainRegion子类
- ✅ SpikeRecorder: record→to_raster() 返回numpy数组
- ✅ build_standard_brain(): 一键构建21区域完整大脑

**可视化工具 (python/wuyun/viz.py):**
- ✅ plot_raster() / plot_connectivity() / plot_activity_bars() / plot_neuromod_timeline() / run_demo()

---

## Step 9: 认知任务演示 ✅

> 目标: 经典认知范式验证涌现行为，暴露系统能力边界

**Task 1: Go/NoGo (BG动作选择 + ACC冲突监控)**
- ✅ ACC冲突检测涌现: NoGo ACC=1383 > Go ACC=1205 (1.15x)
- ⚠️ M1运动相同: 无训练D1/D2权重→相同输入=相同输出

**Task 2: 情绪处理 (Amygdala威胁 + PFC消退 + VTA DA)**
- ✅ 威胁检测: CS+US Amyg=2644 > CS Amyg=2354
- ✅ DA调制: CS+US VTA=404 > CS VTA=356
- ⚠️ PFC消退失败: 级联激活掩盖ITC→CeA局部抑制

**Task 3: Stroop冲突 (ACC→LC-NE→dlPFC) — 全部通过!**
- ✅ ACC冲突检测: Incong=1416 > Cong=1205
- ✅ dlPFC执行控制: Incong=2450 > Cong=2420
- ✅ NE唤醒: Incong=0.263 > Cong=0.254
- 完整通路涌现: ACC检测冲突→LC-NE升高→dlPFC控制增强

---

## Step 10: 工作记忆 + BG在线学习 ✅

> 目标: dlPFC持续性活动 + DA稳定 + BG门控训练

**工作记忆机制 (修改 CorticalRegion, 零新文件):**
- ✅ `enable_working_memory()` — 可选启用, 向后完全兼容
- ✅ L2/3循环自持: 发放→`wm_recurrent_buf_`→下一步注入L2/3 basal
- ✅ DA稳定: `wm_da_gain_ = 1.0 + 2.0 * DA` (D1受体机制)
- ✅ `wm_persistence()` — 活跃L2/3比例 (0~1)

**验证结果:**
- 工作记忆基础: 刺激期301→持续期109 spikes (活动自持)
- DA持续性: DA=0.1→0, DA=0.3→4, DA=0.6→555 (DA稳定WM)
- WM+BG联合: 延迟期BG=308, dlPFC持续性=1.0 (维持→决策)

**生物学对应:** Goldman-Rakic 1995 / Seamans & Yang 2004 / Frank 2005

**系统状态:** 21区域 | **92 测试全通过**

---

## Step 4 补全 ✅

> 目标: 完成Step 4遗留的低优先级项目

**新增区域 (2个新文件):**
- ✅ `SeptalNucleus` (region/limbic/septal_nucleus.h/cpp) — theta起搏器
  - ACh胆碱能 + GABA节律神经元, theta ~6.7Hz (150ms周期)
- ✅ `MammillaryBody` (region/limbic/mammillary_body.h/cpp) — Papez回路中继
  - 内侧核(25)→外侧核(10)

**Papez回路 (3条新投射):**
- ✅ Hippocampus(Sub) → MammillaryBody → ATN(丘脑前核) → ACC

**Hippocampus扩展 (可选, 向后兼容):**
- ✅ Presubiculum (n_presub=25) + HATA (n_hata=15)

**Amygdala扩展 (可选, 向后兼容):**
- ✅ MeA (n_mea=20) + CoA (n_coa=15) + AB (n_ab=20)

**系统状态:** **24区域** | ~3400神经元 | **40投射** | **100 测试全通过**

---

## Step 11: 认知任务验证 ✅

> 目标: 用WM+BG学习验证高级认知功能

**6项认知任务全部通过:**
1. **训练后Go/NoGo** — D1(高DA)=83 > D1(低DA)=82 > D1(无STDP)=66
2. **延迟匹配任务 (DMTS)** — WM延迟=132 (persist=0.62) vs 无WM=0
3. **Papez回路记忆巩固** — ACC(+Papez)=25 vs ACC(无Papez)=0
4. **情绪增强记忆** — Hipp(+情绪)=11054 vs Hipp(中性)=269 (41x增强)
5. **WM引导BG决策** — BG(+WM)=28 > BG(无WM)=25
6. **反转学习** — D1(低DA后)=11 → D1(高DA后)=47 (+327%)

**生物学对应:** Funahashi 1989 / Frank 2004 / Cools 2009 / McGaugh 2004 / Aggleton & Brown 1999

**系统状态:** **106 测试全通过**

---

## Step 12: 注意力机制 ✅

> 目标: PFC→感觉区top-down选择性增益 + ACh/NE精度调制 + VIP去抑制回路

**实现:**
- `set_attention_gain(float gain)` — PFC可选择性放大/抑制任意皮层区
  - gain > 1.0: 注意 (PSP放大 + VIP驱动)
  - gain = 1.0: 正常 (向后兼容)
  - gain < 1.0: 忽略 (PSP衰减)
- VIP去抑制回路: attention→VIP→SST↓→L2/3 apical去抑制→burst增强

**7项测试全部通过:**
1. 基础增益: V1(忽略)=576 < V1(正常)=861 < V1(注意)=1181
2. 选择性注意: V1(注意)=1181 vs V2(忽略)=623 (1.9x)
3. VIP去抑制: gain=1.0→861, 1.3→1037, 2.0→1348
4. NE精度: V1(NE=0.1)=683 → V1(NE=0.9)=1427

**生物学对应:** Desimone & Duncan 1995 / Letzkus et al. 2015 / Feldman & Friston 2010

**系统状态:** **24区域** | 预测编码 + 工作记忆 + **注意力** | **113 测试全通过**

---

## Phase 5 总结

| 指标 | 数值 |
|------|------|
| 区域 | 24 (Phase 4 的 21 + SeptalNucleus + MammillaryBody + ATN) |
| 神经元 | ~3400 |
| 投射 | 40 |
| 测试 | 113 通过 |
| 新增功能 | 预测编码 · Python绑定 · 可视化 · 工作记忆 · 注意力 · Papez回路 |
| 新增类 | SeptalNucleus · MammillaryBody |
| 认知验证 | Go/NoGo · DMTS · Stroop · 情绪记忆 · 反转学习 · Papez巩固 |