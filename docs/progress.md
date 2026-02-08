# 悟韵 (WuYun) v3 开发进度

> 上次更新: 2026-02-09
> 仓库: https://github.com/1600822305/wuyun (agi3 = main, agi2 = v2 分支)

---

## 设计阶段 (2026-02-07)

### 设计文档 (v0.3)
- ✅ 四份设计文档校对完毕 (00设计原则 / 01脑区计划 / 02神经元系统 / 03项目结构)
- ✅ 01脑区文档升级至 **NextBrain 混合方案**: ~97区 (皮层25 + 皮层下72), FreeSurfer 编号

### P0 Python 原型 → 已归档
- ✅ spike/synapse/neuron/core 四模块原型 → `_archived/python_prototype/`

### 架构决策: C++ 核心引擎
- ✅ C++17 / CMake / pybind11 / Google Test / SoA布局 / CSR稀疏 / 事件驱动
- ✅ 理由: Python 100万神经元 ~10秒/step → C++ ~10-50ms/step

---

## 基础构建阶段 (Step 1-11)

### Step 1: C++ 工程骨架 + 基础验证 ✅ (2026-02-07)
> 详细文档: [steps/step01_foundation.md](steps/step01_foundation.md)

SoA 双区室 AdLIF+ 神经元群体 + CSR 稀疏突触组 + STDP/STP 可塑性。
性能: 10K 神经元 146μs/step, 1M 神经元 26ms/step。9 测试通过。

### Step 2: 皮层柱模板 + 地基补全 ✅ (2026-02-07)
> 详细文档: [steps/step02_cortical_column.md](steps/step02_cortical_column.md)

6 层皮层柱 (L4/L2-3/L5/L6/PV/SST/VIP, 18 组突触)。
NMDA Mg²⁺ 阻断 + SpikeBus 跨区域通信 + DA-STDP + 神经调质框架 + 8 种神经元预设。21 测试通过。

### Step 3: 最小可工作大脑 ✅ (2026-02-07)
> 详细文档: [steps/step03_minimal_brain.md](steps/step03_minimal_brain.md)

LGN→V1→dlPFC→BG→MotorThal→M1 完整感觉-决策-运动通路 + VTA DA 奖励信号。
7 区域, 906 神经元。BG 随机映射替代硬编码 (反作弊)。26 测试通过。

### Step 4 系列: 海马 + 杏仁核 + 学习系统 ✅ (2026-02-07)
> 详细文档: [steps/step04_memory_learning.md](steps/step04_memory_learning.md)

- 海马: EC→DG→CA3→CA1→Sub 三突触通路, CA3 自联想 (30%线索→100%补全), DG 稀疏编码
- 杏仁核: La→BLA→CeA 恐惧通路, ITC 消退门控 (96% 有效)
- 学习: STDP 自组织 (86n 发展选择性) + BG DA-STDP (动作选择+反转学习)
- 补全: SeptalNucleus (theta 6.7Hz) + MammillaryBody (Papez 回路) + Hipp/Amyg 扩展
- **里程碑: 从"通电的硬件"变为"能学习的系统"。** 24区域, 40投射, 100 测试通过。

### 修复: V1→dlPFC→BG 信号衰减 ✅ (2026-02-07)

fan-out 3→30%×L4, PSP 25f→35f。全链路打通: V1=7656→dlPFC=4770→BG=3408→M1=1120。57 测试通过。

### Step 5 系列: 扩展脑区 + 调质 + 小脑 + 决策 + 丘脑 ✅ (2026-02-07)
> 详细文档: [steps/step05_extended_brain.md](steps/step05_extended_brain.md)

- 调质广播: LC(NE) + DRN(5-HT) + NBM(ACh), Yerkes-Dodson 倒U型涌现
- 视觉层级: V2(纹理)→V4(形状)→IT(物体), STDP 习惯化涌现
- 小脑: GrC/PC/DCN/MLI/Golgi, CF-LTD 第4种学习规则
- 决策: OFC(价值)/vmPFC(情绪)/ACC(冲突) + MT/PPC 背侧视觉 (双流 what+where)
- 扩展: S1/S2/A1/Gustatory/Piriform + Broca/Wernicke + PMC/SMA/FEF + 9丘脑核
- 46区域, 5409n, ~90投射。121 测试通过。

### Step 6 系列: 预测编码 + 下丘脑 + 意识(GNW) ✅ (2026-02-07)
> 详细文档: [steps/step06_predictive_drive.md](steps/step06_predictive_drive.md)

- 预测编码: L6→L2/3 预测抑制, NE/ACh 精度加权 (Rao-Ballard + Friston)
- 下丘脑: 6核团 (SCN昼夜/VLPO睡眠/Orexin觉醒/PVN应激/LH饥饿/VMH饱腹), flip-flop
- GNW: 30n workspace, 竞争→点火→广播, 9竞争+2广播投射
- 48区域, 5528n, ~109投射。135 测试通过。

### Step 7: Python 绑定 + 可视化 ✅ (2026-02-07)
> 详细文档: [steps/step07_python_bindingsvisual.md](steps/step07_python_bindingsvisual.md)

pybind11 暴露全部 11 种 BrainRegion 子类 + SpikeRecorder + build_standard_brain()。
matplotlib 可视化: raster/connectivity/activity/neuromod 4 种图表 + run_demo()。86 测试通过。

### Step 8: 睡眠 / 海马重放 / 记忆巩固 ✅ (2026-02-07)
> 详细文档: [steps/step08_sleep_replay.md](steps/step08_sleep_replay.md)

- SWR 重放: CA3 bias+jitter→自联想补全→SWR burst (关键: CA3 权重即是记忆, 无需显式存储)
- NREM 慢波: Up/Down ~1Hz, down state 抑制发放 (-8 pA)
- 巩固通路: SWR→CA1→SpikeBus→皮层 up state→STDP 增强 = 系统巩固
- 142 测试通过。

### Step 9: 认知任务 + 感觉输入 ✅ (2026-02-07)
> 详细文档: [steps/step09_sensory_cognitive.md](steps/step09_sensory_cognitive.md)

- 认知范式: Go/NoGo (ACC冲突涌现), 情绪处理 (威胁+DA调制), Stroop (ACC→LC-NE→dlPFC 全通路)
- VisualInput: center-surround ON/OFF, 像素→LGN 电流
- AuditoryInput: tonotopic + onset emphasis
- 149 测试通过。

### Step 10 系列: 工作记忆 + 认知验证 + 注意力 + 规模扩展 ✅ (2026-02-07)
> 详细文档: [steps/step10_wm_scale.md](steps/step10_wm_scale.md)

- 工作记忆: dlPFC L2/3 循环自持 + DA D1 稳定 (Goldman-Rakic 1995)
- 认知验证: 6 项任务 (DMTS/Go-NoGo/Papez/情绪记忆/WM+BG/反转学习)
- 注意力: VIP 去抑制回路 (Letzkus 2013) + NE/ACh 精度调制
- 规模: build_standard_brain(scale) 参数化, scale=1/3/8 (5.5k~44k 神经元)
- 154 测试通过。

### Step 11: REM 睡眠 + 梦境 ✅ (2026-02-07)
> 详细文档: [steps/step11_rem_sleep.md](steps/step11_rem_sleep.md)

SleepCycleManager (AWAKE→NREM→REM→NREM 周期管理), REM 周期增长 (后半夜 REM 延长)。
皮层去同步化噪声 + PGO 波 (梦境视觉) + M1 运动抑制 (防梦游)。
海马 theta (~6Hz) + 创造性重组 (1%/步随机激活 20% CA3)。161 测试通过。

---

## 闭环学习阶段 (Step 13-20)

### Step 13 系列: 闭环 Agent + GridWorld + 学习调优 ✅ (2026-02-07~08)
> 详细文档: [steps/step13_closed_loop.md](steps/step13_closed_loop.md)

- 13-A: 稳态可塑性 — SynapticScaler E/I 自平衡, Scale=3 WM 从 0 恢复到 0.425
- 13-B: GridWorld 闭环 — 10×10 网格, 3×3 视觉, food/danger, M1 WTA 解码
- 13-B+: DA-STDP 修复 — eligibility traces + Phase A/B/C 时序 + MSN up-state + 动作子组
- 13-B++: 调优 v3 — elig clamp(50) + lr 0.005 + NE 动态探索 + ctx 诊断
- 13-C: 视觉通路 — LGN gain 200 + 每步注入 + V1→dlPFC 拓扑映射 + V1 STDP
- 13-D+E: BG 拓扑 — dlPFC→BG 78%匹配 + efference copy
- improvement +0.191, learner advantage +0.017, food +55%。29/29 CTest。

### Step 14: Awake SWR Replay ✅ (2026-02-08)
> 详细文档: [steps/step14_swr_replay.md](steps/step14_swr_replay.md)

EpisodeBuffer 记录 dlPFC spike 快照, 奖励后触发 replay_learning_step (只步进 D1/D2, 保护 GPi/GPe)。
关键决策: 只重放旧正奖励 (防 D2 过强), 5 passes, DA=0.5。
improvement +0.077→+0.120 (+56%), late safety 0.524→0.667 (+27%)。29/29 CTest。

### Step 15 系列: 预测编码闭环 + 环境扩展 ✅ (2026-02-08)
> 详细文档: [steps/step15_predictive_loop.md](steps/step15_predictive_loop.md)

- 15: dlPFC→V1 反馈通路 (促进模式 Bastos 2012, 默认禁用)。3×3 PC 有害, 5×5 PC 有益 (+0.121)
- 15-B: 环境扩展 vision_radius 参数化 (1=3×3, 2=5×5, 3=7×7) + 自动 LGN/V1/dlPFC 缩放
- 15-C: 皮层巩固尝试失败 (awake LTD 主导), 基础设施保留待 NREM 使用
- 29/29 CTest。

### Step 16: 基因层 v1 (遗传算法) ✅ (2026-02-08)
> 详细文档: [steps/step16_genome.md](steps/step16_genome.md)

23 基因直接编码 (学习/探索/重放/视觉/稳态/大小/时序/NE), GA 引擎 (锦标赛/交叉/变异, 16 线程)。
关键教训: 2000 步评估 → 优化短期表现而非学习能力 (短评估陷阱), 需 ≥5000 步 + ≥3 seed。29/29 CTest。

### Step 17: LHb 负RPE + 负经验重放 ✅ (2026-02-08)
> 详细文档: [steps/step17_lhb_negative.md](steps/step17_lhb_negative.md)

LHb 外侧缰核: 惩罚/期望落空→LHb burst→VTA DA pause→D2 NoGo 强化。
17-B: 有了 LHb 受控 DA pause, 负重放变得安全 (2 passes, DA floor=0.05, agent_step≥200)。
完整奖惩回路: 食物→DA burst→D1 Go + 危险→DA pause→D2 NoGo。improvement +0.158 (+32%)。29/29 CTest。

### Step 18: 海马空间记忆闭环 ✅ (2026-02-08)
> 详细文档: [steps/step18_hippocampal_loop.md](steps/step18_hippocampal_loop.md)

EC grid cell 空间编码 (4种空间频率 2D余弦调谐) + CA3 奖励标记 (DA-modulated LTP a_plus×4)。
Hipp→dlPFC 反馈投射 (Sub→EC→dlPFC via SpikeBus)。设计教训: 位置记忆≠导航指令。29/29 CTest。

### Step 19: 杏仁核恐惧回避闭环 ✅ (2026-02-08)
> 详细文档: [steps/step19_amygdala_fear.md](steps/step19_amygdala_fear.md)

La→BLA STDP one-shot 恐惧条件化 (a_plus=0.10, 10× cortical)。
CeA→VTA 直接抑制 + CeA→LHb 间接抑制 = 双重 DA pause → D2 NoGo 强化。
improvement +0.161 (+71%), late safety 0.779 (+35%)。**历史最佳。** 29/29 CTest。

### Step 20: 睡眠巩固闭环 ✅ (2026-02-08)
> 详细文档: [steps/step20_sleep_consolidation.md](steps/step20_sleep_consolidation.md)

NREM SWR offline replay 集成到 ClosedLoopAgent (wake→sleep→replay→wake 周期)。
3×3 环境中 awake replay 已充分, 睡眠过度巩固反而有害 → 默认禁用。
大环境 (10×10+) 中应有价值, enable_sleep_consolidation=true 随时启用。29/29 CTest。

---

## 泛化与优化阶段 (Step 21-38)

### Step 21: 环境升级 10×10 + 5×5 Vision ✅ (2026-02-08)
> 详细文档: [steps/step21_environment_upgrade.md](steps/step21_environment_upgrade.md)

10×10 grid, 5×5 vision (25px), 5 food, 3 danger。自动缩放: LGN=100, V1=447, dlPFC=223。
解锁沉睡子系统: PC(5×5有效) + 睡眠巩固(100格抗遗忘) + 50 episode 缓冲。
5k improvement +0.16, 但 10k 退化 -0.086 (D1 子群无竞争, 权重趋同)。29/29 CTest。

### Step 22: D1 侧向抑制 ✅ (2026-02-08)
> 详细文档: [steps/step22_lateral_inhibition.md](steps/step22_lateral_inhibition.md)

MSN 子群间 GABA 侧枝竞争 (Humphries 2009): winner D1 子群抑制其他子群 → 方向选择性涌现。
**关键突破: 10k 退化修复!** improvement -0.086→+0.005, late safety +40%, food +26%。29/29 CTest。

### Step 23: 泛化能力诊断 ✅ (2026-02-08)
> 详细文档: [steps/step23_generalization_diag.md](steps/step23_generalization_diag.md)

训练 seed=42 后在新 seed 测试: 泛化优势 -6.9% — **训练有害, 系统在"背答案"不是"学道理"**。
根因: V1→dlPFC→BG 直连, 无视觉层级抽象, V1 模式和食物位置强耦合。29/29 CTest。

### Step 24: 视觉层级接入闭环 ✅ (2026-02-08)
> 详细文档: [steps/step24_visual_hierarchy.md](steps/step24_visual_hierarchy.md)

V1→V2→V4→IT→dlPFC 替代 V1→dlPFC 直连 (IT 不变性表征)。brain_steps 15→20。
**泛化翻转 -0.069→+0.042**, 从"背答案"到"学道理"。15×15 大环境 improvement +0.142。28/28 CTest。

### Step 25: DA-STDP 能力诊断 ✅ (2026-02-08)
> 详细文档: [steps/step25_dastdp_diagnosis.md](steps/step25_dastdp_diagnosis.md)

极简任务诊断: 2-armed bandit 权重分化但行为随机, T-maze 连3格都学不会。
两个根本问题: A) BG 权重→行为增益太低, B) 视觉层级是衰减器不是抽象器。30/30 CTest。

### Step 26: 人脑机制修复 ✅ (2026-02-08)
> 详细文档: [steps/step26_brain_mechanism_fix.md](steps/step26_brain_mechanism_fix.md)

- BG 乘法增益 (Surmeier 2007): w→(1+(w-1)×3), 权重差异非线性放大
- Pulvinar tonic: V2=3.0/V4=2.5/IT=2.0 持续驱动, 反馈增益 0.12→0.5
- ACh STDP 门控 (Froemke 2007): 奖励后 STDP gain = 1+|reward|×0.5
- learner advantage +0.0100, 10k improvement +0.072。测试 145秒→48.5秒 (3× 多线程)。

### Step 27: error-gated STDP ✅ (2026-02-08)
> 详细文档: [steps/step27_error_gated_stdp.md](steps/step27_error_gated_stdp.md)

L6→L2/3 预测突触 STDP + error-gated STDP: 只有 regular spike (预测误差) 触发 LTP,
burst (预测匹配) 不更新。发育期逻辑 (dev_period_steps 步无奖励视觉发育)。

### Step 28: 信息量压缩 + SNN 性能优化 ✅ (2026-02-08)
> 详细文档: [steps/step28_compression_perf.md](steps/step28_compression_perf.md)

信息量驱动神经元分配: 1100→120 (9× 压缩), 每个神经元有信息论意义。
树突 mismatch 可塑性 (Sacramento 2018): |V_apical - V_soma| 调制 STDP ≈ backprop。
SNN 优化: 零拷贝 + NMDA查表 + SpikeBus预分配。37秒→2.3秒 (16× 加速)。

### Step 29: Baldwin 进化 ✅ (2026-02-08)
> 详细文档: [steps/step29_baldwin_evolution.md](steps/step29_baldwin_evolution.md)

Baldwin 效应适应度: improvement×3 + late_safety (选择"能学习的大脑"而非"天生就会")。
30代×40体×5seed, 275秒。进化洞察: ne_floor=1.0 "永远探索, 永不利用" (120n 容量限制)。
**泛化 +0.009→+0.667 (74× 提升!)**, trained=0.750 vs fresh=0.083。

### Step 30: 小脑前向预测 + 丘脑门控 ✅ (2026-02-08)
> 详细文档: [steps/step30_cerebellum_thalamus.md](steps/step30_cerebellum_thalamus.md)

小脑 24n (GrC=12/PC=4/DCN=4) 接入闭环: M1 efference copy + V1 context → CF-LTD + DCN→BG 协同。
丘脑 NE/ACh TRN 门控: 高NE→TRN放松→更多信号通过。
**学习链路 10/10 完整**, learner advantage +0.053 (3×)。100代重进化运行中。

### Step 31: Ablation 诊断 ✅ (2026-02-08)
> 详细文档: [steps/step31_ablation.md](steps/step31_ablation.md)

逐模块消融: 120n 规模下 5/10 模块有害 (sleep/cortical STDP/cerebellum/hippocampus/SWR replay)。
根因: 容量不足 (CA3=6, V1=26, GrC=12) 引入噪声权重更新, 干扰核心 BG DA-STDP。
精简后泛化 -0.129→+0.131 翻正, trained safety 0.250→0.833。

### Step 32: Bug 修复 + 重进化 ✅ (2026-02-08)
> 详细文档: [steps/step32_bug_fix_reevolution.md](steps/step32_bug_fix_reevolution.md)

Bug 1: 皮层 STDP LTD/LTP 比例 3.7×→1.2× (生物学正常)。Bug 2: LHb 双重计数移除。
修复后所有模块不再有害 (STDP +0.00, LHb +0.00, sleep -0.20 有用)。
暴露新问题: 灾难性遗忘 (1000步 safety=1.00 → 1500步 safety=0.00)。

### Step 33: 灾难性遗忘修复 ✅ (2026-02-08)
> 详细文档: [steps/step33_catastrophic_forgetting.md](steps/step33_catastrophic_forgetting.md)

- 突触巩固 (元可塑性): per-synapse consolidation score, 巩固突触 lr/6 + decay/6
- 交错回放: 正面回放混入负面经验, 防趋近覆盖趋避
- 杏仁核修复: 移除错误 SpikeBus (兴奋→抑制), STDP 0.10→0.03, CeA 输出加 cap
- **重大突破: 没有任何模块是"有害"的!**
- 关键有用: SWR(-0.88) > LHb(-0.86) > 睡眠(-0.75) > PC(-0.73)。

### Step 35: ACC 前扣带回 ✅ (2026-02-08)
> 详细文档: [steps/step35_acc.md](steps/step35_acc.md)

整合 5 个经典 ACC 模型: 冲突监测(Botvinick) + PRO惊讶(Alexander) + 波动性(Behrens) + 觅食(Kolling) + EVC(Shenhav)。
dACC(12)+vACC(8)+PV(6) 神经元, 替代硬编码 ne_floor。35b: 全接线 (attention+foraging+lr_mod)。
消融: 早期学习快 22×, danger 少 32%。30/30 CTest。

### Step 36: CLS 互补学习系统 ✅ (2026-02-08)
> 详细文档: [steps/step36_cls.md](steps/step36_cls.md)

认知地图 spatial_value_map (位置奖励历史 + 邻域扩散 + 价值梯度→BG) + 睡眠系统巩固。
海马 4-bug 修复: Sub<4n / E/I反转 / 单脉冲注入 / retrieval_bias方向错误。
灾难性遗忘 -0.750→-0.088 (减少 89%), learner advantage +0.036 (+57%)。30/30 CTest。

### Step 37: VTA DA 信号通路修复 ✅ (2026-02-09)
> 详细文档: [steps/step37_vta_da_fix.md](steps/step37_vta_da_fix.md)

3 个 DA 通路 bug: 负 phasic 只持续1步 / Phase A 时序 / 皮层→MSN PSP 衰减太快。
修复: firing-rate-based DA (Grace 1991) + warmup + 步进后读 DA + PSP 0.7→0.9。
DA 恒定 0.300→动态, D1 0→2 fires, 权重范围 5.6×。30/30 CTest。

### Step 38: 丘脑-纹状体直接通路 ✅ (2026-02-09)
> 详细文档: [steps/step38_thalamostriatal.md](steps/step38_thalamostriatal.md)

问题: 视觉层级 14 步延迟 > brain_steps=12, 皮层信号到不了 BG。
方案: LGN→BG 快通路 (delay=1, 显著性粗信号, 不参与 STDP) + 皮层慢通路 (delay=14, 方向精信号)。
38b: ACh 门控巩固 (Hasselmo 1999): 低ACh→保护, 高ACh→允许反转。
**D1 2→36 (18×), 2000步首次正向改善 +0.212。** 30/30 CTest。

---

### Step 39: 皮层信号链三修复 ✅ (2026-02-09)
> 详细文档: [steps/step39_cortical_signal_fix.md](steps/step39_cortical_signal_fix.md)

三瓶颈: brain_steps=12 < 层级延迟14步 / 皮层 PSP_DECAY=0.7 太快 / L4/L5 最小2个神经元断链。
三修复: brain_steps 12→20 / PSP_DECAY 0.7→0.85 / L4/L5 min 2→3。
**皮层 events 34→358 (10.5×), elig 10.9→63.5 (5.8×), weight range 0.0165→0.0464 (2.8×)。** 30/30 CTest。

### Step 40: Phase 1 三区域扩展 ✅ (2026-02-09)
> 详细文档: [steps/step40_phase1_regions.md](steps/step40_phase1_regions.md)

三新区域: NAcc 伏隔核 (16n, 动机/奖赏) + SNc 黑质致密部 (4n, 习惯维持) + SC 上丘 (8n, 快速显著性)。
NAcc 分离动机(ventral)与运动选择(dorsal), SNc tonic DA 稳定已学权重 (70%VTA+30%SNc), SC 皮层下快通道 LGN→SC→BG。
**D1 47→58 (+23%), elig 63.5→114.2 (1.8×), weight range 0.0464→0.1080 (2.3×)。** 30/30 CTest。

---

## 当前系统状态

```
57区域 · ~188闭环神经元 · ~121投射 · 30/30 CTest
默认环境: 10×10 grid, 5×5 vision (25px), 5 food, 3 danger

学习链路 13/13:
  ① V1→V2→V4→IT 视觉层级   ② L6 预测编码 + mismatch STDP
  ③ dlPFC→BG DA-STDP (乘法增益+侧向抑制)   ④ VTA DA burst/pause
  ⑤ ACh STDP 门控 (巩固+反转)   ⑥ 杏仁核 one-shot 恐惧   ⑦ 海马 CA3 + SWR 重放
  ⑧ Baldwin 进化   ⑨ 小脑 CF-LTD + DCN→BG   ⑩ 丘脑 NE/ACh TRN 门控 + 丘脑纹状体通路
  ⑪ NAcc 动机/奖赏整合   ⑫ SNc 习惯维持 (tonic DA)   ⑬ SC 皮层下快速显著性

关键指标:
  D1 发放: 58/50步 (从 0→2→36→47→58)
  皮层→BG events: 616/10步
  Max eligibility: 114.2, Weight range: 0.1080
  learner advantage: +0.036
```
