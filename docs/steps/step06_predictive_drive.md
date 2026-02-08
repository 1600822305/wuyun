# Step 6 系列: 预测编码 + 下丘脑内驱力 + 意识(GNW)

> 日期: 2026-02-07
> 状态: ✅ 完成

## Step 6: 预测编码框架
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
- L4注入 × sensory精度, prediction注入 × prior精度

**验证结果:**
- **预测抑制涌现**: V1(早期无预测)=226 → V1(晚期有预测)=116 (-49%)
- **NE精度**: V1(NE=0.1)=85 → V1(NE=0.5)=187 → V1(NE=0.9)=235
- **ACh精度**: ACh=0.8→prior=0.36, ACh=0.1→prior=0.92
- **层级PC**: V1↔V2↔V4 双向预测+误差
- **向后兼容**: 无PC时 V1=262, PC无反馈时 V1=262 (完全一致)

**生物学对应:**
- L6 → 反馈 → 下级L2/3 apical = 预测信号 (Mumford 1992)
- L2/3 = 感觉(L4 basal) - 预测(apical) = 预测误差 (Rao & Ballard 1999)
- NE = 感觉精度 (意外→LC→NE↑→信任感觉) (Feldman & Friston 2010)
- ACh = 先验精度倒数 (新环境→NBM→ACh↑→不信任预测) (Yu & Dayan 2005)

**系统状态:**
- 21区域 | 3239神经元 | 36投射 | 4调质 | 4学习 | **预测编码**
- **86 测试全通过** (9+6+6+5+7+7+5+4+4+4+5+6+6+6+6), 零回归

## Step 6: 下丘脑内驱力系统
> 目标: 内在动机引擎 — 睡眠/觉醒/应激/摄食

**Hypothalamus 类 (region/limbic/hypothalamus.h/cpp):**
6个核团, 89个神经元:
- **SCN** (n=20) — 昼夜节律起搏器, 正弦振荡 (可配置周期)
- **VLPO** (n=15) — 睡眠促进, GABA/galanin→抑制觉醒中枢
- **Orexin** (n=15) — 觉醒稳定, →LC/DRN/NBM (防止嗜睡发作)
- **PVN** (n=15) — 应激反应, CRH→HPA轴→Amygdala
- **LH** (n=12) — 摄食/饥饿驱力, →VTA (饥饿→动机)
- **VMH** (n=12) — 饱腹/能量平衡

**内部回路:**
- Sleep-wake flip-flop (Saper 2005): VLPO⟷Orexin互相抑制
- SCN→VLPO昼夜门控 (cosine振荡)
- LH⟷VMH摄食平衡 (互相GABA抑制)
- 外部可控: set_sleep_pressure/stress_level/hunger_level/satiety_level

**8条新投射 (→~98条总投射):**
- Orexin→LC/DRN/NBM (觉醒→调质广播)
- Hypothalamus→VTA (饥饿→动机DA)
- Hypothalamus↔Amygdala (应激↔恐惧)
- Insula→Hypothalamus (内感受→驱力)
- Hypothalamus→ACC (驱力→冲突监控)

**7项测试全部通过:**
1. SCN昼夜振荡 + 相位推进
2. Flip-flop: 低压力→wake=0.909, 高压力→wake=0.102
3. 睡眠压力: wake 0.911→0.208 + VLPO=30
4. Orexin稳定: spikes=60, wake=0.977
5. PVN应激: low=0, high=15 spikes + output=0.8
6. LH⟷VMH: 饥饿→LH=24,VMH=0; 饱腹→LH=0,VMH=24
7. 全系统集成: 7区域 + Orexin→LC + wake=0.909

**生物学对应:**
- Saper et al. (2005) Sleep-wake flip-flop switch
- Sakurai (2007) Orexin/hypocretin neural circuit
- Ulrich-Lai & Herman (2009) HPA stress axis

**系统状态:**
- **47区域** | **5498神经元** | **~98投射** | 4调质 | 4学习 | 预测编码 | 工作记忆 | 注意力 | **内驱力**
- **128 测试全通过** (121+7), 零回归

## GNW: 全局工作空间理论
> 目标: Baars/Dehaene 意识访问模型 — 竞争→点火→广播

**GlobalWorkspace 类 (engine/global_workspace.h/cpp):**
30个workspace整合神经元 + 竞争/点火/广播机制:
- **竞争**: 多个皮层区域L5输出→GW, per-region salience累积 (指数衰减防锁定)
- **点火**: 赢者salience超阈值 → ignition (全局点火事件)
- **广播**: 点火后workspace神经元爆发活动→ILN/CeM→全皮层L2/3
- **间隔控制**: min_ignition_gap防止连续点火 (意识是离散采样)

**9条竞争投射 + 2条广播投射 (→~109条总投射):**
竞争输入: V1/IT/PPC/dlPFC/ACC/OFC/Insula/A1/S1 → GW
广播输出: GW → ILN (板内核群) + CeM (中央内侧核)

**可查询状态:**
- `is_ignited()` — 当前是否在点火状态
- `conscious_content_name()` — 当前意识内容 (赢者区域名)
- `ignition_count()` — 累计点火次数
- `winning_salience()` — 当前最高salience值
- `salience_map()` — 全部区域salience

**7项测试全部通过:**
1. 基础点火: step=66 → ignition, count=19
2. 竞争门控: V1(强)胜 A1(弱), content="V1"
3. 广播持续: 广播中=60 > 广播后=0
4. 竞争衰减: peak=24.7 → decayed=0.6 (98%衰减)
5. 点火间隔: gap=50, 200步→4次点火
6. 无输入不点火: ignitions=0
7. 全系统: GW=180, ignitions=10, content="V1"

**生物学对应:**
- Baars (1988) A Cognitive Theory of Consciousness
- Dehaene & Changeux (2011) Experimental and theoretical approaches to conscious processing
- Dehaene, Kerszberg & Changeux (1998) Neuronal model of a global workspace

**系统状态:**
- **48区域** | **5528神经元** | **~109投射** | 4调质 | 4学习 | 预测编码 | 工作记忆 | 注意力 | 内驱力 | **意识(GNW)**
- **135 测试全通过** (128+7), 零回归

## Step 6 剩余 (低优先级)

**6a. 调质系统扩展:**
- ⬜ 5-HT细分: DR (MB-05) + MnR (MB-08)
- ⬜ ACh细分: PTg (MB-09) + LDTg (MB-10) + BF (BF-01)

**6b. 小脑扩展:**
- ⬜ 运动小脑: 前叶+VIIIa/b+绒球 (CB-01, CB-06~07)
- ⬜ 认知小脑: Crus I/II + VIIb (CB-03~05)
- ⬜ 蚓部 (CB-08) + 过渡区 (CB-02)
