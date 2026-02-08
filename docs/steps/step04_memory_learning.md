# Step 4 系列: 海马记忆 + 杏仁核 + 学习系统

> 日期: 2026-02-07
> 状态: ✅ 完成

## Step 4: 海马体 + 杏仁核
- Hippocampus: EC→DG→CA3→CA1→Sub 三突触通路, CA3 自联想, DG 稀疏编码
- Amygdala: La→BLA→CeA 恐惧通路, ITC 消退门控
- GABA 权重符号修复 (双重否定 bug)
- 33 测试通过

## Step 4.5: 9 区域闭环整合
- 6 条跨区域投射: V1→Amyg, dlPFC→Amyg(ITC), dlPFC↔Hipp, Amyg→VTA, BLA→Hipp
- 情绪标记记忆增强 (+19%)
- 40 测试通过

## Step 4.6: 开机学习
- SynapseGroup STDP 集成, CA3 fast STDP (one-shot learning)
- 记忆编码/回忆: 30%线索→100%模式补全 ✓
- 模式分离: DG 稀疏化, 重叠仅 10% ✓
- DA-STDP 强化学习验证 ✓
- **里程碑: 从"通电的硬件"变为"能学习的系统"**

## Step 4.7: 皮层 STDP 自组织
- L4→L2/3, L2/3 recurrent, L2/3→L5 三组 AMPA 突触启用 STDP
- 86 个神经元发展出选择性 (偏好 A 或 B)
- 49 测试通过

## Step 4.8: BG DA-STDP 在线强化学习
- 三因子规则: D1(Gs) DA>baseline→LTP, D2(Gi) DA>baseline→LTD
- 动作选择学习: 奖励 A > 未奖励 B (+63%) ✓
- 反转学习: 偏好成功反转 (+36%) ✓
- 53 测试通过

## Step 4.9: 端到端学习演示
- 9 区域全部启用学习: V1(STDP) + dlPFC(STDP) + BG(DA-STDP) + Hipp(CA3 STDP)
- 3 套独立学习系统同时运行
- 视觉→Amyg→VTA 自然产生 DA (无人工注入)
- 57 测试通过

## Step 4 补全 (后期)
- SeptalNucleus: theta 起搏器 (6.7Hz)
- MammillaryBody: Papez 回路中继
- Hippocampus 扩展: Presubiculum, HATA
- Amygdala 扩展: MeA, CoA, AB
- 24 区域, 40 投射, 100 测试通过

## 修复: V1→dlPFC→BG 信号衰减

> 问题: CorticalRegion::receive_spikes fan-out=3, current=25f → PSP稳态3.1f ≪ 阈值15f

**根因分析:**
- L4 stellate: v_rest=-65, threshold=-50, R_s=1.0 → 需 I>15f 持续
- LGN→V1: 0.62脉冲/步 × 3/50(fan-out/L4) × 25f = 0.93f/步 → 稳态3.1f (远低于阈值)

**修复:**
- ✅ `ColumnConfig::input_psp_regular/burst/fan_out_frac` 可配置参数
- ✅ fan-out: 3固定 → 30%×L4_size (生物学皮层-皮层汇聚)
- ✅ current: 25f/40f → 35f/55f (regular/burst)
- ✅ `CorticalRegion` 存储 `psp_current_regular_/burst_/fan_out_`

**修复后全链路数据:**
```
修复前: LGN=124 → V1=23    → dlPFC=0    → BG=120  → MotorThal=0   → M1=0
修复后: LGN=124 → V1=7656  → dlPFC=4770 → BG=3408 → MotorThal=293 → M1=1120
```
- 额外收获: dlPFC→Hipp通路也打通 (dlPFC=6937→Hipp=18791, CA1=660)
- **57 测试全通过**, 零回归
