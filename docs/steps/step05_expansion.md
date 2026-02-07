# Step 5 系列: 脑区扩展 + 预测编码 + Python

> 日期: 2026-02-07
> 状态: ✅ 完成

## 信号衰减修复
- V1→dlPFC PSP 太弱 → fan-out 30%×L4, current 35f/55f
- 修复后全链路: LGN=124→V1=7656→dlPFC=4770→BG=3408→M1=1120

## Step 5.0: 神经调质广播
- LC_NE (15n, 警觉), DRN_5HT (20n, 耐心), NBM_ACh (15n, 注意力)
- NE 增益调制 → Yerkes-Dodson 倒U型涌现 (无硬编码!)
- 12 区域, 62 测试通过

## Step 5a: 视觉层级 V2/V4/IT
- V2(214n), V4(164n), IT(130n), 前馈+反馈投射
- 层级传播: V1→V2→V4→IT 逐层延迟 ~2ms
- STDP 习惯化涌现: 训练后 IT=4 vs 未训练 IT=697
- 15 区域, 68 测试通过

## Step 5b: 小脑
- Cerebellum (275n): GrC→PC→DCN + MLI + Golgi
- CF-LTD 学习: PC 误差校正 1010→702 (第4种学习规则)
- 16 区域, 74 测试通过

## Step 5c+5d: 决策皮层 + 背侧视觉
- OFC(151n), vmPFC(140n), ACC(135n) — 价值决策三角
- MT(185n), PPC(174n) — 背侧视觉 (where)
- 双流: 腹侧(what) + 背侧(where) + 跨流连接
- 21 区域, 80 测试通过

## Step 6 (预测编码): 皮层预测与误差
- enable_predictive_coding(): 反馈路由, 预测抑制, 精度加权
- NE→感觉精度, ACh→先验精度
- 预测抑制涌现: -49% (Rao & Ballard 1999)
- 86 测试通过

## Step 7 (Python 绑定): pybind11 + 可视化
- 全部 11 种 BrainRegion 绑定
- plot_raster, plot_connectivity, plot_neuromod_timeline
- build_standard_brain() 一键构建

## Step 5 扩展 (后期): 46 区域完整大脑
- 13 个皮层区: S1/S2/A1/Gustatory/Piriform/PCC/Insula/TPJ/Broca/Wernicke/PMC/SMA/FEF
- 9 个丘脑核: VPL/MGN/MD/VA/LP/LD/Pulvinar/CeM/ILN
- ~90 条投射, 体感/听觉/语言/运动/DMN 通路
- 121 测试通过
