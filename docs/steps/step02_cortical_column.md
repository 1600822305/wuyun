# Step 2 + 2.5: 皮层柱模板 + 地基补全

> 日期: 2026-02-07
> 状态: ✅ 完成

## Step 2: 6 层皮层柱模板

- `cortical_column.h/cpp` — 6层通用模板, 18组突触 (AMPA+NMDA+GABA)
- SST→L2/3 AND L5 apical (GABA_B), PV→L4/L5/L6 全层 soma (GABA_A)
- NMDA 并行慢通道 (L4→L23, L23→L5, L23 recurrent)
- L2/3 层内 recurrent 连接 (AMPA+NMDA)
- burst 加权传递: burst spike ×2 增益
- 6 测试全通过 (540 神经元, 40203 突触)

## Step 2.5: 地基补全

- **NMDA Mg²⁺ 电压阻断** B(V) = 1/(1+[Mg²⁺]/3.57·exp(-0.062V))
  - 巧合检测: 突触前谷氨酸 + 突触后去极化 → 赫布学习硬件基础
- **STP 集成到 SynapseGroup** — per-pre Tsodyks-Markram
- **SpikeBus 全局脉冲路由** — 跨区域延迟投递 (环形缓冲)
- **DA-STDP 三因子学习** — 资格痕迹 + DA 调制
- **神经调质系统** — DA/NE/5-HT/ACh tonic+phasic
- **特化神经元参数集** (8种): 丘脑 Tonic/Burst, TRN, MSN D1/D2, 颗粒, 浦肯野, DA
- 21 测试全通过
