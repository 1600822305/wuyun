# Step 56: 丘脑皮层预测反馈 + NMDA 电压门控确认

## 动机

Phase 0 架构审查标记了两个 🔴 最高优先改进项：

1. **NMDA Mg²⁺ 电压门控** — 预测编码的重合检测需要电压依赖的 NMDA 通道
2. **L6→TC 反馈增强** — 预测信号闭环不完整，V1 L6 无法把预测发回 LGN

### 生物学背景

**丘脑皮层预测回路** (Sherman & Guillery 2006, Sillito et al. 2006):

```
感觉世界 → LGN relay (basal, 驱动)
               ↕
           V1 L4 → L2/3 → L5 → L6
               ↑                 |
               |    预测反馈      |
               +---- LGN relay (apical, 调制) ←─┘
```

- **前馈 (driving)**: 感觉输入 → relay basal 树突 → 强兴奋 (20-30 pA)
- **反馈 (modulatory)**: L6 → relay apical 树突 → 弱调制 (12 pA)
- 当预测匹配时：apical 调制 + basal 驱动 → relay burst → 高置信度信号
- 当预测不匹配时：basal 驱动但无 apical → relay tonic → 预测误差信号
- 效果：已预期的输入被抑制，新奇/意外输入优先通过

**NMDA Mg²⁺ 阻断** (Jahr & Stevens 1990, Mayer et al. 1984):

```
B(V) = 1 / (1 + [Mg²⁺]/3.57 × exp(-0.062V))

V = -70 mV (静息): B = 0.04 → 96% 被阻断
V = -30 mV (去极化): B = 0.50 → 50% 通过
V =   0 mV (高去极化): B = 0.95 → 几乎全开
```

NMDA 只在突触后神经元已被 AMPA 去极化时才传导 → **真正的重合检测器**。
这对预测编码至关重要：L4→L2/3 的 NMDA 通路只有在 L2/3 同时收到 basal (前馈)
和 apical (反馈) 输入时才会开放。

## 发现：NMDA 已完整实现

审查代码后发现，NMDA Mg²⁺ 电压门控**已经完整实现**，无需额外工作：

### 已有基础设施

1. **`SynapseParams::mg_conc`** — `NMDA_PARAMS{tau=100, rise=5, e_rev=0, g_max=0.5, mg_conc=1.0}`
2. **B(V) 查找表** — `synapse_group.cpp` 预计算 256 条目 (-100 到 +50 mV)
3. **`step_and_compute()`** — 自动应用 `b_v = nmda_b_lookup(v_post)` 当 `mg_conc > 0`
4. **NMDA 突触组** — 皮层柱内 L4→L2/3, L2/3→L5, L2/3 recurrent 三组 NMDA 并行通路
5. **`deliver_and_inject()`** — 正确传递 `post.v_soma()` 给 `step_and_compute()`

```cpp
// synapse_group.cpp — B(V) 查找表
static void init_nmda_table() {
    for (int i = 0; i < 256; ++i) {
        float v = -100.0f + i * (150.0f / 255.0f);
        nmda_b_table[i] = 1.0f / (1.0f + (1.0f / 3.57f) * std::exp(-0.062f * v));
    }
}

// step_and_compute() — 自动电压门控
float b_v = has_nmda ? nmda_b_lookup(v) : 1.0f;
float i_syn = g_max_ * weights_[s] * g_[s] * b_v * (e_rev_ - v);
```

## 实现：V1→LGN 皮层丘脑反馈

### 问题

`ThalamicRelay::receive_spikes()` 把所有 SpikeBus 脉冲都路由到 relay **basal**（前馈）。
但 L6 反馈应该去 **apical**（调制）。需要区分脉冲来源。

### 设计

利用 `SpikeEvent::region_id` 识别来源。注册皮层反馈源，在 `receive_spikes()` 中
检查来源 → 反馈源走 apical，前馈源走 basal。

### 修改文件

#### `src/region/subcortical/thalamic_relay.h`

```cpp
// 新增接口
void add_cortical_feedback_source(uint32_t region_id);

// 新增成员
std::set<uint32_t> cortical_feedback_sources_;
static constexpr float CORTICAL_FB_CURRENT = 12.0f;  // apical 调制 (弱于前馈 20-30)
```

#### `src/region/subcortical/thalamic_relay.cpp`

```cpp
void ThalamicRelay::receive_spikes(const std::vector<SpikeEvent>& events) {
    for (const auto& evt : events) {
        size_t base = evt.neuron_id % relay_.size();

        if (cortical_feedback_sources_.count(evt.region_id)) {
            // 反馈源 → relay APICAL (调制预测)
            float current = is_burst(...) ? CORTICAL_FB_CURRENT * 1.5f : CORTICAL_FB_CURRENT;
            relay_.inject_apical(base + k, current);
        } else {
            // 前馈源 → relay BASAL (驱动)
            float current = is_burst(...) ? 30.0f : 20.0f;
            relay_.inject_basal(base + k, current);
        }
    }
}

void ThalamicRelay::add_cortical_feedback_source(uint32_t region_id) {
    cortical_feedback_sources_.insert(region_id);
}
```

#### `src/engine/closed_loop_agent.cpp`

```cpp
// build_brain() — 新增 V1→LGN 反馈投射
engine_.add_projection("V1", "LGN", 3);    // v56: L6→TC corticothalamic prediction

// 注册 V1 为 LGN 反馈源
auto* lgn_thal = dynamic_cast<ThalamicRelay*>(lgn_);
if (lgn_thal) {
    lgn_thal->add_cortical_feedback_source(v1_->region_id());
}
```

## 信号流

```
完整的丘脑皮层预测编码回路:

感觉 → LGN relay basal (20-30 pA, 驱动)
                ↓
         LGN → V1 L4 (SpikeBus, delay=2)
                ↓
         V1: L4 → L2/3 (AMPA + NMDA 重合检测)
                ↓              ↓
         L2/3 regular      L2/3 burst
         (预测误差→V2)     (匹配→学习)
                ↓
         L2/3 → L5 → L6 (AMPA + NMDA)
                          ↓
         V1 L6 → LGN relay apical (SpikeBus, delay=3, 12 pA 调制)
                ↑
         预测信号: "我预期看到这个, 抑制已知输入"
```

### 前馈 vs 反馈电流对比

| 来源 | 目标 | 电流 (pA) | 目的 |
|------|------|-----------|------|
| 感觉输入 | relay basal | 20 (regular), 30 (burst) | 驱动中继 |
| V1 L6 反馈 | relay apical | 12 (regular), 18 (burst) | 调制预测 |
| TRN 抑制 | relay basal | GABA_A | 注意力门控 |

## 验证

- 31/31 CTest 零回归
- 新增投射: V1→LGN (delay=3), 总投射数 ~140

## 参考文献

- Sherman SM, Guillery RW (2006) Exploring the Thalamus and Its Role in Cortical Function
- Sillito AM et al. (2006) Always returning: feedback and sensory processing in visual cortex
- Jahr CE, Stevens CF (1990) Voltage dependence of NMDA-activated macroscopic conductances
- Mayer ML et al. (1984) Voltage-dependent block by Mg2+ of NMDA responses
