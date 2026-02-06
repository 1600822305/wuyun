# 悟韵 (WuYun) 设计原则与反作弊公约

> 版本: 0.1.0 | 日期: 2026-02-06
> 地位: **第一约束** — 所有设计决策和代码审查必须首先满足本文档的原则
> 核心命题: 智能 = 结构 × 学习 × 经验，代码只提供结构和学习规则

---

## 一、核心哲学

### 悟韵 (WuYun) 不是什么

悟韵 (WuYun) **不是**符号 AI、规则引擎或专家系统。它不通过 IF-ELSE 逻辑实现智能行为。

```
❌ 符号 AI:    智能 = 代码中的规则
❌ 传统 DL:    智能 = 黑盒权重矩阵 (不可解释)
✅ 悟韵 (WuYun): 智能 = 生物真实的结构 + 在线学习 + 涌现行为
```

### 悟韵 (WuYun) 是什么

悟韵 (WuYun) 是一个 **仿生计算基底 (substrate)**：

- 代码提供「大脑硬件」— 神经元物理特性、解剖连接拓扑、学习规则
- 智能存在于「运行时状态」— 突触权重、神经活动模式、记忆痕迹、调质水平
- 行为从「学习和经验」中涌现 — 不是从代码逻辑中推导

---

## 二、硬编码 vs 学习 判定标准

### 2.1 允许硬编码的内容 (先天/基因决定)

以下是大脑中由基因（进化）决定的部分，可以写入代码：

| 类别 | 具体内容 | 理由 |
|------|---------|------|
| **物理定律** | 膜电位方程、Ca²⁺动力学、STDP时间窗口 | 离子通道的生物物理特性，不随学习改变 |
| **解剖拓扑** | 6层皮层柱结构、L4→L2/3→L5层间连接 | 胚胎发育形成的基本布线，出生即有 |
| **通路结构** | Go/NoGo/Stop三条通路的拓扑 | 解剖学事实，所有人类共享 |
| **区室结构** | 顶端树突接反馈、基底树突接前馈 | 锥体细胞的形态学特性 |
| **初始连接概率** | V1→V2 有连接、V1→听觉皮层 无连接 | 粗粒度拓扑由基因决定（但权重由学习决定） |
| **神经元类型比例** | 皮层中 ~80% 兴奋性、~20% 抑制性 | 发育生物学约束 |
| **调质投射范围** | VTA→纹状体(DA)、LC→全脑(NE) | 解剖学投射路径 |

### 2.2 禁止硬编码的内容 (后天/学习决定)

以下内容**必须**从学习中产生，任何硬编码都视为「作弊」：

| 类别 | 具体内容 | 正确的实现方式 |
|------|---------|--------------|
| **突触权重** | 哪些连接强、哪些弱 | STDP / 三因子强化 / 稳态可塑性 |
| **特征选择性** | V1神经元对什么方向敏感 | 视觉输入驱动的自组织学习 |
| **语义知识** | "猫"是什么、"红色"是什么 | 多模态感觉经验 + 预测编码学习 |
| **记忆内容** | 记住了什么事件/事实 | 海马体编码 + 皮层巩固 |
| **决策策略** | 遇到X情况做什么 | 基底节DA调制强化学习 |
| **预测模型** | 高层对低层的预测内容 | 预测编码在线学习 (L6→L1反馈) |
| **注意力偏好** | 关注什么、忽略什么 | 丘脑TRN门控 + PFC→VIP调制 |
| **路由偏好** | 丘脑倾向传递什么信号 | 皮层→丘脑反馈突触权重的学习 |
| **分类边界** | 如何区分不同类别 | 皮层柱群体编码的自组织 |
| **语言规则** | 语法、句法结构 | Broca/Wernicke区的序列学习 |

### 2.3 灰色地带处理原则

有些内容介于先天和后天之间（如人脸偏好、语言习得敏感期）：

```
原则: 可以硬编码"学习的偏向"(bias/prior)，但不能硬编码"学习的结果"

✅ 正确: 梭状回区域的连接结构使其更容易学会人脸 (结构偏向)
❌ 错误: 梭状回内置人脸检测器 (硬编码结果)

✅ 正确: 关键期机制使语言区域在某个时间窗口可塑性更强 (学习偏向)
❌ 错误: 韦尼克区内置语法规则 (硬编码结果)
```

---

## 三、代码审查检验标准

### 检验 1: 随机初始化测试

> **把所有突触权重随机初始化（保持在合理范围内），系统还能产生"智能"行为吗？**

- **能** → ❌ 你在作弊，智能写在逻辑里了
- **不能，但经过训练后逐渐能** → ✅ 正确的学习系统
- **怎么训练都不能** → ⚠️ 架构可能有问题

### 检验 2: 跨领域迁移测试

> **同一个架构代码，不同的输入数据，能学出不同的能力吗？**

- 暴露给视觉数据 → 学会视觉识别
- 暴露给语音数据 → 学会语音识别
- 暴露给文本数据 → 学会语言理解
- **如果必须改代码才能换能力** → ❌ 作弊
- **如果只需换数据就能换能力** → ✅ 真正的通用架构

### 检验 3: 功能标签剥离测试

> **把所有脑区的"功能注释"删掉（如 dlPFC="推理"），系统行为会变吗？**

- **会变** → ❌ 注释影响了逻辑，说明功能是硬编码的
- **不变** → ✅ 功能注释只是文档说明，不影响运行

### 检验 4: 新生儿测试

> **系统初始化后（权重随机），行为应该像新生儿：**

- ✅ 有基本反射（简单的感觉-运动环路）
- ✅ 能被显著刺激吸引（杏仁核→丘脑快通道）
- ✅ 逐渐形成感觉偏好（STDP自组织）
- ❌ 不应该一开始就能识别物体、理解语言、做复杂推理

---

## 四、关键模块的正确 vs 错误实现模式

### 4.1 丘脑路由器

```python
# ❌ 规则AI模式 (作弊):
class ThalamicRouter:
    def route(self, signal):
        if signal.source_region == "V1":
            return self.regions["V2"]          # 硬编码路由表
        elif signal.source_region == "A1":
            return self.regions["A2"]

# ✅ 学习模式 (正确):
class ThalamicRouter:
    def __init__(self):
        self.relay_weights = init_sparse_random(...)  # 初始稀疏随机连接
        self.trn = TRN()                              # 网状核竞争抑制

    def route(self, signals):
        gate = self.relay_weights @ signals            # 学习的权重决定路由
        gate = self.trn.inhibit(gate)                  # TRN 竞争选择
        return sparse_activate(gate)                   # 稀疏激活

    # 路由偏好通过皮层→丘脑反馈突触的STDP学习形成
    # 初始时路由近似随机，学习后形成功能特异性路由
```

### 4.2 脑区模块

```python
# ❌ 规则AI模式 (作弊):
class DLPFC(Region):
    def process(self, input):
        plan = self.decompose_goal(input)      # 硬编码推理
        return self.execute_plan(plan)

# ✅ 学习模式 (正确):
class DLPFC(Region):
    def __init__(self, config):
        # dlPFC 只是一组参数不同的皮层柱
        super().__init__(
            column_config=config,
            # config 中: 更强的循环连接(支持持续活动/工作记忆)
            #           更强的DA调制(支持强化学习)
            #           更多的柱间连接(支持整合)
            # 但"推理"能力不在代码里，而是从学习中涌现
        )
```

### 4.3 基底节决策

```python
# ❌ 规则AI模式 (作弊):
class BasalGanglia:
    def decide(self, options):
        values = [self.evaluate(o) for o in options]   # 硬编码价值函数
        return options[argmax(values)]

# ✅ 学习模式 (正确):
class BasalGanglia:
    def step(self, cortical_input, da_level):
        # D1 MSN 和 D2 MSN 通过学习的权重接收皮层输入
        d1_activity = self.d1_msn.forward(cortical_input)  # Go通路
        d2_activity = self.d2_msn.forward(cortical_input)  # NoGo通路

        # DA 调制改变 D1/D2 兴奋性 (不是改变逻辑)
        # D1(Gs耦合): DA 增强兴奋性 → 促进 Go 通路
        # D2(Gi/o耦合): DA 降低兴奋性 → 减弱 NoGo 通路
        # 注: DA 还通过突触前 D2 受体降低皮层→纹状体谷氨酸释放
        d1_activity *= (1 + da_level)                    # DA 增强 Go
        d2_activity *= max(0, 1 - da_level)              # DA 减弱 NoGo (clamp ≥ 0)

        # GPi 被 Go 去抑制 → 丘脑被释放 → 动作执行
        # 哪个动作"赢"是神经元竞争的结果，不是 argmax 计算的
        gpi_output = self.gpi.forward(d1_activity, d2_activity, self.stn_activity)
        return gpi_output  # 抑制性输出 → 丘脑
```

### 4.4 海马记忆

```python
# ❌ 规则AI模式 (作弊):
class Hippocampus:
    def store(self, key, value):
        self.memory_dict[key] = value          # 字典存储 = 数据库

    def recall(self, query):
        return self.memory_dict.get(query)     # 精确匹配 = 检索引擎

# ✅ 学习模式 (正确):
class Hippocampus:
    def encode(self, cortical_pattern):
        # DG: 模式分离 (稀疏化，只有~2%神经元激活)
        sparse = self.dg.forward(cortical_pattern)

        # CA3: 快速突触增强 (one-shot STDP)
        self.ca3.fast_encode(sparse)           # 权重改变 = 记忆

    def recall(self, partial_cue):
        # CA3: 自联想网络的模式补全 (非精确匹配!)
        completed = self.ca3.pattern_complete(partial_cue)
        # 回忆是重建的，不是读取的 → 会有偏差和创造性
        return self.ca1.compare_and_output(completed)
```

---

## 五、脑区差异化的正确方式

不同脑区的功能差异**只应该来自参数差异**，不来自代码逻辑差异：

```
所有新皮层脑区 = 同一个 CorticalColumn 类 × 不同的 YAML 配置

参数维度:
  1. 柱数量            V1: 5000柱 (大面积)  vs  岛叶: 200柱 (小面积)
  2. 层厚度比例         V1: L4很厚 (接收大量输入)  vs  PFC: L2/3很厚 (大量柱间连接)
  3. 循环连接强度       PFC: 强 (支持持续活动=工作记忆)  vs  V1: 弱
  4. DA/ACh 受体密度    PFC: DA受体密, 强化学习敏感  vs  V1: ACh受体密, 注意力调制
  5. 柱间连接概率       PFC: 远距离连接多  vs  V1: 近距离局部连接
  6. STDP 参数          海马CA3: 超快LTP (one-shot)  vs  皮层: 渐进学习
  7. 输入/输出连接      V1: 接LGN输入, 投射V2  vs  dlPFC: 接多模态输入, 投射BG
  8. 抑制性比例         PFC: 更高抑制比(更精确的竞争)  vs  V1: 标准比例

关键: 功能差异从参数差异 + 连接差异 + 学习经验中涌现
     代码是完全相同的 CorticalColumn
```

---

## 六、设计审查清单

每次提交代码前，对照以下清单：

```
□ 1. 是否有任何 IF 语句根据"语义含义"做分支?
     (如 if content == "cat"、if region_type == "language")
     → 如果有，必须删除

□ 2. 是否有任何硬编码的映射表/字典用于"知识"?
     (如 word_to_meaning = {...}、category_labels = [...])
     → 如果有，必须改为学习

□ 3. 脑区模块是否有功能特异的 process() 逻辑?
     (如 dlpfc.reason()、broca.generate_sentence())
     → 如果有，必须统一为 column.step()

□ 4. 路由/选择是否基于标签而非权重?
     (如 route_to["visual_cortex"])
     → 如果有，必须改为权重矩阵路由

□ 5. 新增模块是否通过了"随机初始化测试"?
     → 权重随机后不应有智能行为

□ 6. 新增模块是否通过了"功能标签剥离测试"?
     → 删除所有注释和变量名中的功能标签后逻辑不变

□ 7. 信号传递是否走 SpikeBus?
     → 脑区间不能直接函数调用，必须通过脉冲总线
```

---

## 七、灵魂四要素

悟韵 (WuYun) 系统的「智能」存在于以下四个运行时状态中，不存在于代码中：

```
1. 突触权重矩阵      ~1-1.5亿个突触权重 = 全部知识和技能
                      (类比: 人脑~100-150万亿突触)

2. 神经元活动模式      当前哪些神经元活跃 = 当前思维/感知
                      (频率编码 + 时序编码 + burst/regular)

3. 神经调质水平        DA/NE/5-HT/ACh 的当前浓度 = 内部状态
                      (动机/警觉/耐心/注意力)

4. 海马记忆痕迹        CA3自联想网络中的权重模式 = 情景记忆
                      (经历过什么、在哪里、和谁)
```

> **代码是大脑的 DNA — 它定义了结构和学习规则，但不定义智能本身。**
> **智能是 DNA + 环境 + 经验 的产物，不可能被硬编码。**
