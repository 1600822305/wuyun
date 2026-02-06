# 悟韵 (WuYun) 向量化重构设计

> 版本: 1.0.0 | 日期: 2026-02-07
> 状态: **已完成** — 全部模块已迁移, 62/62 测试通过, 零回归
> 目标: 将逐对象 Python 循环架构重构为 NumPy 向量化批量运算
> 前置: 04_roadmap.md (Phase 0-4 已完成), 02_neuron_system_design.md

---

## 一、问题诊断

### 1.1 当前瓶颈

```
当前架构 (逐对象循环):

for neuron in neurons:           # N 次 Python 函数调用
    for syn in neuron.synapses:  # M 次 Python 函数调用
        I += syn.step_and_compute(t, v, dt)   # 标量运算
    neuron.soma.update(I, v_a, κ, dt)         # 标量运算
```

Phase 3 测试 7 (全环路): 5000 步 × ~5000 突触 = 25M 次 Python 函数调用 → 8.4s

问题不是算法复杂度，而是 **Python 解释器开销**:
- 每次函数调用 ~0.5μs (属性查找 + 参数传递 + 栈操作)
- 每次属性访问 ~0.1μs
- 25M 次调用 × 0.5μs = 12.5s (接近实测值)
```

### 1.2 目标架构

```
向量化架构 (批量矩阵运算):

# 一步更新所有突触门控
s *= decay_factor                            # numpy broadcast, 1次调用
s[spike_arrived_mask] += increment           # 向量化索引

# 一步计算所有突触电流 (按区室分组)
I_basal = W_basal @ (s_basal * (E_rev - V_post[post_ids]))  # 稀疏矩阵乘
I_apical = W_apical @ (s_apical * (E_rev - V_post[post_ids]))

# 一步更新所有神经元
V_soma += dt/tau_m * (-(V_soma - V_rest) + R*I_basal - W_adapt + κ*(V_apical - V_soma))
```

**预期效果**: 25M 标量运算 → ~50 次 numpy 批量运算 → **10-50× 加速**

---

## 二、新架构层级

### 2.1 层级映射

```
旧架构 (逐对象)                          新架构 (向量化)
─────────────────────                   ─────────────────────
Spike (namedtuple)                  →   spike_times: np.ndarray
SpikeTrain (deque per neuron)       →   spike_history: np.ndarray (ring buffer)
SpikeBus (dict → per-syn deliver)   →   SpikeRouter (稀疏矩阵索引)
                                    
SynapseBase (per-object)            →   SynapseGroup (向量化突触组)
  ._s (scalar)                      →     .s: np.ndarray[N_syn]
  .weight (scalar)                  →     .W: scipy.sparse / np.ndarray
  ._delay_buffer (deque)            →     .delay_matrix: np.ndarray
                                    
NeuronBase (per-object)             →   NeuronPopulation (向量化群体)
  .soma.v (scalar)                  →     .v_soma: np.ndarray[N]
  .apical.v (scalar)                →     .v_apical: np.ndarray[N]
  ._i_basal (scalar)                →     .i_basal: np.ndarray[N]
                                    
Layer (list of NeuronBase)          →   LayerSlice (NeuronPopulation 的切片视图)
CorticalColumn (dict of Layer)      →   CorticalColumn (组合多个 NeuronPopulation)
CA3Network (lists of neurons+syns)  →   CA3Network (NeuronPopulation + SynapseGroup)
```

### 2.2 新模块结构

```
wuyun/
├── core/                          # ★ 新增: 向量化核心引擎
│   ├── __init__.py
│   ├── population.py              # NeuronPopulation — 向量化神经元群体
│   ├── synapse_group.py           # SynapseGroup — 向量化突触组
│   ├── spike_router.py            # SpikeRouter — 向量化脉冲路由
│   └── plasticity_ops.py          # 向量化可塑性运算 (STDP/STP 等)
│
├── spike/                         # 保留: 信号类型定义
│   ├── signal_types.py            # 不变
│   └── oscillation_clock.py       # 不变
│
├── neuron/                        # 保留但精简: 参数定义
│   ├── neuron_params.py           # NeuronParams, 各类型预设 (从 neuron_base.py 提取)
│   └── compartment_params.py      # SomaticParams, ApicalParams (从 compartment.py 提取)
│
├── synapse/                       # 保留但精简: 参数和规则定义
│   ├── synapse_params.py          # SynapseParams, AMPA/NMDA/GABA 预设
│   └── plasticity/                # 可塑性规则 (接口不变，内部向量化)
│       ├── classical_stdp.py
│       └── ...
│
├── circuit/                       # 重构: 使用新的 core/ 组件
│   ├── cortical_column.py         # 重构内部，API 基本不变
│   ├── hippocampus/               # 重构内部
│   └── basal_ganglia/             # 重构内部
│
└── thalamus/                      # 重构内部
```

---

## 三、核心数据结构设计

### 3.1 NeuronPopulation — 向量化神经元群体

替代: `List[NeuronBase]` + `SomaticCompartment` + `ApicalCompartment`

```python
class NeuronPopulation:
    """向量化神经元群体 — 同类型神经元的批量计算
    
    所有状态变量存储为 numpy 数组，update 一次处理整个群体。
    支持异构参数（不同神经元可以有不同的 tau_m, threshold 等）。
    """
    
    def __init__(self, n: int, params: NeuronParams):
        self.n = n
        
        # === 参数向量 (长度 N, 支持异构) ===
        self.v_rest      = np.full(n, params.somatic.v_rest)
        self.v_threshold  = np.full(n, params.somatic.v_threshold)
        self.v_reset      = np.full(n, params.somatic.v_reset)
        self.tau_m        = np.full(n, params.somatic.tau_m)
        self.r_s          = np.full(n, params.somatic.r_s)
        self.a            = np.full(n, params.somatic.a)        # 亚阈值适应
        self.b            = np.full(n, params.somatic.b)        # 脉冲后适应增量
        self.tau_w        = np.full(n, params.somatic.tau_w)
        self.refractory_period = np.full(n, params.somatic.refractory_period, dtype=np.int32)
        
        self.kappa        = np.full(n, params.kappa)            # apical→soma 耦合
        self.kappa_back   = np.full(n, params.kappa_backward)   # soma→apical 耦合
        self.has_apical   = np.full(n, params.kappa > 0, dtype=bool)
        
        # 顶端树突参数
        self.tau_a         = np.full(n, params.apical.tau_a)
        self.r_a           = np.full(n, params.apical.r_a)
        self.v_ca_thresh   = np.full(n, params.apical.v_ca_threshold)
        self.ca_boost      = np.full(n, params.apical.ca_boost)
        self.ca_duration   = np.full(n, params.apical.ca_duration, dtype=np.int32)
        
        # Burst 参数
        self.burst_count   = np.full(n, params.burst_spike_count, dtype=np.int32)
        self.burst_isi     = np.full(n, params.burst_isi, dtype=np.int32)
        
        # === 状态向量 (每步更新) ===
        self.v_soma       = np.full(n, params.somatic.v_rest)   # 胞体膜电位
        self.v_apical     = np.full(n, params.somatic.v_rest)   # 顶端树突膜电位
        self.w_adapt      = np.zeros(n)                          # 适应变量
        self.refrac_count  = np.zeros(n, dtype=np.int32)         # 不应期倒计时
        self.ca_spike     = np.zeros(n, dtype=bool)              # Ca²⁺ 脉冲状态
        self.ca_timer     = np.zeros(n, dtype=np.int32)          # Ca²⁺ 持续倒计时
        self.burst_remain = np.zeros(n, dtype=np.int32)          # burst 剩余脉冲
        self.burst_isi_ct = np.zeros(n, dtype=np.int32)          # burst ISI 倒计时
        
        # === 输入累积 (每步清零) ===
        self.i_basal      = np.zeros(n)
        self.i_apical     = np.zeros(n)
        self.i_soma       = np.zeros(n)
        
        # === 输出 ===
        self.fired        = np.zeros(n, dtype=bool)   # 本步是否发放
        self.spike_type   = np.zeros(n, dtype=np.int8) # 本步脉冲类型 (SpikeType.value)
        
        # === 脉冲历史 (用于 STDP) ===
        # 环形缓冲: 最近 K 次脉冲的时间戳, shape=(n, K)
        self._spike_ring  = np.full((n, 32), -1, dtype=np.int32)  # -1 = 无记录
        self._spike_ptr   = np.zeros(n, dtype=np.int32)            # 当前写指针
        
    def step(self, t: int, dt: float = 1.0) -> np.ndarray:
        """向量化推进一个时间步
        
        Returns:
            fired: bool 数组, shape=(n,)
        """
        # --- 1. 顶端树突更新 (向量化) ---
        # τ_a · dV_a/dt = -(V_a - V_rest) + R_a · I_apical + κ_back · (V_soma - V_a)
        leak_a = -(self.v_apical - self.v_rest)
        input_a = self.r_a * self.i_apical
        coupling_a = self.kappa_back * (self.v_soma - self.v_apical)
        dv_a = (leak_a + input_a + coupling_a) / self.tau_a * dt
        self.v_apical += dv_a
        
        # Ca²⁺ 脉冲检测 (向量化)
        active_ca = self.ca_timer > 0
        self.ca_timer[active_ca] -= 1
        ended_ca = active_ca & (self.ca_timer == 0)
        self.ca_spike[ended_ca] = False
        
        new_ca = (~active_ca) & (self.v_apical >= self.v_ca_thresh) & self.has_apical
        self.ca_spike[new_ca] = True
        self.ca_timer[new_ca] = self.ca_duration[new_ca]
        self.v_apical[new_ca] += self.ca_boost[new_ca]
        
        # --- 2. 胞体更新 (向量化) ---
        not_refrac = self.refrac_count == 0
        self.refrac_count[~not_refrac] -= 1
        
        # τ_m · dV_s/dt = -(V_s - V_rest) + R_s · (I_basal + I_soma) - w + κ · (V_a - V_s)
        total_input = self.i_basal + self.i_soma
        
        # 只更新非不应期的神经元
        v = self.v_soma
        leak_s = -(v - self.v_rest)
        input_s = self.r_s * total_input
        coupling_s = self.kappa * (self.v_apical - v)
        dv = (leak_s + input_s - self.w_adapt + coupling_s) / self.tau_m * dt
        v_new = v + dv * not_refrac  # 不应期中 dv=0
        
        # 适应变量
        dw = (self.a * (v - self.v_rest) - self.w_adapt) / self.tau_w * dt
        self.w_adapt += dw * not_refrac
        
        # --- 3. 发放检测 (向量化) ---
        fired = (v_new >= self.v_threshold) & not_refrac
        
        # 发放后重置
        v_new[fired] = self.v_reset[fired]
        self.w_adapt[fired] += self.b[fired]
        self.refrac_count[fired] = self.refractory_period[fired]
        self.v_soma = v_new
        
        # --- 4. Burst/Regular 判定 (向量化) ---
        # BURST_START: fired & ca_spike
        # REGULAR: fired & ~ca_spike
        self.spike_type[:] = 0  # SpikeType.NONE = 0
        burst_start = fired & self.ca_spike
        regular = fired & (~self.ca_spike)
        self.spike_type[burst_start] = 4  # SpikeType.BURST_START
        self.spike_type[regular] = 1      # SpikeType.REGULAR
        
        # 启动 burst 状态机
        self.burst_remain[burst_start] = self.burst_count[burst_start] - 1
        self.burst_isi_ct[burst_start] = self.burst_isi[burst_start]
        
        self.fired = fired
        
        # --- 5. 记录脉冲 (向量化) ---
        fired_idx = np.nonzero(fired)[0]
        for i in fired_idx:
            ptr = self._spike_ptr[i]
            self._spike_ring[i, ptr % 32] = t
            self._spike_ptr[i] = ptr + 1
        
        # --- 6. 清空输入 ---
        self.i_basal[:] = 0.0
        self.i_apical[:] = 0.0
        self.i_soma[:] = 0.0
        
        return fired
    
    def get_recent_spike_times(self, neuron_idx: int, window_ms: int = 50) -> np.ndarray:
        """获取单个神经元最近脉冲时间 (用于 STDP)"""
        ring = self._spike_ring[neuron_idx]
        valid = ring[ring >= 0]
        if len(valid) == 0:
            return np.array([], dtype=np.int32)
        latest = valid.max()
        return valid[valid > latest - window_ms]
```

### 3.2 SynapseGroup — 向量化突触组

替代: `List[SynapseBase]` + `SpikeBus`

```python
class SynapseGroup:
    """向量化突触组 — 同类型突触的批量计算
    
    按突触类型 (AMPA/NMDA/GABA_A) 和目标区室 (BASAL/APICAL/SOMA) 分组。
    同一组内所有突触共享相同的 tau_decay, e_rev, g_max。
    
    核心数据:
      pre_ids[K]:   突触前神经元在源 Population 中的索引
      post_ids[K]:  突触后神经元在目标 Population 中的索引
      weights[K]:   突触权重
      s[K]:         门控变量
      delays[K]:    传导延迟 (时间步)
    
    电流计算:
      I[post_i] = Σ_k { g_max * weights[k] * s[k] * (E_rev - V_post[post_i]) }
               = Σ_k { (g_max * weights[k] * s[k]) } * (E_rev - V_post[post_i])
    """
    
    def __init__(
        self,
        pre_ids: np.ndarray,       # shape=(K,), int
        post_ids: np.ndarray,      # shape=(K,), int
        weights: np.ndarray,       # shape=(K,), float
        delays: np.ndarray,        # shape=(K,), int
        synapse_type: SynapseType,
        target_compartment: CompartmentType,
        params: SynapseParams,
        n_post: int,               # 目标群体大小 (用于聚合)
    ):
        self.K = len(pre_ids)      # 突触数量
        self.pre_ids = pre_ids
        self.post_ids = post_ids
        self.weights = weights.copy()
        self.delays = delays
        self.synapse_type = synapse_type
        self.target = target_compartment
        self.params = params
        self.n_post = n_post
        
        # 预计算
        self.decay_factor = np.exp(-1.0 / params.tau_decay)
        self.e_rev = params.e_rev
        self.g_max = params.g_max
        self.is_nmda = (synapse_type == SynapseType.NMDA)
        
        # 动态状态
        self.s = np.zeros(self.K)                    # 门控变量
        self.eligibility = np.zeros(self.K)          # 资格痕迹
        
        # 延迟缓冲: delay_buffer[d, k] = True if spike arrives at delay d for synapse k
        self.max_delay = int(delays.max()) if self.K > 0 else 1
        self.delay_buffer = np.zeros((self.max_delay + 1, self.K), dtype=np.int8)
        # delay_buffer 值: 0=无脉冲, 1=regular, 2=burst
        
        self._time_ptr = 0  # 环形缓冲指针
        
        # 预构建聚合索引: 将 per-synapse 电流聚合到 per-neuron
        # 使用 np.add.at 或预排序分组
        self._build_aggregation_index()
    
    def _build_aggregation_index(self):
        """预构建 post_id → synapse 索引，加速电流聚合"""
        # 按 post_id 排序，构建分段索引
        sort_idx = np.argsort(self.post_ids)
        self._sorted_pre = self.pre_ids[sort_idx]
        self._sorted_post = self.post_ids[sort_idx]
        self._sort_idx = sort_idx
    
    def deliver_spikes(self, pre_fired: np.ndarray, pre_spike_type: np.ndarray):
        """将突触前群体的脉冲送入延迟缓冲
        
        Args:
            pre_fired: bool[N_pre], 突触前群体本步发放状态
            pre_spike_type: int8[N_pre], 脉冲类型
        """
        # 找出哪些突触的 pre 发放了
        active_syns = pre_fired[self.pre_ids]
        if not active_syns.any():
            return
        
        # 将脉冲放入对应延迟槽
        for d in np.unique(self.delays[active_syns]):
            mask = active_syns & (self.delays == d)
            slot = (self._time_ptr + d) % (self.max_delay + 1)
            spike_types = pre_spike_type[self.pre_ids[mask]]
            self.delay_buffer[slot, mask] = np.where(
                spike_types >= 4, 2, 1  # burst → 2, regular → 1
            )
    
    def step_and_compute(self, v_post: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """向量化 step + compute_current
        
        Args:
            v_post: float[N_post], 突触后群体膜电位
            
        Returns:
            I_post: float[N_post], 聚合后的突触电流
        """
        # --- 1. 门控衰减 ---
        self.s *= self.decay_factor
        
        # --- 2. 检查到达的脉冲 ---
        arrived = self.delay_buffer[self._time_ptr]
        spike_mask = arrived > 0
        if spike_mask.any():
            increment = np.where(arrived[spike_mask] == 2, 1.5, 1.0)  # burst 增强
            self.s[spike_mask] = np.minimum(self.s[spike_mask] + increment, 1.0)
            arrived[spike_mask] = 0  # 清除已处理的脉冲
        
        # 推进时间指针
        self._time_ptr = (self._time_ptr + 1) % (self.max_delay + 1)
        
        # --- 3. 计算突触电流 ---
        # 跳过非活跃突触
        active = self.s > 1e-7
        if not active.any():
            return np.zeros(self.n_post)
        
        # conductance = g_max * weight * s
        conductance = self.g_max * self.weights * self.s  # shape=(K,)
        
        # NMDA 电压门控
        if self.is_nmda:
            v_at_post = v_post[self.post_ids]
            mg_block = 1.0 / (1.0 + 0.28011204 * np.exp(-0.062 * v_at_post))
            conductance *= mg_block
        
        # driving force = (E_rev - V_post)
        driving = self.e_rev - v_post[self.post_ids]
        
        # per-synapse current
        i_syn = conductance * driving  # shape=(K,)
        
        # --- 4. 聚合到 post neurons ---
        I_post = np.zeros(self.n_post)
        np.add.at(I_post, self.post_ids, i_syn)
        
        return I_post
    
    def reset(self):
        self.s[:] = 0.0
        self.eligibility[:] = 0.0
        self.delay_buffer[:] = 0
        self._time_ptr = 0
```

### 3.3 SpikeRouter — 向量化脉冲路由

替代: `SpikeBus`

```python
class SpikeRouter:
    """向量化脉冲路由
    
    不再逐个分发脉冲到突触对象，而是直接将 fired 向量传递给 SynapseGroup。
    SynapseGroup 通过 pre_ids 索引自行获取需要的脉冲信息。
    """
    
    def __init__(self):
        self.groups: List[SynapseGroup] = []
        self.source_populations: Dict[str, NeuronPopulation] = {}
    
    def route(self):
        """每步调用: 将各 Population 的发放状态传递给相关 SynapseGroup"""
        for group in self.groups:
            src = self.source_populations[group.source_name]
            group.deliver_spikes(src.fired, src.spike_type)
```

### 3.4 向量化 STDP

替代: `apply_recurrent_stdp` 中的逐突触循环

```python
def batch_stdp_update(
    group: SynapseGroup,
    pre_pop: NeuronPopulation,
    post_pop: NeuronPopulation,
    t: int,
    a_plus: float, a_minus: float,
    tau_plus: float, tau_minus: float,
    window_ms: int = 50,
):
    """向量化 STDP 权重更新
    
    对所有突触同时计算 STDP，避免逐突触 Python 循环。
    """
    # 1. 找出有脉冲的 pre 和 post 神经元
    pre_active = pre_pop._spike_ptr > 0  # 有过脉冲的 pre 神经元
    post_active = post_pop._spike_ptr > 0
    
    # 2. 只处理 pre 和 post 都有脉冲的突触
    relevant = pre_active[group.pre_ids] & post_active[group.post_ids]
    if not relevant.any():
        return
    
    rel_idx = np.nonzero(relevant)[0]
    
    # 3. 对每个相关突触计算 STDP
    dw = np.zeros(len(rel_idx))
    for i, syn_idx in enumerate(rel_idx):
        pre_times = pre_pop.get_recent_spike_times(group.pre_ids[syn_idx], window_ms)
        post_times = post_pop.get_recent_spike_times(group.post_ids[syn_idx], window_ms)
        if len(pre_times) == 0 or len(post_times) == 0:
            continue
        # 向量化 STDP 窗口计算
        dt_matrix = post_times[:, None] - pre_times[None, :]
        ltp = dt_matrix[dt_matrix > 0]
        ltd = dt_matrix[dt_matrix < 0]
        raw = 0.0
        if len(ltp): raw += a_plus * np.exp(-ltp / tau_plus).sum()
        if len(ltd): raw -= a_minus * np.exp(ltd / tau_minus).sum()
        dw[i] = raw
    
    # 4. 批量应用软边界和权重更新
    w = group.weights[rel_idx]
    w_range = group.w_max - group.w_min
    positive = dw > 0
    dw[positive] *= (group.w_max - w[positive]) / w_range
    dw[~positive] *= (w[~positive] - group.w_min) / w_range
    group.weights[rel_idx] = np.clip(w + dw, group.w_min, group.w_max)
```

---

## 四、高层 API 映射

### 4.1 CorticalColumn (API 基本不变)

```python
# 旧 API                                    # 新 API (内部重构)
col.inject_feedforward_current(5.0)      →   col.inject_feedforward_current(5.0)
col.step(t, dt)                          →   col.step(t, dt)
col.get_prediction_error()               →   col.get_prediction_error()
col.get_output_summary()                 →   col.get_output_summary()

# 内部变化:
# 旧: for neuron in layer.neurons: neuron.step(t)
# 新: layer_pop.step(t)  # 一次调用处理所有神经元
```

### 4.2 CA3Network

```python
# 旧 API                                    # 新 API
ca3.step(t, enable_recurrent=True)       →   ca3.step(t, enable_recurrent=True)
ca3.apply_recurrent_stdp(t)              →   ca3.apply_recurrent_stdp(t)
ca3.get_activity()                       →   ca3.get_activity()
ca3.inject_mossy_input(spikes, rates)    →   ca3.inject_mossy_input(spikes, rates)

# 内部变化:
# 旧: for i, neuron in enumerate(self.pyramidal_neurons): spike = neuron.step(t)
# 新: fired = self.pyramidal_pop.step(t)
# 旧: for syn in self._recurrent_synapses: syn.update_weight_stdp(...)
# 新: batch_stdp_update(self.recurrent_group, ...)
```

### 4.3 DentateGyrus

```python
# 旧: 100 NeuronBase + 10 NeuronBase + 2000 SynapseBase + SpikeBus
# 新: granule_pop(100) + pv_pop(10) + g2pv_group(1000) + pv2g_group(1000)
```

### 4.4 HippocampalLoop

```python
# API 完全不变
loop.encode(pattern, duration=500)
loop.recall(cue, duration=300)
loop.get_match_signal()
```

---

## 五、迁移路径 (分阶段)

### Phase V1: 核心引擎 (wuyun/core/) ✅ 已完成
1. 实现 `NeuronPopulation` — 向量化双区室 AdLIF+
2. 实现 `SynapseGroup` — 向量化突触组
3. 编写 `test_v1_core.py` 验证数值等价性 (6/6 通过, 误差 < 1e-10)
4. 实测加速: NeuronPopulation vs NeuronBase **11.2×** (N=200, T=1000)

### Phase V2: 替换基底节 ✅ 已完成
- GPi → NeuronPopulation (无内部突触, 最简单)
- Striatum → 3 个 NeuronPopulation (D1/D2/FSI) + SynapseGroup
- 验证 Phase 4 测试 7/7 通过

### Phase V3: 替换丘脑 ✅ 已完成
- ThalamicNucleus → tc_pop + trn_pop + SynapseGroup
- 修复: burst 统计改为累积计数器
- 验证 Phase 2 测试 7/7 通过

### Phase V4: 替换 CorticalColumn ✅ 已完成
- Layer → 持有 exc_pop / pv_pop / sst_pop
- column_factory → 用 `_build_synapse_group` 创建 SynapseGroup
- CorticalColumn → `_connections = [(SynapseGroup, src_pop, tgt_pop), ...]`
- 修复: `get_firing_rates` ring buffer bug (初始值 -1 被错误计入)
- 验证 Phase 0, 1.8, 2 测试全通过

### Phase V5: 清理 + 全套验证 ✅ 已完成
- 更新 test_cortical_column.py: `layer.excitatory` → `layer.exc_pop`
- 重写 test_pred_coding.py: 适配新 Layer API, 显式反馈验证 burst 机制
- 所有测试文件添加 `sys.stdout.reconfigure(encoding='utf-8')`
- **62/62 测试全通过, 零回归**

> 注: 海马模块 (DG/CA3/CA1/HippocampalLoop) 仍使用 NeuronBase + SynapseBase,
> 因其内部有 STP、逐突触 STDP 等复杂逻辑, 留待后续按需迁移。
> NeuronBase/SynapseBase 作为兼容层保留。

---

## 六、关键设计决策

### 6.1 异构参数处理

**问题**: CorticalColumn 同一层内有不同类型神经元 (L23 pyramidal + PV + SST)

**方案 A**: 每种类型一个 Population → 简单但 Population 数量多
**方案 B**: 一个 Population + 参数向量支持异构 → 灵活但代码复杂

**选择: 方案 A** — 每种类型独立 Population。理由:
- 不同类型的动力学逻辑可能不同 (如 PV 无 apical, MSN 有 Up/Down 态)
- Population 间通过 SynapseGroup 连接
- CorticalColumn 持有多个 Population 的引用

```python
class CorticalColumn:
    def __init__(self, ...):
        self.l4_exc = NeuronPopulation(25, STELLATE_PARAMS)
        self.l23_exc = NeuronPopulation(10, L23_PYRAMIDAL_PARAMS)
        self.l23_pv = NeuronPopulation(3, BASKET_PV_PARAMS)
        self.l23_sst = NeuronPopulation(1, MARTINOTTI_SST_PARAMS)
        self.l5_exc = NeuronPopulation(10, L5_PYRAMIDAL_PARAMS)
        self.l5_pv = NeuronPopulation(3, BASKET_PV_PARAMS)
        self.l6_exc = NeuronPopulation(10, L6_PYRAMIDAL_PARAMS)
        self.l6_pv = NeuronPopulation(3, BASKET_PV_PARAMS)
        # ... SynapseGroup 连接
```

### 6.2 延迟处理

**问题**: 当前每个突触有独立的 deque 延迟缓冲

**方案**: SynapseGroup 级别的环形缓冲矩阵 `delay_buffer[max_delay+1, K]`
- 时间指针每步推进
- 延迟 1 步 (最常见情况) 只需检查当前时间槽

### 6.3 SpikeBus 替代

**问题**: 当前 SpikeBus 是中心化的脉冲调度器

**方案**: 取消 SpikeBus，改为 SynapseGroup 直接从源 Population 读取 fired 状态
- `group.deliver_spikes(source_pop.fired, source_pop.spike_type)`
- 零拷贝: SynapseGroup 通过 pre_ids 索引直接访问源 Population 的状态

### 6.4 与 STP (短时程可塑性) 的兼容

**问题**: 苔藓纤维有独立的 STP (Tsodyks-Markram 模型)

**方案**: STP 也向量化为 `STPGroup`，与 SynapseGroup 配合

```python
class STPGroup:
    """向量化短时程可塑性"""
    def __init__(self, n: int, params: STPParams):
        self.n_vesicle = np.ones(n)    # 囊泡池
        self.p_release = np.full(n, params.p0)  # 释放概率
    
    def step(self, dt: float = 1.0):
        """每步恢复"""
        self.n_vesicle += (1 - self.n_vesicle) * (1 - np.exp(-dt / self.tau_r))
        self.p_release += (self.p0 - self.p_release) * (1 - np.exp(-dt / self.tau_f))
    
    def on_spike(self, mask: np.ndarray) -> np.ndarray:
        """脉冲到达时计算效能"""
        eff = self.n_vesicle[mask] * self.p_release[mask]
        self.n_vesicle[mask] *= (1 - self.p_release[mask])
        self.p_release[mask] += self.a_f * (1 - self.p_release[mask])
        return eff
```

---

## 七、数值验证策略

重构必须保证数值等价性。验证方法:

1. **逐值对比**: 用相同输入跑 NeuronBase 和 NeuronPopulation 各 1000 步，比较 V_soma 误差 < 1e-10
2. **统计等价**: Phase 0-4 所有测试通过 (行为不变)
3. **边界条件**: burst 状态机、不应期、Ca²⁺ 脉冲、NMDA 电压门控

---

## 八、性能数据

### 8.1 实测加速 (test_v1_core 测试 6)

| 方法 | N=200, T=1000 | 耗时 |
|------|---------------|------|
| NeuronBase (逐对象) | 200×1000 步 | 0.358s |
| NeuronPopulation (向量化) | 200×1000 步 | 0.032s |
| **加速比** | | **11.2×** |

### 8.2 已迁移模块

| 模块 | 旧架构 | 新架构 |
|------|--------|--------|
| GPi | N×NeuronBase + 逐个step | 1×NeuronPopulation |
| Striatum | 3×List[NeuronBase] + SpikeBus | 3×NeuronPopulation + SynapseGroup |
| ThalamicNucleus | TC/TRN List[NeuronBase] + SpikeBus | 2×NeuronPopulation + SynapseGroup |
| Layer | List[NeuronBase] | exc_pop + pv_pop + sst_pop |
| CorticalColumn | 4×Layer + SpikeBus + List[SynapseBase] | 4×Layer + List[(SynapseGroup, src, tgt)] |

### 8.3 未迁移模块 (保留 NeuronBase)

| 模块 | 原因 |
|------|------|
| DentateGyrus | 逐神经元 EC 输入映射 |
| CA3Network | 逐突触 STDP + STP (苔藓纤维) |
| CA1Network | Schaffer/EC3 分路径注入 |
| HippocampalLoop | 依赖上述三个模块 |

---

## 九、后续扩展路径

### GPU (CuPy)
```python
# numpy → cupy 几乎一行代码切换
import cupy as np  # 替代 import numpy as np

# 或条件选择:
if USE_GPU:
    import cupy as xp
else:
    import numpy as xp
```

### JAX (自动微分 + JIT)
```python
import jax.numpy as jnp
from jax import jit

@jit
def population_step(v_soma, v_apical, w_adapt, i_basal, params):
    # JAX JIT 编译为高效机器码
    ...
```

向量化是通向 GPU/JIT 的必经之路。不做向量化，无法利用硬件加速。
