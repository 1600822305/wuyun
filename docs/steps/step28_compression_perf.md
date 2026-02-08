# Step 28: 信息量压缩 + 树突 mismatch 可塑性 + SNN 性能优化

> 日期: 2026-02-08
> 状态: ✅ 完成
> 核心突破: 1100→120 神经元 (9×压缩), 37秒→2.3秒 (16×加速)

## 信息量驱动神经元分配

每个神经元有明确信息论意义, 按 `n_pixels` 和 `n_actions` 自动计算:
```
25 像素输入: LGN=25, V1=25, V2=15, V4=8, IT=8, dlPFC=12, BG=8+8, M1=20
总计 ~120 神经元 (之前 ~1100)
```

## 树突 somato-dendritic mismatch (Sacramento/Guerguiev 2018)

`|V_apical - V_soma|` 调制 STDP 幅度, 数学上等价于反向传播:
```cpp
float mismatch = abs(v_apical - v_soma) / 30.0;
float effective_a_plus = a_plus * (0.1 + 0.9 * mismatch);
```
误差大→学得多, 误差小→不学。

## SNN 底层性能优化

- `step_and_compute()` 返回 `const&` 零拷贝 (消除每步 ~50 次 vector 拷贝)
- NMDA B(V) 256 档查表替代 `std::exp()` (每步省 ~30K 次 exp)
- SpikeBus 预分配 `reserve(256)` + 返回引用
- `NeuronPopulation::step()` 用 memset + fire count 合并到主循环
- `deliver_spikes()` 未发放神经元快速跳过

## 结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 闭环神经元 | ~1100 | ~120 |
| 6测试时间 | 37秒 | **2.3秒** (16×) |
| CTest 29套件 | 3.2秒 | **2.9秒** |
| 泛化 | -0.110 | **-0.048** |
| Learner advantage | -0.001 | **+0.011** |
