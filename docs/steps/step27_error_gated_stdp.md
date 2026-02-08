# Step 27: 预测编码学习 + error-gated STDP

> 日期: 2026-02-08
> 状态: ✅ 完成

- L6→L2/3 预测突触组 + STDP: L6 学习预测 L2/3 活动
- error-gated STDP: 只有 regular spike (预测误差) 触发 L4→L2/3 LTP, burst (匹配) 不更新
- `SynapseGroup::apply_stdp_error_gated()` 新接口
- 发育期逻辑: `dev_period_steps` 步无奖励视觉发育
