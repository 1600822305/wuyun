# Step 58: MultiRoomEnv 多房间迷宫环境

## 动机

Step 57 创建了 Environment 抽象接口，将 ClosedLoopAgent 与 GridWorld 解耦。
但如果没有第二个环境实现，接口只是"纸上谈兵"。

MultiRoomEnv 的目标：
1. **验证接口** — 完全独立于 GridWorld 的环境实现，同一个大脑能无缝运行
2. **挑战升级** — 开放场地太简单（随机游走就能觅食），多房间需要导航能力
3. **暴露瓶颈** — 看到 agent 在哪里失败，指导下一步改进方向

## 设计

### 房间布局

```
2×2 rooms, room_w=4, room_h=4 → 11×11 grid

###########
#....#D...#      房间 (0,0)    房间 (1,0)
#F.F......#      4×4 内部空间  4×4 内部空间
#..A.#....#      (x:1-4, y:1-4) (x:6-9, y:1-4)
#....#....#
####D#.####  ← 墙壁行，门道是单格缺口
#....#....#
#D.F.#....#      房间 (0,1)    房间 (1,1)
#...F#....#
#.....F...#
###########

总网格: n_rooms_x * (room_w + 1) + 1 = 2*(4+1)+1 = 11
```

### 与 GridWorld 的关键区别

| 特性 | GridWorld | MultiRoomEnv |
|------|-----------|-------------|
| 依赖 | GridWorld 类 | 完全独立 |
| 墙壁 | 边界 + 可选迷宫 | 房间间墙壁 + 门道 |
| 导航难度 | 开放场地可随机游走 | 必须找到门道穿越 |
| 食物分布 | 全地图随机 | 跨房间分布 |
| 碰撞 | 边界反弹 | 墙壁碰撞 + 轴向滑动 |

### Environment 接口实现

```cpp
class MultiRoomEnv : public Environment {
    // 完全实现 Environment 纯虚接口
    void reset() override;
    void reset_with_seed(uint32_t seed) override;
    std::vector<float> observe() const override;     // 5×5 视觉 patch
    Result step(float dx, float dy) override;        // 连续移动 + 碰撞
    float pos_x/pos_y() const override;              // 海马空间信息
    uint32_t positive/negative_count() const override; // 进化统计
};
```

### 碰撞处理

```
1. 计算目标位置 (nx, ny) = (fx + dx, fy + dy)
2. 目标格子是墙 → 尝试轴向滑动:
   a. 只沿 X 滑: floor(fx+dx) 可通行? → 移动 X
   b. 只沿 Y 滑: floor(fy+dy) 可通行? → 移动 Y
   c. 都不行 → 不动
3. 目标格子可通行 → 正常移动
```

### cell_changed 保护 (v55 danger trap fix)

```cpp
int prev_ix = agent_ix_, prev_iy = agent_iy_;
// ... 移动逻辑 ...
bool cell_changed = (agent_ix_ != prev_ix || agent_iy_ != prev_iy);
if (cell_changed) {
    // 只在换格子时触发 food/danger
}
```

没有这个保护，连续小步在 danger 格子内 → 每步 -1 → 300 步 danger=63。

## 文件

| 文件 | 内容 |
|------|------|
| `src/engine/multi_room_env.h` | MultiRoomConfig + MultiRoomEnv 类声明 |
| `src/engine/multi_room_env.cpp` | 房间生成 + 碰撞 + 观测 + 统计 |
| `tests/cpp/test_multi_room.cpp` | 6 个测试 |
| `tools/benchmark_multiroom.cpp` | 5 seeds × 300 steps 性能诊断 |

## 测试 (6/6 通过)

1. **房间生成** — 2×2 rooms → 11×11 grid，agent 在第一个房间
2. **移动+碰撞** — 墙壁阻挡，边界检测
3. **食物交互** — 500 步随机走，计数正确
4. **观测格式** — 5×5 patch，中心=agent (0.6)
5. **ClosedLoopAgent 闭环** — MultiRoomEnv 插入 agent 跑 100 步无崩溃
6. **Reset** — reset/reset_with_seed 统计归零

Test 5 是核心验证：同一个大脑（64 区域、~252 神经元）通过 Environment 接口
接入完全不同的环境，无需任何脑区代码修改。

## Benchmark 结果 (300 步 × 5 seeds)

```
seed= 42 | food= 2 danger= 0 | ✅ 起始房间觅食
seed= 77 | food= 1 danger= 2 | ✅ 穿门道到下方房间
seed=123 | food= 0 danger= 0 | ❌ 几乎没动
seed=256 | food= 0 danger= 0 | ❌ 卡在角落
seed=789 | food= 2 danger= 0 | ✅ 穿到右侧房间

Avg: food=1.0  danger=0.4
```

### 失败分析

2/5 seeds 卡死，根因：
1. **无墙壁回避** — 撞墙后不知道转向，反复撞同一面墙
2. **探索不足** — M1 noise 固定，长时间无奖赏也不加大搜索范围
3. **门道太窄** — 单格缝隙，随机游走命中概率低

→ Step 59 将添加墙壁回避反射 + 探索驱动重置来解决。

## 验证

- 32/32 CTest 零回归
- Environment 接口完全验证：两个独立实现 (GridWorldEnv + MultiRoomEnv) 都能接入同一个 ClosedLoopAgent
