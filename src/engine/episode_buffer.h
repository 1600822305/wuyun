#pragma once
/**
 * EpisodeBuffer — 经验片段记录器 (Awake SWR Replay)
 *
 * 记录闭环 agent 每个 brain step 的皮层 spike 快照,
 * 供奖励事件后的 "awake sharp-wave ripple" 重放使用。
 *
 * 生物学基础:
 *   - 海马 CA1 place cells 在导航中持续记录空间-行为序列
 *   - 获得奖励后 100-300ms 内, CA3→CA1 产生 awake SWR
 *   - SWR 期间, 最近经验以压缩时间尺度 (~5-20x) 重放
 *   - 重放驱动 VTA DA burst → 纹状体 DA-STDP 二次强化
 *   - 效果: 1 次奖励事件 → 10-20 次突触权重更新
 *
 * 设计: 环形缓冲区, 存储最近 N 个 agent step 的 spike 序列
 */

#include "core/spike_bus.h"
#include <vector>
#include <deque>
#include <cstdint>

namespace wuyun {

/** 单个 brain step 的皮层 spike 快照 */
struct SpikeSnapshot {
    std::vector<SpikeEvent> cortical_events;  // dlPFC → BG 的 spike events
    std::vector<SpikeEvent> sensory_events;   // V1 → dlPFC 的 spike events (皮层巩固用)
    int action_group = -1;                     // 当前探索方向 (efference copy)
};

/** 单个 agent step 的完整经验片段 */
struct Episode {
    std::vector<SpikeSnapshot> steps;  // brain_steps_per_action 个快照
    float reward = 0.0f;              // 该 step 获得的奖励
    int action = -1;                   // 执行的动作 (Action enum)
};

/**
 * 环形缓冲区: 存储最近 max_episodes 个 agent step 的经验
 */
class EpisodeBuffer {
public:
    explicit EpisodeBuffer(size_t max_episodes = 30, size_t brain_steps = 15)
        : max_episodes_(max_episodes)
        , brain_steps_(brain_steps)
    {
        current_.steps.reserve(brain_steps);
    }

    /** 开始记录新的 agent step */
    void begin_episode() {
        current_.steps.clear();
        current_.reward = 0.0f;
        current_.action = -1;
    }

    /** 记录一个 brain step 的 spike 快照 */
    void record_step(const std::vector<SpikeEvent>& cortical_events,
                     int action_group,
                     const std::vector<SpikeEvent>& sensory_events = {}) {
        SpikeSnapshot snap;
        snap.cortical_events = cortical_events;
        snap.sensory_events = sensory_events;
        snap.action_group = action_group;
        current_.steps.push_back(std::move(snap));
    }

    /** 结束当前 episode, 设置奖励和动作 */
    void end_episode(float reward, int action) {
        current_.reward = reward;
        current_.action = action;
        buffer_.push_back(std::move(current_));
        if (buffer_.size() > max_episodes_) {
            buffer_.pop_front();
        }
        current_ = Episode{};
        current_.steps.reserve(brain_steps_);
    }

    /** 获取最近的 N 个 episodes (从最新到最旧) */
    std::vector<const Episode*> recent(size_t n = 5) const {
        std::vector<const Episode*> result;
        size_t count = std::min(n, buffer_.size());
        for (size_t i = 0; i < count; ++i) {
            result.push_back(&buffer_[buffer_.size() - 1 - i]);
        }
        return result;
    }

    /** 获取最近一个有显著奖励的 episode */
    const Episode* last_rewarded(float threshold = 0.05f) const {
        for (auto it = buffer_.rbegin(); it != buffer_.rend(); ++it) {
            if (std::abs(it->reward) > threshold) {
                return &(*it);
            }
        }
        return nullptr;
    }

    size_t size() const { return buffer_.size(); }
    bool empty() const { return buffer_.empty(); }

    /** v53: 清空缓冲区 (反转学习: 旧世界经验不适用新布局) */
    void clear() { buffer_.clear(); }

private:
    size_t max_episodes_;
    size_t brain_steps_;
    std::deque<Episode> buffer_;
    Episode current_;
};

} // namespace wuyun
