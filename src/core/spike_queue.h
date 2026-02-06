#pragma once
/**
 * SpikeQueue — 环形缓冲延迟队列
 *
 * 管理突触传导延迟：脉冲事件入队后，经过指定延迟步数才被投递。
 * 使用环形缓冲实现，O(1) 入队和出队。
 */

#include <vector>
#include <cstdint>

namespace wuyun {

class SpikeQueue {
public:
    /**
     * @param max_delay  最大延迟步数
     * @param n_neurons  神经元数量 (用于分配 fired 缓冲)
     */
    SpikeQueue(int max_delay, size_t n_neurons);

    /**
     * 将发放的神经元按延迟入队
     *
     * @param fired        当前步发放的神经元 (bool 数组)
     * @param delays       每个神经元的延迟步数 (0 = 立即投递)
     * @param current_step 当前仿真时间步
     */
    void enqueue(const std::vector<uint8_t>& fired,
                 const std::vector<int32_t>& delays,
                 int current_step);

    /**
     * 取出当前步到期的所有脉冲事件
     *
     * @param current_step 当前仿真时间步
     * @return 到期发放的神经元 ID 列表
     */
    const std::vector<int32_t>& dequeue(int current_step);

    /** 清空队列 */
    void clear();

private:
    int max_delay_;
    size_t n_neurons_;

    // ring_buffer_[slot] = 该 slot 到期的神经元 ID 列表
    std::vector<std::vector<int32_t>> ring_buffer_;

    // dequeue 返回缓冲 (避免每次分配)
    std::vector<int32_t> dequeue_buf_;
};

} // namespace wuyun
