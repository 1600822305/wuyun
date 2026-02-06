#include "core/spike_queue.h"

namespace wuyun {

SpikeQueue::SpikeQueue(int max_delay, size_t n_neurons)
    : max_delay_(max_delay)
    , n_neurons_(n_neurons)
    , ring_buffer_(static_cast<size_t>(max_delay + 1))
{
}

void SpikeQueue::enqueue(
    const std::vector<uint8_t>& fired,
    const std::vector<int32_t>& delays,
    int current_step
) {
    for (size_t i = 0; i < n_neurons_; ++i) {
        if (!fired[i]) continue;
        int delay = delays[i];
        if (delay < 0) delay = 0;
        if (delay > max_delay_) delay = max_delay_;
        int slot = (current_step + delay) % (max_delay_ + 1);
        ring_buffer_[static_cast<size_t>(slot)].push_back(static_cast<int32_t>(i));
    }
}

const std::vector<int32_t>& SpikeQueue::dequeue(int current_step) {
    int slot = current_step % (max_delay_ + 1);
    dequeue_buf_.swap(ring_buffer_[static_cast<size_t>(slot)]);
    ring_buffer_[static_cast<size_t>(slot)].clear();
    return dequeue_buf_;
}

void SpikeQueue::clear() {
    for (auto& slot : ring_buffer_) {
        slot.clear();
    }
    dequeue_buf_.clear();
}

} // namespace wuyun
