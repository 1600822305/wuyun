#include "core/spike_bus.h"
#include <algorithm>

namespace wuyun {

SpikeBus::SpikeBus(int32_t max_delay)
    : max_delay_(max_delay)
    , delay_buffer_(static_cast<size_t>(max_delay + 1))
{}

uint32_t SpikeBus::register_region(const std::string& name, size_t n_neurons) {
    uint32_t id = static_cast<uint32_t>(region_names_.size());
    region_names_.push_back(name);
    region_sizes_.push_back(n_neurons);
    return id;
}

void SpikeBus::add_projection(uint32_t src_region, uint32_t dst_region,
                               int32_t delay, const std::string& name) {
    projections_.push_back({src_region, dst_region, delay, name});
}

void SpikeBus::submit_spikes(uint32_t region_id,
                              const std::vector<uint8_t>& fired,
                              const std::vector<int8_t>& spike_type,
                              int32_t t) {
    // For each projection from this region, schedule spikes with delay
    for (const auto& proj : projections_) {
        if (proj.src_region != region_id) continue;

        int32_t arrival_t = t + proj.delay;
        size_t slot = static_cast<size_t>(arrival_t % (max_delay_ + 1));

        for (size_t i = 0; i < fired.size(); ++i) {
            if (!fired[i]) continue;
            delay_buffer_[slot].push_back({
                region_id,
                proj.dst_region,
                static_cast<uint32_t>(i),
                spike_type[i],
                arrival_t
            });
        }
    }
}

std::vector<SpikeEvent> SpikeBus::get_arriving_spikes(uint32_t dst_region, int32_t t) const {
    std::vector<SpikeEvent> result;
    size_t slot = static_cast<size_t>(t % (max_delay_ + 1));

    for (const auto& evt : delay_buffer_[slot]) {
        if (evt.timestamp != t) continue;
        if (evt.dst_region != dst_region) continue;
        result.push_back(evt);
    }
    return result;
}

void SpikeBus::advance(int32_t t) {
    // Clear the slot that will be reused next
    int32_t clear_t = t + max_delay_ + 1;
    size_t slot = static_cast<size_t>(clear_t % (max_delay_ + 1));
    delay_buffer_[slot].clear();
}

} // namespace wuyun
