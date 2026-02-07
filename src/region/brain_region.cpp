#include "region/brain_region.h"

namespace wuyun {

BrainRegion::BrainRegion(const std::string& name, size_t n_neurons)
    : name_(name)
    , n_neurons_(n_neurons)
{}

void BrainRegion::register_to_bus(SpikeBus& bus) {
    region_id_ = bus.register_region(name_, n_neurons_);
}

} // namespace wuyun
