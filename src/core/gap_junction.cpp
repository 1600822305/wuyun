#include "core/gap_junction.h"
#include <cmath>

namespace wuyun {

GapJunctionGroup::GapJunctionGroup(size_t n_neurons)
    : n_(n_neurons)
{}

void GapJunctionGroup::add_connection(int32_t a, int32_t b, float g_gap) {
    connections_.push_back({a, b, g_gap});
}

std::vector<float> GapJunctionGroup::compute_currents(
    const std::vector<float>& v_membrane
) const {
    std::vector<float> currents(n_, 0.0f);

    for (const auto& conn : connections_) {
        size_t a = static_cast<size_t>(conn.neuron_a);
        size_t b = static_cast<size_t>(conn.neuron_b);
        float i = conn.g_gap * (v_membrane[a] - v_membrane[b]);
        // Bidirectional: A pulls B toward A, B pulls A toward B
        currents[b] += i;   // Current into B (positive if V_a > V_b)
        currents[a] -= i;   // Current into A (opposite direction)
    }

    return currents;
}

} // namespace wuyun
