/**
 * Debug: trace signal propagation through cortical column layers
 */

#include "circuit/cortical_column.h"
#include "core/types.h"
#include <cstdio>
#include <vector>
#include <numeric>

using namespace wuyun;

int main() {
    printf("=== Column Signal Propagation Debug ===\n\n");

    ColumnConfig cfg;
    cfg.n_l23_pyramidal = 50;
    cfg.n_l4_stellate = 30;
    cfg.n_l5_pyramidal = 30;
    cfg.n_l6_pyramidal = 20;
    cfg.n_pv_basket = 10;
    cfg.n_sst_martinotti = 5;
    cfg.n_vip = 3;
    CorticalColumn col(cfg);

    printf("Neurons: L4=%zu L23=%zu L5=%zu L6=%zu PV=%zu SST=%zu VIP=%zu\n",
           cfg.n_l4_stellate, cfg.n_l23_pyramidal, cfg.n_l5_pyramidal,
           cfg.n_l6_pyramidal, cfg.n_pv_basket, cfg.n_sst_martinotti, cfg.n_vip);
    printf("Total synapses: %zu\n\n", col.total_synapses());

    std::vector<float> ff(cfg.n_l4_stellate, 15.0f);
    std::vector<float> fb_l23(cfg.n_l23_pyramidal, 20.0f);
    std::vector<float> fb_l5(cfg.n_l5_pyramidal, 20.0f);

    // Phase 1: Feedforward only
    printf("--- Phase 1: Feedforward only (15.0 to L4) ---\n");
    for (int t = 0; t < 50; ++t) {
        col.inject_feedforward(ff);
        auto out = col.step(t);

        // Count fired per layer
        size_t l4_fired = 0, l23_fired = 0, l5_fired = 0, l6_fired = 0;
        size_t pv_fired = 0, sst_fired = 0, vip_fired = 0;

        for (size_t i = 0; i < col.l4().size(); ++i)  l4_fired  += col.l4().fired()[i];
        for (size_t i = 0; i < col.l23().size(); ++i) l23_fired += col.l23().fired()[i];
        for (size_t i = 0; i < col.l5().size(); ++i)  l5_fired  += col.l5().fired()[i];
        for (size_t i = 0; i < col.l6().size(); ++i)  l6_fired  += col.l6().fired()[i];

        if (l4_fired > 0 || l23_fired > 0 || l5_fired > 0 || l6_fired > 0) {
            printf("  t=%3d | L4:%2zu L23:%2zu L5:%2zu L6:%2zu | reg=%zu burst=%zu drive=%zu\n",
                   t, l4_fired, l23_fired, l5_fired, l6_fired,
                   out.n_regular, out.n_burst, out.n_drive);
        }
    }

    // Check L4 membrane potential
    printf("\n  L4 v_soma[0..4]: ");
    for (size_t i = 0; i < 5 && i < col.l4().size(); ++i) {
        printf("%.1f ", col.l4().v_soma()[i]);
    }
    printf("\n  L23 v_soma[0..4]: ");
    for (size_t i = 0; i < 5 && i < col.l23().size(); ++i) {
        printf("%.1f ", col.l23().v_soma()[i]);
    }
    printf("\n");

    // Phase 2: FF + FB
    printf("\n--- Phase 2: FF(15) + FB(20 to L23 apical) ---\n");
    CorticalColumn col2(cfg);
    for (int t = 0; t < 50; ++t) {
        col2.inject_feedforward(ff);
        col2.inject_feedback(fb_l23, fb_l5);
        auto out = col2.step(t);

        size_t l4_fired = 0, l23_fired = 0;
        for (size_t i = 0; i < col2.l4().size(); ++i)  l4_fired  += col2.l4().fired()[i];
        for (size_t i = 0; i < col2.l23().size(); ++i) l23_fired += col2.l23().fired()[i];

        if (l4_fired > 0 || l23_fired > 0) {
            printf("  t=%3d | L4:%2zu L23:%2zu | reg=%zu burst=%zu\n",
                   t, l4_fired, l23_fired, out.n_regular, out.n_burst);
        }
    }

    printf("\n  L23 v_soma[0..4]: ");
    for (size_t i = 0; i < 5 && i < col2.l23().size(); ++i) {
        printf("%.1f ", col2.l23().v_soma()[i]);
    }
    printf("\n  L23 v_apical[0..4]: ");
    for (size_t i = 0; i < 5 && i < col2.l23().size(); ++i) {
        printf("%.1f ", col2.l23().v_apical()[i]);
    }
    printf("\n");

    return 0;
}
