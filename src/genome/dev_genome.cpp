#include "genome/dev_genome.h"
#include <cstdio>
#include <sstream>
#include <algorithm>

namespace wuyun {

// 固定条形码: LGN 和 BG 的分子身份 (不进化)
const float DevGenome::LGN_BARCODE[BARCODE_DIM] =
    {1.0f, 0.8f, 0.1f, 0.0f, 0.2f, 0.0f, 0.1f, 0.0f};
const float DevGenome::BG_BARCODE[BARCODE_DIM] =
    {0.0f, 0.1f, 0.8f, 1.0f, 0.0f, 0.2f, 0.1f, 0.0f};

DevGenome::DevGenome() {
    // 皮层条形码
    for (int t = 0; t < N_CORTICAL_TYPES; ++t) {
        for (int d = 0; d < BARCODE_DIM; ++d) {
            char name[32];
            snprintf(name, sizeof(name), "ctx%d_bc%d", t, d);
            cortical_barcode[t][d] = {name, 0.5f, 0.0f, 1.0f};
        }
        char dn[32], in[32];
        snprintf(dn, sizeof(dn), "ctx%d_div", t);
        snprintf(in, sizeof(in), "ctx%d_inh", t);
        cortical_division[t] = {dn, 4.0f, 2.0f, 7.0f};
        cortical_inh_frac[t] = {in, 0.2f, 0.05f, 0.4f};
    }
    // 兼容性矩阵
    for (int i = 0; i < BARCODE_DIM; ++i) {
        for (int j = 0; j < BARCODE_DIM; ++j) {
            char name[16];
            snprintf(name, sizeof(name), "W%d%d", i, j);
            float def = (i == j) ? 0.3f : 0.0f;
            w_connect[i][j] = {name, def, -1.0f, 1.0f};
        }
    }
    // 皮层→BG 接口条形码
    for (int d = 0; d < BARCODE_DIM; ++d) {
        char name[16];
        snprintf(name, sizeof(name), "c2bg_%d", d);
        // 默认: 与 BG 条形码兼容 (维度 2,3 高)
        cortical_to_bg[d] = {name, BG_BARCODE[d], 0.0f, 1.0f};
    }
}

// =============================================================================
// 条形码兼容性
// =============================================================================

float DevGenome::barcode_compat(const float bc_a[BARCODE_DIM],
                                 const float bc_b[BARCODE_DIM]) const {
    float result = 0.0f;
    for (int i = 0; i < BARCODE_DIM; ++i)
        for (int j = 0; j < BARCODE_DIM; ++j)
            result += bc_a[i] * w_connect[i][j].value * bc_b[j];
    return result;
}

float DevGenome::conn_prob_from_compat(float compat) const {
    float x = compat - connect_threshold.value;
    return 1.0f / (1.0f + std::exp(-x * 3.0f));
}

// =============================================================================
// 基因操作
// =============================================================================

std::vector<Gene*> DevGenome::all_genes() {
    std::vector<Gene*> g;
    // 固定回路参数
    g.push_back(&bg_size); g.push_back(&da_stdp_lr); g.push_back(&bg_gain);
    g.push_back(&vta_size); g.push_back(&da_phasic_gain);
    g.push_back(&thal_size); g.push_back(&thal_gate);
    g.push_back(&lgn_gain); g.push_back(&lgn_baseline);
    g.push_back(&motor_noise); g.push_back(&reward_scale);
    g.push_back(&amyg_size); g.push_back(&hipp_size);
    g.push_back(&homeo_target); g.push_back(&homeo_eta);
    g.push_back(&ne_floor); g.push_back(&replay_passes); g.push_back(&dev_period);
    // 先验基因
    g.push_back(&hedonic_gain); g.push_back(&fear_valence);
    g.push_back(&sensory_motor); g.push_back(&explore_drive); g.push_back(&approach_bias);
    // v52: 反射弧基因 + 一次学习
    g.push_back(&sc_approach); g.push_back(&pag_freeze); g.push_back(&novelty_boost);
    // 皮层条形码
    for (int t = 0; t < N_CORTICAL_TYPES; ++t)
        for (int d = 0; d < BARCODE_DIM; ++d)
            g.push_back(&cortical_barcode[t][d]);
    // 皮层大小+抑制
    for (int t = 0; t < N_CORTICAL_TYPES; ++t) g.push_back(&cortical_division[t]);
    for (int t = 0; t < N_CORTICAL_TYPES; ++t) g.push_back(&cortical_inh_frac[t]);
    // 兼容性矩阵
    for (int i = 0; i < BARCODE_DIM; ++i)
        for (int j = 0; j < BARCODE_DIM; ++j)
            g.push_back(&w_connect[i][j]);
    g.push_back(&connect_threshold);
    // 接口条形码
    for (int d = 0; d < BARCODE_DIM; ++d) g.push_back(&cortical_to_bg[d]);
    return g;
}

std::vector<const Gene*> DevGenome::all_genes() const {
    std::vector<const Gene*> g;
    g.push_back(&bg_size); g.push_back(&da_stdp_lr); g.push_back(&bg_gain);
    g.push_back(&vta_size); g.push_back(&da_phasic_gain);
    g.push_back(&thal_size); g.push_back(&thal_gate);
    g.push_back(&lgn_gain); g.push_back(&lgn_baseline);
    g.push_back(&motor_noise); g.push_back(&reward_scale);
    g.push_back(&amyg_size); g.push_back(&hipp_size);
    g.push_back(&homeo_target); g.push_back(&homeo_eta);
    g.push_back(&ne_floor); g.push_back(&replay_passes); g.push_back(&dev_period);
    g.push_back(&hedonic_gain); g.push_back(&fear_valence);
    g.push_back(&sensory_motor); g.push_back(&explore_drive); g.push_back(&approach_bias);
    g.push_back(&sc_approach); g.push_back(&pag_freeze); g.push_back(&novelty_boost);
    for (int t = 0; t < N_CORTICAL_TYPES; ++t)
        for (int d = 0; d < BARCODE_DIM; ++d)
            g.push_back(&cortical_barcode[t][d]);
    for (int t = 0; t < N_CORTICAL_TYPES; ++t) g.push_back(&cortical_division[t]);
    for (int t = 0; t < N_CORTICAL_TYPES; ++t) g.push_back(&cortical_inh_frac[t]);
    for (int i = 0; i < BARCODE_DIM; ++i)
        for (int j = 0; j < BARCODE_DIM; ++j)
            g.push_back(&w_connect[i][j]);
    g.push_back(&connect_threshold);
    for (int d = 0; d < BARCODE_DIM; ++d) g.push_back(&cortical_to_bg[d]);
    return g;
}

size_t DevGenome::n_genes() const { return all_genes().size(); }

void DevGenome::randomize(std::mt19937& rng) {
    for (Gene* gene : all_genes()) {
        std::uniform_real_distribution<float> dist(gene->min_val, gene->max_val);
        gene->value = dist(rng);
    }
}

void DevGenome::mutate(std::mt19937& rng, float mutation_rate, float sigma) {
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    for (Gene* gene : all_genes()) {
        if (coin(rng) < mutation_rate) gene->mutate(rng, sigma);
    }
}

DevGenome DevGenome::crossover(const DevGenome& a, const DevGenome& b, std::mt19937& rng) {
    DevGenome child;
    auto ag = a.all_genes();
    auto bg_genes = b.all_genes();
    auto cg = child.all_genes();
    std::uniform_int_distribution<int> coin(0, 1);
    for (size_t i = 0; i < cg.size(); ++i)
        cg[i]->value = coin(rng) ? ag[i]->value : bg_genes[i]->value;
    return child;
}

std::string DevGenome::summary() const {
    char buf[256];
    int total_n = 0;
    for (int t = 0; t < N_CORTICAL_TYPES; ++t)
        total_n += 1 << std::clamp(static_cast<int>(cortical_division[t].value), 2, 7);
    // 加固定区域估算
    total_n += static_cast<int>(20 * bg_size.value);  // BG ~20
    total_n += static_cast<int>(4 * vta_size.value);   // VTA ~4

    snprintf(buf, sizeof(buf),
        "fit=%.4f ctx=%dn bg=%.1f lr=%.4f noise=%.0f",
        fitness, total_n, bg_size.value, da_stdp_lr.value, motor_noise.value);
    return std::string(buf);
}

std::string DevGenome::to_json() const {
    std::ostringstream ss;
    ss << "{\n";
    for (const Gene* gene : all_genes())
        ss << "  \"" << gene->name << "\": " << gene->value << ",\n";
    ss << "  \"fitness\": " << fitness << "\n}\n";
    return ss.str();
}

} // namespace wuyun
