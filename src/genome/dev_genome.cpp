#include "genome/dev_genome.h"
#include <cstdio>
#include <sstream>

namespace wuyun {

std::vector<Gene*> DevGenome::all_genes() {
    std::vector<Gene*> genes;
    // Phase A: 增殖 + 连接
    for (int i = 0; i < 5; ++i) genes.push_back(&division_rounds[i]);
    genes.push_back(&inhibitory_prob);
    for (int i = 0; i < 3; ++i) genes.push_back(&growth_gradient[i]);
    genes.push_back(&connection_radius);
    genes.push_back(&recurrent_prob);
    for (int i = 0; i < 25; ++i) genes.push_back(&cross_connect[i]);
    genes.push_back(&da_stdp_lr);
    genes.push_back(&homeostatic_eta);
    genes.push_back(&homeostatic_target);
    genes.push_back(&sensory_gain);
    genes.push_back(&motor_noise);
    genes.push_back(&reward_gain);
    // Phase B: 导向分子 + 分化
    for (int i = 0; i < 8; ++i) genes.push_back(&guidance_cx[i]);
    for (int i = 0; i < 8; ++i) genes.push_back(&guidance_cy[i]);
    for (int i = 0; i < 8; ++i) genes.push_back(&guidance_sigma[i]);
    for (int i = 0; i < 8; ++i) genes.push_back(&guidance_amp[i]);
    for (int i = 0; i < 8; ++i) genes.push_back(&guidance_attract[i]);
    for (int i = 0; i < 40; ++i) genes.push_back(&receptor_expr[i]);
    genes.push_back(&da_gradient);
    genes.push_back(&nmda_gradient);
    // Phase C: 修剪
    genes.push_back(&pruning_threshold);
    genes.push_back(&critical_period);
    genes.push_back(&spontaneous_rate);
    return genes;
}

std::vector<const Gene*> DevGenome::all_genes() const {
    std::vector<const Gene*> genes;
    for (int i = 0; i < 5; ++i) genes.push_back(&division_rounds[i]);
    genes.push_back(&inhibitory_prob);
    for (int i = 0; i < 3; ++i) genes.push_back(&growth_gradient[i]);
    genes.push_back(&connection_radius);
    genes.push_back(&recurrent_prob);
    for (int i = 0; i < 25; ++i) genes.push_back(&cross_connect[i]);
    genes.push_back(&da_stdp_lr);
    genes.push_back(&homeostatic_eta);
    genes.push_back(&homeostatic_target);
    genes.push_back(&sensory_gain);
    genes.push_back(&motor_noise);
    genes.push_back(&reward_gain);
    for (int i = 0; i < 8; ++i) genes.push_back(&guidance_cx[i]);
    for (int i = 0; i < 8; ++i) genes.push_back(&guidance_cy[i]);
    for (int i = 0; i < 8; ++i) genes.push_back(&guidance_sigma[i]);
    for (int i = 0; i < 8; ++i) genes.push_back(&guidance_amp[i]);
    for (int i = 0; i < 8; ++i) genes.push_back(&guidance_attract[i]);
    for (int i = 0; i < 40; ++i) genes.push_back(&receptor_expr[i]);
    genes.push_back(&da_gradient);
    genes.push_back(&nmda_gradient);
    genes.push_back(&pruning_threshold);
    genes.push_back(&critical_period);
    genes.push_back(&spontaneous_rate);
    return genes;
}

size_t DevGenome::n_genes() const { return all_genes().size(); }

void DevGenome::randomize(std::mt19937& rng) {
    for (Gene* g : all_genes()) {
        std::uniform_real_distribution<float> dist(g->min_val, g->max_val);
        g->value = dist(rng);
    }
}

void DevGenome::mutate(std::mt19937& rng, float mutation_rate, float sigma) {
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    for (Gene* g : all_genes()) {
        if (coin(rng) < mutation_rate) {
            g->mutate(rng, sigma);
        }
    }
}

DevGenome DevGenome::crossover(const DevGenome& a, const DevGenome& b, std::mt19937& rng) {
    DevGenome child;
    auto a_genes = a.all_genes();
    auto b_genes = b.all_genes();
    auto c_genes = child.all_genes();

    std::uniform_int_distribution<int> coin(0, 1);
    for (size_t i = 0; i < c_genes.size(); ++i) {
        c_genes[i]->value = coin(rng) ? a_genes[i]->value : b_genes[i]->value;
    }
    return child;
}

std::string DevGenome::summary() const {
    char buf[256];
    int n_total = 0;
    for (int i = 0; i < 5; ++i) {
        n_total += (1 << static_cast<int>(division_rounds[i].value));
    }
    snprintf(buf, sizeof(buf),
        "fit=%.4f div=%d+%d+%d+%d+%d=%dn inh=%.0f%% lr=%.4f",
        fitness,
        1 << static_cast<int>(division_rounds[0].value),
        1 << static_cast<int>(division_rounds[1].value),
        1 << static_cast<int>(division_rounds[2].value),
        1 << static_cast<int>(division_rounds[3].value),
        1 << static_cast<int>(division_rounds[4].value),
        n_total,
        inhibitory_prob.value * 100.0f,
        da_stdp_lr.value);
    return std::string(buf);
}

std::string DevGenome::to_json() const {
    std::ostringstream ss;
    ss << "{\n";
    for (const Gene* g : all_genes()) {
        ss << "  \"" << g->name << "\": " << g->value << ",\n";
    }
    ss << "  \"fitness\": " << fitness << ",\n";
    ss << "  \"generation\": " << generation << "\n";
    ss << "}\n";
    return ss.str();
}

} // namespace wuyun
