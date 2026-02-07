#pragma once
/**
 * Genome — 可进化的闭环参数基因组
 *
 * 将 ClosedLoopAgent 的 ~25 个关键参数编码为基因组,
 * 支持遗传算法的随机初始化、交叉、变异和适应度评估。
 *
 * 生物学对应:
 *   全局基因 → BDNF, COMT 等影响全脑特性的基因
 *   区域基因 → PAX6 (V1大小), FOXP2 (语言区) 等区域特异基因
 *   增益基因 → SCN系列 (兴奋性), GRIN2B (NMDA/学习) 等
 *
 * v1: 直接编码, 纯遗传算法, ~25 个闭环参数
 */

#include <vector>
#include <random>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cmath>

namespace wuyun {

struct AgentConfig;  // Forward declaration

// =============================================================================
// Gene: 单个基因 (浮点参数 + 范围约束)
// =============================================================================

struct Gene {
    std::string name;
    float value;
    float min_val;
    float max_val;

    void clamp() { value = std::clamp(value, min_val, max_val); }

    // Gaussian mutation: value += N(0, sigma * range)
    void mutate(std::mt19937& rng, float sigma = 0.1f) {
        float range = max_val - min_val;
        std::normal_distribution<float> dist(0.0f, sigma * range);
        value += dist(rng);
        clamp();
    }
};

// =============================================================================
// Genome: 完整基因组 (~25 个闭环参数)
// =============================================================================

struct Genome {
    // --- Global learning genes ---
    Gene da_stdp_lr         {"da_stdp_lr",         0.03f,   0.005f,  0.08f};
    Gene reward_scale       {"reward_scale",        1.5f,    0.3f,    5.0f};
    Gene cortical_a_plus    {"cortical_a_plus",     0.005f,  0.001f,  0.02f};
    Gene cortical_a_minus   {"cortical_a_minus",    0.006f,  0.001f,  0.02f};  // stored positive
    Gene cortical_w_max     {"cortical_w_max",      1.5f,    0.5f,    3.0f};

    // --- Exploration genes ---
    Gene exploration_noise  {"exploration_noise",   55.0f,   20.0f,   100.0f};
    Gene bg_to_m1_gain      {"bg_to_m1_gain",       8.0f,    2.0f,    25.0f};
    Gene attractor_ratio    {"attractor_ratio",     0.6f,    0.3f,    0.9f};   // noise → attractor drive
    Gene background_ratio   {"background_ratio",    0.1f,    0.02f,   0.3f};   // noise → background

    // --- Replay genes ---
    Gene replay_passes      {"replay_passes",       5.0f,    1.0f,    15.0f};  // int, rounded
    Gene replay_da_scale    {"replay_da_scale",     0.5f,    0.1f,    1.0f};

    // --- Visual encoding genes ---
    Gene lgn_gain           {"lgn_gain",            200.0f,  50.0f,   500.0f};
    Gene lgn_baseline       {"lgn_baseline",        5.0f,    1.0f,    20.0f};
    Gene lgn_noise          {"lgn_noise",           2.0f,    0.5f,    8.0f};

    // --- Homeostatic genes ---
    Gene homeostatic_target {"homeostatic_target",  5.0f,    1.0f,    15.0f};
    Gene homeostatic_eta    {"homeostatic_eta",     0.001f,  0.0001f, 0.01f};

    // --- Brain size genes (scale factors) ---
    Gene v1_size            {"v1_size",             1.0f,    0.5f,    2.5f};
    Gene dlpfc_size         {"dlpfc_size",          1.0f,    0.5f,    2.5f};
    Gene bg_size            {"bg_size",             1.0f,    0.5f,    2.0f};

    // --- Timing genes ---
    Gene brain_steps        {"brain_steps",         15.0f,   8.0f,    25.0f};  // int, rounded
    Gene reward_steps       {"reward_steps",        5.0f,    2.0f,    10.0f};  // int, rounded

    // --- NE exploration modulation ---
    Gene ne_food_scale      {"ne_food_scale",       3.0f,    1.0f,    8.0f};   // food_rate × this
    Gene ne_floor           {"ne_floor",            0.7f,    0.4f,    1.0f};   // min noise_scale

    // --- Metadata ---
    float fitness = 0.0f;
    int   generation = 0;

    // =========================================================================
    // Access all genes as a flat vector (for generic GA operations)
    // =========================================================================
    std::vector<Gene*> all_genes();
    std::vector<const Gene*> all_genes() const;
    size_t n_genes() const;

    // =========================================================================
    // Core operations
    // =========================================================================

    /** Randomize all genes uniformly within their ranges */
    void randomize(std::mt19937& rng);

    /** Mutate ~mutation_rate fraction of genes */
    void mutate(std::mt19937& rng, float mutation_rate = 0.15f, float sigma = 0.1f);

    /** Uniform crossover: each gene has 50% chance from parent a or b */
    static Genome crossover(const Genome& a, const Genome& b, std::mt19937& rng);

    /** Convert genome to AgentConfig (for building a ClosedLoopAgent) */
    AgentConfig to_agent_config() const;

    /** JSON serialization */
    std::string to_json() const;
    static Genome from_json(const std::string& json);

    /** Summary string (one-liner) */
    std::string summary() const;
};

} // namespace wuyun
