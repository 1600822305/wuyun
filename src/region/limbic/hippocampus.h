#pragma once
/**
 * Hippocampus — 海马记忆系统
 *
 * 实现海马体的核心三突触通路 (trisynaptic circuit):
 *   EC(L2) → DG(模式分离) → CA3(自联想/模式补全) → CA1(比较/输出) → Sub(分发)
 *   EC(L3) → CA1 (直接通路, 绕过 DG/CA3)
 *
 * 关键特性 (按 01 文档 §2.1.3):
 *   - DG: GRANULE_CELL_PARAMS, 极稀疏激活 (~2%), 高阈值
 *   - CA3: PLACE_CELL_PARAMS, 1-2% 自联想循环连接 (recurrent)
 *   - CA1: PLACE_CELL_PARAMS, 双区室 theta 相位进动
 *   - Subiculum: 标准锥体, 多目标输出
 *
 * 遵守 00 文档反作弊原则:
 *   - 记忆内容存在于突触权重中, 不是字典/数据库
 *   - 回忆是 CA3 自联想网络的模式补全, 不是精确匹配
 *   - 模式分离是 DG 稀疏编码的涌现结果, 不是算法
 *
 * 设计文档: docs/01_brain_region_plan.md §2.1.3
 */

#include "region/brain_region.h"
#include "core/population.h"
#include "core/synapse_group.h"

namespace wuyun {

struct HippocampusConfig {
    std::string name = "Hippocampus";

    // --- Population sizes ---
    size_t n_ec       = 80;   // 内嗅皮层 (grid cells)
    size_t n_dg       = 200;  // 齿状回 (granule cells, ~2% activate)
    size_t n_ca3      = 60;   // CA3 (place cells, autoassociative)
    size_t n_ca1      = 80;   // CA1 (place cells, comparator)
    size_t n_sub      = 40;   // 下托 (output)
    size_t n_presub   = 0;    // 前下托 (head direction, →乳头体+EC)
    size_t n_hata     = 0;    // 海马-杏仁核过渡区 (→Amygdala)

    // Inhibitory interneurons (one per subregion for E/I balance)
    size_t n_dg_inh   = 20;   // DG basket cells
    size_t n_ca3_inh  = 10;   // CA3 basket cells
    size_t n_ca1_inh  = 15;   // CA1 basket cells

    // --- Connection probabilities ---
    // Trisynaptic path
    float p_ec_to_dg    = 0.20f;  // Perforant path (EC L2 → DG)
    float p_dg_to_ca3   = 0.05f;  // Mossy fiber (sparse but strong)
    float p_ca3_to_ca3  = 0.02f;  // CA3 recurrent (autoassociative, ~1-2%)
    float p_ca3_to_ca1  = 0.15f;  // Schaffer collateral
    float p_ca1_to_sub  = 0.20f;  // CA1 → Subiculum
    float p_sub_to_ec   = 0.10f;  // Subiculum → EC (output loop)
    // Direct path (bypasses DG/CA3)
    float p_ec_to_ca1   = 0.10f;  // EC L3 → CA1 (direct)
    // Feedback
    float p_ca3_to_dg   = 0.03f;  // CA3 → DG feedback (backprojection)
    // Presubiculum + HATA connections (only active if n_presub/n_hata > 0)
    float p_ca1_to_presub = 0.15f;  // CA1 → Presub
    float p_presub_to_ec  = 0.10f;  // Presub → EC (head direction feedback)
    float p_ca1_to_hata   = 0.10f;  // CA1 → HATA
    float w_presub        = 0.5f;
    float w_hata          = 0.5f;

    // Inhibitory
    float p_ec_to_dg_inh  = 0.30f;  // EC → DG basket (feedforward inhibition)
    float p_dg_to_dg_inh  = 0.40f;  // DG → DG basket (feedback inhibition)
    float p_dg_inh_to_dg  = 0.50f;  // DG basket → DG (dense inhibitory blanket)
    float p_ca3_to_ca3_inh = 0.20f;
    float p_ca3_inh_to_ca3 = 0.30f;
    float p_ca1_to_ca1_inh = 0.20f;
    float p_ca1_inh_to_ca1 = 0.30f;

    // --- Synapse weights ---
    float w_ec_dg      = 0.8f;   // Perforant path (moderate, DG has high threshold)
    float w_dg_ca3     = 2.0f;   // Mossy fiber (very strong, few synapses)
    float w_ca3_ca3    = 0.3f;   // Recurrent (weak individual, many together)
    float w_ca3_ca1    = 0.6f;   // Schaffer collateral
    float w_ca1_sub    = 0.5f;
    float w_sub_ec     = 0.4f;
    float w_ec_ca1     = 0.4f;   // Direct path
    float w_ca3_dg_fb  = 0.2f;   // Feedback
    float w_inh        = 1.5f;   // Inhibitory weight (positive; GABA e_rev handles sign)
    float w_exc_to_inh = 1.2f;   // Excitatory → inhibitory (strong to drive DG basket cells)

    // --- CA3 STDP (one-shot memory encoding) ---
    bool  ca3_stdp_enabled = true;   // Enable STDP on CA3 recurrent synapses
    float ca3_stdp_a_plus  = 0.05f;  // Fast LTP (5x cortical, one-shot learning)
    float ca3_stdp_a_minus = -0.06f; // LTD (slightly stronger for competition)
    float ca3_stdp_tau     = 20.0f;  // Time window (ms)
    float ca3_stdp_w_max   = 2.0f;   // Max weight (higher than initial 0.3)

    // --- Sleep / SWR (Sharp-Wave Ripple) replay ---
    float swr_noise_amp     = 12.0f;  // Subthreshold noise injected into CA3 during sleep
    size_t swr_duration     = 5;      // Steps per SWR burst event
    size_t swr_refractory   = 25;     // Min steps between SWR events
    float swr_ca3_threshold = 0.15f;  // CA3 firing fraction to detect SWR onset
    float swr_boost         = 20.0f;  // Extra CA3 drive during active SWR (amplify replay)
};

class Hippocampus : public BrainRegion {
public:
    explicit Hippocampus(const HippocampusConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_all_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_all_; }

    // --- Hippocampus-specific interface ---

    /** Inject cortical input to EC (皮层→内嗅皮层) */
    void inject_cortical_input(const std::vector<float>& currents);

    /** Get CA1 output firing (for readout by downstream regions) */
    const NeuronPopulation& ca1() const { return ca1_; }
    const NeuronPopulation& ca3() const { return ca3_; }
    const NeuronPopulation& dg()  const { return dg_; }
    const NeuronPopulation& ec()  const { return ec_; }
    const NeuronPopulation& sub() const { return sub_; }
    const NeuronPopulation& presub() const { return presub_; }
    const NeuronPopulation& hata() const { return hata_; }
    bool has_presub() const { return config_.n_presub > 0; }
    bool has_hata()   const { return config_.n_hata > 0; }

    /** Get DG activation sparsity (fraction of DG neurons active) */
    float dg_sparsity() const;

    // --- Sleep / SWR replay interface ---

    /** Enable sleep replay mode (SWR generation via CA3 noise → pattern completion) */
    void enable_sleep_replay()  { sleep_replay_ = true; }
    void disable_sleep_replay() { sleep_replay_ = false; in_swr_ = false; }
    bool sleep_replay_enabled() const { return sleep_replay_; }

    /** Is a sharp-wave ripple currently active? */
    bool is_swr() const { return in_swr_; }

    /** Total SWR events since construction */
    uint32_t swr_count() const { return swr_count_; }

    /** CA3 firing fraction during last SWR (replay strength proxy) */
    float last_replay_strength() const { return last_replay_strength_; }

    // --- REM theta / creative recombination interface ---

    /** Enable REM theta mode (theta oscillation + creative recombination) */
    void enable_rem_theta()  { rem_theta_ = true; sleep_replay_ = false; }
    void disable_rem_theta() { rem_theta_ = false; }
    bool rem_theta_enabled() const { return rem_theta_; }

    /** REM theta phase [0, 1) */
    float rem_theta_phase() const { return rem_theta_phase_; }

    /** Number of creative recombination events during REM */
    uint32_t rem_recombination_count() const { return rem_recomb_count_; }

private:
    void build_synapses();
    void aggregate_state();

    HippocampusConfig config_;

    // --- 5+2 excitatory populations ---
    NeuronPopulation ec_;       // 内嗅皮层 (grid cells)
    NeuronPopulation dg_;       // 齿状回 (granule cells)
    NeuronPopulation ca3_;      // CA3 (place cells, autoassociative)
    NeuronPopulation ca1_;      // CA1 (place cells, comparator)
    NeuronPopulation sub_;      // 下托 (output)
    NeuronPopulation presub_;   // 前下托 (optional, head direction)
    NeuronPopulation hata_;     // HATA (optional, Hipp-Amyg transition)

    // --- 3 inhibitory populations (E/I balance) ---
    NeuronPopulation dg_inh_;   // DG basket cells
    NeuronPopulation ca3_inh_;  // CA3 basket cells
    NeuronPopulation ca1_inh_;  // CA1 basket cells

    // --- Trisynaptic path synapses ---
    SynapseGroup syn_ec_to_dg_;       // Perforant path
    SynapseGroup syn_dg_to_ca3_;      // Mossy fiber
    SynapseGroup syn_ca3_to_ca3_;     // CA3 recurrent (autoassociative)
    SynapseGroup syn_ca3_to_ca1_;     // Schaffer collateral
    SynapseGroup syn_ca1_to_sub_;     // CA1 → Sub
    SynapseGroup syn_sub_to_ec_;      // Sub → EC (output loop)
    SynapseGroup syn_ec_to_ca1_;      // Direct path EC→CA1
    SynapseGroup syn_ca3_to_dg_fb_;   // CA3 → DG feedback
    SynapseGroup syn_ca1_to_presub_;  // CA1 → Presub (optional)
    SynapseGroup syn_presub_to_ec_;   // Presub → EC (optional)
    SynapseGroup syn_ca1_to_hata_;    // CA1 → HATA (optional)

    // --- Inhibitory synapses ---
    SynapseGroup syn_ec_to_dg_inh_;   // EC → DG inh (feedforward inhibition)
    SynapseGroup syn_dg_to_dg_inh_;   // DG exc → DG inh (feedback inhibition)
    SynapseGroup syn_dg_inh_to_dg_;   // DG inh → DG exc
    SynapseGroup syn_ca3_to_ca3_inh_; // CA3 exc → CA3 inh
    SynapseGroup syn_ca3_inh_to_ca3_; // CA3 inh → CA3 exc
    SynapseGroup syn_ca1_to_ca1_inh_; // CA1 exc → CA1 inh
    SynapseGroup syn_ca1_inh_to_ca1_; // CA1 inh → CA1 exc

    // --- PSP buffer for cross-region input ---
    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_ec_;

    // --- Sleep / SWR state ---
    bool     sleep_replay_       = false;
    bool     in_swr_             = false;
    uint32_t swr_count_          = 0;
    int32_t  swr_timer_          = 0;     // Countdown during active SWR
    int32_t  swr_refractory_cd_  = 0;     // Refractory countdown between SWRs
    float    last_replay_strength_ = 0.0f;

    void try_generate_swr(int32_t t);

    // --- REM theta state ---
    bool     rem_theta_          = false;
    float    rem_theta_phase_    = 0.0f;
    uint32_t rem_recomb_count_   = 0;
    static constexpr float REM_THETA_FREQ = 0.006f;  // ~6Hz theta
    static constexpr float REM_THETA_AMP  = 10.0f;   // Theta modulation amplitude
    static constexpr float REM_RECOMB_PROB = 0.01f;  // Creative recombination probability/step

    void try_rem_theta(int32_t t);

    // --- Aggregate firing state ---
    std::vector<uint8_t> fired_all_;
    std::vector<int8_t>  spike_type_all_;
};

} // namespace wuyun
