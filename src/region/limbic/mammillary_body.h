#pragma once
/**
 * MammillaryBody — 乳头体 (Papez回路中继站)
 *
 * Papez回路: Hippocampus(Sub) → 乳头体 → 丘脑前核(ATN) → ACC → EC → Hipp
 *
 * 功能:
 *   - 从海马下托(Sub)接收输出
 *   - 中继到丘脑前核 (Anterior Thalamic Nucleus)
 *   - 参与空间记忆和情节记忆巩固
 *
 * 生物学参考:
 *   - Papez 1937: A proposed mechanism of emotion
 *   - Vann & Aggleton 2004: The mammillary bodies - two memory systems in one?
 */

#include "region/brain_region.h"
#include "core/population.h"
#include "core/synapse_group.h"

namespace wuyun {

struct MammillaryConfig {
    std::string name = "MammillaryBody";

    size_t n_medial  = 25;   // 内侧乳头体核 (主要, →ATN)
    size_t n_lateral = 10;   // 外侧乳头体核 (辅助)

    float p_medial_to_lateral = 0.25f;
    float w_medial_lateral    = 1.0f;
};

class MammillaryBody : public BrainRegion {
public:
    explicit MammillaryBody(const MammillaryConfig& config);

    void step(int32_t t, float dt = 1.0f) override;
    void receive_spikes(const std::vector<SpikeEvent>& events) override;
    void submit_spikes(SpikeBus& bus, int32_t t) override;
    void inject_external(const std::vector<float>& currents) override;

    const std::vector<uint8_t>& fired()      const override { return fired_all_; }
    const std::vector<int8_t>&  spike_type()  const override { return spike_type_all_; }

    const NeuronPopulation& medial()  const { return medial_; }
    const NeuronPopulation& lateral() const { return lateral_; }

private:
    void aggregate_state();

    MammillaryConfig config_;

    NeuronPopulation medial_;    // 内侧核 (主输出)
    NeuronPopulation lateral_;   // 外侧核

    SynapseGroup syn_med_to_lat_;

    static constexpr float PSP_DECAY = 0.7f;
    std::vector<float> psp_medial_;  // Sub → medial

    std::vector<uint8_t> fired_all_;
    std::vector<int8_t>  spike_type_all_;
};

} // namespace wuyun
