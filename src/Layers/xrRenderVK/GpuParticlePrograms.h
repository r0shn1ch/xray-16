#pragma once
#include "gpu_particles/gpu_particle_types.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace xray::render::vulkan
{
struct ParticleEffectDef;
// Stable program IDs and a contiguous action table for GPU upload. A table
// belongs to one immutable ParticleCatalog and must outlive its emitters.
class GpuParticlePrograms
{
public:
    bool intern(const ParticleEffectDef& definition, uint32_t& id, std::string& error);
    bool set_material(uint32_t id, uint32_t material, uint32_t blend, uint32_t variant, std::string& error);
    const std::vector<fg::GpuPapiProgram>& programs() const { return programs_; }
    const std::vector<fg::GpuPapiAction>& actions() const { return actions_; }
    const std::vector<std::string>& names() const { return names_; }
    bool dirty() const { return dirty_; }
    void mark_uploaded() { dirty_ = false; }
    uint64_t revision() const { return revision_; }
    void mark_uploaded(uint64_t revision) { if (revision == revision_) dirty_ = false; }
    void clear();
private:
    std::vector<fg::GpuPapiProgram> programs_;
    std::vector<fg::GpuPapiAction> actions_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, uint32_t> ids_;
    bool dirty_{};
    uint64_t revision_{};
};
}
