#pragma once
#include "GpuParticleCommands.h"
#include "GpuParticlePrograms.h"
#include "xrCore/_matrix.h"
#include <cstdint>
#include <vector>

namespace xray::render::vulkan
{
struct ParticleEffectDef;
// Owner of generation-safe root handles. GPU status must be applied after the
// corresponding runtime readback; callers serialize access to this object.
class GpuParticleEmitters
{
public:
    using Handle = uint64_t;
    GpuParticleEmitters(GpuParticlePrograms& programs, GpuParticleCommands& commands, uint32_t roots);
    bool register_program(const ParticleEffectDef& definition, uint32_t& id, std::string& error)
    { return programs_.intern(definition, id, error); }
    bool create(const ParticleEffectDef& definition, Handle& handle, std::string& error);
    bool destroy(Handle handle, std::string& error);
    bool play(Handle handle, std::string& error);
    bool stop(Handle handle, bool deferred, std::string& error);
    bool set_parent(Handle handle, const Fmatrix& transform, const Fvector& velocity, bool local, std::string& error);
    bool set_hud(Handle handle, bool hud, std::string& error);
    bool tick(Handle handle, uint32_t milliseconds, std::string& error);
    bool children(Handle handle, uint32_t birth, uint32_t play, uint32_t death, uint32_t flags, std::string& error);
    bool apply(const std::vector<fg::GpuPapiStatus>& statuses, std::string& error);
    bool snapshot(Handle handle, fg::GpuPapiStatus& status) const;
    bool visible_index(Handle handle, uint32_t& index) const;
    // Includes programs of children that may still be alive after a change
    // to the root's child configuration. Invalidated on destroy/reuse.
    const std::vector<uint32_t>* draw_programs(Handle handle) const;
    bool has_child_effects(Handle handle) const;
private:
    struct Root { uint32_t generation{}, lifecycle_serial{}; bool live{}, retiring{}, has_snapshot{}, hud{}, has_children{};
        fg::GpuPapiStatus snapshot{}; std::vector<uint32_t> draw_programs; };
    Root* find(Handle handle);
    const Root* find(Handle handle) const;
    bool enqueue(Handle handle, fg::GpuPapiCommand command, bool lifecycle, std::string& error);
    GpuParticlePrograms& programs_; GpuParticleCommands& commands_;
    std::vector<Root> roots_;
};
}
