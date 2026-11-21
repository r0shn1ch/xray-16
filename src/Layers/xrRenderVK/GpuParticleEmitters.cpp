#include "xrEngine/stdafx.h"
#include "GpuParticleEmitters.h"
#include "ParticleCatalog.h"
#include <algorithm>

namespace xray::render::vulkan
{
GpuParticleEmitters::GpuParticleEmitters(GpuParticlePrograms& programs, GpuParticleCommands& commands, uint32_t roots)
    : programs_(programs), commands_(commands), roots_(roots) {}
const GpuParticleEmitters::Root* GpuParticleEmitters::find(Handle handle) const
{
    const uint32_t low = static_cast<uint32_t>(handle), generation = static_cast<uint32_t>(handle >> 32);
    if (!low || low - 1 >= roots_.size()) return nullptr;
    const auto& root = roots_[low - 1];
    return root.live && root.generation == generation ? &root : nullptr;
}
GpuParticleEmitters::Root* GpuParticleEmitters::find(Handle handle)
{ return const_cast<Root*>(static_cast<const GpuParticleEmitters*>(this)->find(handle)); }
bool GpuParticleEmitters::create(const ParticleEffectDef& definition, Handle& handle, std::string& error)
{
    size_t index = 0;
    while (index < roots_.size() && (roots_[index].live || roots_[index].retiring)) ++index;
    if (index >= roots_.size() || index >= UINT32_MAX)
    { error = "GPU particle emitter capacity exhausted"; return false; }
    uint32_t program = 0, generation = 0, serial = 0;
    if (!programs_.intern(definition, program, error) || !commands_.begin_generation(static_cast<uint32_t>(index), error)) return false;
    fg::GpuPapiCommand create{}; create.type = 0; create.p0 = program;
    if (!commands_.enqueue(static_cast<uint32_t>(index), create, error) ||
        !commands_.state(static_cast<uint32_t>(index), generation, serial)) return false;
    Root root{}; root.generation = generation; root.lifecycle_serial = serial; root.live = true;
    root.draw_programs.push_back(program);
    roots_[index] = std::move(root); handle = (uint64_t(generation) << 32) | (index + 1); error.clear(); return true;
}
bool GpuParticleEmitters::enqueue(Handle handle, fg::GpuPapiCommand command, bool lifecycle, std::string& error)
{
    auto* root = find(handle);
    if (!root) { error = "invalid GPU particle emitter handle"; return false; }
    const uint32_t index = static_cast<uint32_t>(handle) - 1;
    if (!commands_.enqueue(index, command, error)) return false;
    if (lifecycle)
    {
        uint32_t generation, serial;
        if (!commands_.state(index, generation, serial)) { error = "GPU particle command state unavailable"; return false; }
        root->lifecycle_serial = serial;
    }
    error.clear(); return true;
}
bool GpuParticleEmitters::destroy(Handle handle, std::string& error)
{
    fg::GpuPapiCommand command{}; command.type = 1;
    if (!enqueue(handle, command, true, error)) return false;
    auto* root = find(handle); root->live = false; root->retiring = true; root->has_snapshot = false; return true;
}
bool GpuParticleEmitters::play(Handle handle, std::string& error)
{ fg::GpuPapiCommand command{}; command.type = 2; return enqueue(handle, command, true, error); }
bool GpuParticleEmitters::stop(Handle handle, bool deferred, std::string& error)
{ fg::GpuPapiCommand command{}; command.type = 3; command.p0 = deferred; return enqueue(handle, command, true, error); }
bool GpuParticleEmitters::set_parent(Handle handle, const Fmatrix& m, const Fvector& v, bool local, std::string& error)
{
    fg::GpuPapiCommand command{}; command.type = 4; command.p0 = local;
    command.basisX.set(m.i.x, m.i.y, m.i.z, 0); command.basisY.set(m.j.x, m.j.y, m.j.z, 0);
    command.basisZ.set(m.k.x, m.k.y, m.k.z, 0); command.origin.set(m.c.x, m.c.y, m.c.z, 1);
    command.velocity.set(v.x, v.y, v.z, 0); return enqueue(handle, command, false, error);
}
bool GpuParticleEmitters::set_hud(Handle handle, bool hud, std::string& error)
{
    auto* root = find(handle);
    if (!root) { error = "invalid GPU particle emitter handle"; return false; }
    if (root->hud == hud) { error.clear(); return true; }
    fg::GpuPapiCommand command{}; command.type = 5; command.p0 = hud;
    if (!enqueue(handle, command, false, error)) return false;
    root->hud = hud; return true;
}
bool GpuParticleEmitters::tick(Handle handle, uint32_t milliseconds, std::string& error)
{ fg::GpuPapiCommand command{}; command.type = 6; command.p0 = milliseconds; return enqueue(handle, command, false, error); }
bool GpuParticleEmitters::children(Handle handle, uint32_t birth, uint32_t play, uint32_t death, uint32_t flags, std::string& error)
{
    const auto valid = [&](uint32_t id) { return id == GPU_PAPI_INVALID || id < programs_.programs().size(); };
    if (!valid(birth) || !valid(play) || !valid(death)) { error = "invalid GPU particle child program"; return false; }
    if (((flags & 32u) && (birth == GPU_PAPI_INVALID || !(programs_.programs()[birth].flags & (1u << 14)))) ||
        ((flags & 2u) && play == GPU_PAPI_INVALID) ||
        ((flags & 64u) && (death == GPU_PAPI_INVALID || !(programs_.programs()[death].flags & (1u << 14)))))
    { error = "missing or unlimited GPU particle child effect"; return false; }
    if (!(flags & 32u)) birth = GPU_PAPI_INVALID;
    if (!(flags & 2u)) play = GPU_PAPI_INVALID;
    if (!(flags & 64u)) death = GPU_PAPI_INVALID;
    fg::GpuPapiCommand command{}; command.type = 7; command.p0 = birth; command.p1 = play; command.p2 = death; command.p3 = flags;
    if (!enqueue(handle, command, false, error)) return false;
    auto& programs = find(handle)->draw_programs;
    for (uint32_t id : {birth, play, death})
        if (id != GPU_PAPI_INVALID && std::find(programs.begin(), programs.end(), id) == programs.end())
            programs.push_back(id);
    return true;
}
bool GpuParticleEmitters::apply(const std::vector<fg::GpuPapiStatus>& statuses, std::string& error)
{
    if (statuses.size() != roots_.size()) { error = "GPU particle emitter status count mismatch"; return false; }
    for (size_t i = 0; i < roots_.size(); ++i)
    {
        auto& root = roots_[i]; const auto& status = statuses[i];
        if (status.generation == root.generation && status.error && status.retired)
        {
            root.live = root.retiring = root.has_snapshot = false;
            continue;
        }
        if (status.generation != root.generation || status.serial < root.lifecycle_serial ||
            (root.has_snapshot && status.serial < root.snapshot.serial)) continue;
        if (root.retiring && status.retired) { root.retiring = false; root.has_snapshot = false; }
        else if (root.live) { root.snapshot = status; root.has_snapshot = true; }
    }
    error.clear(); return true;
}
bool GpuParticleEmitters::snapshot(Handle handle, fg::GpuPapiStatus& status) const
{
    const auto* root = find(handle);
    if (!root || !root->has_snapshot || root->snapshot.serial < root->lifecycle_serial) return false;
    status = root->snapshot; return true;
}
bool GpuParticleEmitters::visible_index(Handle handle, uint32_t& index) const
{
    if (!find(handle)) return false;
    index = (static_cast<uint32_t>(handle) - 1) * 2; return true;
}
const std::vector<uint32_t>* GpuParticleEmitters::draw_programs(Handle handle) const
{
    const auto* root = find(handle);
    return root ? &root->draw_programs : nullptr;
}
}
