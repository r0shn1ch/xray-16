#include "xrEngine/stdafx.h"
#include "GpuParticleMaterials.h"
#include "GameTextureFactory.h"
#include "ParticleCatalog.h"
#include "ParticleVisual.h"
#include "LevelPassGraph.h"
#include <algorithm>
#include <cstring>

namespace xray::render::vulkan
{
bool GpuParticleMaterials::sync(GpuParticlePrograms& programs, const ParticleCatalog& catalog,
    GameTextureFactory& textures, DeferredPass& pass, std::string& error)
{
    if (programs.names().size() != programs.programs().size())
    { error = "GPU particle program names and table disagree"; return false; }
    if (entries_.size() > programs.programs().size())
    { error = "clear GPU particle materials before replacing the program table"; return false; }
    entries_.resize(programs.programs().size());
    for (size_t i = 0; i < entries_.size(); ++i)
    {
        if (entries_[i].resolved)
        {
            if (entries_[i].name != programs.names()[i])
            { error = "clear GPU particle materials before replacing the program table"; return false; }
            continue;
        }
        const auto* definition = catalog.effect(programs.names()[i]);
        if (!definition) { error = "GPU particle material definition is missing: " + programs.names()[i]; return false; }
        Entry entry{}; entry.name = definition->name;
        if (!(definition->flags & 1u)) { entry.resolved = true; entries_[i] = std::move(entry); continue; }
        SurfaceMode surface{}; MaterialDefinition material;
        if (!textures.surface_mode(definition->shader, definition->texture, surface, error,
                &entry.alpha_ref, &entry.blend, true, false, false, false, nullptr, nullptr, &material))
        { error = "GPU particle material '" + definition->name + "': " + error; return false; }
        if (entry.blend < 0 || entry.blend > 5)
        { error = "unsupported GPU particle blend mode for '" + definition->name + "'"; return false; }
        entry.screen = particle_uses_screen_program(definition->shader, material.class_name);
        entry.distortion = definition->shader == "particles\\xdistort";
        entry.priority = material.priority; entry.matrices = material.matrices;
        const bool clamp = (material.class_name == "S_SET   " || material.class_name == "PARTICLE") && material.pass.clamp;
        if (!textures.material(definition->texture, pass, entry.set, error, clamp)) return false;
        const uint32_t variant = entry.distortion ? 2u : entry.screen ? 1u : 0u;
        if (!programs.set_material(static_cast<uint32_t>(i), static_cast<uint32_t>(i), entry.blend, variant, error))
        { textures.release_material(entry.set, pass); return false; }
        entry.resolved = true; entries_[i] = std::move(entry);
    }
    error.clear(); return true;
}
void GpuParticleMaterials::clear(GameTextureFactory& textures, DeferredPass& pass)
{
    for (auto& entry : entries_) if (entry.set) textures.release_material(entry.set, pass);
    entries_.clear();
}
bool GpuParticleMaterials::fragment(uint32_t program, bool hud, std::string& name, bool& gbuffer,
    std::string& error) const
{
    const auto* entry = get(program);
    if (!entry || !entry->resolved || !entry->set)
    { error = "GPU particle material has no supported color pass"; return false; }
    if (entry->distortion)
    {
        name = hud ? "vk\\gpu_particle_hud_distortion.ps" : "vk\\gpu_particle_distortion.ps";
        gbuffer = false; error.clear(); return true;
    }
    const bool deferred = !hud && !entry->screen && entry->blend == 0;
    RenderPassGraph graph;
    if (!build_render_pass_graph({RenderConsumer::Particle,
            deferred ? SurfaceMode::AlphaTest : SurfaceMode::Transparent,
            hud, false, false, false, 0, entry->blend, 0, false, -1, entry->screen}, graph, error))
        return false;
    const std::string& source = graph.passes[0].fragment;
    if (source.compare(0, 3, "vk\\") != 0)
    { error = "unexpected GPU particle fragment path"; return false; }
    name = "vk\\gpu_" + source.substr(3); gbuffer = deferred; error.clear(); return true;
}
bool GpuParticleMaterials::constants(uint32_t program, float time, const float* view, const DeferredPass& pass,
    std::array<float, 64>& output, std::string& error) const
{
    const auto* entry = get(program);
    if (!entry || !entry->resolved || !entry->set || !view)
    { error = "GPU particle material constants unavailable"; return false; }
    std::array<float, 64> data{};
    data[16] = static_cast<float>(entry->alpha_ref) / 255.f;
    data[17] = 1.f;
    const SurfaceParameters surface = pass.material_surface(entry->set);
    static_assert(sizeof(surface) == 8 * sizeof(float));
    std::memcpy(data.data() + 20, &surface, sizeof(surface));
    float* rows = data.data() + 32;
    rows[0] = rows[5] = rows[8] = rows[13] = rows[16] = rows[21] = rows[26] = 1.f;
    for (size_t slot = 0; slot < 3; ++slot)
    {
        if (!entry->matrices.enabled[slot]) continue;
        const auto matrix = entry->matrices.values[slot].evaluate(time, view);
        const size_t first = slot == 0 ? 0 : slot == 1 ? 2 : 4;
        for (size_t row = 0; row < (slot == 2 ? 3u : 2u); ++row)
            std::copy(matrix[row].begin(), matrix[row].end(), rows + (first + row) * 4);
        if (slot == 2 && entry->matrices.values[slot].mode == 2) rows[27] = 1.f;
    }
    output = data; error.clear(); return true;
}
}
