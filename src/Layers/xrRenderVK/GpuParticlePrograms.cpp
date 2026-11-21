#include "xrEngine/stdafx.h"
#include "GpuParticlePrograms.h"
#include "GpuParticleTranslate.h"
#include "ParticleCatalog.h"

namespace xray::render::vulkan
{
bool GpuParticlePrograms::intern(const ParticleEffectDef& definition, uint32_t& id, std::string& error)
{
    if (definition.name.empty()) { error = "GPU particle definition has no name"; return false; }
    const auto found = ids_.find(definition.name);
    if (found != ids_.end()) { id = found->second; error.clear(); return true; }
    if (programs_.size() >= UINT32_MAX)
    { error = "GPU particle program table overflow"; return false; }
    fg::GpuPapiProgram program{};
    xr_vector<fg::GpuPapiAction> translated;
    xr_string reason;
    if (!TranslateGpuParticleDefinition(definition, program, translated, reason))
    { error = reason.c_str(); return false; }
    if (translated.size() > UINT32_MAX - actions_.size())
    { error = "GPU particle action table overflow"; return false; }
    program.actionFirst = static_cast<uint32_t>(actions_.size());
    program.materialID = GPU_PAPI_INVALID;
    const auto index = static_cast<uint32_t>(programs_.size());
    actions_.insert(actions_.end(), translated.begin(), translated.end());
    programs_.push_back(program);
    names_.push_back(definition.name);
    ids_.emplace(definition.name, index);
    id = index; dirty_ = true; ++revision_; error.clear(); return true;
}
bool GpuParticlePrograms::set_material(uint32_t id, uint32_t material, uint32_t blend, uint32_t variant,
    std::string& error)
{
    if (id >= programs_.size() || blend > 5 || variant > 2)
    { error = "invalid GPU particle material mapping"; return false; }
    auto& program = programs_[id];
    if (program.materialID != material || program.blendMode != blend || program.shaderVariant != variant)
    { program.materialID = material; program.blendMode = blend; program.shaderVariant = variant; dirty_ = true; ++revision_; }
    error.clear(); return true;
}
void GpuParticlePrograms::clear()
{
    programs_.clear(); actions_.clear(); names_.clear(); ids_.clear(); dirty_ = false; ++revision_;
}
}
