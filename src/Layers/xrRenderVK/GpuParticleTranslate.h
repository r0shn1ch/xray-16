#pragma once
#include "gpu_particles/gpu_particle_types.h"

namespace xray::render::vulkan
{
struct ParticleEffectDef;
// Converts validated on-disk PAPI actions into the Vulkan compute shader ABI.
// Outputs remain unchanged on failure.
bool TranslateGpuParticleDefinition(const ParticleEffectDef& definition, fg::GpuPapiProgram& program,
    xr_vector<fg::GpuPapiAction>& actions, xr_string& error);
}
