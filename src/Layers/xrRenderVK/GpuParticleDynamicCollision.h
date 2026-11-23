#pragma once
#include "GpuParticleCollisionData.h"

namespace xray::render::vulkan
{
// Refreshes game-object collision shapes and skinned dynamic-mesh triangles.
// Existing static collision is preserved; output is unchanged on failure.
bool update_gpu_particle_dynamic_collision(GpuParticleCollisionInput& data, bool enabled,
    bool& changed, std::string& error);
}
