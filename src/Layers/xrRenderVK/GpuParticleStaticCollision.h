#pragma once
#include "GpuParticleCollisionData.h"

namespace xray::render::vulkan
{
// Builds the stackless collision tree consumed by gpu_particle_collision.h.
// Dynamic sections in output are retained; output is unchanged on failure.
bool build_gpu_particle_static_collision(const std::vector<GpuParticleCollisionTriangle>& triangles,
    GpuParticleCollisionInput& output, std::string& error);
}
