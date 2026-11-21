#pragma once
#include "GpuParticleCompute.h"

namespace xray::render::vulkan
{
GpuParticleCompute::Source engine_gpu_particle_shader_source();
// Loads the packaged shader assets through the same game filesystem overlay as
// the other Vulkan shaders, then constructs the GPU particle pipelines.
bool initialize_engine_gpu_particle_compute(GpuParticleCompute& compute, VkDevice device,
    PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits, uint32_t slots,
    VkPipelineCache cache, std::string& error);
}
