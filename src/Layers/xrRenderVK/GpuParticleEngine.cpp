#include "xrEngine/stdafx.h"
#include "GpuParticleEngine.h"

namespace xray::render::vulkan
{
GpuParticleCompute::Source engine_gpu_particle_shader_source()
{
    return [](const std::string& name, std::vector<uint8_t>& bytes, std::string& why) {
        string_path path;
        if (!FS.exist(path, "$game_shaders$", name.c_str()))
        { why = "missing GPU particle shader: " + name; return false; }
        IReader* reader = FS.r_open(path);
        if (!reader) { why = "could not open GPU particle shader: " + name; return false; }
        const auto* begin = static_cast<const uint8_t*>(reader->pointer());
        if (reader->length()) bytes.assign(begin, begin + reader->length());
        FS.r_close(reader);
        return true;
    };
}
bool initialize_engine_gpu_particle_compute(GpuParticleCompute& compute, VkDevice device,
    PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits, uint32_t slots,
    VkPipelineCache cache, std::string& error)
{
    return compute.initialize_from_source(device, get, limits, slots, cache, engine_gpu_particle_shader_source(), error);
}
}
