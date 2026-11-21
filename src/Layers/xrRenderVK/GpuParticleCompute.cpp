#include "GpuParticleCompute.h"
#include <algorithm>
#include <cstring>

namespace xray::render::vulkan
{
bool GpuParticleCompute::initialize_from_source(VkDevice device, PFN_vkGetDeviceProcAddr get,
    const VkPhysicalDeviceLimits& limits, uint32_t slots, VkPipelineCache cache,
    const Source& source, std::string& error)
{
    destroy();
    if (!source) { error = "GPU particle shader source unavailable"; return false; }
    const char* names[]{"vk/gpu_particle_init.cs.spv", "vk/gpu_particle_simulate.cs.spv",
        "vk/gpu_particle_compact.cs.spv", "vk/gpu_particle_motion_capture.cs.spv"};
    std::array<std::vector<uint32_t>, 4> words;
    std::array<Shader, 4> shaders{};
    for (size_t i = 0; i < words.size(); ++i)
    {
        std::vector<uint8_t> bytes;
        if (!source(names[i], bytes, error)) return false;
        if (bytes.size() < 20 || bytes.size() % 4 || bytes.size() > 16 * 1024 * 1024)
        { error = "invalid GPU particle shader asset: " + std::string(names[i]); return false; }
        words[i].resize(bytes.size() / 4);
        std::memcpy(words[i].data(), bytes.data(), bytes.size());
        if (words[i][0] != 0x07230203u)
        { error = "invalid GPU particle SPIR-V magic: " + std::string(names[i]); return false; }
        shaders[i] = {words[i].data(), bytes.size()};
    }
    return initialize(device, get, limits, slots, cache, shaders, error);
}
bool GpuParticleCompute::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get,
    const VkPhysicalDeviceLimits& limits, uint32_t slots, VkPipelineCache cache,
    const std::array<Shader, 4>& shaders, std::string& error)
{
    destroy();
    if (limits.maxPerStageDescriptorStorageBuffers < 13 || limits.maxDescriptorSetStorageBuffers < 13 ||
        limits.maxPerStageDescriptorUniformBuffers < 2 || limits.maxDescriptorSetUniformBuffers < 2 ||
        limits.maxPerStageResources < 14 || limits.maxComputeWorkGroupInvocations < 64 ||
        limits.maxComputeWorkGroupSize[0] < 64 || !limits.maxComputeWorkGroupCount[0])
    { error = "device limits do not support GPU particle compute shaders"; return false; }
    using Binding = StorageComputePass::Binding;
    const auto layout = [](uint32_t count, uint32_t uniforms) {
        std::vector<Binding> bindings;
        for (uint32_t i = 0; i < count; ++i)
            bindings.push_back({i, i < uniforms ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER});
        return bindings;
    };
    if (!init_.initialize(device, get, slots, layout(5, 1), 0, limits.maxStorageBufferRange,
            limits.maxUniformBufferRange, cache, shaders[0].code, shaders[0].bytes, error) ||
        !simulate_.initialize(device, get, slots, layout(14, 1), 0, limits.maxStorageBufferRange,
            limits.maxUniformBufferRange, cache, shaders[1].code, shaders[1].bytes, error) ||
        !compact_.initialize(device, get, slots, layout(10, 2), 0, limits.maxStorageBufferRange,
            limits.maxUniformBufferRange, cache, shaders[2].code, shaders[2].bytes, error) ||
        !capture_motion_.initialize(device, get, slots, layout(3, 1), 0, limits.maxStorageBufferRange,
            limits.maxUniformBufferRange, cache, shaders[3].code, shaders[3].bytes, error))
    { destroy(); return false; }
    max_groups_ = std::min(limits.maxComputeWorkGroupCount[0], 65535u);
    error.clear(); return true;
}
bool GpuParticleCompute::record(Pass pass, VkCommandBuffer command, uint32_t slot,
    const std::vector<VkDescriptorBufferInfo>& buffers, uint32_t groups, std::string& error) const
{
    if (!groups || groups > max_groups_)
    { error = "invalid GPU particle compute group count"; return false; }
    switch (pass)
    {
    case Pass::Initialize: return init_.record(command, slot, buffers, nullptr, groups, 1, error);
    case Pass::Simulate: return simulate_.record(command, slot, buffers, nullptr, groups, 1, error);
    case Pass::Compact: return compact_.record(command, slot, buffers, nullptr, groups, 1, error);
    case Pass::CaptureMotion: return capture_motion_.record(command, slot, buffers, nullptr, groups, 1, error);
    }
    error = "invalid GPU particle compute pass"; return false;
}
void GpuParticleCompute::destroy()
{
    capture_motion_.destroy(); compact_.destroy(); simulate_.destroy(); init_.destroy(); max_groups_ = 0;
}
}
