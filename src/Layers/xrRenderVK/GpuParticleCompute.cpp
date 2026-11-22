#include "GpuParticleCompute.h"
#include <algorithm>
#include <cstring>

namespace xray::render::vulkan
{
bool GpuParticleCompute::initialize_from_source(VkDevice device, PFN_vkGetDeviceProcAddr get,
    const VkPhysicalDeviceLimits& limits, uint32_t slots, VkPipelineCache cache,
    const Source& source, std::string& error, nvrhi::IDevice* nvrhi_device)
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
    if (!initialize(device, get, limits, slots, cache, shaders, error)) return false;
    if (nvrhi_device)
    {
        try
        {
            nvrhi::ShaderDesc shader_desc;
            shader_desc.shaderType = nvrhi::ShaderType::Compute;
            shader_desc.debugName = "GPU particle pool initialization";
            auto shader = nvrhi_device->createShader(shader_desc, shaders[0].code, shaders[0].bytes);
            nvrhi::BindingLayoutDesc bindings;
            bindings.visibility = nvrhi::ShaderType::Compute;
            // These SPIR-V modules use explicit vk::binding numbers rather
            // than NVRHI's default HLSL register-class offsets.
            bindings.bindingOffsets = nvrhi::VulkanBindingOffsets{}
                .setShaderResourceOffset(0).setSamplerOffset(0)
                .setConstantBufferOffset(0).setUnorderedAccessViewOffset(0);
            bindings.addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0));
            for (uint32_t i = 1; i <= 4; ++i)
                bindings.addItem(nvrhi::BindingLayoutItem::RawBuffer_UAV(i));
            auto layout = nvrhi_device->createBindingLayout(bindings);
            nvrhi::BufferDesc constants;
            constants.byteSize = 32;
            constants.isConstantBuffer = true;
            constants.isVolatile = true;
            constants.maxVersions = std::max(slots, 1u) * 2;
            constants.debugName = "GPU particle initialization constants";
            auto uniform = nvrhi_device->createBuffer(constants);
            nvrhi::ComputePipelineHandle pipeline;
            if (shader && layout && uniform)
                pipeline = nvrhi_device->createComputePipeline(
                    nvrhi::ComputePipelineDesc().setComputeShader(shader).addBindingLayout(layout));
            if (pipeline)
            {
                nvrhi_device_ = nvrhi_device;
                nvrhi_init_shader_ = std::move(shader);
                nvrhi_init_layout_ = std::move(layout);
                nvrhi_init_constants_ = std::move(uniform);
                nvrhi_init_pipeline_ = std::move(pipeline);
            }
        }
        catch (const std::exception&)
        {
            // The Vulkan implementation remains available if NVRHI cannot
            // create this optional pipeline on the current device.
        }
    }
    error.clear(); return true;
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

bool GpuParticleCompute::record_nvrhi_initialize(nvrhi::ICommandList& command,
    const void* constants, size_t constant_bytes, const std::array<nvrhi::IBuffer*, 4>& buffers,
    uint32_t groups, std::string& error) const
{
    if (!nvrhi_device_ || !nvrhi_init_pipeline_ || !constants || constant_bytes != 32 ||
        !groups || groups > max_groups_ || std::any_of(buffers.begin(), buffers.end(), [](auto* b) { return !b; }))
    { error = "invalid NVRHI particle initialization"; return false; }
    try
    {
        nvrhi::BindingSetDesc descriptor;
        descriptor.addItem(nvrhi::BindingSetItem::ConstantBuffer(0, nvrhi_init_constants_));
        for (uint32_t i = 0; i < buffers.size(); ++i)
        {
            descriptor.addItem(nvrhi::BindingSetItem::RawBuffer_UAV(i + 1, buffers[i]));
            // The Vulkan graph has already transitioned each resource to UAV.
            command.beginTrackingBufferState(buffers[i], nvrhi::ResourceStates::UnorderedAccess);
        }
        auto binding = nvrhi_device_->createBindingSet(descriptor, nvrhi_init_layout_);
        if (!binding) { error = "NVRHI particle initialization binding creation failed"; return false; }
        command.writeBuffer(nvrhi_init_constants_, constants, constant_bytes);
        command.setComputeState(nvrhi::ComputeState().setPipeline(nvrhi_init_pipeline_).addBindingSet(binding));
        command.dispatch(groups);
    }
    catch (const std::exception& exception)
    { error = std::string("NVRHI particle initialization failed: ") + exception.what(); return false; }
    error.clear(); return true;
}
void GpuParticleCompute::destroy()
{
    nvrhi_init_pipeline_ = nullptr; nvrhi_init_layout_ = nullptr;
    nvrhi_init_shader_ = nullptr; nvrhi_init_constants_ = nullptr; nvrhi_device_ = nullptr;
    capture_motion_.destroy(); compact_.destroy(); simulate_.destroy(); init_.destroy(); max_groups_ = 0;
}
}
