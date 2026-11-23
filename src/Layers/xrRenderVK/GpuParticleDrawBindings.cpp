#include "GpuParticleDrawBindings.h"
#include <algorithm>

namespace xray::render::vulkan
{
bool GpuParticleDrawBindings::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get,
    const VkPhysicalDeviceLimits& limits, uint32_t slots, std::string& error)
{
    destroy();
    if (!device || !get || !slots || slots > 4096 || limits.maxPerStageDescriptorStorageBuffers < 5 ||
        limits.maxDescriptorSetStorageBuffers < 5 || limits.maxPerStageDescriptorUniformBuffers < 2 ||
        limits.maxDescriptorSetUniformBuffers < 2 || !limits.maxStorageBufferRange ||
        limits.maxPerStageResources < 6 || limits.maxUniformBufferRange < 256)
    { error = "device limits do not support GPU particle draw bindings"; return false; }
    auto create_layout = reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(get(device, "vkCreateDescriptorSetLayout"));
    auto create_pool = reinterpret_cast<PFN_vkCreateDescriptorPool>(get(device, "vkCreateDescriptorPool"));
    auto allocate = reinterpret_cast<PFN_vkAllocateDescriptorSets>(get(device, "vkAllocateDescriptorSets"));
    destroy_layout_ = reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(get(device, "vkDestroyDescriptorSetLayout"));
    destroy_pool_ = reinterpret_cast<PFN_vkDestroyDescriptorPool>(get(device, "vkDestroyDescriptorPool"));
    update_ = reinterpret_cast<PFN_vkUpdateDescriptorSets>(get(device, "vkUpdateDescriptorSets"));
    bind_ = reinterpret_cast<PFN_vkCmdBindDescriptorSets>(get(device, "vkCmdBindDescriptorSets"));
    if (!create_layout || !create_pool || !allocate || !destroy_layout_ || !destroy_pool_ || !update_ || !bind_)
    { error = "GPU particle draw binding procedures unavailable"; destroy(); return false; }
    device_ = device; max_storage_ = limits.maxStorageBufferRange; max_uniform_ = limits.maxUniformBufferRange;
    uniform_alignment_ = std::max<VkDeviceSize>(limits.minUniformBufferOffsetAlignment, 1);
    VkDescriptorSetLayoutBinding bindings[7]{};
    for (uint32_t i = 0; i < 7; ++i)
        bindings[i] = {i, i >= 5 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            1, VkShaderStageFlags(i == 6 ? VK_SHADER_STAGE_FRAGMENT_BIT : i == 5 ?
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT), nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 7; li.pBindings = bindings;
    if (create_layout(device, &li, nullptr, &layout_) != VK_SUCCESS)
    { error = "GPU particle draw descriptor layout creation failed"; destroy(); return false; }
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, slots * 5},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, slots * 2}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = slots; pi.poolSizeCount = 2; pi.pPoolSizes = sizes;
    if (create_pool(device, &pi, nullptr, &pool_) != VK_SUCCESS)
    { error = "GPU particle draw descriptor pool creation failed"; destroy(); return false; }
    sets_.resize(slots); std::vector<VkDescriptorSetLayout> layouts(slots, layout_);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_; ai.descriptorSetCount = slots; ai.pSetLayouts = layouts.data();
    if (allocate(device, &ai, sets_.data()) != VK_SUCCESS)
    { error = "GPU particle draw descriptor allocation failed"; destroy(); return false; }
    error.clear(); return true;
}
bool GpuParticleDrawBindings::bind(VkCommandBuffer command, VkPipelineLayout pipeline_layout, uint32_t slot,
    const GpuParticleBuffers& buffers, VkDescriptorBufferInfo params,
    VkDescriptorBufferInfo material_params, std::string& error) const
{
    if (!device_ || !command || !pipeline_layout || slot >= sets_.size() || !params.buffer ||
        params.range < sizeof(GpuParticleDrawParams) || params.range > max_uniform_ || params.offset % uniform_alignment_ ||
        !material_params.buffer || material_params.range < 256 || material_params.range > max_uniform_ ||
        material_params.offset % uniform_alignment_)
    { error = "invalid GPU particle draw binding"; return false; }
    const GpuParticleBuffers::Resource resources[]{GpuParticleBuffers::Particles, GpuParticleBuffers::Emitters,
        GpuParticleBuffers::Programs, GpuParticleBuffers::Indices, GpuParticleBuffers::BucketOffsets};
    VkDescriptorBufferInfo infos[7]{};
    for (size_t i = 0; i < 5; ++i)
    {
        const auto& buffer = buffers.get(resources[i]);
        if (!buffer.handle() || !buffer.size() || buffer.size() > max_storage_)
        { error = "invalid GPU particle draw buffer"; return false; }
        infos[i] = {buffer.handle(), 0, buffer.size()};
    }
    infos[5] = params; infos[6] = material_params;
    VkWriteDescriptorSet writes[7]{};
    for (uint32_t i = 0; i < 7; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = sets_[slot];
        writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = i >= 5 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    update_(device_, 7, writes, 0, nullptr);
    bind_(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &sets_[slot], 0, nullptr);
    error.clear(); return true;
}
void GpuParticleDrawBindings::destroy()
{
    if (device_ && pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
    if (device_ && layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    device_ = VK_NULL_HANDLE; pool_ = VK_NULL_HANDLE; layout_ = VK_NULL_HANDLE; sets_.clear();
    max_storage_ = max_uniform_ = uniform_alignment_ = 0;
}
}
