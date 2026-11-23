#include "RaySurface.h"
#include "RaySurfaceShaders.h"
#include "ComputePipeline.h"
#include "ShaderModule.h"

namespace xray::render::vulkan
{
namespace
{
template <typename T> T procedure(VkDevice device, PFN_vkGetDeviceProcAddr get, const char* name)
{
    return reinterpret_cast<T>(get(device, name));
}
}

bool RaySurfacePass::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t frame_slots,
    const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceDescriptorIndexingProperties& indexing,
    const BindlessTextureTable& textures, VkPipelineCache cache, std::string& error)
{
    destroy();
    if (!device || !get || !frame_slots || frame_slots > LevelRayScene::FrameSlots ||
        !limits.maxStorageBufferRange || limits.maxPerStageDescriptorStorageBuffers < 6 ||
        limits.maxDescriptorSetStorageBuffers < 6 || limits.maxBoundDescriptorSets < 2 ||
        indexing.maxPerStageDescriptorUpdateAfterBindStorageBuffers < 6 ||
        indexing.maxDescriptorSetUpdateAfterBindStorageBuffers < 6 ||
        indexing.maxPerStageUpdateAfterBindResources < 6 ||
        !textures.layout() || !textures.set() || !textures.capacity() ||
        textures.capacity() > indexing.maxPerStageDescriptorUpdateAfterBindSampledImages ||
        textures.capacity() > indexing.maxPerStageDescriptorUpdateAfterBindSamplers ||
        textures.capacity() > indexing.maxDescriptorSetUpdateAfterBindSampledImages ||
        textures.capacity() > indexing.maxDescriptorSetUpdateAfterBindSamplers ||
        textures.capacity() > indexing.maxPerStageUpdateAfterBindResources - 6)
    {
        error = "invalid ray surface pass configuration";
        return false;
    }
    device_ = device;
    max_storage_range_ = limits.maxStorageBufferRange;
    texture_set_ = textures.set();
    texture_capacity_ = textures.capacity();
    const auto create_set_layout = procedure<PFN_vkCreateDescriptorSetLayout>(device, get, "vkCreateDescriptorSetLayout");
    destroy_set_layout_ = procedure<PFN_vkDestroyDescriptorSetLayout>(device, get, "vkDestroyDescriptorSetLayout");
    const auto create_pool = procedure<PFN_vkCreateDescriptorPool>(device, get, "vkCreateDescriptorPool");
    destroy_pool_ = procedure<PFN_vkDestroyDescriptorPool>(device, get, "vkDestroyDescriptorPool");
    const auto allocate_sets = procedure<PFN_vkAllocateDescriptorSets>(device, get, "vkAllocateDescriptorSets");
    update_sets_ = procedure<PFN_vkUpdateDescriptorSets>(device, get, "vkUpdateDescriptorSets");
    const auto create_layout = procedure<PFN_vkCreatePipelineLayout>(device, get, "vkCreatePipelineLayout");
    destroy_layout_ = procedure<PFN_vkDestroyPipelineLayout>(device, get, "vkDestroyPipelineLayout");
    const auto create_module = procedure<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule");
    const auto destroy_module = procedure<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule");
    const auto create_pipeline = procedure<PFN_vkCreateComputePipelines>(device, get, "vkCreateComputePipelines");
    destroy_pipeline_ = procedure<PFN_vkDestroyPipeline>(device, get, "vkDestroyPipeline");
    bind_pipeline_ = procedure<PFN_vkCmdBindPipeline>(device, get, "vkCmdBindPipeline");
    bind_sets_ = procedure<PFN_vkCmdBindDescriptorSets>(device, get, "vkCmdBindDescriptorSets");
    push_constants_ = procedure<PFN_vkCmdPushConstants>(device, get, "vkCmdPushConstants");
    dispatch_ = procedure<PFN_vkCmdDispatch>(device, get, "vkCmdDispatch");
    barrier_ = procedure<PFN_vkCmdPipelineBarrier>(device, get, "vkCmdPipelineBarrier");
    const auto fail = [&](const char* message) { error = message; destroy(); return false; };
    if (!create_set_layout || !destroy_set_layout_ || !create_pool || !destroy_pool_ || !allocate_sets ||
        !update_sets_ || !create_layout || !destroy_layout_ || !create_module || !destroy_module ||
        !create_pipeline || !destroy_pipeline_ || !bind_pipeline_ || !bind_sets_ || !push_constants_ ||
        !dispatch_ || !barrier_)
        return fail("ray surface compute procedures unavailable");

    VkDescriptorSetLayoutBinding bindings[6]{};
    for (uint32_t i = 0; i < 6; ++i)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 6;
    set_info.pBindings = bindings;
    if (create_set_layout(device, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("ray surface descriptor layout creation failed");
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 3};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    const VkDescriptorSetLayout set_layouts[]{set_layout_, textures.layout()};
    layout_info.setLayoutCount = 2;
    layout_info.pSetLayouts = set_layouts;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    if (create_layout(device, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("ray surface pipeline layout creation failed");
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frame_slots * 6};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = frame_slots;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (create_pool(device, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("ray surface descriptor pool creation failed");
    sets_.resize(frame_slots);
    std::vector<VkDescriptorSetLayout> layouts(frame_slots, set_layout_);
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool_;
    allocate.descriptorSetCount = frame_slots;
    allocate.pSetLayouts = layouts.data();
    if (allocate_sets(device, &allocate, sets_.data()) != VK_SUCCESS)
        return fail("ray surface descriptor allocation failed");
    ShaderModule module;
    if (!module.initialize(device, {create_module, destroy_module}, ray_surface_shaders::Reconstruct,
            sizeof(ray_surface_shaders::Reconstruct), error))
    {
        destroy();
        return false;
    }
    if (!create_compute_pipeline(device, cache, layout_, module.handle(), "main",
            create_pipeline, destroy_pipeline_, pipeline_, error))
    {
        destroy();
        return false;
    }
    error.clear();
    return true;
}

bool RaySurfacePass::record(VkCommandBuffer command, uint32_t frame_slot, const LevelRayScene& scene,
    const BufferResource& rays, const BufferResource& hits, const BufferResource& surfaces,
    uint32_t count, std::string& error) const
{
    if (!pipeline_ || !command || frame_slot >= sets_.size() || frame_slot >= LevelRayScene::FrameSlots ||
        (uint64_t(count) + 63) / 64 > uint64_t(65535) * 65535)
    {
        error = "invalid ray surface dispatch";
        return false;
    }
    if (!count) { error.clear(); return true; }
    const BufferResource* buffers[]{&rays, &hits, &scene.hit_batch_buffer(), &scene.hit_geometry_buffer(),
        &scene.texture_transform_buffer(frame_slot), &surfaces};
    const VkDeviceSize ranges[]{VkDeviceSize(count) * sizeof(RayQueryInput),
        VkDeviceSize(count) * sizeof(RayQueryHit), buffers[2]->size(), buffers[3]->size(),
        buffers[4]->size(), VkDeviceSize(count) * sizeof(RaySurface)};
    VkDescriptorBufferInfo infos[6]{};
    VkWriteDescriptorSet writes[6]{};
    for (uint32_t i = 0; i < 6; ++i)
    {
        if (!buffers[i]->handle() || !ranges[i] || ranges[i] > buffers[i]->size() || ranges[i] > max_storage_range_)
        {
            error = "ray surface buffer missing or exceeds storage descriptor range";
            return false;
        }
        infos[i] = {buffers[i]->handle(), 0, ranges[i]};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = sets_[frame_slot];
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    update_sets_(device_, 6, writes, 0, nullptr);
    bind_pipeline_(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    const VkDescriptorSet descriptor_sets[]{sets_[frame_slot], texture_set_};
    bind_sets_(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 2, descriptor_sets, 0, nullptr);
    const uint32_t groups = static_cast<uint32_t>((uint64_t(count) + 63) / 64);
    const uint32_t groups_x = groups < 65535u ? groups : 65535u;
    const uint32_t constants[]{count, groups_x, texture_capacity_};
    push_constants_(command, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), constants);
    dispatch_(command, groups_x, (groups + groups_x - 1) / groups_x, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    barrier_(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    error.clear();
    return true;
}

void RaySurfacePass::destroy()
{
    if (device_ && pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
    if (device_ && layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    if (device_ && pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
    if (device_ && set_layout_ && destroy_set_layout_) destroy_set_layout_(device_, set_layout_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
    sets_.clear();
    device_ = VK_NULL_HANDLE;
    max_storage_range_ = 0;
    texture_set_ = VK_NULL_HANDLE;
    texture_capacity_ = 0;
}
}
