#include "RayQueryBatch.h"
#include "RayQueryBatchShaders.h"
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

bool RayQueryBatch::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get,
    uint32_t frame_slots, VkPipelineCache cache,
    const BindlessTextureTable& textures, std::string& error)
{
    destroy();
    if (!device || !get || !frame_slots || frame_slots > 16 ||
        !textures.layout() || !textures.set() || !textures.capacity())
    {
        error = "ray query batch requires a device and frame slots";
        return false;
    }
    device_ = device;
    texture_set_ = textures.set();
    texture_capacity_ = textures.capacity();
    const auto create_set_layout = procedure<PFN_vkCreateDescriptorSetLayout>(device, get,
        "vkCreateDescriptorSetLayout");
    destroy_set_layout_ = procedure<PFN_vkDestroyDescriptorSetLayout>(device, get,
        "vkDestroyDescriptorSetLayout");
    const auto create_pool = procedure<PFN_vkCreateDescriptorPool>(device, get, "vkCreateDescriptorPool");
    destroy_pool_ = procedure<PFN_vkDestroyDescriptorPool>(device, get, "vkDestroyDescriptorPool");
    const auto allocate_sets = procedure<PFN_vkAllocateDescriptorSets>(device, get, "vkAllocateDescriptorSets");
    update_sets_ = procedure<PFN_vkUpdateDescriptorSets>(device, get, "vkUpdateDescriptorSets");
    const auto create_layout = procedure<PFN_vkCreatePipelineLayout>(device, get, "vkCreatePipelineLayout");
    destroy_layout_ = procedure<PFN_vkDestroyPipelineLayout>(device, get, "vkDestroyPipelineLayout");
    const auto create_module = procedure<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule");
    const auto destroy_module = procedure<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule");
    const auto create_pipeline = procedure<PFN_vkCreateComputePipelines>(device, get,
        "vkCreateComputePipelines");
    destroy_pipeline_ = procedure<PFN_vkDestroyPipeline>(device, get, "vkDestroyPipeline");
    bind_pipeline_ = procedure<PFN_vkCmdBindPipeline>(device, get, "vkCmdBindPipeline");
    bind_sets_ = procedure<PFN_vkCmdBindDescriptorSets>(device, get, "vkCmdBindDescriptorSets");
    push_constants_ = procedure<PFN_vkCmdPushConstants>(device, get, "vkCmdPushConstants");
    dispatch_ = procedure<PFN_vkCmdDispatch>(device, get, "vkCmdDispatch");
    barrier_ = procedure<PFN_vkCmdPipelineBarrier>(device, get, "vkCmdPipelineBarrier");
    const auto fail = [&](const char* message) {
        error = message;
        destroy();
        return false;
    };
    if (!create_set_layout || !destroy_set_layout_ || !create_pool || !destroy_pool_ ||
        !allocate_sets || !update_sets_ || !create_layout || !destroy_layout_ ||
        !create_module || !destroy_module || !create_pipeline || !destroy_pipeline_ ||
        !bind_pipeline_ || !bind_sets_ || !push_constants_ || !dispatch_ || !barrier_)
        return fail("ray query compute procedures unavailable");

    const VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 6;
    set_info.pBindings = bindings;
    if (create_set_layout(device, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("ray query descriptor layout creation failed");
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 3};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    const VkDescriptorSetLayout set_layouts[]{set_layout_, textures.layout()};
    layout_info.setLayoutCount = 2;
    layout_info.pSetLayouts = set_layouts;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    if (create_layout(device, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("ray query pipeline layout creation failed");
    const VkDescriptorPoolSize pool_sizes[]{
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, frame_slots},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frame_slots * 5}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = frame_slots;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;
    if (create_pool(device, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("ray query descriptor pool creation failed");
    sets_.resize(frame_slots);
    std::vector<VkDescriptorSetLayout> layouts(frame_slots, set_layout_);
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool_;
    allocate.descriptorSetCount = frame_slots;
    allocate.pSetLayouts = layouts.data();
    if (allocate_sets(device, &allocate, sets_.data()) != VK_SUCCESS)
        return fail("ray query descriptor allocation failed");
    ShaderModule module;
    if (!module.initialize(device, {create_module, destroy_module},
            ray_query_batch_shaders::Query, sizeof(ray_query_batch_shaders::Query), error))
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

bool RayQueryBatch::record(VkCommandBuffer command, uint32_t frame_slot,
    const LevelRayScene& scene, const BufferResource& rays,
    const BufferResource& hits, uint32_t ray_count, std::string& error) const
{
    if (!ready() || !command || frame_slot >= sets_.size() || frame_slot >= LevelRayScene::FrameSlots || !scene.handle() ||
        !texture_set_ || !texture_capacity_ ||
        !rays.handle() || !hits.handle() ||
        !scene.hit_batch_buffer().handle() || !scene.hit_geometry_buffer().handle() ||
        VkDeviceSize(ray_count) * sizeof(RayQueryInput) > rays.size() ||
        VkDeviceSize(ray_count) * sizeof(RayQueryHit) > hits.size() ||
        (uint64_t(ray_count) + 63) / 64 > uint64_t(MaxGroupsX) * 65535)
    {
        error = "invalid ray query batch recording request";
        return false;
    }
    if (!ray_count)
    {
        error.clear();
        return true;
    }
    VkWriteDescriptorSetAccelerationStructureKHR acceleration{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    acceleration.accelerationStructureCount = 1;
    const VkAccelerationStructureKHR scene_handle = scene.handle();
    acceleration.pAccelerationStructures = &scene_handle;
    VkDescriptorBufferInfo ray_info{rays.handle(), 0, VkDeviceSize(ray_count) * sizeof(RayQueryInput)};
    VkDescriptorBufferInfo hit_info{hits.handle(), 0, VkDeviceSize(ray_count) * sizeof(RayQueryHit)};
    VkDescriptorBufferInfo batch_info{scene.hit_batch_buffer().handle(), 0,
        scene.hit_batch_buffer().size()};
    VkDescriptorBufferInfo geometry_info{scene.hit_geometry_buffer().handle(), 0,
        scene.hit_geometry_buffer().size()};
    VkDescriptorBufferInfo transforms_info{scene.texture_transform_buffer(frame_slot).handle(), 0,
        scene.texture_transform_buffer(frame_slot).size()};
    VkWriteDescriptorSet writes[6]{};
    for (uint32_t index = 0; index < 6; ++index)
    {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = sets_[frame_slot];
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
    }
    writes[0].pNext = &acceleration;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &ray_info;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &hit_info;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &batch_info;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[4].pBufferInfo = &geometry_info;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[5].pBufferInfo = &transforms_info;
    update_sets_(device_, 6, writes, 0, nullptr);
    bind_pipeline_(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    const VkDescriptorSet descriptor_sets[]{sets_[frame_slot], texture_set_};
    bind_sets_(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 2,
        descriptor_sets, 0, nullptr);
    const uint32_t groups = static_cast<uint32_t>((uint64_t(ray_count) + 63) / 64);
    const uint32_t groups_x = groups < MaxGroupsX ? groups : MaxGroupsX;
    const uint32_t groups_y = (groups + groups_x - 1) / groups_x;
    const uint32_t constants[]{ray_count, texture_capacity_, groups_x};
    push_constants_(command, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(constants), constants);
    dispatch_(command, groups_x, groups_y, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    barrier_(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    error.clear();
    return true;
}

void RayQueryBatch::destroy()
{
    if (device_ && pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
    if (device_ && layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    if (device_ && pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
    if (device_ && set_layout_ && destroy_set_layout_) destroy_set_layout_(device_, set_layout_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
    texture_set_ = VK_NULL_HANDLE;
    texture_capacity_ = 0;
    layout_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
    sets_.clear();
    device_ = VK_NULL_HANDLE;
    destroy_set_layout_ = nullptr;
    destroy_pool_ = nullptr;
    destroy_layout_ = nullptr;
    destroy_pipeline_ = nullptr;
    update_sets_ = nullptr;
    bind_pipeline_ = nullptr;
    bind_sets_ = nullptr;
    push_constants_ = nullptr;
    dispatch_ = nullptr;
    barrier_ = nullptr;
}
}
