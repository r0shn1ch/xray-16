#include "ReferencePathTracePass.h"
#include "ReferencePathTraceShaders.h"
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

bool ReferencePathTracePass::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get,
    uint32_t frame_slots, VkDeviceSize max_storage_range, VkPipelineCache cache,
    const BindlessTextureTable& textures, VkDescriptorSetLayout projection_layout,
    std::string& error)
{
    destroy();
    if (!device || !get || !frame_slots || frame_slots > LevelRayScene::FrameSlots ||
        !max_storage_range || !textures.layout() || !textures.set() || !textures.capacity())
    {
        error = "invalid reference path trace configuration";
        return false;
    }
    device_ = device;
    max_storage_range_ = max_storage_range;
    texture_set_ = textures.set();
    texture_capacity_ = textures.capacity();
    if (!projection_layout)
    {
        if (!projection_fallback_.initialize(device, get, 1, error))
        { destroy(); return false; }
        projection_layout = projection_fallback_.layout();
    }
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
    const auto fail = [&](const char* message) { error = message; destroy(); return false; };
    if (!create_set_layout || !destroy_set_layout_ || !create_pool || !destroy_pool_ || !allocate_sets ||
        !update_sets_ || !create_layout || !destroy_layout_ || !create_module || !destroy_module ||
        !create_pipeline || !destroy_pipeline_ || !bind_pipeline_ || !bind_sets_ || !push_constants_ || !dispatch_)
        return fail("reference path trace Vulkan procedures unavailable");

    VkDescriptorSetLayoutBinding bindings[11]{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
        VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    for (uint32_t i = 1; i < 10; ++i)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[10] = {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
        VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 11;
    set_info.pBindings = bindings;
    if (create_set_layout(device, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("reference path trace descriptor layout creation failed");

    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReferencePathTraceParameters)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    const VkDescriptorSetLayout layouts[]{set_layout_, textures.layout(), projection_layout};
    layout_info.setLayoutCount = 3;
    layout_info.pSetLayouts = layouts;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    if (create_layout(device, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("reference path trace pipeline layout creation failed");

    const VkDescriptorPoolSize pool_sizes[]{{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, frame_slots},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frame_slots * 9},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, frame_slots}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = frame_slots;
    pool_info.poolSizeCount = 3;
    pool_info.pPoolSizes = pool_sizes;
    if (create_pool(device, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("reference path trace descriptor pool creation failed");
    sets_.resize(frame_slots);
    std::vector<VkDescriptorSetLayout> set_layouts(frame_slots, set_layout_);
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool_;
    allocate.descriptorSetCount = frame_slots;
    allocate.pSetLayouts = set_layouts.data();
    if (allocate_sets(device, &allocate, sets_.data()) != VK_SUCCESS)
        return fail("reference path trace descriptor allocation failed");

    ShaderModule module;
    if (!module.initialize(device, {create_module, destroy_module}, reference_path_trace_shaders::Trace,
            sizeof(reference_path_trace_shaders::Trace), error))
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

bool ReferencePathTracePass::record(VkCommandBuffer command, uint32_t frame_slot,
    const LevelRayScene& scene, const BufferResource& primary_rays,
    const BufferResource& primary_surfaces, const BufferResource& previous_history,
    const BufferResource& next_history, const BufferResource& local_lights,
    VkDescriptorSet projection_set,
    VkImageView motion_view, VkSampler motion_sampler,
    const BufferResource& radiance,
    const ReferencePathTraceParameters& parameters, std::string& error) const
{
    const uint32_t count = parameters.dimensions[0];
    if (!pipeline_ || !command || frame_slot >= sets_.size() || frame_slot >= LevelRayScene::FrameSlots ||
        !scene.handle() || !texture_set_ || !texture_capacity_ || !primary_rays.handle() ||
        !primary_surfaces.handle() || !radiance.handle() || !scene.hit_batch_buffer().handle() ||
        !scene.hit_geometry_buffer().handle() || !scene.texture_transform_buffer(frame_slot).handle() ||
        !previous_history.handle() || !next_history.handle() || !local_lights.handle() ||
        !motion_view || !motion_sampler ||
        parameters.dimensions[2] != texture_capacity_ || parameters.dimensions[1] == 0 ||
        VkDeviceSize(count) * sizeof(RayQueryInput) > primary_rays.size() ||
        VkDeviceSize(count) * sizeof(RaySurface) > primary_surfaces.size() ||
        VkDeviceSize(count) * sizeof(ReferencePathTraceHistoryCell) > previous_history.size() ||
        VkDeviceSize(count) * sizeof(ReferencePathTraceHistoryCell) > next_history.size() ||
        VkDeviceSize(count) * sizeof(float) * 4 > radiance.size() ||
        VkDeviceSize(parameters.dimensions[1]) * 64 > UINT32_MAX)
    {
        error = "invalid reference path trace recording request";
        return false;
    }
    if (!count) { error.clear(); return true; }

    VkWriteDescriptorSetAccelerationStructureKHR acceleration{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    acceleration.accelerationStructureCount = 1;
    const VkAccelerationStructureKHR scene_handle = scene.handle();
    acceleration.pAccelerationStructures = &scene_handle;
    const VkDescriptorBufferInfo infos[]{
        {primary_rays.handle(), 0, VkDeviceSize(count) * sizeof(RayQueryInput)},
        {primary_surfaces.handle(), 0, VkDeviceSize(count) * sizeof(RaySurface)},
        {scene.hit_batch_buffer().handle(), 0, scene.hit_batch_buffer().size()},
        {scene.hit_geometry_buffer().handle(), 0, scene.hit_geometry_buffer().size()},
        {scene.texture_transform_buffer(frame_slot).handle(), 0, scene.texture_transform_buffer(frame_slot).size()},
        {radiance.handle(), 0, VkDeviceSize(count) * sizeof(float) * 4},
        {previous_history.handle(), 0, VkDeviceSize(count) * sizeof(ReferencePathTraceHistoryCell)},
        {next_history.handle(), 0, VkDeviceSize(count) * sizeof(ReferencePathTraceHistoryCell)},
        {local_lights.handle(), 0, local_lights.size()}};
    VkDescriptorImageInfo motion_info{motion_sampler, motion_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[11]{};
    for (uint32_t i = 0; i < 11; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = sets_[frame_slot];
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
    }
    writes[0].pNext = &acceleration;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    for (uint32_t i = 1; i < 10; ++i)
    {
        if (!infos[i - 1].buffer || !infos[i - 1].range || infos[i - 1].range > max_storage_range_)
        { error = "reference path trace descriptor exceeds storage buffer limits"; return false; }
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i - 1];
    }
    writes[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[10].pImageInfo = &motion_info;
    update_sets_(device_, 11, writes, 0, nullptr);
    bind_pipeline_(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    const VkDescriptorSet descriptor_sets[]{sets_[frame_slot], texture_set_,
        projection_set ? projection_set : projection_fallback_.set()};
    bind_sets_(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 3, descriptor_sets, 0, nullptr);
    push_constants_(command, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(parameters), &parameters);
    const uint32_t groups_x = parameters.dimensions[1];
    const uint64_t groups = (uint64_t(count) + 63u) / 64u;
    const uint32_t groups_y = static_cast<uint32_t>((groups + groups_x - 1u) / groups_x);
    dispatch_(command, groups_x, groups_y, 1);
    error.clear();
    return true;
}

void ReferencePathTracePass::destroy()
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
    projection_fallback_.destroy();
    destroy_set_layout_ = nullptr;
    destroy_pool_ = nullptr;
    destroy_layout_ = nullptr;
    destroy_pipeline_ = nullptr;
    update_sets_ = nullptr;
    bind_pipeline_ = nullptr;
    bind_sets_ = nullptr;
    push_constants_ = nullptr;
    dispatch_ = nullptr;
}
}
