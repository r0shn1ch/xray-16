#include "xrEngine/stdafx.h"
#include "ClusterVisibility.h"
#include "ClusterVisibilityShaders.h"
#include "ComputePipeline.h"
#include "FrameImageGraph.h"
#include "ShaderModule.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace xray::render::vulkan
{
namespace
{
template <typename T>
T procedure(VkDevice device, PFN_vkGetDeviceProcAddr get, const char* name)
{
    return reinterpret_cast<T>(get(device, name));
}

constexpr uint32_t LocalSize = 64;
constexpr uint32_t MaxDispatchGroupsX = 65535;
constexpr float TargetErrorPixels = 1.f;
constexpr float Pi = 3.14159265358979323846f;
}

ClusterVisibility::ClusterVisibility(ClusterVisibility&& other) noexcept
{
    *this = std::move(other);
}

ClusterVisibility& ClusterVisibility::operator=(ClusterVisibility&& other) noexcept
{
    if (this == &other)
        return *this;
    destroy();
    device_ = std::exchange(other.device_, nullptr);
    groups_ = std::move(other.groups_);
    clusters_ = std::move(other.clusters_);
    commands_ = std::move(other.commands_);
    previous_mvp_ = std::move(other.previous_mvp_);
    set_layout_ = std::exchange(other.set_layout_, VK_NULL_HANDLE);
    pool_ = std::exchange(other.pool_, VK_NULL_HANDLE);
    layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
    pipeline_ = std::exchange(other.pipeline_, VK_NULL_HANDLE);
    destroy_set_layout_ = std::exchange(other.destroy_set_layout_, nullptr);
    destroy_pool_ = std::exchange(other.destroy_pool_, nullptr);
    destroy_layout_ = std::exchange(other.destroy_layout_, nullptr);
    destroy_pipeline_ = std::exchange(other.destroy_pipeline_, nullptr);
    destroy_sampler_ = std::exchange(other.destroy_sampler_, nullptr);
    destroy_image_ = std::exchange(other.destroy_image_, nullptr);
    free_memory_ = std::exchange(other.free_memory_, nullptr);
    destroy_image_view_ = std::exchange(other.destroy_image_view_, nullptr);
    clear_color_image_ = std::exchange(other.clear_color_image_, nullptr);
    bind_pipeline_ = std::exchange(other.bind_pipeline_, nullptr);
    bind_sets_ = std::exchange(other.bind_sets_, nullptr);
    update_sets_ = std::exchange(other.update_sets_, nullptr);
    push_constants_ = std::exchange(other.push_constants_, nullptr);
    dispatch_ = std::exchange(other.dispatch_, nullptr);
    barrier_ = std::exchange(other.barrier_, nullptr);
    fallback_hiz_image_ = std::exchange(other.fallback_hiz_image_, VK_NULL_HANDLE);
    fallback_hiz_memory_ = std::exchange(other.fallback_hiz_memory_, VK_NULL_HANDLE);
    fallback_hiz_view_ = std::exchange(other.fallback_hiz_view_, VK_NULL_HANDLE);
    fallback_hiz_sampler_ = std::exchange(other.fallback_hiz_sampler_, VK_NULL_HANDLE);
    fallback_hiz_initialized_ = std::exchange(other.fallback_hiz_initialized_, false);
    command_count_ = std::exchange(other.command_count_, 0);
    sets_ = other.sets_;
    other.sets_ = {};
    spans_ = std::move(other.spans_);
    group_count_ = std::exchange(other.group_count_, 0);
    cluster_count_ = std::exchange(other.cluster_count_, 0);
    return *this;
}

bool ClusterVisibility::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkPipelineCache cache,
    const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& buffers,
    const std::vector<ClusterCullGroup>& groups, const std::vector<ClusterCullRecord>& clusters,
    const std::vector<ClusterCullSpan>& spans, uint32_t command_count,
    VkDeviceSize max_storage_buffer_range, std::string& error)
{
    destroy();
    if (!device || !get || groups.empty() || clusters.empty() || spans.empty() || !command_count ||
        groups.size() > std::numeric_limits<uint32_t>::max() ||
        clusters.size() > std::numeric_limits<uint32_t>::max() ||
        groups.size() > std::numeric_limits<VkDeviceSize>::max() / sizeof(ClusterCullGroup) ||
        clusters.size() > std::numeric_limits<VkDeviceSize>::max() / sizeof(ClusterCullRecord))
    {
        error = "Cluster DAG compute culling needs valid metadata and Vulkan procedures";
        return false;
    }
    const VkDeviceSize group_bytes = groups.size() * sizeof(ClusterCullGroup);
    const VkDeviceSize cluster_bytes = clusters.size() * sizeof(ClusterCullRecord);
    const VkDeviceSize command_bytes = VkDeviceSize(command_count) * 2u * sizeof(VkDrawIndexedIndirectCommand);
    if (!max_storage_buffer_range || group_bytes > max_storage_buffer_range ||
        cluster_bytes > max_storage_buffer_range || command_bytes > max_storage_buffer_range)
    {
        error = "Cluster DAG metadata exceeds the Vulkan storage-buffer range";
        return false;
    }

    device_ = device;
    command_count_ = command_count;
    group_count_ = static_cast<uint32_t>(groups.size());
    cluster_count_ = static_cast<uint32_t>(clusters.size());
    spans_ = spans;
    auto create_set_layout = procedure<PFN_vkCreateDescriptorSetLayout>(device, get, "vkCreateDescriptorSetLayout");
    destroy_set_layout_ = procedure<PFN_vkDestroyDescriptorSetLayout>(device, get, "vkDestroyDescriptorSetLayout");
    auto create_pool = procedure<PFN_vkCreateDescriptorPool>(device, get, "vkCreateDescriptorPool");
    destroy_pool_ = procedure<PFN_vkDestroyDescriptorPool>(device, get, "vkDestroyDescriptorPool");
    auto allocate_sets = procedure<PFN_vkAllocateDescriptorSets>(device, get, "vkAllocateDescriptorSets");
    auto create_layout = procedure<PFN_vkCreatePipelineLayout>(device, get, "vkCreatePipelineLayout");
    destroy_layout_ = procedure<PFN_vkDestroyPipelineLayout>(device, get, "vkDestroyPipelineLayout");
    auto create_pipeline = procedure<PFN_vkCreateComputePipelines>(device, get, "vkCreateComputePipelines");
    destroy_pipeline_ = procedure<PFN_vkDestroyPipeline>(device, get, "vkDestroyPipeline");
    auto create_image = procedure<PFN_vkCreateImage>(device, get, "vkCreateImage");
    destroy_image_ = procedure<PFN_vkDestroyImage>(device, get, "vkDestroyImage");
    auto get_image_requirements = procedure<PFN_vkGetImageMemoryRequirements>(device, get,
        "vkGetImageMemoryRequirements");
    auto allocate_memory = procedure<PFN_vkAllocateMemory>(device, get, "vkAllocateMemory");
    free_memory_ = procedure<PFN_vkFreeMemory>(device, get, "vkFreeMemory");
    auto bind_image_memory = procedure<PFN_vkBindImageMemory>(device, get, "vkBindImageMemory");
    auto create_image_view = procedure<PFN_vkCreateImageView>(device, get, "vkCreateImageView");
    destroy_image_view_ = procedure<PFN_vkDestroyImageView>(device, get, "vkDestroyImageView");
    auto create_sampler = procedure<PFN_vkCreateSampler>(device, get, "vkCreateSampler");
    destroy_sampler_ = procedure<PFN_vkDestroySampler>(device, get, "vkDestroySampler");
    clear_color_image_ = procedure<PFN_vkCmdClearColorImage>(device, get, "vkCmdClearColorImage");
    bind_pipeline_ = procedure<PFN_vkCmdBindPipeline>(device, get, "vkCmdBindPipeline");
    bind_sets_ = procedure<PFN_vkCmdBindDescriptorSets>(device, get, "vkCmdBindDescriptorSets");
    update_sets_ = procedure<PFN_vkUpdateDescriptorSets>(device, get, "vkUpdateDescriptorSets");
    push_constants_ = procedure<PFN_vkCmdPushConstants>(device, get, "vkCmdPushConstants");
    dispatch_ = procedure<PFN_vkCmdDispatch>(device, get, "vkCmdDispatch");
    barrier_ = procedure<PFN_vkCmdPipelineBarrier>(device, get, "vkCmdPipelineBarrier");
    const auto fail = [&](const char* reason) {
        error = reason;
        destroy();
        return false;
    };
    if (!create_set_layout || !destroy_set_layout_ || !create_pool || !destroy_pool_ || !allocate_sets ||
        !update_sets_ || !create_layout || !destroy_layout_ || !create_pipeline || !destroy_pipeline_ ||
        !create_image || !destroy_image_ || !get_image_requirements || !allocate_memory || !free_memory_ ||
        !bind_image_memory || !create_image_view || !destroy_image_view_ || !create_sampler ||
        !destroy_sampler_ || !clear_color_image_ || !bind_pipeline_ || !bind_sets_ ||
        !push_constants_ || !dispatch_ || !barrier_)
        return fail("Cluster DAG compute procedures are unavailable");

    VkImageCreateInfo fallback_image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    fallback_image.imageType = VK_IMAGE_TYPE_2D;
    fallback_image.format = VK_FORMAT_R8G8B8A8_UNORM;
    fallback_image.extent = {1, 1, 1};
    fallback_image.mipLevels = fallback_image.arrayLayers = 1;
    fallback_image.samples = VK_SAMPLE_COUNT_1_BIT;
    fallback_image.tiling = VK_IMAGE_TILING_OPTIMAL;
    fallback_image.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    fallback_image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (create_image(device_, &fallback_image, nullptr, &fallback_hiz_image_) != VK_SUCCESS)
        return fail("Cluster DAG Hi-Z fallback image creation failed");
    VkMemoryRequirements fallback_requirements{};
    get_image_requirements(device_, fallback_hiz_image_, &fallback_requirements);
    uint32_t fallback_memory_type = UINT32_MAX;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        if ((fallback_requirements.memoryTypeBits & (1u << i)) &&
            (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        {
            fallback_memory_type = i;
            break;
        }
    if (fallback_memory_type == UINT32_MAX)
        return fail("Cluster DAG Hi-Z fallback needs device-local image memory");
    VkMemoryAllocateInfo fallback_allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    fallback_allocation.allocationSize = fallback_requirements.size;
    fallback_allocation.memoryTypeIndex = fallback_memory_type;
    if (allocate_memory(device_, &fallback_allocation, nullptr, &fallback_hiz_memory_) != VK_SUCCESS ||
        bind_image_memory(device_, fallback_hiz_image_, fallback_hiz_memory_, 0) != VK_SUCCESS)
        return fail("Cluster DAG Hi-Z fallback image allocation failed");
    VkImageViewCreateInfo fallback_view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    fallback_view.image = fallback_hiz_image_;
    fallback_view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    fallback_view.format = VK_FORMAT_R8G8B8A8_UNORM;
    fallback_view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (create_image_view(device_, &fallback_view, nullptr, &fallback_hiz_view_) != VK_SUCCESS)
        return fail("Cluster DAG Hi-Z fallback image view creation failed");
    VkSamplerCreateInfo fallback_sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    fallback_sampler.magFilter = fallback_sampler.minFilter = VK_FILTER_NEAREST;
    fallback_sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    fallback_sampler.addressModeU = fallback_sampler.addressModeV = fallback_sampler.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (create_sampler(device_, &fallback_sampler, nullptr, &fallback_hiz_sampler_) != VK_SUCCESS)
        return fail("Cluster DAG Hi-Z fallback sampler creation failed");

    if (!groups_.initialize(device_, group_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            memory, buffers, error, true) ||
        !groups_.write(0, groups.data(), static_cast<size_t>(group_bytes), error) ||
        !clusters_.initialize(device_, cluster_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            memory, buffers, error, true) ||
        !clusters_.write(0, clusters.data(), static_cast<size_t>(cluster_bytes), error))
    {
        destroy();
        return false;
    }
    for (BufferResource& output : commands_)
        if (!output.initialize(device_, command_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                memory, buffers, error, true))
        {
            destroy();
            return false;
        }
    std::array<float, 16> initial_mvp{};
    initial_mvp[0] = initial_mvp[5] = initial_mvp[10] = initial_mvp[15] = 1.f;
    for (BufferResource& previous : previous_mvp_)
        if (!previous.initialize(device_, sizeof(initial_mvp), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                memory, buffers, error, true) ||
            !previous.write(0, initial_mvp.data(), sizeof(initial_mvp), error))
        {
            destroy();
            return false;
        }

    const VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 5;
    set_info.pBindings = bindings;
    if (create_set_layout(device_, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("Cluster DAG descriptor-set layout creation failed");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    if (create_layout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("Cluster DAG pipeline-layout creation failed");

    const VkDescriptorPoolSize pool_sizes[]{
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3u * FrameContext::FramesInFlight},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, FrameContext::FramesInFlight},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, FrameContext::FramesInFlight}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = FrameContext::FramesInFlight;
    pool_info.poolSizeCount = 3;
    pool_info.pPoolSizes = pool_sizes;
    if (create_pool(device_, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("Cluster DAG descriptor-pool creation failed");
    std::array<VkDescriptorSetLayout, FrameContext::FramesInFlight> layouts{};
    layouts.fill(set_layout_);
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = pool_;
    allocation.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocation.pSetLayouts = layouts.data();
    if (allocate_sets(device_, &allocation, sets_.data()) != VK_SUCCESS)
        return fail("Cluster DAG descriptor-set allocation failed");

    for (uint32_t frame = 0; frame < sets_.size(); ++frame)
    {
        const VkDescriptorBufferInfo infos[]{
            {groups_.handle(), 0, groups_.size()},
            {clusters_.handle(), 0, clusters_.size()},
            {commands_[frame].handle(), 0, commands_[frame].size()}};
        const VkDescriptorImageInfo fallback_info{fallback_hiz_sampler_, fallback_hiz_view_,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkDescriptorBufferInfo previous_info{previous_mvp_[frame].handle(), 0, previous_mvp_[frame].size()};
        VkWriteDescriptorSet writes[5]{};
        for (uint32_t binding = 0; binding < 3; ++binding)
        {
            writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding].dstSet = sets_[frame];
            writes[binding].dstBinding = binding;
            writes[binding].descriptorCount = 1;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].pBufferInfo = &infos[binding];
        }
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = sets_[frame];
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].pImageInfo = &fallback_info;
        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = sets_[frame];
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[4].pBufferInfo = &previous_info;
        update_sets_(device_, 5, writes, 0, nullptr);
    }

    ShaderModule shader;
    if (!shader.initialize(device_,
            {procedure<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
             procedure<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")},
            cluster_visibility_shaders::Cull, sizeof(cluster_visibility_shaders::Cull), error))
    {
        destroy();
        return false;
    }
    if (!create_compute_pipeline(device_, cache, layout_, shader.handle(), "main", create_pipeline,
            destroy_pipeline_, pipeline_, error))
    {
        destroy();
        return false;
    }
    error.clear();
    return true;
}

bool ClusterVisibility::record_view(VkCommandBuffer command, VkDescriptorSet set, const float (&mvp)[16],
    const float (&camera)[3], float projection_scale, uint32_t output_view, std::string& error,
    VkExtent2D hiz_source_extent) const
{
    if (!command || !set || !std::isfinite(projection_scale) || projection_scale <= 0.f)
    {
        error = "Cluster DAG culling received an invalid frame or projection";
        return false;
    }
    bind_pipeline_(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    bind_sets_(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set, 0, nullptr);
    for (const ClusterCullSpan& span : spans_)
    {
        if (!span.group_count || !span.cluster_count || span.group_first > group_count_ ||
            span.group_count > group_count_ - span.group_first || span.cluster_first > cluster_count_ ||
            span.cluster_count > cluster_count_ - span.cluster_first || span.command_first > command_count_ ||
            span.cluster_count > command_count_ - span.command_first)
        {
            error = "Cluster DAG culling span is outside packed metadata";
            return false;
        }
        for (uint32_t chunk = 0; chunk < span.cluster_count;)
        {
            const uint32_t chunk_count = std::min(span.cluster_count - chunk, MaxDispatchGroupsX * LocalSize);
            PushConstants push{};
            std::copy_n(mvp, 16, push.mvp);
            std::copy_n(camera, 3, push.camera_projection_scale);
            push.camera_projection_scale[3] = projection_scale;
            push.ranges[0] = span.group_first;
            push.ranges[1] = span.group_count;
            push.ranges[2] = span.cluster_first;
            push.ranges[3] = span.cluster_count;
            push.draw[0] = output_view * command_count_ + span.command_first;
            push.draw[1] = span.first_index;
            push.draw[2] = static_cast<uint32_t>(span.vertex_offset);
            uint32_t target_bits{};
            std::memcpy(&target_bits, &TargetErrorPixels, sizeof(target_bits));
            push.quality[0] = target_bits;
            push.quality[1] = chunk;
            push.quality[2] = hiz_source_extent.width;
            push.quality[3] = hiz_source_extent.height;
            push_constants_(command, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            dispatch_(command, (chunk_count + LocalSize - 1) / LocalSize, 1, 1);
            chunk += chunk_count;
        }
    }
    error.clear();
    return true;
}

bool ClusterVisibility::record(const FrameRecordingContext& frame, const float (&camera_mvp)[16],
    const float (&sun_mvp)[16], const float (&camera)[3], float fov_degrees, uint32_t shadow_height,
    std::string& error, const float* previous_mvp, VkImage previous_hiz_image,
    VkImageView previous_hiz, VkSampler hiz_sampler,
    const FrameImageSharing* previous_hiz_sharing, uint32_t previous_hiz_levels, bool hiz_valid) const
{
    if (!ready() || frame.frame_index >= sets_.size() || frame.extent.height == 0 || !shadow_height ||
        !std::isfinite(fov_degrees) || !std::isfinite(camera[0]) || !std::isfinite(camera[1]) ||
        !std::isfinite(camera[2]))
    {
        error = "Cluster DAG culling is unavailable for this frame";
        return false;
    }
    const float half_fov = std::clamp(fov_degrees, 1.f, 179.f) * (Pi / 360.f);
    const float projection_scale = float(frame.extent.height) * 0.5f / std::tan(half_fov);
    const float shadow_projection_scale = float(shadow_height) * 0.5f / std::tan(half_fov);
    if (!previous_mvp_[frame.frame_index].write(0,
            previous_mvp ? previous_mvp : camera_mvp, sizeof(float) * 16, error))
        return false;

    const bool use_hiz = hiz_valid && previous_mvp && previous_hiz_image && previous_hiz && hiz_sampler &&
        previous_hiz_sharing && previous_hiz_levels;
    const VkDescriptorImageInfo hiz_info{
        use_hiz ? hiz_sampler : fallback_hiz_sampler_,
        use_hiz ? previous_hiz : fallback_hiz_view_,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet hiz_write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    hiz_write.dstSet = sets_[frame.frame_index];
    hiz_write.dstBinding = 3;
    hiz_write.descriptorCount = 1;
    hiz_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    hiz_write.pImageInfo = &hiz_info;
    update_sets_(device_, 1, &hiz_write, 0, nullptr);

    FrameImageGraph graph(FrameSubmissionGraph::Queue::Graphics);
    FrameImageGraph::BufferResource groups, clusters, previous_mvp_resource, commands;
    FrameImageGraph::Resource fallback_hiz;
    FrameImageGraph::Resource previous_hiz_resource;
    const VkDeviceSize group_bytes = VkDeviceSize(group_count_) * sizeof(ClusterCullGroup);
    const VkDeviceSize cluster_bytes = VkDeviceSize(cluster_count_) * sizeof(ClusterCullRecord);
    const VkDeviceSize matrix_bytes = sizeof(float) * 16;
    const VkDeviceSize command_bytes = VkDeviceSize(command_count_) * 2u * sizeof(VkDrawIndexedIndirectCommand);
    ImageStateTracker image_states;
    if (!graph.import_buffer(groups_.handle(), 0, group_bytes, BufferUse::HostWrite, groups, error) ||
        !graph.import_buffer(clusters_.handle(), 0, cluster_bytes, BufferUse::HostWrite, clusters, error) ||
        !graph.import_buffer(previous_mvp_[frame.frame_index].handle(), 0, matrix_bytes,
            BufferUse::HostWrite, previous_mvp_resource, error) ||
        !graph.import_buffer(commands_[frame.frame_index].handle(), 0, command_bytes,
            BufferUse::Undefined, commands, error))
        return false;

    const bool initialize_fallback_hiz = !use_hiz && !fallback_hiz_initialized_;
    if (!use_hiz)
    {
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const ImageUse initial_use = fallback_hiz_initialized_
            ? ImageUse::ComputeSampled : ImageUse::Undefined;
        if (!image_states.register_image(fallback_hiz_image_, range, initial_use, error))
            return false;
        if (!graph.import_image(fallback_hiz_image_, range, fallback_hiz, error,
                FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {}, initial_use))
            return false;
        if (initialize_fallback_hiz)
        {
            const auto clear = graph.add_command_pass("cluster DAG fallback Hi-Z clear",
                [this](VkCommandBuffer command) {
                VkClearColorValue color{};
                color.float32[0] = color.float32[1] = color.float32[2] = color.float32[3] = 1.f;
                const VkImageSubresourceRange clear_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                clear_color_image_(command, fallback_hiz_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    &color, 1, &clear_range);
            });
            FrameImageGraph::Resource cleared_fallback_hiz;
            if (!graph.write(clear, fallback_hiz, ImageUse::TransferDestination,
                    cleared_fallback_hiz, error))
                return false;
            fallback_hiz = cleared_fallback_hiz;
        }
    }
    else
    {
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, previous_hiz_levels, 0, 1};
        const bool previous_hiz_was_generated_compute = frame.compute_command_buffer && frame.submission_graph;
        const ImageUse previous_hiz_initial_use = previous_hiz_was_generated_compute
            ? ImageUse::ComputeSampled : ImageUse::SampledGraphicsCompute;
        if (!image_states.register_image(previous_hiz_image, range, previous_hiz_initial_use, error) ||
            !graph.import_image(previous_hiz_image, range, previous_hiz_resource, error,
                FrameSubmissionGraph::Queue::Graphics, 0,
                previous_hiz_sharing->mode, previous_hiz_sharing->queue_families,
                previous_hiz_initial_use))
            return false;
    }

    const auto cull = graph.add_checked_command_pass("cluster DAG camera and sun culling",
        [&, use_hiz](VkCommandBuffer command, std::string& record_error) {
        if (!record_view(command, sets_[frame.frame_index], camera_mvp, camera,
                projection_scale, 0, record_error, use_hiz ? frame.extent : VkExtent2D{}) ||
            !record_view(command, sets_[frame.frame_index], sun_mvp, camera,
                shadow_projection_scale, 1, record_error))
            return false;
        record_error.clear();
        return true;
    });
    FrameImageGraph::BufferResource written_commands;
    if (!graph.read_buffer(cull, groups, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(cull, clusters, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(cull, previous_mvp_resource, BufferUse::ComputeRead, error) ||
        !graph.write_buffer(cull, commands, BufferUse::ComputeWrite, written_commands, error))
        return false;
    if (!use_hiz && !graph.read(cull, fallback_hiz, ImageUse::ComputeSampled, error))
        return false;
    if (use_hiz && !graph.read(cull, previous_hiz_resource, ImageUse::ComputeSampled, error))
        return false;

    // This no-op consumer represents the first shadow draw, which is recorded
    // immediately after culling returns. Its indirect read makes the graph
    // emit the barrier for indirect draws and for compute reading the camera commands.
    const auto draw_consumer = graph.add_pass("cluster DAG draw command consumers", [] {});
    if (!graph.read_buffer(draw_consumer, written_commands, BufferUse::IndirectAndComputeRead, error))
        return false;

    if (!graph.execute(frame.command_buffer, image_states, {barrier_}, error,
            frame.command_queue_flags, frame.semaphore_handoff, FrameSubmissionGraph::Queue::Graphics))
        return false;
    if (initialize_fallback_hiz)
        fallback_hiz_initialized_ = true;
    error.clear();
    return true;
}

bool ClusterVisibility::record_shadow_view(const FrameRecordingContext& frame,
    const float (&shadow_mvp)[16], const float (&camera)[3], float fov_degrees, std::string& error) const
{
    if (!ready() || frame.frame_index >= sets_.size() || frame.extent.height == 0 ||
        !std::isfinite(fov_degrees) || !std::isfinite(camera[0]) || !std::isfinite(camera[1]) ||
        !std::isfinite(camera[2]))
    {
        error = "Cluster DAG shadow culling is unavailable for this frame";
        return false;
    }
    const float half_fov = std::clamp(fov_degrees, 1.f, 179.f) * (Pi / 360.f);
    const float projection_scale = float(frame.extent.height) * 0.5f / std::tan(half_fov);
    if (!std::isfinite(projection_scale) || projection_scale <= 0.f)
    {
        error = "Cluster DAG shadow culling received an invalid projection";
        return false;
    }

    FrameImageGraph graph(FrameSubmissionGraph::Queue::Graphics);
    FrameImageGraph::BufferResource groups, clusters, previous_mvp_resource, commands;
    const VkDeviceSize group_bytes = VkDeviceSize(group_count_) * sizeof(ClusterCullGroup);
    const VkDeviceSize cluster_bytes = VkDeviceSize(cluster_count_) * sizeof(ClusterCullRecord);
    const VkDeviceSize matrix_bytes = sizeof(float) * 16;
    const VkDeviceSize command_offset_bytes = command_offset(1, 0);
    const VkDeviceSize command_bytes = VkDeviceSize(command_count_) * sizeof(VkDrawIndexedIndirectCommand);
    if (!graph.import_buffer(groups_.handle(), 0, group_bytes, BufferUse::HostWrite, groups, error) ||
        !graph.import_buffer(clusters_.handle(), 0, cluster_bytes, BufferUse::HostWrite, clusters, error) ||
        !graph.import_buffer(previous_mvp_[frame.frame_index].handle(), 0, matrix_bytes,
            BufferUse::HostWrite, previous_mvp_resource, error) ||
        // View 1 is reused after each shadow draw. Treat the prior segment as
        // read/write so graph execution orders both the previous indirect draw
        // and the previous compute write before this overwrite.
        !graph.import_buffer(commands_[frame.frame_index].handle(), command_offset_bytes, command_bytes,
            BufferUse::AllCommandsReadWrite, commands, error))
        return false;

    const auto cull = graph.add_checked_command_pass("cluster DAG local shadow culling",
        [&, projection_scale](VkCommandBuffer command, std::string& record_error) {
        return record_view(command, sets_[frame.frame_index], shadow_mvp, camera,
            projection_scale, 1, record_error);
    });
    FrameImageGraph::BufferResource written_commands;
    if (!graph.read_buffer(cull, groups, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(cull, clusters, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(cull, previous_mvp_resource, BufferUse::ComputeRead, error) ||
        !graph.write_buffer(cull, commands, BufferUse::ComputeWrite, written_commands, error))
        return false;

    const auto shadow_indirect_consumer = graph.add_pass("cluster DAG local shadow indirect draw", [] {});
    if (!graph.read_buffer(shadow_indirect_consumer, written_commands, BufferUse::IndirectRead, error))
        return false;

    ImageStateTracker image_states;
    if (!graph.execute(frame.command_buffer, image_states, {barrier_}, error,
            frame.command_queue_flags, frame.semaphore_handoff, FrameSubmissionGraph::Queue::Graphics))
        return false;
    error.clear();
    return true;
}

void ClusterVisibility::destroy()
{
    if (device_ && pipeline_ && destroy_pipeline_)
        destroy_pipeline_(device_, pipeline_, nullptr);
    if (device_ && layout_ && destroy_layout_)
        destroy_layout_(device_, layout_, nullptr);
    if (device_ && pool_ && destroy_pool_)
        destroy_pool_(device_, pool_, nullptr);
    if (device_ && set_layout_ && destroy_set_layout_)
        destroy_set_layout_(device_, set_layout_, nullptr);
    if (device_ && fallback_hiz_sampler_ && destroy_sampler_)
        destroy_sampler_(device_, fallback_hiz_sampler_, nullptr);
    if (device_ && fallback_hiz_view_ && destroy_image_view_)
        destroy_image_view_(device_, fallback_hiz_view_, nullptr);
    if (device_ && fallback_hiz_image_ && destroy_image_)
        destroy_image_(device_, fallback_hiz_image_, nullptr);
    if (device_ && fallback_hiz_memory_ && free_memory_)
        free_memory_(device_, fallback_hiz_memory_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
    sets_ = {};
    spans_.clear();
    group_count_ = 0;
    cluster_count_ = 0;
    groups_.destroy();
    clusters_.destroy();
    for (BufferResource& output : commands_)
        output.destroy();
    for (BufferResource& previous : previous_mvp_)
        previous.destroy();
    command_count_ = 0;
    device_ = VK_NULL_HANDLE;
    destroy_set_layout_ = nullptr;
    destroy_pool_ = nullptr;
    destroy_layout_ = nullptr;
    destroy_pipeline_ = nullptr;
    destroy_sampler_ = nullptr;
    destroy_image_ = nullptr;
    free_memory_ = nullptr;
    destroy_image_view_ = nullptr;
    clear_color_image_ = nullptr;
    bind_pipeline_ = nullptr;
    bind_sets_ = nullptr;
    update_sets_ = nullptr;
    push_constants_ = nullptr;
    dispatch_ = nullptr;
    barrier_ = nullptr;
    fallback_hiz_image_ = VK_NULL_HANDLE;
    fallback_hiz_memory_ = VK_NULL_HANDLE;
    fallback_hiz_view_ = VK_NULL_HANDLE;
    fallback_hiz_sampler_ = VK_NULL_HANDLE;
    fallback_hiz_initialized_ = false;
}
} // namespace xray::render::vulkan
