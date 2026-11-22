#include "xrEngine/stdafx.h"
#include "SmokeTrailNoise.h"
#include "FrameImageGraph.h"
#include "FrameBufferPool.h"
#include "FrameImagePool.h"
#include "ShaderModule.h"
#include <cmath>
#include <cstring>

namespace xray::render::vulkan
{
bool SmokeTrailNoise::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
    const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
    uint32_t size, uint32_t frames, VkPipelineCache cache, const GpuParticleCompute::Source& source,
    std::string& error, uint32_t graphics_queue_family, uint32_t compute_queue_family,
    FrameImagePool* image_pool)
{
    destroy();
    if (!size || size > 256) { error = "invalid smoke noise volume size"; return false; }
    const uint64_t bytes = uint64_t(size) * size * size * 4;
    const uint32_t groups = (size + 3) / 4;
    if (!device || !get || !source || size > limits.maxImageDimension3D || !frames || frames > 5 ||
        bytes > limits.maxStorageBufferRange || limits.maxComputeWorkGroupInvocations < 64 ||
        limits.maxComputeWorkGroupSize[0] < 4 || limits.maxComputeWorkGroupSize[1] < 4 || limits.maxComputeWorkGroupSize[2] < 4 ||
        groups > limits.maxComputeWorkGroupCount[0] || groups * groups > limits.maxComputeWorkGroupCount[1] ||
        groups * groups > 65535)
    { error = "invalid smoke noise configuration or device limits"; return false; }
#define NOISE_PROC(type, name) reinterpret_cast<type>(get(device, name))
    auto create_image = NOISE_PROC(PFN_vkCreateImage, "vkCreateImage");
    auto requirements = NOISE_PROC(PFN_vkGetImageMemoryRequirements, "vkGetImageMemoryRequirements");
    auto allocate = NOISE_PROC(PFN_vkAllocateMemory, "vkAllocateMemory");
    auto bind = NOISE_PROC(PFN_vkBindImageMemory, "vkBindImageMemory");
    auto create_view = NOISE_PROC(PFN_vkCreateImageView, "vkCreateImageView");
    auto create_sampler = NOISE_PROC(PFN_vkCreateSampler, "vkCreateSampler");
    destroy_image_ = NOISE_PROC(PFN_vkDestroyImage, "vkDestroyImage"); free_memory_ = NOISE_PROC(PFN_vkFreeMemory, "vkFreeMemory");
    destroy_view_ = NOISE_PROC(PFN_vkDestroyImageView, "vkDestroyImageView");
    destroy_sampler_ = NOISE_PROC(PFN_vkDestroySampler, "vkDestroySampler");
    barrier_ = NOISE_PROC(PFN_vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
    copy_ = NOISE_PROC(PFN_vkCmdCopyBufferToImage, "vkCmdCopyBufferToImage");
#undef NOISE_PROC
    if (!create_image || !requirements || !allocate || !bind || !create_view || !create_sampler || !destroy_image_ ||
        !free_memory_ || !destroy_view_ || !destroy_sampler_ || !barrier_ || !copy_)
    { error = "smoke noise Vulkan procedures unavailable"; destroy(); return false; }
    device_ = device; size_ = size;
    graphics_queue_family_ = graphics_queue_family;
    compute_queue_family_ = compute_queue_family;
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_3D; ii.format = VK_FORMAT_R8G8B8A8_UNORM; ii.extent = {size, size, size};
    ii.mipLevels = ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const uint32_t queue_families[]{graphics_queue_family, compute_queue_family};
    if (graphics_queue_family != UINT32_MAX && compute_queue_family != UINT32_MAX &&
        graphics_queue_family != compute_queue_family)
    {
        ii.sharingMode = VK_SHARING_MODE_CONCURRENT;
        ii.queueFamilyIndexCount = 2;
        ii.pQueueFamilyIndices = queue_families;
    }
    image_sharing_mode_ = ii.sharingMode;
    image_sharing_queue_families_.clear();
    if (image_sharing_mode_ == VK_SHARING_MODE_CONCURRENT)
        image_sharing_queue_families_.assign(queue_families, queue_families + 2);
    image_pool_ = image_pool;
    if (image_pool_)
    {
        if (!image_pool_->acquire(ii, image_, error)) { destroy(); return false; }
    }
    else
    {
        if (create_image(device, &ii, nullptr, &image_) != VK_SUCCESS)
        { error = "smoke noise image creation failed"; destroy(); return false; }
        VkMemoryRequirements req{}; requirements(device, image_, &req);
        uint32_t type = 0;
        while (type < memory.memoryTypeCount && (!(req.memoryTypeBits & (1u << type)) ||
            !(memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))) ++type;
        if (type == memory.memoryTypeCount) { error = "smoke noise image memory unavailable"; destroy(); return false; }
        VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mi.allocationSize = req.size; mi.memoryTypeIndex = type;
        if (allocate(device, &mi, nullptr, &memory_) != VK_SUCCESS || bind(device, image_, memory_, 0) != VK_SUCCESS)
        { error = "smoke noise image allocation failed"; destroy(); return false; }
    }
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = image_;
    vi.viewType = VK_IMAGE_VIEW_TYPE_3D; vi.format = ii.format; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (create_view(device, &vi, nullptr, &view_) != VK_SUCCESS)
    { error = "smoke noise image view failed"; destroy(); return false; }
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR; si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    if (create_sampler(device, &si, nullptr, &sampler_) != VK_SUCCESS ||
        !buffer_.initialize(device, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory, dispatch, error, true))
    { if (error.empty()) error = "smoke noise sampler or buffer failed"; destroy(); return false; }
    std::vector<uint8_t> shader;
    if (!source("vk/smoke_noise.cs.spv", shader, error) || !has_spirv_entry(shader.data(), shader.size(), 5, "main"))
    { if (error.empty()) error = "invalid smoke noise shader"; destroy(); return false; }
    std::vector<uint32_t> words(shader.size() / 4); std::memcpy(words.data(), shader.data(), shader.size());
    const std::vector<StorageComputePass::Binding> bindings{{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}};
    if (!compute_.initialize(device, get, frames, bindings, 0, limits.maxStorageBufferRange,
            limits.maxUniformBufferRange, cache, words.data(), shader.size(), error)) { destroy(); return false; }
    for (uint32_t i = 0; i < frames; ++i)
    {
        auto uniform = std::make_unique<GpuParticleUniforms>();
        if (!uniform->initialize(device, limits, memory, dispatch, 1, error)) { destroy(); return false; }
        uniforms_.push_back(std::move(uniform));
    }
    error.clear(); return true;
}
bool SmokeTrailNoise::record(VkCommandBuffer command, uint32_t slot, float time, float tile_scale,
    bool first, std::string& error, VkQueueFlags queue_flags, bool semaphore_handoff,
    FrameSubmissionGraph* submission_graph,
    FrameSubmissionGraph::Pass compute_submission_pass,
    FrameSubmissionGraph::Pass graphics_continuation_pass,
    uint64_t previous_graphics_submission,
    FrameSubmissionGraph::Queue recording_queue, VkQueueFlags graphics_queue_flags,
    RequestSubmissionCommandBuffer request_submission_command_buffer,
    FrameBufferPool* buffer_pool, framegraph::ResourceRetirement* retirement)
{
    if (!command || slot >= uniforms_.size() || !image_ || !std::isfinite(time) || !std::isfinite(tile_scale) || tile_scale <= 0.f)
    { error = "invalid smoke noise frame"; return false; }
    if (semaphore_handoff && request_submission_command_buffer)
    {
        if (!submission_graph || compute_submission_pass == FrameSubmissionGraph::InvalidPass ||
            graphics_continuation_pass == FrameSubmissionGraph::InvalidPass)
        {
            error = "smoke noise generated groups have no frame submission mapping";
            return false;
        }
        struct Params { float time, scale; uint32_t size; float padding; } params{time, tile_scale, size_, 0.f};
        auto& uniform = *uniforms_[slot]; uniform.reset(); VkDescriptorBufferInfo constants{};
        if (!uniform.append(&params, sizeof(params), constants, error)) return false;
        const bool transient = buffer_pool && retirement && slot < 32;
        const uint32_t groups_x = (size_ + 3) / 4;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        FrameImageGraph graph(recording_queue);
        FrameImageGraph::BufferResource buffer_resource, generated_buffer;
        FrameImageGraph::Resource image_resource, uploaded_image;
        const bool has_buffer = transient
            ? graph.declare_buffer(buffer_.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT, buffer_resource, error)
            : graph.import_buffer(buffer_.handle(), 0, buffer_.size(), BufferUse::AllCommandsReadWrite,
                buffer_resource, error, recording_queue, 0, buffer_.sharing_mode(),
                buffer_.sharing_queue_families());
        if (!has_buffer ||
            !graph.import_image(image_, range, image_resource, error,
                FrameSubmissionGraph::Queue::Graphics, first ? 0 : previous_graphics_submission,
                image_sharing_mode_, image_sharing_queue_families_))
            return false;
        const auto generate = graph.add_checked_command_pass("generate smoke noise volume", recording_queue,
            [&](VkCommandBuffer recording_command, std::string& pass_error) {
                const VkDescriptorBufferInfo storage{graph.buffer_handle(generated_buffer), 0, buffer_.size()};
                return compute_.record(recording_command, slot, {constants, storage}, nullptr,
                    groups_x, groups_x * groups_x, pass_error);
            });
        if (!graph.write_buffer(generate, buffer_resource, BufferUse::ComputeWrite,
                generated_buffer, error))
            return false;
        const VkBufferImageCopy region{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            {0, 0, 0}, {size_, size_, size_}};
        const auto upload_image = graph.add_command_pass("upload smoke noise image", recording_queue,
            [&](VkCommandBuffer recording_command) {
                copy_(recording_command, graph.buffer_handle(generated_buffer), image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &region);
            });
        if (!graph.read_buffer(upload_image, generated_buffer, BufferUse::TransferSource, error) ||
            !graph.write(upload_image, image_resource, ImageUse::TransferDestination, uploaded_image, error))
            return false;
        const auto sample = graph.add_pass("sample smoke noise image",
            FrameSubmissionGraph::Queue::Graphics, {});
        if (!graph.read(sample, uploaded_image, ImageUse::Sampled, error))
            return false;
        if (transient && !graph.materialize_buffers(*buffer_pool, error))
            return false;

        ImageStateTracker image_states;
        if (!image_states.register_image(image_, range,
                first ? ImageUse::Undefined : ImageUse::Sampled, error))
            return false;
        std::vector<FrameImageGraph::QueueGroup> queue_groups;
        std::vector<FrameSubmissionGraph::Pass> submission_for_pass;
        if (!graph.generate_submission_groups(*submission_graph,
                [this](FrameSubmissionGraph::Queue queue) {
                    if (queue == FrameSubmissionGraph::Queue::Compute) return compute_queue_family_;
                    if (queue == FrameSubmissionGraph::Queue::Graphics) return graphics_queue_family_;
                    return VK_QUEUE_FAMILY_IGNORED;
                }, queue_groups, submission_for_pass, error))
            return false;
        FrameImageGraph::ExecutionState execution;
        if (!graph.begin_execution(execution, error))
            return false;
        const ImageStateDispatch state_dispatch{barrier_};
        bool recorded = true;
        for (const FrameImageGraph::QueueGroup& group : queue_groups)
        {
            const FrameSubmissionGraph::Pass submission = group.submission;
            VkCommandBuffer group_command = VK_NULL_HANDLE;
            if (submission == FrameSubmissionGraph::InvalidPass ||
                !request_submission_command_buffer(submission, group.queue, group_command, error))
            {
                recorded = false;
                break;
            }
            const VkQueueFlags group_flags = group.queue == FrameSubmissionGraph::Queue::Compute ?
                queue_flags : group.queue == FrameSubmissionGraph::Queue::Copy ?
                    VK_QUEUE_TRANSFER_BIT : graphics_queue_flags;
            if (!graph.execute_group(group, group_command, image_states, execution, state_dispatch,
                    error, group_flags, semaphore_handoff))
            {
                recorded = false;
                break;
            }
        }
        // Earlier queue groups can reference this buffer even if a later group fails.
        if (transient)
            graph.retire_allocations(*retirement, 1u << slot);
        if (recorded) error.clear();
        return recorded;
    }
    if (semaphore_handoff)
    {
        if (!submission_graph || compute_submission_pass == FrameSubmissionGraph::InvalidPass ||
            graphics_continuation_pass == FrameSubmissionGraph::InvalidPass)
        {
            error = "smoke noise async image has no frame submission mapping";
            return false;
        }
        FrameImageGraph graph(recording_queue);
        FrameImageGraph::Resource image_resource, sampled_image;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (!graph.import_image(image_, range, image_resource, error,
                FrameSubmissionGraph::Queue::Graphics, first ? 0 : previous_graphics_submission,
                image_sharing_mode_, image_sharing_queue_families_))
            return false;
        const auto generate = graph.add_pass("generate smoke noise image", {});
        if (!graph.write(generate, image_resource, ImageUse::TransferDestination,
                sampled_image, error))
            return false;
        const auto draw = graph.add_pass("sample smoke noise image",
            FrameSubmissionGraph::Queue::Graphics, {});
        if (!graph.read(draw, sampled_image, ImageUse::Sampled, error) ||
            !graph.connect_submission_dependencies(*submission_graph,
                [compute_submission_pass, graphics_continuation_pass](FrameSubmissionGraph::Queue queue) {
                    if (queue == FrameSubmissionGraph::Queue::Compute) return compute_submission_pass;
                    if (queue == FrameSubmissionGraph::Queue::Graphics) return graphics_continuation_pass;
                    return FrameSubmissionGraph::InvalidPass;
                },
                [this](FrameSubmissionGraph::Queue queue) {
                    if (queue == FrameSubmissionGraph::Queue::Compute) return compute_queue_family_;
                    if (queue == FrameSubmissionGraph::Queue::Graphics) return graphics_queue_family_;
                    return VK_QUEUE_FAMILY_IGNORED;
                }, error))
            return false;
    }
    struct Params { float time, scale; uint32_t size; float padding; } params{time, tile_scale, size_, 0.f};
    auto& uniform = *uniforms_[slot]; uniform.reset(); VkDescriptorBufferInfo constants{};
    if (!uniform.append(&params, sizeof(params), constants, error)) return false;
    const VkDescriptorBufferInfo storage{buffer_.handle(), 0, buffer_.size()};
    const uint32_t groups = (size_ + 3) / 4;
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier_(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &before, 0, nullptr, 0, nullptr);
    if (!compute_.record(command, slot, {constants, storage}, nullptr, groups, groups * groups, error)) return false;
    // StorageComputePass dispatches z=1; the noise shader flattens Z into Y groups below.
    VkImageMemoryBarrier image{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    image.oldLayout = first ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image.srcQueueFamilyIndex = image.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image.image = image_; image.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    image.srcAccessMask = first || semaphore_handoff ? 0 : VK_ACCESS_SHADER_READ_BIT;
    image.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    const VkPipelineStageFlags source_stage = first || semaphore_handoff ?
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
    barrier_(command, source_stage,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image);
    VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {size_, size_, size_};
    copy_(command, buffer_.handle(), image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    image.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; image.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image.dstAccessMask = semaphore_handoff ? 0 : VK_ACCESS_SHADER_READ_BIT;
    const VkPipelineStageFlags destination_stage = semaphore_handoff ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT :
        ((queue_flags & VK_QUEUE_GRAPHICS_BIT) ? VK_PIPELINE_STAGE_VERTEX_SHADER_BIT :
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    barrier_(command, VK_PIPELINE_STAGE_TRANSFER_BIT, destination_stage, 0, 0, nullptr, 0, nullptr, 1, &image);
    error.clear(); return true;
}
void SmokeTrailNoise::destroy()
{
    uniforms_.clear(); compute_.destroy(); buffer_.destroy();
    if (device_ && sampler_ && destroy_sampler_) destroy_sampler_(device_, sampler_, nullptr);
    if (device_ && view_ && destroy_view_) destroy_view_(device_, view_, nullptr);
    if (image_ && image_pool_) image_pool_->release(image_);
    else if (device_ && image_ && destroy_image_) destroy_image_(device_, image_, nullptr);
    image_pool_ = nullptr;
    if (device_ && memory_ && free_memory_) free_memory_(device_, memory_, nullptr);
    device_ = VK_NULL_HANDLE; image_ = VK_NULL_HANDLE; memory_ = VK_NULL_HANDLE; view_ = VK_NULL_HANDLE; sampler_ = VK_NULL_HANDLE; size_ = 0;
    image_sharing_mode_ = VK_SHARING_MODE_EXCLUSIVE;
    image_sharing_queue_families_.clear();
    graphics_queue_family_ = compute_queue_family_ = VK_QUEUE_FAMILY_IGNORED;
}
}
