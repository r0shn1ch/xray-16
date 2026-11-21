#include "GBufferTargets.h"
#include "FrameImageGraph.h"

namespace xray::render::vulkan
{
bool negotiate_gbuffer_msaa(VkPhysicalDevice physical, const VkPhysicalDeviceProperties& properties,
    VkFormat color_format, VkFormat depth_format, VkSampleCountFlagBits requested,
    PFN_vkGetPhysicalDeviceImageFormatProperties image_properties,
    const VkPhysicalDeviceDepthStencilResolveProperties& resolve_properties,
    PFN_vkCreateRenderPass2 create_render_pass, GBufferMsaaConfig& result, std::string& error,
    VkFormat scene_color_format)
{
    result = {};
    const uint32_t count = static_cast<uint32_t>(requested);
    constexpr uint32_t allowed = VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT | VK_SAMPLE_COUNT_8_BIT |
        VK_SAMPLE_COUNT_16_BIT | VK_SAMPLE_COUNT_32_BIT | VK_SAMPLE_COUNT_64_BIT;
    if (!physical || !image_properties || !create_render_pass || !(count & allowed) || (count & (count - 1)) ||
        color_format == VK_FORMAT_UNDEFINED || depth_format == VK_FORMAT_UNDEFINED)
    { error = "invalid MSAA capability request"; return false; }
    VkImageFormatProperties color{}, depth{}, scene_color{};
    if (image_properties(physical, color_format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 0, &color) != VK_SUCCESS ||
        image_properties(physical, depth_format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0, &depth) != VK_SUCCESS)
    { error = "MSAA attachment formats or usages are unsupported"; return false; }
    if (scene_color_format != VK_FORMAT_UNDEFINED &&
        image_properties(physical, scene_color_format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 0, &scene_color) != VK_SUCCESS)
    { error = "MSAA scene color format or usage is unsupported"; return false; }
    const VkSampleCountFlags supported = color.sampleCounts & depth.sampleCounts &
        (scene_color_format == VK_FORMAT_UNDEFINED ? ~VkSampleCountFlags{0} : scene_color.sampleCounts) &
        properties.limits.framebufferColorSampleCounts & properties.limits.framebufferDepthSampleCounts;
    VkSampleCountFlagBits selected = VK_SAMPLE_COUNT_1_BIT;
    for (uint32_t candidate = count; candidate >= 2; candidate >>= 1)
        if (supported & candidate) { selected = static_cast<VkSampleCountFlagBits>(candidate); break; }
    if (selected == VK_SAMPLE_COUNT_1_BIT)
    { error = "no supported multisample count for color and depth"; return false; }
    const bool stencil = depth_format == VK_FORMAT_D16_UNORM_S8_UINT || depth_format == VK_FORMAT_D24_UNORM_S8_UINT ||
        depth_format == VK_FORMAT_D32_SFLOAT_S8_UINT;
    VkResolveModeFlags depth_modes = resolve_properties.supportedDepthResolveModes;
    const bool independent_none = resolve_properties.independentResolve || resolve_properties.independentResolveNone;
    if (stencil && !independent_none) depth_modes &= resolve_properties.supportedStencilResolveModes;
    const VkResolveModeFlagBits mode = depth_modes & VK_RESOLVE_MODE_MIN_BIT ? VK_RESOLVE_MODE_MIN_BIT :
        depth_modes & VK_RESOLVE_MODE_SAMPLE_ZERO_BIT ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT : VK_RESOLVE_MODE_NONE;
    if (mode == VK_RESOLVE_MODE_NONE)
    { error = "device has no compatible depth/stencil resolve mode"; return false; }
    result.samples = selected; result.depth_resolve = mode;
    result.stencil_resolve = stencil && !independent_none ? mode : VK_RESOLVE_MODE_NONE;
    result.create_render_pass = create_render_pass;
    error.clear(); return true;
}

bool GBufferTargets::create_attachment(VkFormat format, VkImageUsageFlags usage,
    VkImageAspectFlags aspect, Attachment& target, std::string& error, VkSampleCountFlagBits samples)
{
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = format;
    image.extent = {extent_.width, extent_.height, 1};
    image.mipLevels = image.arrayLayers = 1;
    image.samples = samples;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = usage;
    sharing_.apply(image);
    if (image_pool_)
    {
        if (!image_pool_->acquire(image, target.image, error))
        {
            error = "Vulkan G-buffer image: " + error;
            return false;
        }
    }
    else if (vk_.create_image(device_, &image, nullptr, &target.image) != VK_SUCCESS)
    {
        error = "could not create Vulkan G-buffer image";
        return false;
    }
    if (!image_pool_)
    {
        VkMemoryRequirements requirements{};
        vk_.get_image_memory_requirements(device_, target.image, &requirements);
        uint32_t memory_type = UINT32_MAX;
        for (uint32_t i = 0; i < memory_.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memory_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            {
                memory_type = i;
                break;
            }
        if (memory_type == UINT32_MAX)
        {
            error = "no device-local memory for G-buffer image";
            return false;
        }
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memory_type;
        if (vk_.allocate_memory(device_, &allocation, nullptr, &target.memory) != VK_SUCCESS ||
            vk_.bind_image_memory(device_, target.image, target.memory, 0) != VK_SUCCESS)
        {
            error = "could not allocate Vulkan G-buffer image memory";
            return false;
        }
    }
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = target.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange.aspectMask = aspect;
    view.subresourceRange.levelCount = view.subresourceRange.layerCount = 1;
    if (vk_.create_image_view(device_, &view, nullptr, &target.view) != VK_SUCCESS)
    {
        error = "could not create Vulkan G-buffer image view";
        return false;
    }
    return true;
}

bool GBufferTargets::initialize(VkPhysicalDevice physical_device, VkDevice device, VkExtent2D extent, uint32_t image_count,
    VkFormat depth_format, const VkPhysicalDeviceMemoryProperties& memory,
    const FrameDispatch& dispatch, PFN_vkCreateSampler create_sampler,
    PFN_vkDestroySampler destroy_sampler, std::string& error,
    FrameImagePool* image_pool, bool enable_depth_readback, const GBufferMsaaConfig* msaa,
    const FrameImageSharing* sharing)
{
    destroy();
    if (!physical_device || !device || !extent.width || !extent.height || !image_count ||
        image_count > 16 || depth_format == VK_FORMAT_UNDEFINED ||
        !dispatch.get_format_properties ||
        !dispatch.create_image || !dispatch.destroy_image || !dispatch.allocate_memory ||
        !dispatch.free_memory || !dispatch.bind_image_memory || !dispatch.get_image_memory_requirements ||
        !dispatch.create_image_view || !dispatch.destroy_image_view ||
        !dispatch.create_framebuffer || !dispatch.destroy_framebuffer ||
        !dispatch.create_render_pass || !dispatch.destroy_render_pass ||
        !dispatch.cmd_begin_render_pass || !dispatch.cmd_end_render_pass ||
        !dispatch.cmd_pipeline_barrier || !dispatch.cmd_copy_image ||
        !create_sampler || !destroy_sampler)
    {
        error = "G-buffer targets need a device, formats and complete image procedures";
        return false;
    }
    const VkFormat color = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormatProperties color_support{};
    dispatch.get_format_properties(physical_device, color, &color_support);
    if ((color_support.optimalTilingFeatures &
            (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) !=
        (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
    {
        error = "RGBA8 cannot be used as a sampled G-buffer attachment";
        return false;
    }
    VkFormatProperties depth_support{};
    dispatch.get_format_properties(physical_device, depth_format, &depth_support);
    if ((depth_support.optimalTilingFeatures &
            (VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) !=
        (VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
    {
        error = "selected depth format cannot be stored and sampled as a G-buffer attachment";
        return false;
    }
    device_ = device;
    extent_ = extent;
    memory_ = memory;
    vk_ = dispatch;
    destroy_sampler_ = destroy_sampler;
    image_pool_ = image_pool;
    sharing_ = sharing ? *sharing : FrameImageSharing{};
    samples_ = msaa ? msaa->samples : VK_SAMPLE_COUNT_1_BIT;
    const bool multisampled = samples_ != VK_SAMPLE_COUNT_1_BIT;
    if (!(multisampled ? create_msaa_gbuffer_render_pass(device, color, color, color, depth_format, samples_,
            msaa->depth_resolve, msaa->stencil_resolve, msaa->create_render_pass, pass_, error) :
            create_gbuffer_render_pass(device, color, color, color, depth_format, vk_, pass_, error)))
    {
        destroy();
        return false;
    }
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.maxLod = 0;
    if (create_sampler(device, &sampler, nullptr, &sampler_) != VK_SUCCESS)
    {
        error = "could not create G-buffer sampler";
        destroy();
        return false;
    }
    targets_.resize(image_count);
    for (auto& target : targets_)
    {
        if (multisampled &&
            (!create_attachment(color, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                 target.multisampled[0], error, samples_) ||
             !create_attachment(color, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                 target.multisampled[1], error, samples_) ||
             !create_attachment(color, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                 target.multisampled[2], error, samples_) ||
             !create_attachment(depth_format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                 VK_IMAGE_ASPECT_DEPTH_BIT, target.multisampled[MultisampledDepthAttachment], error, samples_) ||
             !create_attachment(depth_format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                 VK_IMAGE_ASPECT_DEPTH_BIT, target.scene_depth, error, samples_)))
        { destroy(); return false; }
        if (!create_attachment(color, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, target.albedo, error) ||
            !create_attachment(color, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, target.normal, error) ||
            !create_attachment(color, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, target.surface, error) ||
            !create_attachment(depth_format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT, target.depth, error) ||
            !create_attachment(depth_format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                (enable_depth_readback ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0),
                VK_IMAGE_ASPECT_DEPTH_BIT, target.sampled_depth, error))
        {
            destroy();
            return false;
        }
        const VkImageView views[]{target.multisampled[0].view, target.multisampled[1].view,
            target.multisampled[2].view, target.multisampled[MultisampledDepthAttachment].view, target.albedo.view,
            target.normal.view, target.surface.view, target.depth.view};
        VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebuffer.renderPass = pass_;
        framebuffer.attachmentCount = multisampled ? 8 : 4;
        framebuffer.pAttachments = multisampled ? views : views + 4;
        framebuffer.width = extent.width;
        framebuffer.height = extent.height;
        framebuffer.layers = 1;
        if (vk_.create_framebuffer(device, &framebuffer, nullptr, &target.framebuffer) != VK_SUCCESS)
        {
            if (error.empty()) error = "could not create G-buffer framebuffer";
            destroy();
            return false;
        }
    }
    error.clear();
    return true;
}

bool GBufferTargets::begin(const FrameRecordingContext& frame, FrameRecordingContext& geometry_frame) const
{
    if (!pass_ || !frame.command_buffer || frame.image_index >= targets_.size() ||
        frame.extent.width != extent_.width || frame.extent.height != extent_.height) return false;
    VkClearValue clear[4]{};
    clear[0].color = {{0, 0, 0, 0}};
    clear[1].color = {{0.5f, 0.5f, 1.0f, 0}};
    clear[2].color = {{0, 0, 0, 0}};
    clear[3].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    info.renderPass = pass_;
    info.framebuffer = targets_[frame.image_index].framebuffer;
    info.renderArea.extent = extent_;
    info.clearValueCount = 4;
    info.pClearValues = clear;
    vk_.cmd_begin_render_pass(frame.command_buffer, &info, VK_SUBPASS_CONTENTS_INLINE);
    // Preserve queue families, submission graph and generated-command-buffer
    // callbacks from the frame. Geometry consumers use this copy while the
    // render pass is active; reconstructing only the first six fields silently
    // dropped the frame's asynchronous scheduling context.
    geometry_frame = frame;
    geometry_frame.render_pass = pass_;
    geometry_frame.framebuffer = info.framebuffer;
    geometry_frame.extent = extent_;
    return true;
}

void GBufferTargets::end(VkCommandBuffer command) const
{
    if (command && pass_) vk_.cmd_end_render_pass(command);
}

uint32_t GBufferTargets::graph_attachment_count() const
{
    return samples_ == VK_SAMPLE_COUNT_1_BIT ? 4u : 8u;
}

VkImage GBufferTargets::graph_attachment_image(uint32_t image_index, uint32_t attachment_index) const
{
    if (image_index >= targets_.size() || attachment_index >= graph_attachment_count())
        return VK_NULL_HANDLE;
    const Target& target = targets_[image_index];
    if (samples_ != VK_SAMPLE_COUNT_1_BIT && attachment_index < target.multisampled.size())
        return target.multisampled[attachment_index].image;
    switch (attachment_index - (samples_ == VK_SAMPLE_COUNT_1_BIT ? 0u : 4u))
    {
    case 0: return target.albedo.image;
    case 1: return target.normal.image;
    case 2: return target.surface.image;
    case 3: return target.depth.image;
    default: return VK_NULL_HANDLE;
    }
}

VkImageAspectFlags GBufferTargets::graph_attachment_aspect(uint32_t attachment_index) const
{
    if (attachment_index >= graph_attachment_count())
        return 0;
    const bool depth_attachment = attachment_index == 3 ||
        (samples_ != VK_SAMPLE_COUNT_1_BIT && attachment_index == 7);
    return depth_attachment ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
}

bool GBufferTargets::lighting_samples_attachment(uint32_t attachment_index) const
{
    const VkImageAspectFlags aspect = graph_attachment_aspect(attachment_index);
    return aspect == VK_IMAGE_ASPECT_COLOR_BIT ||
        (samples_ != VK_SAMPLE_COUNT_1_BIT &&
            attachment_index == MultisampledDepthAttachment);
}

bool GBufferTargets::copy_depth(const FrameRecordingContext& frame) const
{
    if (!frame.command_buffer || frame.image_index >= targets_.size()) return false;
    const Target& target = targets_[frame.image_index];
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    ImageStateTracker states;
    std::string error;
    // The G-buffer render pass just wrote depth and left it read-only. The
    // sampled copy is fully overwritten each frame, including after pool reuse.
    if (!states.register_image(target.depth.image, range, ImageUse::DepthStencilProduced, error) ||
        !states.register_image(target.sampled_depth.image, range, ImageUse::Undefined, error))
        return false;
    FrameImageGraph graph;
    FrameImageGraph::Resource depth, sampled;
    if (!graph.import_image(target.depth.image, range, depth, error) ||
        !graph.import_image(target.sampled_depth.image, range, sampled, error))
        return false;
    const auto copy = graph.add_checked_command_pass("G-buffer depth copy",
        FrameSubmissionGraph::Queue::Graphics,
        [this, image_index = frame.image_index](
            VkCommandBuffer recording_command, std::string& pass_error) {
            if (!record_depth_copy(recording_command, image_index))
            {
                pass_error = "G-buffer depth copy recording failed";
                return false;
            }
            return true;
        });
    if (!graph.read(copy, depth, ImageUse::TransferSource, error) ||
        !graph.write(copy, sampled, ImageUse::TransferDestination, sampled, error))
        return false;
    const auto finish = graph.add_pass("G-buffer depth ready", [] {});
    const ImageUse sampled_use = (frame.command_queue_flags & VK_QUEUE_COMPUTE_BIT)
        ? ImageUse::SampledGraphicsCompute : ImageUse::Sampled;
    if (!graph.read(finish, depth, ImageUse::DepthStencilReadOnly, error) ||
        !graph.read(finish, sampled, sampled_use, error))
        return false;
    return graph.execute(frame.command_buffer, states, {vk_.cmd_pipeline_barrier}, error,
        frame.command_queue_flags);
}

bool GBufferTargets::copy_scene_depth(const FrameRecordingContext& frame) const
{
    if (samples_ == VK_SAMPLE_COUNT_1_BIT)
        return true;
    if (!frame.command_buffer || frame.image_index >= targets_.size())
        return false;
    const Target& target = targets_[frame.image_index];
    if (!target.multisampled[MultisampledDepthAttachment].image || !target.scene_depth.image)
        return false;
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    ImageStateTracker states;
    FrameImageGraph graph(FrameSubmissionGraph::Queue::Graphics);
    FrameImageGraph::Resource source, destination, copied;
    std::string error;
    if (!states.register_image(target.multisampled[MultisampledDepthAttachment].image, range, ImageUse::DepthStencilProduced, error) ||
        !states.register_image(target.scene_depth.image, range, ImageUse::Undefined, error) ||
        !graph.import_image(target.multisampled[MultisampledDepthAttachment].image, range, source, error,
            FrameSubmissionGraph::Queue::Graphics, 0, sharing_.mode,
            sharing_.queue_families, ImageUse::DepthStencilProduced) ||
        !graph.import_image(target.scene_depth.image, range, destination, error,
            FrameSubmissionGraph::Queue::Graphics, 0, sharing_.mode,
            sharing_.queue_families, ImageUse::Undefined))
        return false;
    const auto copy = graph.add_checked_command_pass("G-buffer scene depth copy",
        FrameSubmissionGraph::Queue::Graphics,
        [this, index = frame.image_index](VkCommandBuffer command, std::string& pass_error) {
            VkImageCopy region{};
            region.srcSubresource.aspectMask = region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            region.srcSubresource.layerCount = region.dstSubresource.layerCount = 1;
            region.extent = {extent_.width, extent_.height, 1};
            vk_.cmd_copy_image(command, targets_[index].multisampled[MultisampledDepthAttachment].image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, targets_[index].scene_depth.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            pass_error.clear();
            return true;
        });
    if (!graph.read(copy, source, ImageUse::TransferSource, error) ||
        !graph.write(copy, destination, ImageUse::TransferDestination, copied, error))
        return false;
    const auto ready = graph.add_pass("G-buffer scene depth ready", [] {});
    if (!graph.read(ready, source, ImageUse::DepthStencilReadOnly, error) ||
        !graph.read(ready, copied, ImageUse::DepthStencilReadOnly, error))
        return false;
    return graph.execute(frame.command_buffer, states, {vk_.cmd_pipeline_barrier}, error,
        frame.graphics_queue_flags);
}

bool GBufferTargets::record_depth_copy(VkCommandBuffer command, uint32_t image_index) const
{
    if (!command || image_index >= targets_.size() || !vk_.cmd_copy_image)
        return false;
    const Target& target = targets_[image_index];
    VkImageCopy region{};
    region.srcSubresource.aspectMask = region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    region.srcSubresource.layerCount = region.dstSubresource.layerCount = 1;
    region.extent = {extent_.width, extent_.height, 1};
    vk_.cmd_copy_image(command, target.depth.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        target.sampled_depth.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    return true;
}

VkDescriptorSet GBufferTargets::lighting_set(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].lighting : VK_NULL_HANDLE;
}

VkDescriptorSet GBufferTargets::multisampled_lighting_set(uint32_t index) const
{
    if (index >= targets_.size()) return VK_NULL_HANDLE;
    const auto& target = targets_[index];
    return target.multisampled_lighting ? target.multisampled_lighting : target.lighting;
}

VkImageView GBufferTargets::depth_view(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].depth.view : VK_NULL_HANDLE;
}

VkImage GBufferTargets::depth_image(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].depth.image : VK_NULL_HANDLE;
}

VkImageView GBufferTargets::scene_depth_view(uint32_t index) const
{
    if (index >= targets_.size()) return VK_NULL_HANDLE;
    const auto& target = targets_[index];
    return samples_ == VK_SAMPLE_COUNT_1_BIT ? target.depth.view : target.scene_depth.view;
}

VkImage GBufferTargets::scene_depth_image(uint32_t index) const
{
    if (index >= targets_.size()) return VK_NULL_HANDLE;
    const auto& target = targets_[index];
    return samples_ == VK_SAMPLE_COUNT_1_BIT ? target.depth.image : target.scene_depth.image;
}

VkImageView GBufferTargets::sampled_depth_view(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].sampled_depth.view : VK_NULL_HANDLE;
}

VkImage GBufferTargets::sampled_depth_image(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].sampled_depth.image : VK_NULL_HANDLE;
}

VkImageView GBufferTargets::normal_view(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].normal.view : VK_NULL_HANDLE;
}

VkImageView GBufferTargets::albedo_view(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].albedo.view : VK_NULL_HANDLE;
}

VkImage GBufferTargets::albedo_image(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].albedo.image : VK_NULL_HANDLE;
}

VkImage GBufferTargets::normal_image(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].normal.image : VK_NULL_HANDLE;
}

VkImageView GBufferTargets::surface_view(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].surface.view : VK_NULL_HANDLE;
}

VkImage GBufferTargets::surface_image(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].surface.image : VK_NULL_HANDLE;
}

void GBufferTargets::destroy()
{
    if (device_)
    {
        for (auto& target : targets_)
        {
            if (target.framebuffer) vk_.destroy_framebuffer(device_, target.framebuffer, nullptr);
            for (Attachment* attachment : {&target.albedo, &target.normal, &target.surface,
                    &target.depth, &target.sampled_depth, &target.scene_depth,
                    &target.multisampled[0], &target.multisampled[1],
                    &target.multisampled[2], &target.multisampled[MultisampledDepthAttachment]})
            {
                if (attachment->view) vk_.destroy_image_view(device_, attachment->view, nullptr);
                if (attachment->image)
                {
                    if (image_pool_) image_pool_->release(attachment->image);
                    else vk_.destroy_image(device_, attachment->image, nullptr);
                }
                if (attachment->memory) vk_.free_memory(device_, attachment->memory, nullptr);
            }
        }
        if (sampler_) destroy_sampler_(device_, sampler_, nullptr);
        if (pass_) vk_.destroy_render_pass(device_, pass_, nullptr);
    }
    targets_.clear();
    sharing_ = {};
    device_ = VK_NULL_HANDLE;
    samples_ = VK_SAMPLE_COUNT_1_BIT;
    pass_ = VK_NULL_HANDLE;
    sampler_ = VK_NULL_HANDLE;
    destroy_sampler_ = nullptr;
    image_pool_ = nullptr;
    vk_ = {};
}
}
