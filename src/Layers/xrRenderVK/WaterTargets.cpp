#include "WaterTargets.h"
#include "FrameImageGraph.h"

#include <utility>

namespace xray::render::vulkan
{
bool WaterTargets::make_image(Image& target, std::string& error)
{
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = format_;
    image.extent = {extent_.width, extent_.height, 1};
    image.mipLevels = image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (image_pool_)
    {
        if (!image_pool_->acquire(image, target.image, error))
        { error = "water color target: " + error; return false; }
    }
    else
    {
        if (vk_.create_image(device_, &image, nullptr, &target.image) != VK_SUCCESS)
        { error = "cannot create water color target"; return false; }
        VkMemoryRequirements requirements{};
        vk_.get_image_memory_requirements(device_, target.image, &requirements);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < memory_.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memory_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            { type = i; break; }
        if (type == UINT32_MAX)
        { error = "no device-local memory for water color target"; return false; }
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        if (vk_.allocate_memory(device_, &allocation, nullptr, &target.memory) != VK_SUCCESS ||
            vk_.bind_image_memory(device_, target.image, target.memory, 0) != VK_SUCCESS)
        { error = "cannot allocate water color target"; return false; }
    }
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = target.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format_;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.levelCount = view.subresourceRange.layerCount = 1;
    const VkImageSubresourceRange capture_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (!capture_states_.register_image(target.image, capture_range, ImageUse::Undefined, error))
    { error = "cannot track water capture image: " + error; return false; }
    if (vk_.create_image_view(device_, &view, nullptr, &target.view) != VK_SUCCESS)
    { error = "cannot create water color view"; return false; }
    return true;
}

bool WaterTargets::initialize(VkPhysicalDevice physical, VkDevice device, VkFormat format, VkExtent2D extent,
    uint32_t count, const VkPhysicalDeviceMemoryProperties& memory,
    const FrameDispatch& dispatch, const BufferResourceDispatch& buffers,
    std::string& error, bool refraction_only, FrameImagePool* image_pool)
{
    destroy();
    if (!physical || !device || format == VK_FORMAT_UNDEFINED || !extent.width || !extent.height ||
        !count || count > 16 || !dispatch.create_image || !dispatch.destroy_image ||
        !dispatch.get_format_properties ||
        !dispatch.get_image_memory_requirements || !dispatch.allocate_memory ||
        !dispatch.free_memory || !dispatch.bind_image_memory ||
        !dispatch.create_image_view || !dispatch.destroy_image_view ||
        !dispatch.cmd_copy_image || !dispatch.cmd_pipeline_barrier)
    { error = "water targets require complete Vulkan image and copy procedures"; return false; }
    VkFormatProperties properties{};
    dispatch.get_format_properties(physical, format, &properties);
    if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
    { error = "swapchain color format cannot be sampled by water targets"; return false; }
    device_ = device;
    format_ = format;
    extent_ = extent;
    memory_ = memory;
    vk_ = dispatch;
    image_pool_ = image_pool;
    targets_.resize(count);
    for (auto& target : targets_)
        if (!make_image(target.refraction, error) || (!refraction_only && (!make_image(target.field, error) ||
            !target.constants.initialize(device, sizeof(WaterSceneUniform),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                memory, buffers, error))))
        { destroy(); return false; }
    error.clear();
    return true;
}

bool WaterTargets::write_uniform(uint32_t index, const WaterSceneUniform& value,
    std::string& error)
{
    return index < targets_.size() &&
        targets_[index].constants.write(0, &value, sizeof(value), error);
}

bool WaterTargets::capture(VkCommandBuffer command, VkImage source, uint32_t index)
{
    return capture_images(command, source, index, false);
}

bool WaterTargets::capture_refraction(VkCommandBuffer command, VkImage source, uint32_t index)
{
    return capture(command, source, index);
}

bool WaterTargets::capture_distortion(VkCommandBuffer command, VkImage source, uint32_t index)
{
    return capture_images(command, source, index, true);
}

bool WaterTargets::capture_images(VkCommandBuffer command, VkImage source, uint32_t index, bool distortion)
{
    if (!command || !source || index >= targets_.size() ||
        (distortion && !targets_[index].field.image)) return false;
    const auto& target = targets_[index];
    const VkImage destination = distortion ? target.field.image : target.refraction.image;
    ImageStateTracker candidate = capture_states_;
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    FrameImageGraph graph(FrameSubmissionGraph::Queue::Graphics);
    FrameImageGraph::Resource captured_image{};
    std::string error;
    if (!candidate.has_image(destination) &&
        !candidate.register_image(destination, range, ImageUse::Undefined, error))
        return false;
    if (!graph.import_image(destination, range, captured_image, error,
            FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {},
            ImageUse::Undefined))
        return false;
    const auto copy = graph.add_checked_command_pass("water image capture",
        FrameSubmissionGraph::Queue::Graphics,
        [this, source, destination](VkCommandBuffer recording_command, std::string& pass_error) {
            VkImageCopy region{};
            region.srcSubresource.aspectMask = region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.layerCount = region.dstSubresource.layerCount = 1;
            region.extent = {extent_.width, extent_.height, 1};
            vk_.cmd_copy_image(recording_command, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            pass_error.clear();
            return true;
        });
    FrameImageGraph::Resource copied_image{};
    if (!graph.write(copy, captured_image, ImageUse::TransferDestination, copied_image, error))
        return false;
    const auto sampled = graph.add_pass("water image ready for sampling", [] {});
    if (!graph.read(sampled, copied_image, ImageUse::Sampled, error) ||
        !graph.execute(command, candidate, {vk_.cmd_pipeline_barrier}, error,
            VK_QUEUE_GRAPHICS_BIT, false, FrameSubmissionGraph::Queue::Graphics))
        return false;
    capture_states_ = std::move(candidate);
    return true;
}

VkImageView WaterTargets::refraction(uint32_t index) const
{ return index < targets_.size() ? targets_[index].refraction.view : VK_NULL_HANDLE; }
VkImageView WaterTargets::distortion_field(uint32_t index) const
{ return index < targets_.size() ? targets_[index].field.view : VK_NULL_HANDLE; }
VkBuffer WaterTargets::uniform(uint32_t index) const
{ return index < targets_.size() ? targets_[index].constants.handle() : VK_NULL_HANDLE; }

void WaterTargets::destroy()
{
    capture_states_ = {};
    if (device_)
        for (auto& target : targets_)
        {
            target.constants.destroy();
            for (Image* image : {&target.refraction, &target.field})
            {
                if (image->view) vk_.destroy_image_view(device_, image->view, nullptr);
                if (image->image)
                {
                    if (image_pool_) image_pool_->release(image->image);
                    else vk_.destroy_image(device_, image->image, nullptr);
                }
                if (image->memory) vk_.free_memory(device_, image->memory, nullptr);
            }
        }
    targets_.clear();
    device_ = VK_NULL_HANDLE;
    format_ = VK_FORMAT_UNDEFINED;
    extent_ = {};
    vk_ = {};
    image_pool_ = nullptr;
}
}
