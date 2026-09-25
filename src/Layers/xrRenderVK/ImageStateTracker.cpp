#include "ImageStateTracker.h"

#include <algorithm>

namespace xray::render::vulkan
{
namespace
{
struct ImageUseInfo
{
    VkImageLayout layout;
    VkAccessFlags access;
    VkPipelineStageFlags stage;
};

ImageUseInfo describe(ImageUse use)
{
    switch (use)
    {
    case ImageUse::Undefined:
        return {VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    case ImageUse::TransferSource:
        return {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
    case ImageUse::TransferDestination:
        return {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
    case ImageUse::Sampled:
        return {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    case ImageUse::ColorAttachment:
        return {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    case ImageUse::DepthStencilAttachment:
        return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT};
    case ImageUse::Present:
        return {VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 0, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT};
    }
    return {VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
}
}

bool ImageStateTracker::register_image(VkImage image, const VkImageSubresourceRange& range,
    ImageUse initial_use, std::string& error)
{
    if (image == VK_NULL_HANDLE || range.aspectMask == 0 || range.levelCount == 0 || range.layerCount == 0)
    {
        error = "Vulkan image state registration has an invalid image range";
        return false;
    }
    if (std::any_of(m_images.begin(), m_images.end(), [image](const ImageState& state)
        { return state.image == image; }))
    {
        error = "Vulkan image is already registered with the state tracker";
        return false;
    }
    try
    {
        m_images.push_back({image, range, initial_use});
    }
    catch (...)
    {
        error = "Vulkan image state tracking allocation failed";
        return false;
    }
    error.clear();
    return true;
}

bool ImageStateTracker::transition(VkCommandBuffer command, VkImage image, ImageUse next_use,
    const ImageStateDispatch& vk, std::string& error)
{
    if (command == VK_NULL_HANDLE || !vk.cmd_pipeline_barrier)
    {
        error = "Vulkan image transition procedures are unavailable";
        return false;
    }
    const auto found = std::find_if(m_images.begin(), m_images.end(), [image](const ImageState& state)
        { return state.image == image; });
    if (found == m_images.end())
    {
        error = "Vulkan image is not registered with the state tracker";
        return false;
    }
    if (found->use == next_use)
    {
        error.clear();
        return true;
    }
    if (next_use == ImageUse::Undefined)
    {
        error = "Vulkan image transitions cannot target the undefined layout";
        return false;
    }

    const ImageUseInfo before = describe(found->use);
    const ImageUseInfo after = describe(next_use);
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = before.access;
    barrier.dstAccessMask = after.access;
    barrier.oldLayout = before.layout;
    barrier.newLayout = after.layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = found->range;
    vk.cmd_pipeline_barrier(command, before.stage, after.stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    found->use = next_use;
    error.clear();
    return true;
}

bool ImageStateTracker::forget_image(VkImage image)
{
    const auto found = std::find_if(m_images.begin(), m_images.end(), [image](const ImageState& state)
        { return state.image == image; });
    if (found == m_images.end())
        return false;
    m_images.erase(found);
    return true;
}
}
