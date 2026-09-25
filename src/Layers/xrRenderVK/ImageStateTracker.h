#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace xray::render::vulkan
{
enum class ImageUse
{
    Undefined,
    TransferSource,
    TransferDestination,
    Sampled,
    ColorAttachment,
    DepthStencilAttachment,
    Present
};

struct ImageStateDispatch
{
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier{};
};

// Tracks whole-image usage on one externally synchronized graphics queue.
// Swapchain and image lifetime remain owned by their renderer resources.
class ImageStateTracker
{
public:
    bool register_image(VkImage image, const VkImageSubresourceRange& range,
        ImageUse initial_use, std::string& error);
    bool transition(VkCommandBuffer command, VkImage image, ImageUse next_use,
        const ImageStateDispatch& vk, std::string& error);
    bool forget_image(VkImage image);

private:
    struct ImageState
    {
        VkImage image = VK_NULL_HANDLE;
        VkImageSubresourceRange range{};
        ImageUse use = ImageUse::Undefined;
    };

    std::vector<ImageState> m_images;
};
}
