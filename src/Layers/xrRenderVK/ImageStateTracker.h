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
    // A render pass ended in transfer-source layout, but no copy/readback ran;
    // retain the actual color-attachment producer for the next dependency.
    ColorAttachmentWriteTransferSource,
    TransferDestination,
    TransferDestinationGeneral,
    Sampled,
    VertexSampled,
    ComputeSampled,
    SampledGraphicsCompute,
    StorageRead,
    StorageWrite,
    StorageReadWrite,
    ColorAttachment,
    DepthStencilAttachment,
    // Import-only state after a depth render pass writes and ends read-only.
    DepthStencilProduced,
    DepthStencilReadOnly,
    Present
};

inline bool image_use_supported_on_queue(ImageUse use, VkQueueFlags flags)
{
    switch (use)
    {
    case ImageUse::TransferSource:
    case ImageUse::TransferDestination:
    case ImageUse::TransferDestinationGeneral:
        return (flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)) != 0;
    case ImageUse::Sampled:
    case ImageUse::VertexSampled:
    case ImageUse::ColorAttachment:
    case ImageUse::ColorAttachmentWriteTransferSource:
    case ImageUse::DepthStencilAttachment:
    case ImageUse::DepthStencilProduced:
    case ImageUse::DepthStencilReadOnly:
    case ImageUse::Present:
        return (flags & VK_QUEUE_GRAPHICS_BIT) != 0;
    case ImageUse::ComputeSampled:
    case ImageUse::StorageRead:
    case ImageUse::StorageWrite:
    case ImageUse::StorageReadWrite:
        return (flags & VK_QUEUE_COMPUTE_BIT) != 0;
    case ImageUse::SampledGraphicsCompute:
        return (flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
    case ImageUse::Undefined:
        return true;
    }
    return false;
}

struct ImageStateDispatch
{
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier{};
};

// Tracks image subresources across externally synchronized command recording.
// Semaphore handoffs bridge queues, and paired release/acquire calls track
// exclusive family transfers. Resource lifetime remains renderer-owned.
class ImageStateTracker
{
public:
    bool register_image(VkImage image, const VkImageSubresourceRange& range,
        ImageUse initial_use, std::string& error);
    bool transition(VkCommandBuffer command, VkImage image, ImageUse next_use,
        const ImageStateDispatch& vk, std::string& error,
        VkQueueFlags queue_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
        bool semaphore_handoff = false);
    bool transition(VkCommandBuffer command, VkImage image, const VkImageSubresourceRange& range,
        ImageUse next_use, const ImageStateDispatch& vk, std::string& error,
        VkQueueFlags queue_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
        bool semaphore_handoff = false);
    bool release_ownership(VkCommandBuffer command, VkImage image,
        const VkImageSubresourceRange& range, ImageUse source_use, ImageUse destination_use,
        uint32_t source_family, uint32_t destination_family,
        const ImageStateDispatch& vk, std::string& error, VkQueueFlags queue_flags);
    bool acquire_ownership(VkCommandBuffer command, VkImage image,
        const VkImageSubresourceRange& range, ImageUse source_use, ImageUse destination_use,
        uint32_t source_family, uint32_t destination_family,
        const ImageStateDispatch& vk, std::string& error, VkQueueFlags queue_flags);
    bool can_transition(VkImage image, const VkImageSubresourceRange& range,
        ImageUse next_use, std::string& error) const;
    // Records a layout transition performed implicitly by the finalLayout of a
    // render pass after its command recorder has finished successfully.
    bool set_render_pass_final_state(VkImage image, const VkImageSubresourceRange& range,
        ImageUse final_use, std::string& error);
    bool has_image(VkImage image) const;
    bool forget_image(VkImage image);

private:
    struct ImageState
    {
        VkImage image = VK_NULL_HANDLE;
        VkImageSubresourceRange range{};
        std::vector<VkImageAspectFlagBits> aspects;
        std::vector<ImageUse> uses;
        std::vector<uint64_t> pending_ownership;
    };

    bool transition_range(VkCommandBuffer command, VkImage image, const VkImageSubresourceRange* range,
        ImageUse next_use, const ImageStateDispatch& vk, std::string& error,
        VkQueueFlags queue_flags, bool semaphore_handoff);
    bool transition_ownership(VkCommandBuffer command, VkImage image,
        const VkImageSubresourceRange& range, ImageUse source_use, ImageUse destination_use,
        uint32_t source_family, uint32_t destination_family,
        const ImageStateDispatch& vk, std::string& error, VkQueueFlags queue_flags,
        bool acquire);

    std::vector<ImageState> m_images;
};
}
