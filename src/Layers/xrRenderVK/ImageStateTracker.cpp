#include "ImageStateTracker.h"

#include <algorithm>
#include <limits>
#include <utility>

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

struct ImageBarrier
{
    VkImageMemoryBarrier barrier{};
    ImageUse previous_use = ImageUse::Undefined;
};

ImageUseInfo describe(ImageUse use)
{
    switch (use)
    {
    case ImageUse::Undefined:
        return {VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    case ImageUse::TransferSource:
        return {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
    case ImageUse::ColorAttachmentWriteTransferSource:
        return {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    case ImageUse::TransferDestination:
        return {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
    case ImageUse::TransferDestinationGeneral:
        return {VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
    case ImageUse::Sampled:
        return {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};
    case ImageUse::VertexSampled:
        return {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT};
    case ImageUse::ComputeSampled:
        return {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    case ImageUse::SampledGraphicsCompute:
        return {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    case ImageUse::StorageRead:
        return {VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    case ImageUse::StorageWrite:
        return {VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    case ImageUse::StorageReadWrite:
        return {VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    case ImageUse::ColorAttachment:
        return {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    case ImageUse::DepthStencilAttachment:
        return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT};
    case ImageUse::DepthStencilProduced:
        return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT};
    case ImageUse::DepthStencilReadOnly:
        return {VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};
    case ImageUse::Present:
        return {VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, 0, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT};
    }
    return {VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
}

bool valid_use(ImageUse use)
{
    switch (use)
    {
    case ImageUse::Undefined:
    case ImageUse::TransferSource:
    case ImageUse::ColorAttachmentWriteTransferSource:
    case ImageUse::TransferDestination:
    case ImageUse::TransferDestinationGeneral:
    case ImageUse::Sampled:
    case ImageUse::VertexSampled:
    case ImageUse::ComputeSampled:
    case ImageUse::SampledGraphicsCompute:
    case ImageUse::StorageRead:
    case ImageUse::StorageWrite:
    case ImageUse::StorageReadWrite:
    case ImageUse::ColorAttachment:
    case ImageUse::DepthStencilAttachment:
    case ImageUse::DepthStencilProduced:
    case ImageUse::DepthStencilReadOnly:
    case ImageUse::Present:
        return true;
    }
    return false;
}

bool valid_range(const VkImageSubresourceRange& range)
{
    const uint32_t remaining = std::numeric_limits<uint32_t>::max();
    return range.aspectMask != 0 && range.levelCount != 0 && range.layerCount != 0 &&
        range.levelCount != remaining && range.layerCount != remaining &&
        range.baseMipLevel <= remaining - range.levelCount && range.baseArrayLayer <= remaining - range.layerCount;
}

bool contains_range(const VkImageSubresourceRange& outer, const VkImageSubresourceRange& inner)
{
    if (!valid_range(inner) || (inner.aspectMask & outer.aspectMask) != inner.aspectMask ||
        inner.baseMipLevel < outer.baseMipLevel || inner.baseArrayLayer < outer.baseArrayLayer)
        return false;

    const uint32_t mip_offset = inner.baseMipLevel - outer.baseMipLevel;
    const uint32_t layer_offset = inner.baseArrayLayer - outer.baseArrayLayer;
    return mip_offset <= outer.levelCount && inner.levelCount <= outer.levelCount - mip_offset &&
        layer_offset <= outer.layerCount && inner.layerCount <= outer.layerCount - layer_offset;
}

bool writes_image(ImageUse use)
{
    return use == ImageUse::StorageWrite || use == ImageUse::StorageReadWrite ||
        use == ImageUse::ColorAttachment || use == ImageUse::DepthStencilAttachment ||
        use == ImageUse::DepthStencilProduced;
}

size_t state_index(size_t aspect_index, uint32_t mip_level, uint32_t array_layer,
    const VkImageSubresourceRange& image_range)
{
    const size_t mip_count = image_range.levelCount;
    const size_t layer_count = image_range.layerCount;
    const size_t mip_offset = mip_level - image_range.baseMipLevel;
    const size_t layer_offset = array_layer - image_range.baseArrayLayer;
    return (aspect_index * mip_count + mip_offset) * layer_count + layer_offset;
}
}

bool ImageStateTracker::register_image(VkImage image, const VkImageSubresourceRange& range,
    ImageUse initial_use, std::string& error)
{
    if (image == VK_NULL_HANDLE || !valid_range(range) || !valid_use(initial_use))
    {
        error = "Vulkan image state registration has an invalid image range or state";
        return false;
    }
    if (std::any_of(m_images.begin(), m_images.end(), [image](const ImageState& state)
        { return state.image == image; }))
    {
        error = "Vulkan image is already registered with the state tracker";
        return false;
    }

    ImageState state;
    state.image = image;
    state.range = range;
    try
    {
        VkImageAspectFlags remaining_aspects = range.aspectMask;
        while (remaining_aspects)
        {
            const VkImageAspectFlags aspect = remaining_aspects & (0u - remaining_aspects);
            state.aspects.push_back(static_cast<VkImageAspectFlagBits>(aspect));
            remaining_aspects &= ~aspect;
        }

        const size_t mip_count = range.levelCount;
        const size_t layer_count = range.layerCount;
        if (mip_count > std::numeric_limits<size_t>::max() / layer_count ||
            mip_count * layer_count > std::numeric_limits<size_t>::max() / state.aspects.size())
        {
            error = "Vulkan image subresource range is too large to track";
            return false;
        }
        const size_t state_count = mip_count * layer_count * state.aspects.size();
        state.uses.resize(state_count, initial_use);
        state.pending_ownership.resize(state_count, 0);
        m_images.push_back(std::move(state));
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
    const ImageStateDispatch& vk, std::string& error, VkQueueFlags queue_flags, bool semaphore_handoff)
{
    return transition_range(command, image, nullptr, next_use, vk, error, queue_flags, semaphore_handoff);
}

bool ImageStateTracker::transition(VkCommandBuffer command, VkImage image,
    const VkImageSubresourceRange& range, ImageUse next_use, const ImageStateDispatch& vk, std::string& error,
    VkQueueFlags queue_flags, bool semaphore_handoff)
{
    return transition_range(command, image, &range, next_use, vk, error, queue_flags, semaphore_handoff);
}

bool ImageStateTracker::release_ownership(VkCommandBuffer command, VkImage image,
    const VkImageSubresourceRange& range, ImageUse source_use, ImageUse destination_use,
    uint32_t source_family, uint32_t destination_family,
    const ImageStateDispatch& vk, std::string& error, VkQueueFlags queue_flags)
{
    return transition_ownership(command, image, range, source_use, destination_use,
        source_family, destination_family, vk, error, queue_flags, false);
}

bool ImageStateTracker::acquire_ownership(VkCommandBuffer command, VkImage image,
    const VkImageSubresourceRange& range, ImageUse source_use, ImageUse destination_use,
    uint32_t source_family, uint32_t destination_family,
    const ImageStateDispatch& vk, std::string& error, VkQueueFlags queue_flags)
{
    return transition_ownership(command, image, range, source_use, destination_use,
        source_family, destination_family, vk, error, queue_flags, true);
}

bool ImageStateTracker::transition_ownership(VkCommandBuffer command, VkImage image,
    const VkImageSubresourceRange& range, ImageUse source_use, ImageUse destination_use,
    uint32_t source_family, uint32_t destination_family,
    const ImageStateDispatch& vk, std::string& error, VkQueueFlags queue_flags, bool acquire)
{
    if (command == VK_NULL_HANDLE || !vk.cmd_pipeline_barrier ||
        source_family == VK_QUEUE_FAMILY_IGNORED ||
        destination_family == VK_QUEUE_FAMILY_IGNORED ||
        source_family == destination_family || !valid_use(source_use) ||
        !valid_use(destination_use) || source_use == ImageUse::Undefined ||
        destination_use == ImageUse::Undefined || destination_use == ImageUse::DepthStencilProduced)
    {
        error = "Vulkan image ownership transfer has invalid procedures, families, or states";
        return false;
    }
    if (!image_use_supported_on_queue(acquire ? destination_use : source_use, queue_flags))
    {
        error = "Vulkan image ownership transfer state is unsupported by this queue family";
        return false;
    }

    const auto found = std::find_if(m_images.begin(), m_images.end(), [image](const ImageState& state)
        { return state.image == image; });
    if (found == m_images.end())
    {
        error = "Vulkan image ownership transfer references an unregistered image";
        return false;
    }
    if (!contains_range(found->range, range))
    {
        error = "Vulkan image ownership transfer range is outside the registered subresources";
        return false;
    }

    const uint64_t ownership_token = (uint64_t(source_family) << 32) | destination_family;
    for (size_t aspect_index = 0; aspect_index < found->aspects.size(); ++aspect_index)
    {
        const VkImageAspectFlags aspect = found->aspects[aspect_index];
        if (!(range.aspectMask & aspect))
            continue;
        for (uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.levelCount; ++mip)
        {
            for (uint32_t layer = range.baseArrayLayer;
                layer < range.baseArrayLayer + range.layerCount; ++layer)
            {
                const size_t index = state_index(aspect_index, mip, layer, found->range);
                if (found->uses[index] != source_use)
                {
                    error = "Vulkan image ownership transfer source state does not match the tracked state";
                    return false;
                }
                if (acquire ? found->pending_ownership[index] != ownership_token :
                        found->pending_ownership[index] != 0)
                {
                    error = acquire ?
                        "Vulkan image ownership acquire has no matching release" :
                        "Vulkan image already has a pending ownership release";
                    return false;
                }
            }
        }
    }

    const ImageUseInfo before = describe(source_use);
    const ImageUseInfo after = describe(destination_use);
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = acquire ? 0 : before.access;
    barrier.dstAccessMask = acquire ? after.access : 0;
    barrier.oldLayout = before.layout;
    barrier.newLayout = after.layout;
    barrier.srcQueueFamilyIndex = source_family;
    barrier.dstQueueFamilyIndex = destination_family;
    barrier.image = image;
    barrier.subresourceRange = range;
    vk.cmd_pipeline_barrier(command,
        acquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : before.stage,
        acquire ? after.stage : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    for (size_t aspect_index = 0; aspect_index < found->aspects.size(); ++aspect_index)
    {
        const VkImageAspectFlags aspect = found->aspects[aspect_index];
        if (!(range.aspectMask & aspect))
            continue;
        for (uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.levelCount; ++mip)
        {
            for (uint32_t layer = range.baseArrayLayer;
                layer < range.baseArrayLayer + range.layerCount; ++layer)
            {
                const size_t index = state_index(aspect_index, mip, layer, found->range);
                if (acquire)
                {
                    found->uses[index] = destination_use;
                    found->pending_ownership[index] = 0;
                }
                else
                    found->pending_ownership[index] = ownership_token;
            }
        }
    }
    error.clear();
    return true;
}

bool ImageStateTracker::can_transition(VkImage image, const VkImageSubresourceRange& range,
    ImageUse next_use, std::string& error) const
{
    if (!valid_use(next_use) || next_use == ImageUse::Undefined ||
        next_use == ImageUse::DepthStencilProduced)
    {
        error = "Vulkan image transitions cannot target an invalid or undefined state";
        return false;
    }
    const auto found = std::find_if(m_images.begin(), m_images.end(), [image](const ImageState& state)
        { return state.image == image; });
    if (found == m_images.end())
    {
        error = "Vulkan image is not registered with the state tracker";
        return false;
    }
    if (!contains_range(found->range, range))
    {
        error = "Vulkan image transition range is outside the registered subresources";
        return false;
    }
    error.clear();
    return true;
}

bool ImageStateTracker::set_render_pass_final_state(VkImage image,
    const VkImageSubresourceRange& range, ImageUse final_use, std::string& error)
{
    if (!valid_use(final_use) || final_use == ImageUse::Undefined ||
        final_use == ImageUse::DepthStencilProduced)
    {
        error = "Vulkan render pass reported an invalid final image state";
        return false;
    }
    auto found = std::find_if(m_images.begin(), m_images.end(), [image](const ImageState& state)
        { return state.image == image; });
    if (found == m_images.end() || !contains_range(found->range, range))
    {
        error = "Vulkan render pass final state is outside the tracked image range";
        return false;
    }

    for (size_t aspect_index = 0; aspect_index < found->aspects.size(); ++aspect_index)
    {
        const VkImageAspectFlags aspect = found->aspects[aspect_index];
        if (!(range.aspectMask & aspect))
            continue;
        for (uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.levelCount; ++mip)
        {
            for (uint32_t layer = range.baseArrayLayer;
                layer < range.baseArrayLayer + range.layerCount; ++layer)
            {
                const size_t index = state_index(aspect_index, mip, layer, found->range);
                if (found->pending_ownership[index])
                {
                    error = "Vulkan render pass final state has a pending queue-family transfer";
                    return false;
                }
                found->uses[index] = final_use;
            }
        }
    }
    error.clear();
    return true;
}

bool ImageStateTracker::has_image(VkImage image) const
{
    return image != VK_NULL_HANDLE && std::any_of(m_images.begin(), m_images.end(),
        [image](const ImageState& state) { return state.image == image; });
}

bool ImageStateTracker::transition_range(VkCommandBuffer command, VkImage image,
    const VkImageSubresourceRange* requested_range, ImageUse next_use, const ImageStateDispatch& vk,
    std::string& error, VkQueueFlags queue_flags, bool semaphore_handoff)
{
    if (command == VK_NULL_HANDLE || !vk.cmd_pipeline_barrier)
    {
        error = "Vulkan image transition procedures are unavailable";
        return false;
    }
    if (!valid_use(next_use) || next_use == ImageUse::Undefined ||
        next_use == ImageUse::DepthStencilProduced)
    {
        error = "Vulkan image transitions cannot target an invalid or undefined state";
        return false;
    }

    const auto found = std::find_if(m_images.begin(), m_images.end(), [image](const ImageState& state)
        { return state.image == image; });
    if (found == m_images.end())
    {
        error = "Vulkan image is not registered with the state tracker";
        return false;
    }

    const VkImageSubresourceRange& range = requested_range ? *requested_range : found->range;
    if (!contains_range(found->range, range))
    {
        error = "Vulkan image transition range is outside the registered subresources";
        return false;
    }

    try
    {
        std::vector<ImageBarrier> barriers;
        std::vector<size_t> previous_row;
        std::vector<size_t> current_row;
        VkPipelineStageFlags source_stages = 0;
        const ImageUseInfo after = describe(next_use);
        bool has_transition = false;

        for (size_t aspect_index = 0; aspect_index < found->aspects.size(); ++aspect_index)
        {
            const VkImageAspectFlags aspect = found->aspects[aspect_index];
            if (!(range.aspectMask & aspect))
                continue;

            previous_row.clear();
            for (uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.levelCount; ++mip)
            {
                current_row.clear();
                const uint32_t layer_end = range.baseArrayLayer + range.layerCount;
                uint32_t layer = range.baseArrayLayer;
                while (layer < layer_end)
                {
                    const ImageUse previous_use = found->uses[state_index(aspect_index, mip, layer, found->range)];
                    if (previous_use == next_use && !writes_image(next_use))
                    {
                        ++layer;
                        continue;
                    }

                    const uint32_t first_layer = layer++;
                    while (layer < layer_end &&
                        found->uses[state_index(aspect_index, mip, layer, found->range)] == previous_use)
                        ++layer;
                    const uint32_t run_layer_count = layer - first_layer;
                    has_transition = true;
                    const bool source_supported = image_use_supported_on_queue(previous_use, queue_flags);
                    source_stages |= semaphore_handoff && !source_supported ?
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : describe(previous_use).stage;

                    size_t barrier_index = barriers.size();
                    for (size_t previous_index : previous_row)
                    {
                        ImageBarrier& candidate = barriers[previous_index];
                        VkImageSubresourceRange& candidate_range = candidate.barrier.subresourceRange;
                        if (candidate.previous_use == previous_use && candidate_range.aspectMask == aspect &&
                            candidate_range.baseArrayLayer == first_layer &&
                            candidate_range.layerCount == run_layer_count &&
                            candidate_range.baseMipLevel + candidate_range.levelCount == mip)
                        {
                            barrier_index = previous_index;
                            ++candidate_range.levelCount;
                            break;
                        }
                    }

                    if (barrier_index == barriers.size())
                    {
                        const ImageUseInfo before = describe(previous_use);
                        const bool use_semaphore_acquire = semaphore_handoff && !source_supported;
                        ImageBarrier entry;
                        entry.previous_use = previous_use;
                        entry.barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                        entry.barrier.srcAccessMask = use_semaphore_acquire ? 0 : before.access;
                        entry.barrier.dstAccessMask = after.access;
                        entry.barrier.oldLayout = before.layout;
                        entry.barrier.newLayout = after.layout;
                        entry.barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        entry.barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        entry.barrier.image = image;
                        entry.barrier.subresourceRange = {aspect, mip, 1, first_layer, run_layer_count};
                        barriers.push_back(entry);
                        barrier_index = barriers.size() - 1;
                    }
                    current_row.push_back(barrier_index);
                }
                previous_row.swap(current_row);
            }
        }

        if (!has_transition)
        {
            error.clear();
            return true;
        }
        if (barriers.size() > std::numeric_limits<uint32_t>::max())
        {
            error = "Vulkan image transition generated too many barriers";
            return false;
        }

        std::vector<VkImageMemoryBarrier> vk_barriers;
        vk_barriers.reserve(barriers.size());
        for (const ImageBarrier& barrier : barriers)
            vk_barriers.push_back(barrier.barrier);
        vk.cmd_pipeline_barrier(command, source_stages, after.stage, 0, 0, nullptr, 0, nullptr,
            static_cast<uint32_t>(vk_barriers.size()), vk_barriers.data());

        for (size_t aspect_index = 0; aspect_index < found->aspects.size(); ++aspect_index)
        {
            const VkImageAspectFlags aspect = found->aspects[aspect_index];
            if (!(range.aspectMask & aspect))
                continue;
            for (uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.levelCount; ++mip)
                for (uint32_t layer = range.baseArrayLayer; layer < range.baseArrayLayer + range.layerCount; ++layer)
                    found->uses[state_index(aspect_index, mip, layer, found->range)] = next_use;
        }
    }
    catch (...)
    {
        error = "Vulkan image transition allocation failed";
        return false;
    }

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
