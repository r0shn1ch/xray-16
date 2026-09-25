#include "src/Layers/xrRenderVK/ImageStateTracker.h"

#include <cassert>
#include <cstdint>
#include <string>

namespace
{
uint32_t barrier_calls = 0;
VkPipelineStageFlags source_stage{};
VkPipelineStageFlags destination_stage{};
VkImageMemoryBarrier last_barrier{};

void VKAPI_CALL capture_barrier(VkCommandBuffer, VkPipelineStageFlags source, VkPipelineStageFlags destination,
    VkDependencyFlags, uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*,
    uint32_t image_count, const VkImageMemoryBarrier* images)
{
    ++barrier_calls;
    source_stage = source;
    destination_stage = destination;
    assert(image_count == 1);
    last_barrier = images[0];
}
}

int main()
{
    using namespace xray::render::vulkan;
    const auto fake_image = reinterpret_cast<VkImage>(static_cast<uintptr_t>(1));
    const auto fake_command = reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(2));
    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 4;
    range.layerCount = 6;

    ImageStateTracker states;
    ImageStateDispatch dispatch{capture_barrier};
    std::string error;
    assert(states.register_image(fake_image, range, ImageUse::Undefined, error));
    assert(!states.register_image(fake_image, range, ImageUse::Undefined, error));
    assert(states.transition(fake_command, fake_image, ImageUse::TransferDestination, dispatch, error));
    assert(barrier_calls == 1);
    assert(source_stage == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    assert(destination_stage == VK_PIPELINE_STAGE_TRANSFER_BIT);
    assert(last_barrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED);
    assert(last_barrier.newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    assert(last_barrier.dstAccessMask == VK_ACCESS_TRANSFER_WRITE_BIT);
    assert(last_barrier.subresourceRange.levelCount == 4);
    assert(last_barrier.subresourceRange.layerCount == 6);

    assert(states.transition(fake_command, fake_image, ImageUse::TransferDestination, dispatch, error));
    assert(barrier_calls == 1);
    assert(states.transition(fake_command, fake_image, ImageUse::Sampled, dispatch, error));
    assert(barrier_calls == 2);
    assert(source_stage == VK_PIPELINE_STAGE_TRANSFER_BIT);
    assert(destination_stage == (VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT));
    assert(last_barrier.oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    assert(last_barrier.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    assert(last_barrier.srcAccessMask == VK_ACCESS_TRANSFER_WRITE_BIT);
    assert(last_barrier.dstAccessMask == VK_ACCESS_SHADER_READ_BIT);
    assert(!states.transition(fake_command, fake_image, ImageUse::Undefined, dispatch, error));
    assert(barrier_calls == 2);

    const auto unknown_image = reinterpret_cast<VkImage>(static_cast<uintptr_t>(3));
    assert(!states.transition(fake_command, unknown_image, ImageUse::Sampled, dispatch, error));
    assert(states.forget_image(fake_image));
    assert(!states.forget_image(fake_image));
}
