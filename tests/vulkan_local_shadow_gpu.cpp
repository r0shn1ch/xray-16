#include "../src/Layers/xrRenderVK/LocalShadowTargets.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace xray::render::vulkan;
static void check(VkResult result)
{
    if (result != VK_SUCCESS)
        throw std::runtime_error("Vulkan result " + std::to_string(result));
}
int main()
{
    VkInstance instance{};
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    check(vkCreateInstance(&instance_info, nullptr, &instance));
    uint32_t count = 1;
    VkPhysicalDevice gpu{};
    check(vkEnumeratePhysicalDevices(instance, &count, &gpu));
    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &families, nullptr);
    std::vector<VkQueueFamilyProperties> properties(families);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &families, properties.data());
    uint32_t family = 0;
    while (family < families && !(properties[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
        ++family;
    if (family == families)
        throw std::runtime_error("no graphics queue");
    float priority = 1;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkDevice device{};
    check(vkCreateDevice(gpu, &device_info, nullptr, &device));
    VkQueue queue{};
    vkGetDeviceQueue(device, family, 0, &queue);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(gpu, &memory);
    FrameDispatch dispatch{};
    dispatch.create_image = vkCreateImage;
    dispatch.destroy_image = vkDestroyImage;
    dispatch.get_image_memory_requirements = vkGetImageMemoryRequirements;
    dispatch.allocate_memory = vkAllocateMemory;
    dispatch.free_memory = vkFreeMemory;
    dispatch.bind_image_memory = vkBindImageMemory;
    dispatch.create_image_view = vkCreateImageView;
    dispatch.destroy_image_view = vkDestroyImageView;
    dispatch.create_framebuffer = vkCreateFramebuffer;
    dispatch.destroy_framebuffer = vkDestroyFramebuffer;
    dispatch.create_render_pass = vkCreateRenderPass;
    dispatch.destroy_render_pass = vkDestroyRenderPass;
    dispatch.cmd_begin_render_pass = vkCmdBeginRenderPass;
    dispatch.cmd_end_render_pass = vkCmdEndRenderPass;
    dispatch.cmd_copy_image = vkCmdCopyImage;
    dispatch.cmd_pipeline_barrier = vkCmdPipelineBarrier;
    const BufferResourceDispatch buffers{vkCreateBuffer,   vkDestroyBuffer, vkGetBufferMemoryRequirements,
                                         vkAllocateMemory, vkFreeMemory,    vkBindBufferMemory,
                                         vkMapMemory,      vkUnmapMemory};
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = family;
    VkCommandPool pool{};
    check(vkCreateCommandPool(device, &pool_info, nullptr, &pool));
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer command{};
    check(vkAllocateCommandBuffers(device, &allocation, &command));
    for (VkFormat format : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT})
    {
        LocalShadowTargets targets;
        std::string error;
        if (!targets.initialize(device, format, 2, memory, dispatch, buffers, vkCreateSampler, vkDestroySampler, error))
            throw std::runtime_error(error);
        BufferResource readback;
        if (!readback.initialize(device, 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory,
                                 buffers, error))
            throw std::runtime_error(error);
        check(vkResetCommandPool(device, pool, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(command, &begin));
        FrameRecordingContext frame{command, VK_NULL_HANDLE, VK_NULL_HANDLE, {512, 512}, 0, 0}, shadow;
        assert(targets.initialize_layers(frame));
        float matrix[16]{};
        matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1;
        assert(targets.begin_face(frame, 1, 1, shadow, true));
        VkClearAttachment clear{};
        clear.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        clear.clearValue.depthStencil = {.2f, 0};
        VkClearRect rect{{{0, 0}, {LocalShadowSize, LocalShadowSize}}, 0, 1};
        vkCmdClearAttachments(command, 1, &clear, 1, &rect);
        targets.end(command);
        targets.remember_face(1, 1, matrix);
        assert(targets.cached_face(1, 1, matrix));
        assert(targets.copy_cached_face(frame, 1, 1));
        assert(targets.begin_face(frame, 1, 1, shadow, false, true));
        clear.clearValue.depthStencil.depth = .1f;
        rect.rect.extent = {1, 1};
        vkCmdClearAttachments(command, 1, &clear, 1, &rect);
        targets.end(command);
        // Depth written by dynamic casters must never pollute the shared static cache.
        auto read_depth = [&](uint32_t index, VkDeviceSize offset) {
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.image = targets.image(index);
            barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.subresourceRange = {
                VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT |
                                   (format == VK_FORMAT_D24_UNORM_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)),
                0, 1, 7, 1};
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(command,
                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            VkBufferImageCopy copy{};
            copy.bufferOffset = offset;
            copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 7, 1};
            copy.imageExtent = {1, 1, 1};
            vkCmdCopyImageToBuffer(command, targets.image(index), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   readback.handle(), 1, &copy);
        };
        read_depth(0, 0);
        frame.image_index = 1;
        frame.frame_index = 1;
        assert(targets.initialize_layers(frame));
        assert(targets.copy_cached_face(frame, 1, 1));
        assert(targets.begin_face(frame, 1, 1, shadow, false, true));
        targets.end(command);
        read_depth(1, 4);
        VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        host.srcQueueFamilyIndex = host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host.buffer = readback.handle();
        host.size = 8;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                             &host, 0, nullptr);
        check(vkEndCommandBuffer(command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(queue));
        uint32_t values[2]{};
        assert(readback.read(0, values, 8, error));
        for (unsigned i = 0; i < 2; ++i)
        {
            float actual{};
            if (format == VK_FORMAT_D32_SFLOAT)
                std::memcpy(&actual, &values[i], 4);
            else
                actual = float(values[i] & 0xffffff) / 16777215.f;
            if (std::abs(actual - (i ? .2f : .1f)) > .001f)
                throw std::runtime_error("static shadow cache was overwritten by dynamic depth");
        }
        readback.destroy();
        targets.destroy();
    }
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::cout << "local-shadow GPU cache: D32/D24 static preservation + dynamic overlay OK\n";
}
