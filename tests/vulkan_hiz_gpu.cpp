#include "src/Layers/xrRenderVK/HiZPyramid.h"
#include <vulkan/vulkan.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace xray::render::vulkan;

namespace
{
void check(VkResult result)
{
    if (result != VK_SUCCESS) throw std::runtime_error("Vulkan Hi-Z result " + std::to_string(result));
}

struct Fixture
{
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkImage source{};
    VkImageView source_view{};
    VkDeviceMemory source_memory{};
    VkImage motion{};
    VkImageView motion_view{};
    VkDeviceMemory motion_memory{};
    VkBuffer upload{}, readback{};
    VkDeviceMemory upload_memory{}, readback_memory{};
    VkBuffer motion_upload{};
    VkDeviceMemory motion_upload_memory{};
    VkPhysicalDeviceMemoryProperties memory{};

    ~Fixture()
    {
        if (device)
        {
            vkDeviceWaitIdle(device);
            if (motion_view) vkDestroyImageView(device, motion_view, nullptr);
            if (motion) vkDestroyImage(device, motion, nullptr);
            if (motion_memory) vkFreeMemory(device, motion_memory, nullptr);
            if (source_view) vkDestroyImageView(device, source_view, nullptr);
            if (source) vkDestroyImage(device, source, nullptr);
            if (source_memory) vkFreeMemory(device, source_memory, nullptr);
            if (upload) vkDestroyBuffer(device, upload, nullptr);
            if (readback) vkDestroyBuffer(device, readback, nullptr);
            if (motion_upload) vkDestroyBuffer(device, motion_upload, nullptr);
            if (upload_memory) vkFreeMemory(device, upload_memory, nullptr);
            if (readback_memory) vkFreeMemory(device, readback_memory, nullptr);
            if (motion_upload_memory) vkFreeMemory(device, motion_upload_memory, nullptr);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    VkDeviceMemory allocate(VkMemoryRequirements requirements, VkMemoryPropertyFlags flags)
    {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memory.memoryTypes[i].propertyFlags & flags) == flags)
            {
                VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                info.allocationSize = requirements.size;
                info.memoryTypeIndex = i;
                VkDeviceMemory result{};
                check(vkAllocateMemory(device, &info, nullptr, &result));
                return result;
            }
        throw std::runtime_error("Hi-Z GPU test memory type unavailable");
    }

    void buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer& result, VkDeviceMemory& allocation)
    {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = usage;
        check(vkCreateBuffer(device, &info, nullptr, &result));
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, result, &requirements);
        allocation = allocate(requirements,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkBindBufferMemory(device, result, allocation, 0));
    }
};

void barrier(VkCommandBuffer command, VkImage image, VkImageLayout old_layout,
    VkImageLayout new_layout, VkAccessFlags source, VkAccessFlags destination,
    VkPipelineStageFlags before, VkPipelineStageFlags after, uint32_t mips,
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT)
{
    VkImageMemoryBarrier change{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    change.srcAccessMask = source;
    change.dstAccessMask = destination;
    change.oldLayout = old_layout;
    change.newLayout = new_layout;
    change.srcQueueFamilyIndex = change.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    change.image = image;
    change.subresourceRange = {aspect, 0, mips, 0, 1};
    vkCmdPipelineBarrier(command, before, after, 0, 0, nullptr, 0, nullptr, 1, &change);
}
}

int run_test(int argc, char** argv)
{
    Fixture vk;
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    check(vkCreateInstance(&instance_info, nullptr, &vk.instance));
    uint32_t count{};
    check(vkEnumeratePhysicalDevices(vk.instance, &count, nullptr));
    if (!count) return 77;
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(vk.instance, &count, devices.data()));
    const int selected = argc > 1 ? std::stoi(argv[1]) : -1;
    uint32_t family = UINT32_MAX;
    VkQueueFlags selected_family_flags{};
    for (size_t index = 0; index < devices.size(); ++index)
    {
        if (selected >= 0 && size_t(selected) != index) continue;
        VkFormatProperties support{};
        vkGetPhysicalDeviceFormatProperties(devices[index], VK_FORMAT_R32_SFLOAT, &support);
        if ((support.optimalTilingFeatures &
                (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) !=
            (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) continue;
        VkFormatProperties depth_support{};
        vkGetPhysicalDeviceFormatProperties(devices[index], VK_FORMAT_D32_SFLOAT, &depth_support);
        if ((depth_support.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) == 0)
            continue;
        VkFormatProperties motion_support{};
        vkGetPhysicalDeviceFormatProperties(devices[index], VK_FORMAT_R8G8B8A8_UNORM, &motion_support);
        if ((motion_support.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) == 0)
            continue;
        uint32_t families{};
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families, nullptr);
        std::vector<VkQueueFamilyProperties> queues(families);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families, queues.data());
        for (uint32_t i = 0; i < families; ++i)
            if ((queues[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
                (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            {
                vk.physical = devices[index];
                family = i;
                selected_family_flags = queues[i].queueFlags;
                break;
            }
        if (vk.physical) break;
    }
    if (!vk.physical) return 77;
    float priority = 1.f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    check(vkCreateDevice(vk.physical, &device_info, nullptr, &vk.device));
    vkGetDeviceQueue(vk.device, family, 0, &vk.queue);
    vkGetPhysicalDeviceMemoryProperties(vk.physical, &vk.memory);
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(vk.device, &pool_info, nullptr, &vk.pool));
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = vk.pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(vk.device, &command_info, &vk.command));

    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_D32_SFLOAT;
    image_info.extent = {5, 3, 1};
    image_info.mipLevels = image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    check(vkCreateImage(vk.device, &image_info, nullptr, &vk.source));
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(vk.device, vk.source, &requirements);
    vk.source_memory = vk.allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkBindImageMemory(vk.device, vk.source, vk.source_memory, 0));
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = vk.source;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_D32_SFLOAT;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(vk.device, &view_info, nullptr, &vk.source_view));
    VkImageCreateInfo motion_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    motion_info.imageType = VK_IMAGE_TYPE_2D;
    motion_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    motion_info.extent = {5, 3, 1};
    motion_info.mipLevels = motion_info.arrayLayers = 1;
    motion_info.samples = VK_SAMPLE_COUNT_1_BIT;
    motion_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    motion_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    check(vkCreateImage(vk.device, &motion_info, nullptr, &vk.motion));
    vkGetImageMemoryRequirements(vk.device, vk.motion, &requirements);
    vk.motion_memory = vk.allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkBindImageMemory(vk.device, vk.motion, vk.motion_memory, 0));
    view_info.image = vk.motion;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(vk.device, &view_info, nullptr, &vk.motion_view));
    vk.buffer(15 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk.upload, vk.upload_memory);
    vk.buffer(3 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk.readback, vk.readback_memory);
    vk.buffer(5 * 3 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        vk.motion_upload, vk.motion_upload_memory);
    std::array<uint8_t, 5 * 3 * 4> dynamic_mask{};
    dynamic_mask[(5 * 3 - 1) * 4 + 3] = 255;
    void* motion_mapped{};
    check(vkMapMemory(vk.device, vk.motion_upload_memory, 0, dynamic_mask.size(), 0, &motion_mapped));
    std::memcpy(motion_mapped, dynamic_mask.data(), dynamic_mask.size());
    vkUnmapMemory(vk.device, vk.motion_upload_memory);

    FrameDispatch image_dispatch{};
    image_dispatch.get_format_properties = vkGetPhysicalDeviceFormatProperties;
    image_dispatch.create_image = vkCreateImage;
    image_dispatch.destroy_image = vkDestroyImage;
    image_dispatch.get_image_memory_requirements = vkGetImageMemoryRequirements;
    image_dispatch.allocate_memory = vkAllocateMemory;
    image_dispatch.free_memory = vkFreeMemory;
    image_dispatch.bind_image_memory = vkBindImageMemory;
    image_dispatch.create_image_view = vkCreateImageView;
    image_dispatch.destroy_image_view = vkDestroyImageView;
    image_dispatch.cmd_pipeline_barrier = vkCmdPipelineBarrier;
    FrameImagePool image_pool;
    if (!image_pool.initialize(vk.device, vk.memory,
            {vkCreateImage, vkDestroyImage, vkGetImageMemoryRequirements,
             vkAllocateMemory, vkFreeMemory, vkBindImageMemory}))
        throw std::runtime_error("Hi-Z image pool initialization failed");
    check(vkResetCommandBuffer(vk.command, 0));
    VkCommandBufferBeginInfo mask_begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(vk.command, &mask_begin));
    barrier(vk.command, vk.motion, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 1);
    VkBufferImageCopy mask_upload{};
    mask_upload.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    mask_upload.imageExtent = {5, 3, 1};
    vkCmdCopyBufferToImage(vk.command, vk.motion_upload, vk.motion,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &mask_upload);
    barrier(vk.command, vk.motion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 1);
    check(vkEndCommandBuffer(vk.command));
    VkSubmitInfo mask_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    mask_submit.commandBufferCount = 1;
    mask_submit.pCommandBuffers = &vk.command;
    check(vkQueueSubmit(vk.queue, 1, &mask_submit, VK_NULL_HANDLE));
    check(vkQueueWaitIdle(vk.queue));
    check(vkResetCommandBuffer(vk.command, 0));
    HiZPyramid pyramid;
    std::string error;
    if (!pyramid.initialize(vk.physical, vk.device, vkGetDeviceProcAddr, {5, 3},
            {vk.source_view}, {vk.source}, {vk.motion_view}, {vk.motion},
            vk.memory, image_dispatch, VK_NULL_HANDLE, error, &image_pool))
        throw std::runtime_error(error);
    if (pyramid.levels() != 2 || pyramid.extent().width != 2 || pyramid.extent().height != 1 ||
        !pyramid.view(0) || !pyramid.image(0))
        throw std::runtime_error("Hi-Z pyramid dimensions are wrong");

    for (uint32_t iteration = 0; iteration < 3; ++iteration)
    {
        std::array<float, 15> input{};
        for (uint32_t i = 0; i < input.size(); ++i)
            input[i] = iteration == 0 ? float(i) / 16.f : iteration == 1 ? .25f : .5f;
        if (iteration == 2) input.back() = 1.f; // Far-plane clear on an odd edge.
        void* mapped{};
        check(vkMapMemory(vk.device, vk.upload_memory, 0, sizeof(input), 0, &mapped));
        std::memcpy(mapped, input.data(), sizeof(input));
        vkUnmapMemory(vk.device, vk.upload_memory);
        check(vkResetCommandBuffer(vk.command, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(vk.command, &begin));
        barrier(vk.command, vk.source, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 1, VK_IMAGE_ASPECT_DEPTH_BIT);
        VkBufferImageCopy upload{};
        upload.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        upload.imageExtent = {5, 3, 1};
        vkCmdCopyBufferToImage(vk.command, vk.upload, vk.source,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &upload);
        barrier(vk.command, vk.source, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 1, VK_IMAGE_ASPECT_DEPTH_BIT);
        FrameRecordingContext frame{};
        frame.command_buffer = vk.command;
        frame.command_queue_flags = selected_family_flags;
        if (!pyramid.record(frame, true)) throw std::runtime_error("Hi-Z record failed");
        barrier(vk.command, pyramid.image(0), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, pyramid.levels());
        VkBufferImageCopy copies[2]{};
        const VkExtent3D extents[]{{2, 1, 1}, {1, 1, 1}};
        const VkDeviceSize offsets[]{0, 2 * sizeof(float)};
        for (uint32_t mip = 0; mip < 2; ++mip)
        {
            copies[mip].bufferOffset = offsets[mip];
            copies[mip].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
            copies[mip].imageExtent = extents[mip];
        }
        vkCmdCopyImageToBuffer(vk.command, pyramid.image(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk.readback, 2, copies);
        check(vkEndCommandBuffer(vk.command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &vk.command;
        check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(vk.queue));
        check(vkMapMemory(vk.device, vk.readback_memory, 0, 3 * sizeof(float), 0, &mapped));
        const std::array<float, 3> expected = iteration == 0
            ? std::array<float, 3>{11.f / 16, 1.f, 1.f}
            : iteration == 1 ? std::array<float, 3>{.25f, 1.f, 1.f}
                             : std::array<float, 3>{.5f, 1.f, 1.f};
        const float* result = static_cast<const float*>(mapped);
        for (uint32_t i = 0; i < expected.size(); ++i)
            if (std::abs(result[i] - expected[i]) > 1e-5f)
                throw std::runtime_error("Hi-Z GPU reduction mismatch at " + std::to_string(i) +
                    ": got " + std::to_string(result[i]) + ", expected " + std::to_string(expected[i]));
        vkUnmapMemory(vk.device, vk.readback_memory);
    }
    check(vkDeviceWaitIdle(vk.device));
    const VkImage reused_image = pyramid.image(0);
    pyramid.destroy();
    if (!pyramid.initialize(vk.physical, vk.device, vkGetDeviceProcAddr, {5, 3},
            {vk.source_view}, {vk.source}, {vk.motion_view}, {vk.motion},
            vk.memory, image_dispatch, VK_NULL_HANDLE, error, &image_pool) ||
        pyramid.image(0) != reused_image)
        throw std::runtime_error("Hi-Z image was not reused after GPU-idle rebuild: " + error);
    pyramid.destroy();
    image_pool.reset();
    return 0;
}

int main(int argc, char** argv)
{
    try
    {
        return run_test(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "Vulkan Hi-Z GPU test failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::fprintf(stderr, "Vulkan Hi-Z GPU test failed with an unknown exception\n");
        return EXIT_FAILURE;
    }
}
