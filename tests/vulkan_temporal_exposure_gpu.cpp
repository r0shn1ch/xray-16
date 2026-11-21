#include "src/Layers/xrRenderVK/TemporalExposure.h"
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
void check(VkResult value)
{
    if (value != VK_SUCCESS) throw std::runtime_error("Vulkan exposure result " + std::to_string(value));
}
struct Fixture
{
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkImage scene{};
    VkImageView scene_view{};
    VkDeviceMemory scene_memory{};
    VkBuffer upload{}, readback{};
    VkDeviceMemory upload_memory{}, readback_memory{};
    VkPhysicalDeviceMemoryProperties memory{};
    ~Fixture()
    {
        if (device)
        {
            vkDeviceWaitIdle(device);
            if (scene_view) vkDestroyImageView(device, scene_view, nullptr);
            if (scene) vkDestroyImage(device, scene, nullptr);
            if (scene_memory) vkFreeMemory(device, scene_memory, nullptr);
            if (upload) vkDestroyBuffer(device, upload, nullptr);
            if (readback) vkDestroyBuffer(device, readback, nullptr);
            if (upload_memory) vkFreeMemory(device, upload_memory, nullptr);
            if (readback_memory) vkFreeMemory(device, readback_memory, nullptr);
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
                VkDeviceMemory output{};
                check(vkAllocateMemory(device, &info, nullptr, &output));
                return output;
            }
        throw std::runtime_error("exposure GPU memory type unavailable");
    }
    void buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer& output, VkDeviceMemory& allocation)
    {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = usage;
        check(vkCreateBuffer(device, &info, nullptr, &output));
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, output, &requirements);
        allocation = allocate(requirements,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkBindBufferMemory(device, output, allocation, 0));
    }
};
void transition(VkCommandBuffer command, VkImage image, VkImageLayout before_layout,
    VkImageLayout after_layout, VkAccessFlags before_access, VkAccessFlags after_access,
    VkPipelineStageFlags before_stage, VkPipelineStageFlags after_stage, uint32_t mips = 1)
{
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = before_access;
    barrier.dstAccessMask = after_access;
    barrier.oldLayout = before_layout;
    barrier.newLayout = after_layout;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
    vkCmdPipelineBarrier(command, before_stage, after_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
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
    for (size_t index = 0; index < devices.size(); ++index)
    {
        if (selected >= 0 && size_t(selected) != index) continue;
        VkFormatProperties exposure_support{}, scene_support{};
        vkGetPhysicalDeviceFormatProperties(devices[index], VK_FORMAT_R32_SFLOAT, &exposure_support);
        vkGetPhysicalDeviceFormatProperties(devices[index], VK_FORMAT_R32G32B32A32_SFLOAT, &scene_support);
        if ((exposure_support.optimalTilingFeatures &
                (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) !=
                (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ||
            !(scene_support.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) continue;
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
                break;
            }
        if (vk.physical) break;
    }
    if (!vk.physical) return 77;
    const float priority = 1.f;
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
    image_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image_info.extent = {16, 16, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    check(vkCreateImage(vk.device, &image_info, nullptr, &vk.scene));
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(vk.device, vk.scene, &requirements);
    vk.scene_memory = vk.allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkBindImageMemory(vk.device, vk.scene, vk.scene_memory, 0));
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = vk.scene;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(vk.device, &view_info, nullptr, &vk.scene_view));
    vk.buffer(256 * 4 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk.upload, vk.upload_memory);
    vk.buffer(sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk.readback, vk.readback_memory);
    FrameDispatch dispatch{};
    dispatch.get_format_properties = vkGetPhysicalDeviceFormatProperties;
    dispatch.create_image = vkCreateImage;
    dispatch.destroy_image = vkDestroyImage;
    dispatch.get_image_memory_requirements = vkGetImageMemoryRequirements;
    dispatch.allocate_memory = vkAllocateMemory;
    dispatch.free_memory = vkFreeMemory;
    dispatch.bind_image_memory = vkBindImageMemory;
    dispatch.create_image_view = vkCreateImageView;
    dispatch.destroy_image_view = vkDestroyImageView;
    FrameImagePool image_pool;
    if (!image_pool.initialize(vk.device, vk.memory,
            {vkCreateImage, vkDestroyImage, vkGetImageMemoryRequirements,
             vkAllocateMemory, vkFreeMemory, vkBindImageMemory}))
        throw std::runtime_error("exposure image pool initialization failed");
    TemporalExposure exposure;
    std::string error;
    if (!exposure.initialize(vk.physical, vk.device, vkGetDeviceProcAddr,
            {vk.scene_view}, vk.memory, dispatch, VK_NULL_HANDLE, error, &image_pool))
        throw std::runtime_error(error);
    for (uint32_t iteration = 0; iteration < 6; ++iteration)
    {
        if (iteration == 2 || iteration == 5) exposure.reset_history();
        const float luminance = iteration == 0 || iteration == 4 ? 1.f :
            iteration == 3 ? .5f : .25f;
        std::array<float, 256 * 4> pixels{};
        for (size_t pixel = 0; pixel < 256; ++pixel)
        {
            const float value = iteration == 5 ? (pixel < 100 ? .5f : 1.f) : luminance;
            pixels[pixel * 4] = pixels[pixel * 4 + 1] = pixels[pixel * 4 + 2] = value;
            pixels[pixel * 4 + 3] = 1.f;
        }
        if (iteration == 3)
            pixels[0] = pixels[1] = pixels[2] = 1000.f;
        if (iteration == 4)
            pixels[0] = pixels[1] = pixels[2] = 0.f;
        void* mapped{};
        check(vkMapMemory(vk.device, vk.upload_memory, 0, sizeof(pixels), 0, &mapped));
        std::memcpy(mapped, pixels.data(), sizeof(pixels));
        vkUnmapMemory(vk.device, vk.upload_memory);
        exposure.set_parameters(1.f, 0.1f, std::log(2.f), 1.f);
        check(vkResetCommandBuffer(vk.command, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(vk.command, &begin));
        transition(vk.command, vk.scene, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 1);
        VkBufferImageCopy upload_region{};
        upload_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        upload_region.imageExtent = {16, 16, 1};
        vkCmdCopyBufferToImage(vk.command, vk.upload, vk.scene,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &upload_region);
        transition(vk.command, vk.scene, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 1);
        transition(vk.command, exposure.image(), iteration ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            iteration ? VK_ACCESS_SHADER_READ_BIT : 0,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            iteration ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        FrameRecordingContext frame{};
        frame.command_buffer = vk.command;
        frame.extent = {16, 16};
        if (!exposure.record(frame)) throw std::runtime_error("exposure recording failed");
        transition(vk.command, exposure.image(), VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy output_region{};
        output_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        output_region.imageExtent = {1, 1, 1};
        vkCmdCopyImageToBuffer(vk.command, exposure.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk.readback, 1, &output_region);
        transition(vk.command, exposure.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        check(vkEndCommandBuffer(vk.command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &vk.command;
        check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(vk.queue));
        check(vkMapMemory(vk.device, vk.readback_memory, 0, sizeof(float), 0, &mapped));
        const float result = *static_cast<const float*>(mapped);
        vkUnmapMemory(vk.device, vk.readback_memory);
        const float expected = iteration == 5 ? std::exp2(88.f / 231.f) :
            iteration == 0 ? 1.f : iteration == 1 || iteration == 4 ? 1.5f : 2.f;
        if (std::abs(result - expected) > 1e-5f)
            throw std::runtime_error("temporal exposure mismatch: " + std::to_string(result) +
                " expected " + std::to_string(expected));
        exposure.commit_frame();
    }
    check(vkDeviceWaitIdle(vk.device));
    const VkImage reused_image = exposure.image();
    exposure.destroy();
    if (!exposure.initialize(vk.physical, vk.device, vkGetDeviceProcAddr,
            {vk.scene_view}, vk.memory, dispatch, VK_NULL_HANDLE, error, &image_pool) ||
        exposure.image() != reused_image)
        throw std::runtime_error("exposure image was not reused after GPU-idle rebuild: " + error);
    exposure.destroy();
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
        std::fprintf(stderr, "Vulkan temporal exposure GPU test failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::fprintf(stderr, "Vulkan temporal exposure GPU test failed with an unknown exception\n");
        return EXIT_FAILURE;
    }
}
