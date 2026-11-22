#include "Layers/xrRenderVK/ClusterVisibility.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace xray::render::vulkan;

namespace
{
struct Fixture
{
    VkInstance instance{};
    VkDevice device{};
    VkCommandPool command_pool{};
    VkImageView hiz_view{};
    VkSampler hiz_sampler{};

    ~Fixture()
    {
        if (device)
        {
            vkDeviceWaitIdle(device);
            if (hiz_sampler) vkDestroySampler(device, hiz_sampler, nullptr);
            if (hiz_view) vkDestroyImageView(device, hiz_view, nullptr);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

struct HiZSampling
{
    VkDevice device{};
    VkImageView view{};
    VkSampler sampler{};
    ~HiZSampling()
    {
        if (sampler) vkDestroySampler(device, sampler, nullptr);
        if (view) vkDestroyImageView(device, view, nullptr);
    }
};

struct DeviceDrain
{
    VkDevice device{};
    ~DeviceDrain() { if (device) vkDeviceWaitIdle(device); }
};

bool same_command(const VkDrawIndexedIndirectCommand& actual, uint32_t index_count,
    uint32_t instance_count, uint32_t first_index, int32_t vertex_offset)
{
    return actual.indexCount == index_count && actual.instanceCount == instance_count &&
        actual.firstIndex == first_index && actual.vertexOffset == vertex_offset && actual.firstInstance == 0;
}
}

int main(int argc, char** argv)
{
    int requested_device = -1;
    for (int argument = 1; argument < argc; ++argument)
    {
        if (std::string(argv[argument]) != "--device-index" || ++argument >= argc)
        {
            std::cerr << "usage: vulkan_cluster_visibility_gpu_test [--device-index N]\n";
            return 1;
        }
        try
        {
            size_t parsed = 0;
            requested_device = std::stoi(argv[argument], &parsed);
            if (parsed != std::string(argv[argument]).size() || requested_device < 0)
                throw std::invalid_argument("invalid adapter index");
        }
        catch (const std::exception&)
        {
            std::cerr << "invalid --device-index value\n";
            return 1;
        }
    }

    Fixture fixture;
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    if (vkCreateInstance(&instance_info, nullptr, &fixture.instance) != VK_SUCCESS)
    {
        std::cerr << "Vulkan instance creation failed\n";
        return 77;
    }

    uint32_t device_count = 0;
    if (vkEnumeratePhysicalDevices(fixture.instance, &device_count, nullptr) != VK_SUCCESS || !device_count)
    {
        std::cerr << "no Vulkan physical device is available\n";
        return 77;
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    if (vkEnumeratePhysicalDevices(fixture.instance, &device_count, devices.data()) != VK_SUCCESS)
    {
        std::cerr << "Vulkan physical-device enumeration failed\n";
        return 77;
    }
    if (requested_device >= 0 && static_cast<uint32_t>(requested_device) >= device_count)
    {
        std::cerr << "selected Vulkan physical-device index is out of range\n";
        return 1;
    }

    VkPhysicalDevice physical{};
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory{};
    VkQueueFamilyProperties selected_queue{};
    uint32_t selected_family = UINT32_MAX;
    for (uint32_t index = 0; index < device_count; ++index)
    {
        if (requested_device >= 0 && index != static_cast<uint32_t>(requested_device))
            continue;
        VkPhysicalDeviceProperties candidate_properties{};
        vkGetPhysicalDeviceProperties(devices[index], &candidate_properties);
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &family_count, families.data());
        for (uint32_t family = 0; family < family_count; ++family)
        {
            const VkQueueFlags required = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
            if (families[family].queueCount && (families[family].queueFlags & required) == required)
            {
                physical = devices[index];
                properties = candidate_properties;
                selected_family = family;
                selected_queue = families[family];
                vkGetPhysicalDeviceMemoryProperties(physical, &memory);
                break;
            }
        }
        if (physical) break;
    }
    if (!physical)
    {
        std::cerr << "selected device has no graphics and compute queue family\n";
        return 77;
    }
    std::cout << properties.deviceName << ": Cluster DAG GPU visibility\n";

    const float priority = 1.f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = selected_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    if (vkCreateDevice(physical, &device_info, nullptr, &fixture.device) != VK_SUCCESS)
    {
        std::cerr << "Vulkan logical-device creation failed\n";
        return 77;
    }
    VkQueue queue{};
    vkGetDeviceQueue(fixture.device, selected_family, 0, &queue);
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = selected_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    require(vkCreateCommandPool(fixture.device, &pool_info, nullptr, &fixture.command_pool) == VK_SUCCESS,
        "Vulkan command-pool creation failed");

    const BufferResourceDispatch buffers{vkCreateBuffer, vkDestroyBuffer, vkGetBufferMemoryRequirements,
        vkAllocateMemory, vkFreeMemory, vkBindBufferMemory, vkMapMemory, vkUnmapMemory,
        nullptr, selected_family, selected_family, selected_family};
    const std::vector<ClusterCullGroup> groups{
        {{0.f, 0.f, .5f, .1f}, 10.f, {}},
        {{0.f, 0.f, .5f, .1f}, 10.f, {}}};
    const std::vector<ClusterCullRecord> clusters{
        {{.2f, 0.f, .5f, .1f}, -1, 0, 4, 3},
        {{4.f, 0.f, .5f, .1f}, -1, 0, 20, 6},
        {{0.f, 0.f, .5f, .1f}, -1, 0, 4, 3},
        {{4.f, 0.f, .5f, .1f}, -1, 0, 20, 6}};
    // Each progressive window owns local group/cluster indices and a separate
    // packed index range and indirect-command segment.
    const std::vector<ClusterCullSpan> spans{
        {0, 1, 0, 2, 0, 100, 3},
        {1, 1, 2, 2, 2, 200, 7}};
    const uint32_t command_count = 4;
    std::string error;
    ClusterVisibility visibility;
    require(visibility.initialize(fixture.device, vkGetDeviceProcAddr, VK_NULL_HANDLE, memory,
            buffers, groups, clusters, spans, command_count,
            properties.limits.maxStorageBufferRange, error), error);

    BufferResource readback;
    BufferResource depth_upload;
    FrameImagePool hiz_pool;
    HiZSampling hiz_sampling{fixture.device};
    const VkDeviceSize command_bytes = sizeof(VkDrawIndexedIndirectCommand) * command_count * 2;
    require(readback.initialize(fixture.device, command_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            memory, buffers, error), error);
    const std::array<float, 2> clear_depths{1.f, 1.f};
    require(depth_upload.initialize(fixture.device, sizeof(clear_depths), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            memory, buffers, error), error);
    require(depth_upload.write(0, clear_depths.data(), sizeof(clear_depths), error), error);
    DeviceDrain device_drain{fixture.device};

    VkCommandBufferAllocateInfo command_allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_allocation.commandPool = fixture.command_pool;
    command_allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocation.commandBufferCount = 1;
    VkCommandBuffer command{};
    require(vkAllocateCommandBuffers(fixture.device, &command_allocation, &command) == VK_SUCCESS,
        "Vulkan command-buffer allocation failed");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    require(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS, "Vulkan command-buffer begin failed");

    constexpr float identity[16]{
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f};
    float sun_mvp[16];
    std::copy(std::begin(identity), std::end(identity), std::begin(sun_mvp));
    sun_mvp[12] = -4.f;
    const float camera[3]{0.f, 0.f, 0.f};
    FrameRecordingContext frame{command, VK_NULL_HANDLE, VK_NULL_HANDLE, {64, 64}, 0, 0};
    frame.command_queue_flags = selected_queue.queueFlags;
    frame.graphics_queue_flags = selected_queue.queueFlags;
    frame.compute_queue_flags = selected_queue.queueFlags;
    require(visibility.record(frame, identity, sun_mvp, camera, 60.f, 64, error), error);

    VkBufferMemoryBarrier command_read{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    command_read.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    command_read.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    command_read.srcQueueFamilyIndex = command_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    command_read.buffer = visibility.command_buffer(frame.frame_index);
    command_read.offset = 0;
    command_read.size = command_bytes;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 1, &command_read, 0, nullptr);
    const VkBufferCopy command_copy{0, 0, command_bytes};
    vkCmdCopyBuffer(command, visibility.command_buffer(frame.frame_index), readback.handle(), 1, &command_copy);
    VkMemoryBarrier host_read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host_read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &host_read, 0, nullptr, 0, nullptr);
    require(vkEndCommandBuffer(command) == VK_SUCCESS, "Vulkan command-buffer end failed");

    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence{};
    require(vkCreateFence(fixture.device, &fence_info, nullptr, &fence) == VK_SUCCESS,
        "Vulkan fence creation failed");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    const bool completed = vkQueueSubmit(queue, 1, &submit, fence) == VK_SUCCESS &&
        vkWaitForFences(fixture.device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    vkDestroyFence(fixture.device, fence, nullptr);
    require(completed, "Cluster DAG compute dispatch did not complete");

    std::array<VkDrawIndexedIndirectCommand, 8> commands{};
    require(readback.read(0, commands.data(), sizeof(commands), error), error);
    require(same_command(commands[0], 3, 1, 104, 3) &&
            same_command(commands[1], 0, 0, 120, 3) &&
            same_command(commands[2], 3, 1, 204, 7) &&
            same_command(commands[3], 0, 0, 220, 7),
        "camera culling did not preserve independent progressive-window command and index ranges");
    require(same_command(commands[4], 0, 0, 104, 3) &&
            same_command(commands[5], 6, 1, 120, 3) &&
            same_command(commands[6], 0, 0, 204, 7) &&
            same_command(commands[7], 6, 1, 220, 7),
        "sun culling did not preserve progressive-window ranges in its independent shadow view");

    require(vkResetCommandBuffer(command, 0) == VK_SUCCESS &&
            vkBeginCommandBuffer(command, &begin) == VK_SUCCESS,
        "local-shadow command-buffer reset failed");
    require(visibility.record_shadow_view(frame, identity, camera, 60.f, error), error);
    command_read.offset = visibility.command_offset(1, 0);
    command_read.size = sizeof(VkDrawIndexedIndirectCommand) * command_count;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 1, &command_read, 0, nullptr);
    const VkBufferCopy local_shadow_copy{command_read.offset, 0, command_read.size};
    vkCmdCopyBuffer(command, visibility.command_buffer(frame.frame_index), readback.handle(),
        1, &local_shadow_copy);
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &host_read, 0, nullptr, 0, nullptr);
    require(vkEndCommandBuffer(command) == VK_SUCCESS, "local-shadow command-buffer end failed");
    VkFence local_shadow_fence{};
    require(vkCreateFence(fixture.device, &fence_info, nullptr, &local_shadow_fence) == VK_SUCCESS,
        "local-shadow fence creation failed");
    submit.pCommandBuffers = &command;
    const bool local_shadow_completed = vkQueueSubmit(queue, 1, &submit, local_shadow_fence) == VK_SUCCESS &&
        vkWaitForFences(fixture.device, 1, &local_shadow_fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    vkDestroyFence(fixture.device, local_shadow_fence, nullptr);
    require(local_shadow_completed, "Cluster DAG local-shadow dispatch did not complete");
    require(readback.read(0, commands.data(), sizeof(VkDrawIndexedIndirectCommand) * command_count, error), error);
    require(same_command(commands[0], 3, 1, 104, 3) &&
            same_command(commands[1], 0, 0, 120, 3) &&
            same_command(commands[2], 3, 1, 204, 7) &&
            same_command(commands[3], 0, 0, 220, 7),
        "local-shadow culling did not refresh both progressive-window command ranges");

    // A 65x65 source yields a 32x32 base. At mip 2 the first cluster spans
    // columns 3 through 5. A clear texel in column 4 must prevent occlusion.
    constexpr uint32_t hiz_levels = 4;
    require(hiz_pool.initialize(fixture.device, memory, {vkCreateImage, vkDestroyImage,
        vkGetImageMemoryRequirements, vkAllocateMemory, vkFreeMemory, vkBindImageMemory}),
        "Hi-Z image pool initialization failed");
    VkImageCreateInfo hiz_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    hiz_info.imageType = VK_IMAGE_TYPE_2D;
    hiz_info.format = VK_FORMAT_R32_SFLOAT;
    hiz_info.extent = {32, 32, 1};
    hiz_info.mipLevels = hiz_levels;
    hiz_info.arrayLayers = 1;
    hiz_info.samples = VK_SAMPLE_COUNT_1_BIT;
    hiz_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImage hiz_image{};
    require(hiz_pool.acquire(hiz_info, hiz_image, error), error);
    const VkImageSubresourceRange hiz_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, hiz_levels, 0, 1};
    VkImageViewCreateInfo hiz_view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    hiz_view_info.image = hiz_image;
    hiz_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    hiz_view_info.format = hiz_info.format;
    hiz_view_info.subresourceRange = hiz_range;
    require(vkCreateImageView(fixture.device, &hiz_view_info, nullptr, &hiz_sampling.view) == VK_SUCCESS,
        "Hi-Z image view creation failed");
    VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.maxLod = float(hiz_levels - 1);
    require(vkCreateSampler(fixture.device, &sampler_info, nullptr, &hiz_sampling.sampler) == VK_SUCCESS,
        "Hi-Z sampler creation failed");

    frame.extent = {65, 65};
    const FrameImageSharing hiz_sharing;
    for (const bool has_gap : {false, true})
    {
        require(vkResetCommandBuffer(command, 0) == VK_SUCCESS &&
                vkBeginCommandBuffer(command, &begin) == VK_SUCCESS,
            "Hi-Z command-buffer reset failed");
        VkImageMemoryBarrier hiz_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        hiz_barrier.srcAccessMask = has_gap ? VK_ACCESS_SHADER_READ_BIT : 0;
        hiz_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        hiz_barrier.oldLayout = has_gap ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
        hiz_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        hiz_barrier.srcQueueFamilyIndex = hiz_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hiz_barrier.image = hiz_image;
        hiz_barrier.subresourceRange = hiz_range;
        vkCmdPipelineBarrier(command, has_gap ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT :
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
            0, nullptr, 1, &hiz_barrier);
        if (has_gap)
        {
            const std::array<VkBufferImageCopy, 2> gaps{{
                {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 2, 0, 1}, {4, 3, 0}, {1, 1, 1}},
                {sizeof(float), 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 3, 0, 1}, {2, 1, 0}, {1, 1, 1}}}};
            vkCmdCopyBufferToImage(command, depth_upload.handle(), hiz_image, hiz_barrier.newLayout,
                static_cast<uint32_t>(gaps.size()), gaps.data());
        }
        else
        {
            const VkClearColorValue blocked{{.1f, 0.f, 0.f, 0.f}};
            vkCmdClearColorImage(command, hiz_image, hiz_barrier.newLayout, &blocked, 1, &hiz_range);
        }
        hiz_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        hiz_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        hiz_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        hiz_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &hiz_barrier);
        require(visibility.record(frame, identity, sun_mvp, camera, 60.f, 64, error,
            identity, hiz_image, hiz_sampling.view, hiz_sampling.sampler, &hiz_sharing, hiz_levels, true), error);
        command_read.offset = 0;
        command_read.size = command_bytes;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &command_read, 0, nullptr);
        vkCmdCopyBuffer(command, visibility.command_buffer(frame.frame_index), readback.handle(), 1, &command_copy);
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &host_read, 0, nullptr, 0, nullptr);
        require(vkEndCommandBuffer(command) == VK_SUCCESS &&
                vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS &&
                vkQueueWaitIdle(queue) == VK_SUCCESS, "Cluster DAG Hi-Z dispatch did not complete");
        require(readback.read(0, commands.data(), sizeof(commands), error), error);
        if (!same_command(commands[0], has_gap ? 3 : 0, has_gap ? 1 : 0, 104, 3))
        {
            std::cerr << (has_gap ? "Hi-Z skipped a clear intermediate texel"
                                  : "Hi-Z did not reject a fully occluded cluster") << '\n';
            return 1;
        }
    }
    vkDestroyImageView(fixture.device, hiz_sampling.view, nullptr);
    hiz_sampling.view = VK_NULL_HANDLE;
    vkFreeCommandBuffers(fixture.device, fixture.command_pool, 1, &command);
    std::cout << properties.deviceName << ": camera, sun, local-shadow and Hi-Z indirect culling passed\n";
    return 0;
}
