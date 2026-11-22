#include "src/Layers/xrRenderVK/TemporalResolve.h"
#include <vulkan/vulkan.h>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>

using namespace xray::render::vulkan;

namespace
{
void check(VkResult value)
{ if (value != VK_SUCCESS) throw std::runtime_error("temporal resolve Vulkan result " + std::to_string(value)); }

struct Fixture
{
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkPhysicalDeviceMemoryProperties memory{};
    std::array<VkImage, 4> input{};
    std::array<VkDeviceMemory, 4> input_memory{};
    std::array<VkImageView, 4> input_view{};
    VkBuffer upload{}, readback{};
    VkDeviceMemory upload_memory{}, readback_memory{};
    ~Fixture()
    {
        if (device)
        {
            vkDeviceWaitIdle(device);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            for (auto view : input_view) if (view) vkDestroyImageView(device, view, nullptr);
            for (auto image : input) if (image) vkDestroyImage(device, image, nullptr);
            for (auto allocation : input_memory) if (allocation) vkFreeMemory(device, allocation, nullptr);
            if (upload) vkDestroyBuffer(device, upload, nullptr);
            if (readback) vkDestroyBuffer(device, readback, nullptr);
            if (upload_memory) vkFreeMemory(device, upload_memory, nullptr);
            if (readback_memory) vkFreeMemory(device, readback_memory, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
    VkDeviceMemory allocate(VkMemoryRequirements req, VkMemoryPropertyFlags flags)
    {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((req.memoryTypeBits & (1u << i)) &&
                (memory.memoryTypes[i].propertyFlags & flags) == flags)
            {
                VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                info.allocationSize = req.size;
                info.memoryTypeIndex = i;
                VkDeviceMemory result{};
                check(vkAllocateMemory(device, &info, nullptr, &result));
                return result;
            }
        throw std::runtime_error("temporal resolve memory type unavailable");
    }
    void make_image(size_t index, VkImageUsageFlags usage,
        VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT,
        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT)
    {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {2, 2, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        check(vkCreateImage(device, &info, nullptr, &input[index]));
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, input[index], &req);
        input_memory[index] = allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkBindImageMemory(device, input[index], input_memory[index], 0));
        if (index == 3) return;
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = input[index];
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = info.format;
        view.subresourceRange = {aspect, 0, 1, 0, 1};
        check(vkCreateImageView(device, &view, nullptr, &input_view[index]));
    }
    void make_buffer(VkDeviceSize bytes, VkBufferUsageFlags usage,
        VkBuffer& result, VkDeviceMemory& allocation)
    {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = usage;
        check(vkCreateBuffer(device, &info, nullptr, &result));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, result, &req);
        allocation = allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkBindBufferMemory(device, result, allocation, 0));
    }
};

void transition(VkCommandBuffer command, VkImage image, VkImageLayout old_layout,
    VkImageLayout new_layout, VkAccessFlags source, VkAccessFlags destination,
    VkPipelineStageFlags before, VkPipelineStageFlags after,
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT)
{
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = source;
    barrier.dstAccessMask = destination;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, before, after, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

float half_to_float(uint16_t bits)
{
    const uint32_t sign = (bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 31u;
    const uint32_t mantissa = bits & 1023u;
    uint32_t result = sign;
    if (exponent == 0)
        return mantissa == 0 ? 0.f : std::ldexp(float(mantissa), -24);
    if (exponent == 31) result |= 0x7f800000u | (mantissa << 13);
    else result |= ((exponent + 112u) << 23) | (mantissa << 13);
    float value{};
    std::memcpy(&value, &result, sizeof(value));
    return value;
}
}

int main(int argc, char** argv) try
{
    Fixture vk;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    check(vkCreateInstance(&instance_info, nullptr, &vk.instance));
    uint32_t count{};
    check(vkEnumeratePhysicalDevices(vk.instance, &count, nullptr));
    if (!count) return 77;
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(vk.instance, &count, devices.data()));
    const int selected = argc > 1 ? std::stoi(argv[1]) : 0;
    if (selected < 0 || uint32_t(selected) >= count) return 77;
    vk.physical = devices[selected];
    VkFormatProperties support{};
    vkGetPhysicalDeviceFormatProperties(vk.physical, VK_FORMAT_R16G16B16A16_SFLOAT, &support);
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT;
    if ((support.optimalTilingFeatures & required) != required) return 77;
    VkFormatProperties depth_support{};
    vkGetPhysicalDeviceFormatProperties(vk.physical, VK_FORMAT_D32_SFLOAT, &depth_support);
    if (!(depth_support.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) return 77;
    VkFormatProperties presentation_support{};
    vkGetPhysicalDeviceFormatProperties(vk.physical, VK_FORMAT_B8G8R8A8_UNORM, &presentation_support);
    if (!(presentation_support.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) return 77;
    uint32_t family_count{};
    vkGetPhysicalDeviceQueueFamilyProperties(vk.physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(vk.physical, &family_count, families.data());
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; ++i)
        if ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { family = i; break; }
    if (family == UINT32_MAX) return 77;
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
    vk.make_image(0, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    vk.make_image(1, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    vk.make_image(2, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
    vk.make_image(3, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_FORMAT_B8G8R8A8_UNORM);
    vk.make_buffer(80, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk.upload, vk.upload_memory);
    constexpr VkDeviceSize blit_offset = 64;
    constexpr VkDeviceSize readback_size = blit_offset + 2 * 2 * 4;
    vk.make_buffer(readback_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk.readback, vk.readback_memory);
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(vk.device, &pool_info, nullptr, &vk.pool));
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = vk.pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(vk.device, &command_info, &vk.command));
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
    TemporalResolve temporal;
    std::string error;
    if (!temporal.initialize(vk.physical, vk.device, vkGetDeviceProcAddr, {2, 2},
            {vk.input_view[0]}, {vk.input_view[1]}, {vk.input_view[2]}, vk.memory, dispatch,
            VK_NULL_HANDLE, error, nullptr, true)) throw std::runtime_error(error);
    temporal.set_active(true);
    TemporalResolve indirect_temporal;
    if (!indirect_temporal.initialize(vk.physical, vk.device, vkGetDeviceProcAddr, {1, 1},
            {vk.input_view[0]}, {vk.input_view[1]}, {vk.input_view[2]}, vk.memory, dispatch,
            VK_NULL_HANDLE, error, nullptr, true, true)) throw std::runtime_error(error);
    indirect_temporal.set_active(true);
    TemporalResolve upscaled_temporal;
    if (!upscaled_temporal.initialize(vk.physical, vk.device, vkGetDeviceProcAddr, {4, 4},
            {vk.input_view[0]}, {vk.input_view[1]}, {vk.input_view[2]}, vk.memory, dispatch,
            VK_NULL_HANDLE, error, nullptr, true, false, true)) throw std::runtime_error(error);
    upscaled_temporal.set_active(true);
    const std::array<uint16_t, 16> motion{};
    for (uint32_t iteration = 0; iteration < 10; ++iteration)
    {
        if (iteration == 2)
        {
            temporal.reset_history();
            indirect_temporal.reset_history();
            upscaled_temporal.reset_history();
            if (indirect_temporal.history_valid()) throw std::runtime_error("indirect history reset failed");
        }
        check(vkResetCommandBuffer(vk.command, 0));
        void* mapped{};
        check(vkMapMemory(vk.device, vk.upload_memory, 0, 80, 0, &mapped));
        auto* words = static_cast<uint16_t*>(mapped);
        for (uint32_t pixel = 0; pixel < 4; ++pixel)
        {
            // Frame five changes a static pixel from black to red while its
            // neighbors still span the old and new colors. Neighborhood
            // clamping alone cannot reject the stale history there.
            const uint16_t red = iteration == 9 && pixel == 1 ? 0 : iteration == 0 ? 0x3400 :
                (iteration == 7 || (iteration == 6 && pixel == 2)) ? 0x3c00 :
                (iteration == 5 ? (pixel == 0 ? 0x3c00 : 0) : (pixel % 2 ? 0x3c00 : 0));
            words[pixel * 4] = red;
            words[pixel * 4 + 1] = words[pixel * 4 + 2] = 0;
            if (iteration == 0 && pixel == 3)
                words[pixel * 4 + 1] = 0x7c00; // An infinite channel in a zero-weight history tap.
            words[pixel * 4 + 3] = 0x3c00;
        }
        std::memcpy(words + 16, motion.data(), 32);
        if (iteration == 3) words[16] = 0x3800; // Previous pixel is one texel to the right.
        if (iteration == 6) words[16] = 0x3400; // Halfway across a depth discontinuity.
        if (iteration == 8) words[16] = 0x3400;
        for (uint32_t pixel = 0; pixel < 4; ++pixel)
            words[16 + pixel * 4 + 2] = iteration >= 4 && pixel == 0 ? 0x3266 : 0x3a66;
        if (iteration == 9)
        {
            // Match the LOD billboard velocity contract: keep its dynamic-depth
            // marker in alpha, but use invalid previous depth to reject history.
            words[16 + 1 * 4 + 1] = 0x3800; // Reproject pixel 1 to prior bright pixel 3.
            words[16 + 1 * 4 + 2] = 0xbc00; // -1 previous depth rejects that history.
        }
        const std::array<float, 4> depths{
            // Frame three reprojects to an old .8-depth pixel while the
            // current depth is .3. Comparing current depth would reject it.
            iteration >= 4 ? .2f : (iteration == 3 ? .3f : .8f),
            iteration == 3 ? .3f : .8f, iteration == 6 ? .2f : .8f, .8f};
        std::memcpy(words + 32, depths.data(), sizeof(depths));
        vkUnmapMemory(vk.device, vk.upload_memory);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(vk.command, &begin));
        transition(vk.command, vk.input[0], iteration ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL :
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            iteration ? VK_ACCESS_SHADER_READ_BIT : 0, VK_ACCESS_TRANSFER_WRITE_BIT,
            iteration ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {2, 2, 1};
        vkCmdCopyBufferToImage(vk.command, vk.upload, vk.input[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        transition(vk.command, vk.input[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        if (iteration == 0 || iteration == 3 || iteration == 4 || iteration == 6 || iteration == 7 || iteration == 8)
        {
            transition(vk.command, vk.input[1], iteration == 0 ? VK_IMAGE_LAYOUT_UNDEFINED :
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                iteration == 0 ? 0 : VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                iteration == 0 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);
            copy.bufferOffset = 32;
            vkCmdCopyBufferToImage(vk.command, vk.upload, vk.input[1],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            transition(vk.command, vk.input[1], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        if (iteration == 0 || iteration == 3 || iteration == 4 || iteration == 6 || iteration == 7)
        {
            transition(vk.command, vk.input[2], iteration == 0 ? VK_IMAGE_LAYOUT_UNDEFINED :
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                iteration == 0 ? 0 : VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                iteration == 0 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
            copy.bufferOffset = 64;
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            vkCmdCopyBufferToImage(vk.command, vk.upload, vk.input[2],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            transition(vk.command, vk.input[2], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
        }
        const uint32_t output = temporal.output_slot();
        transition(vk.command, temporal.image(output ^ 1), iteration == 0 ?
            VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, iteration == 0 ? 0 : VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_SHADER_READ_BIT, iteration == 0 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT :
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        const VkImageLayout previous_output_layout = iteration == 0 ? VK_IMAGE_LAYOUT_UNDEFINED :
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        const VkAccessFlags previous_output_access = iteration == 0 ? 0 : VK_ACCESS_SHADER_READ_BIT;
        const VkPipelineStageFlags previous_output_stage = iteration == 0 ?
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        transition(vk.command, temporal.image(output), previous_output_layout, VK_IMAGE_LAYOUT_GENERAL,
            previous_output_access, VK_ACCESS_SHADER_WRITE_BIT,
            previous_output_stage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        FrameRecordingContext frame{};
        frame.command_buffer = vk.command;
        frame.extent = {2, 2};
        if (!temporal.record(frame)) throw std::runtime_error("temporal resolve record failed");
        transition(vk.command, temporal.image(output), VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        copy.bufferOffset = 0;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vkCmdCopyImageToBuffer(vk.command, temporal.image(output), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk.readback, 1, &copy);
        if (iteration == 3)
        {
            transition(vk.command, vk.input[3], VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkImageBlit region{};
            region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.srcOffsets[1] = region.dstOffsets[1] = {2, 2, 1};
            vkCmdBlitImage(vk.command, temporal.image(output), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                vk.input[3], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);
            transition(vk.command, vk.input[3], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            copy.bufferOffset = blit_offset;
            vkCmdCopyImageToBuffer(vk.command, vk.input[3], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                vk.readback, 1, &copy);
        }
        const uint32_t indirect_output = indirect_temporal.output_slot();
        FrameRecordingContext indirect_frame = frame;
        indirect_frame.extent = {1, 1};
        if (!indirect_temporal.record_standalone(indirect_frame))
            throw std::runtime_error("standalone indirect temporal record failed");
        transition(vk.command, indirect_temporal.image(indirect_output), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        copy.bufferOffset = 40;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageExtent = {1, 1, 1};
        vkCmdCopyImageToBuffer(vk.command, indirect_temporal.image(indirect_output), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk.readback, 1, &copy);
        transition(vk.command, indirect_temporal.image(indirect_output), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        const uint32_t upscale_output = upscaled_temporal.output_slot();
        upscaled_temporal.set_jitter(iteration == 2 ? .125f : 0.f, 0.f);
        FrameRecordingContext upscale_frame = frame; upscale_frame.extent = {4, 4};
        if (!upscaled_temporal.record_standalone(upscale_frame))
            throw std::runtime_error("upscaled temporal resolve record failed");
        transition(vk.command, upscaled_temporal.image(upscale_output), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        copy.bufferOffset = 48; copy.imageOffset = {1, 0, 0}; copy.imageExtent = {1, 1, 1};
        vkCmdCopyImageToBuffer(vk.command, upscaled_temporal.image(upscale_output), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk.readback, 1, &copy);
        copy.bufferOffset = 56; copy.imageOffset = {0, 0, 0};
        vkCmdCopyImageToBuffer(vk.command, upscaled_temporal.image(upscale_output), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk.readback, 1, &copy);
        transition(vk.command, upscaled_temporal.image(upscale_output), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        check(vkEndCommandBuffer(vk.command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &vk.command;
        check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(vk.queue));
        check(vkMapMemory(vk.device, vk.readback_memory, 0, readback_size, 0, &mapped));
        const auto* values = static_cast<const uint16_t*>(mapped);
        if (std::abs(half_to_float(values[23]) - .8f) > .01f)
            throw std::runtime_error("indirect temporal did not sample scaled depth");
        if (iteration == 1 && std::abs(half_to_float(values[20]) - .09f) > .01f)
            throw std::runtime_error("indirect temporal discarded valid history in a dark neighborhood");
        if (iteration == 2 && std::abs(half_to_float(values[20])) > .01f)
            throw std::runtime_error("indirect temporal reused history after reset");
        if (iteration == 2 && std::abs(half_to_float(values[24]) - .5f) > .01f)
            throw std::runtime_error("temporal upscale did not compensate input jitter");
        if (iteration == 1 && std::abs(half_to_float(values[28]) - .225f) > .01f)
            throw std::runtime_error("temporal upscale did not accumulate full-resolution history");
        if (iteration == 2 && std::abs(half_to_float(values[28])) > .01f)
            throw std::runtime_error("temporal upscale reused history after reset");
        if (iteration == 8 && std::abs(half_to_float(values[24])) > .01f)
            throw std::runtime_error("temporal upscale mixed foreground and background depth layers");
        if (iteration == 9 && std::abs(half_to_float(values[4])) > .01f)
            throw std::runtime_error("temporal resolve reused history for a changing LOD billboard facet");
        const float red = half_to_float(values[0]);
        if (iteration == 1 && half_to_float(values[1]) != 0.f)
            throw std::runtime_error("non-finite temporal history contaminated a neighboring pixel");
        // At the depth edge only one of two equal-weight history taps survives.
        const float expected = iteration == 0 ? .25f : iteration == 1 ? .225f :
            iteration == 3 ? .9f : iteration == 6 ? 1.f / 3.f :
            iteration == 5 || iteration == 7 ? 1.f : 0.f;
        if (std::abs(red - expected) > .01f)
            throw std::runtime_error("temporal resolve mismatch at frame " + std::to_string(iteration) +
                ": " + std::to_string(red));
        if (iteration == 4 && std::abs(half_to_float(values[3]) - .2f) > .01f)
            throw std::runtime_error("temporal resolve did not store new depth history");
        if (iteration == 3)
        {
            const auto* bytes = static_cast<const uint8_t*>(mapped);
            if (std::abs(int(bytes[blit_offset + 2]) - 230) > 4)
                throw std::runtime_error("temporal resolve RGBA16F-to-BGRA8 blit mismatch");
        }
        vkUnmapMemory(vk.device, vk.readback_memory);
        temporal.commit_frame();
        indirect_temporal.commit_standalone();
        upscaled_temporal.commit_standalone();
    }
    temporal.destroy();
    indirect_temporal.destroy();
    upscaled_temporal.destroy();
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
