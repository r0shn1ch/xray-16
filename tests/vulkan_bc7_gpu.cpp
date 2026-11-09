// Compare the production BC7 fallback with Vulkan's independent BC7 decoder.
// No shader compiler or pre-generated reference images are needed: native
// compressed and software-expanded textures are blitted into RGBA8 readback.
#include "../src/Layers/xrRenderVK/Bc7Decoder.h"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
void check(VkResult result)
{
    if (result != VK_SUCCESS)
        throw std::runtime_error("Vulkan result " + std::to_string(result));
}

struct Fixture
{
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool pool{};
    VkPhysicalDeviceMemoryProperties memory{};
    std::vector<VkImage> images;
    std::vector<VkBuffer> buffers;
    std::vector<VkDeviceMemory> allocations;

    ~Fixture()
    {
        if (device)
        {
            vkDeviceWaitIdle(device);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            for (auto image : images) vkDestroyImage(device, image, nullptr);
            for (auto buffer : buffers) vkDestroyBuffer(device, buffer, nullptr);
            for (auto allocation : allocations) vkFreeMemory(device, allocation, nullptr);
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
                allocations.push_back(result);
                return result;
            }
        throw std::runtime_error("BC7 fixture has no suitable memory type");
    }

    VkBuffer buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkDeviceMemory& allocation)
    {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        VkBuffer result{};
        check(vkCreateBuffer(device, &info, nullptr, &result));
        buffers.push_back(result);
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, result, &requirements);
        allocation = allocate(requirements,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkBindBufferMemory(device, result, allocation, 0));
        return result;
    }

    VkImage image(VkFormat format, uint32_t width, uint32_t height)
    {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {width, height, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VkImage result{};
        check(vkCreateImage(device, &info, nullptr, &result));
        images.push_back(result);
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, result, &requirements);
        check(vkBindImageMemory(device, result,
            allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0));
        return result;
    }
};

void transition(VkCommandBuffer command, VkImage image, VkImageLayout from, VkImageLayout to,
    VkAccessFlags before, VkAccessFlags after)
{
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = before;
    barrier.dstAccessMask = after;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, from == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT :
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        0, nullptr, 0, nullptr, 1, &barrier);
}

constexpr unsigned Columns = 16, Blocks = 8 * 64, Width = Columns * 4, Height = Blocks / Columns * 4;
constexpr size_t CompressedBytes = Blocks * 16, RgbaBytes = Width * Height * 4;

void bits(uint8_t* block, unsigned offset, unsigned count, unsigned value)
{
    for (unsigned bit = 0; bit < count; ++bit, ++offset)
        block[offset / 8] = uint8_t((block[offset / 8] & ~(1u << (offset % 8))) |
            (((value >> bit) & 1u) << (offset % 8)));
}

void run(Fixture& vk, bool srgb, const std::vector<uint8_t>& compressed, const std::vector<uint8_t>& decoded)
{
    const VkImage native = vk.image(srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK, Width, Height);
    const VkImage fallback = vk.image(srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM, Width, Height);
    const std::array<VkImage, 2> output{vk.image(VK_FORMAT_R8G8B8A8_UNORM, Width, Height),
        vk.image(VK_FORMAT_R8G8B8A8_UNORM, Width, Height)};
    VkDeviceMemory upload_memory{}, readback_memory{};
    const auto upload = vk.buffer(CompressedBytes + RgbaBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, upload_memory);
    const auto readback = vk.buffer(RgbaBytes * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT, readback_memory);
    void* data{};
    check(vkMapMemory(vk.device, upload_memory, 0, VK_WHOLE_SIZE, 0, &data));
    std::memcpy(data, compressed.data(), compressed.size());
    std::memcpy(static_cast<uint8_t*>(data) + CompressedBytes, decoded.data(), decoded.size());
    vkUnmapMemory(vk.device, upload_memory);

    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = vk.pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer command{};
    check(vkAllocateCommandBuffers(vk.device, &allocation, &command));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin));
    for (auto image : {native, fallback, output[0], output[1]})
        transition(command, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {Width, Height, 1};
    vkCmdCopyBufferToImage(command, upload, native, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    copy.bufferOffset = CompressedBytes;
    vkCmdCopyBufferToImage(command, upload, fallback, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    for (auto image : {native, fallback})
        transition(command, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkImageBlit region{};
    region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffsets[1] = region.dstOffsets[1] = {int32_t(Width), int32_t(Height), 1};
    for (unsigned i = 0; i < 2; ++i)
    {
        vkCmdBlitImage(command, i ? fallback : native, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            output[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);
        transition(command, output[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        copy.bufferOffset = i * RgbaBytes;
        vkCmdCopyImageToBuffer(command, output[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &copy);
    }
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &host, 0, nullptr, 0, nullptr);
    check(vkEndCommandBuffer(command));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE));
    check(vkQueueWaitIdle(vk.queue));
    check(vkMapMemory(vk.device, readback_memory, 0, VK_WHOLE_SIZE, 0, &data));
    const auto* pixels = static_cast<const uint8_t*>(data);
    unsigned maximum = 0;
    for (size_t i = 0; i < RgbaBytes; ++i)
    {
        const int expected = [&] {
            const double encoded = decoded[i] / 255.0;
            return !srgb || i % 4 == 3 ? int(decoded[i]) : int(std::lround(255 *
                (encoded <= .04045 ? encoded / 12.92 : std::pow((encoded + .055) / 1.055, 2.4))));
        }();
        const unsigned difference = unsigned(std::abs(int(pixels[i]) - int(pixels[RgbaBytes + i])));
        maximum = std::max(maximum, difference);
        if (difference > 1 || std::abs(int(pixels[i]) - expected) > 1)
        {
            const size_t texel = i / 4, block = (texel / Width / 4) * Columns + texel % Width / 4;
            std::cerr << "BC7 " << (srgb ? "sRGB" : "UNORM") << " mode=" << block / 64
                      << " vector=" << block % 64 << " texel=" << texel << " channel=" << i % 4
                      << " native=" << int(pixels[i]) << " fallback=" << int(pixels[RgbaBytes + i])
                      << " expected=" << expected << '\n';
            vkUnmapMemory(vk.device, readback_memory);
            throw std::runtime_error("BC7 software fallback disagrees with native Vulkan decoding");
        }
    }
    vkUnmapMemory(vk.device, readback_memory);
    vkFreeCommandBuffers(vk.device, vk.pool, 1, &command);
    std::cout << (srgb ? "sRGB" : "UNORM") << ": all 8 modes / 512 blocks, max byte difference=" << maximum << '\n';
}
}

int main()
{
    try
    {
        Fixture vk;
        VkInstanceCreateInfo instance{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        check(vkCreateInstance(&instance, nullptr, &vk.instance));
        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(vk.instance, &count, nullptr));
        std::vector<VkPhysicalDevice> devices(count);
        if (count) check(vkEnumeratePhysicalDevices(vk.instance, &count, devices.data()));
        for (auto device : devices)
        {
            VkPhysicalDeviceFeatures features{};
            vkGetPhysicalDeviceFeatures(device, &features);
            bool supported = features.textureCompressionBC;
            for (auto format : {VK_FORMAT_BC7_UNORM_BLOCK, VK_FORMAT_BC7_SRGB_BLOCK, VK_FORMAT_R8G8B8A8_SRGB})
            {
                VkFormatProperties properties{};
                vkGetPhysicalDeviceFormatProperties(device, format, &properties);
                supported &= (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0;
            }
            VkFormatProperties output{};
            vkGetPhysicalDeviceFormatProperties(device, VK_FORMAT_R8G8B8A8_UNORM, &output);
            supported &= (output.optimalTilingFeatures & (VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT)) ==
                (VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT);
            if (supported) { vk.physical = device; break; }
        }
        if (!vk.physical)
        {
            std::cout << "SKIP: no Vulkan device with native BC7 and required blit formats\n";
            return 77;
        }
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(vk.physical, &properties);
        std::cout << "BC7 oracle device: " << properties.deviceName << '\n';
        vkGetPhysicalDeviceMemoryProperties(vk.physical, &vk.memory);
        uint32_t count_families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(vk.physical, &count_families, nullptr);
        std::vector<VkQueueFamilyProperties> families(count_families);
        vkGetPhysicalDeviceQueueFamilyProperties(vk.physical, &count_families, families.data());
        uint32_t family = 0;
        while (family < count_families && !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++family;
        if (family == count_families) throw std::runtime_error("no graphics queue for BC7 blits");
        const float priority = 1.f;
        VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue.queueFamilyIndex = family;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        VkPhysicalDeviceFeatures enabled{};
        enabled.textureCompressionBC = VK_TRUE;
        VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device.queueCreateInfoCount = 1;
        device.pQueueCreateInfos = &queue;
        device.pEnabledFeatures = &enabled;
        check(vkCreateDevice(vk.physical, &device, nullptr, &vk.device));
        vkGetDeviceQueue(vk.device, family, 0, &vk.queue);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = family;
        check(vkCreateCommandPool(vk.device, &pool, nullptr, &vk.pool));

        std::vector<uint8_t> compressed(CompressedBytes), decoded(RgbaBytes);
        uint32_t random = 0x94d03b71u;
        for (unsigned block = 0; block < Blocks; ++block)
        {
            auto* bytes = compressed.data() + block * 16;
            for (unsigned i = 0; i < 16; ++i)
            {
                random = random * 1664525u + 1013904223u;
                bytes[i] = uint8_t(random >> 24);
            }
            const unsigned mode = block / 64, vector = block % 64;
            bits(bytes, 0, mode + 1, 1u << mode);
            if (mode == 0) bits(bytes, mode + 1, 4, vector % 16);
            else if (mode == 1 || mode == 2 || mode == 3 || mode == 7) bits(bytes, mode + 1, 6, vector);
            else if (mode == 4) bits(bytes, mode + 1, 3, vector % 8); // Rotation and index selector.
            else if (mode == 5) bits(bytes, mode + 1, 2, vector % 4); // Rotation.
            xray::render::vulkan::decode_bc7_block(bytes,
                decoded.data() + ((block / Columns * 4) * Width + block % Columns * 4) * 4, Width * 4);
        }
        run(vk, false, compressed, decoded);
        run(vk, true, compressed, decoded);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
