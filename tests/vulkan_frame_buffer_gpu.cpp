#include "src/Layers/xrRenderVK/FrameImageGraph.h"

#include <vulkan/vulkan.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace xray::render::vulkan;

namespace
{
void check(VkResult result)
{
    if (result != VK_SUCCESS)
        throw std::runtime_error("Vulkan buffer graph test failed: " + std::to_string(result));
}

struct Fixture
{
    VkInstance instance{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkBuffer source{}, destination{};
    VkDeviceMemory source_memory{}, destination_memory{};

    ~Fixture()
    {
        if (device)
        {
            vkDeviceWaitIdle(device);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            if (source) vkDestroyBuffer(device, source, nullptr);
            if (destination) vkDestroyBuffer(device, destination, nullptr);
            if (source_memory) vkFreeMemory(device, source_memory, nullptr);
            if (destination_memory) vkFreeMemory(device, destination_memory, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    void make_buffer(VkPhysicalDevice physical, VkBufferUsageFlags usage,
        VkBuffer& buffer, VkDeviceMemory& memory)
    {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = 64;
        info.usage = usage;
        check(vkCreateBuffer(device, &info, nullptr, &buffer));
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(physical, &properties);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (properties.memoryTypes[i].propertyFlags &
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            { type = i; break; }
        if (type == UINT32_MAX) throw std::runtime_error("coherent memory unavailable");
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        check(vkAllocateMemory(device, &allocation, nullptr, &memory));
        check(vkBindBufferMemory(device, buffer, memory, 0));
    }
};
}

int main(int argc, char** argv)
{
    Fixture vk;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    check(vkCreateInstance(&instance_info, nullptr, &vk.instance));
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(vk.instance, &count, nullptr));
    if (!count) return 77;
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(vk.instance, &count, devices.data()));
    const int selected = argc > 1 ? std::stoi(argv[1]) : 0;
    if (selected < 0 || uint32_t(selected) >= count) return 77;
    VkPhysicalDevice physical = devices[selected];
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; ++i)
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { family = i; break; }
    if (family == UINT32_MAX) return 77;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    check(vkCreateDevice(physical, &device_info, nullptr, &vk.device));
    vkGetDeviceQueue(vk.device, family, 0, &vk.queue);
    vk.make_buffer(physical, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        vk.source, vk.source_memory);
    vk.make_buffer(physical, VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk.destination, vk.destination_memory);
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = family;
    check(vkCreateCommandPool(vk.device, &pool_info, nullptr, &vk.pool));
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = vk.pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(vk.device, &command_info, &vk.command));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(vk.command, &begin));

    FrameImageGraph graph;
    FrameImageGraph::BufferResource source, destination, filled, copied;
    std::string error;
    if (!graph.import_buffer(vk.source, 0, 64, BufferUse::Undefined, source, error) ||
        !graph.import_buffer(vk.destination, 0, 64, BufferUse::Undefined, destination, error))
        throw std::runtime_error(error);
    const auto readback = graph.add_pass("host readback", {});
    const auto copy = graph.add_pass("copy", [&] {
        VkBufferCopy region{0, 0, 64};
        vkCmdCopyBuffer(vk.command, vk.source, vk.destination, 1, &region);
    });
    const auto fill = graph.add_pass("fill", [&] {
        vkCmdFillBuffer(vk.command, vk.source, 0, 64, 0x1234abcd);
    });
    if (!graph.write_buffer(fill, source, BufferUse::TransferDestination, filled, error) ||
        !graph.read_buffer(copy, filled, BufferUse::TransferSource, error) ||
        !graph.write_buffer(copy, destination, BufferUse::TransferDestination, copied, error) ||
        !graph.read_buffer(readback, copied, BufferUse::HostRead, error))
        throw std::runtime_error(error);
    ImageStateTracker empty_states;
    if (!graph.execute(vk.command, empty_states, {vkCmdPipelineBarrier}, error))
        throw std::runtime_error(error);
    check(vkEndCommandBuffer(vk.command));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &vk.command;
    check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE));
    check(vkQueueWaitIdle(vk.queue));
    void* mapped = nullptr;
    check(vkMapMemory(vk.device, vk.destination_memory, 0, 64, 0, &mapped));
    const auto* words = static_cast<const uint32_t*>(mapped);
    for (uint32_t i = 0; i < 16; ++i)
        if (words[i] != 0x1234abcd) throw std::runtime_error("buffer graph copied incorrect data");
    vkUnmapMemory(vk.device, vk.destination_memory);
}
