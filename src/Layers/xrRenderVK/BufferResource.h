#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nvrhi
{
class IBuffer;
class IDevice;
}

namespace xray::render::vulkan
{
struct BufferMemoryBlock;
struct BufferResourceDispatch
{
    PFN_vkCreateBuffer create_buffer{};
    PFN_vkDestroyBuffer destroy_buffer{};
    PFN_vkGetBufferMemoryRequirements get_buffer_memory_requirements{};
    PFN_vkAllocateMemory allocate_memory{};
    PFN_vkFreeMemory free_memory{};
    PFN_vkBindBufferMemory bind_buffer_memory{};
    PFN_vkMapMemory map_memory{};
    PFN_vkUnmapMemory unmap_memory{};
    nvrhi::IDevice* nvrhi_device{};
    uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;
    uint32_t compute_queue_family = VK_QUEUE_FAMILY_IGNORED;
    uint32_t transfer_queue_family = VK_QUEUE_FAMILY_IGNORED;
};

// A small mapped-buffer owner for host-visible coherent memory and an
// allocation owner for device-local buffers. Callers synchronize GPU use
// before modifying bytes that may still be read by an in-flight submission.
class BufferResource
{
public:
    BufferResource() = default;
    ~BufferResource();
    BufferResource(const BufferResource&) = delete;
    BufferResource& operator=(const BufferResource&) = delete;
    BufferResource(BufferResource&& other) noexcept;
    BufferResource& operator=(BufferResource&& other) noexcept;

    bool initialize(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags required_memory_properties,
        const VkPhysicalDeviceMemoryProperties& memory_properties,
        const BufferResourceDispatch& dispatch, std::string& error, bool pooled = false,
        VkQueueFlags shared_queue_flags = 0, VkResult* allocation_result = nullptr);
    bool write(VkDeviceSize offset, const void* data, size_t size, std::string& error);
    // Caller waits for the GPU write before reading coherent host memory.
    bool read(VkDeviceSize offset, void* data, size_t size, std::string& error) const;
    void destroy();

    VkBuffer handle() const { return m_buffer; }
    VkDeviceSize size() const { return m_size; }
    VkSharingMode sharing_mode() const { return m_sharing_mode; }
    const std::vector<uint32_t>& sharing_queue_families() const { return m_sharing_queue_families; }
    bool host_visible() const { return m_mapped != nullptr; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkSharingMode m_sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
    std::vector<uint32_t> m_sharing_queue_families;
    nvrhi::IBuffer* m_nvrhi_buffer{};
    VkDeviceSize m_size = 0;
    void* m_mapped = nullptr;
    std::shared_ptr<BufferMemoryBlock> m_pool_block;
    VkDeviceSize m_pool_offset{};
    BufferResourceDispatch m_vk{};
};
}
