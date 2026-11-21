#pragma once

#include "BufferResource.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
// Exact-match pool for frame-graph buffer resources. Call release only after
// the last GPU use completes; tick evicts released buffers after four frames.
class FrameBufferPool
{
public:
    FrameBufferPool() = default;
    ~FrameBufferPool() { reset(); }
    FrameBufferPool(const FrameBufferPool&) = delete;
    FrameBufferPool& operator=(const FrameBufferPool&) = delete;

    bool initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
        const BufferResourceDispatch& dispatch);
    bool initialized() const { return device_ != VK_NULL_HANDLE; }
    bool acquire(VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags required_memory_properties, bool pooled_memory,
        BufferResource*& buffer, std::string& error,
        VkQueueFlags shared_queue_flags = 0);
    bool release(BufferResource* buffer);
    void tick();
    void reset();

private:
    struct Descriptor
    {
        VkDeviceSize size{};
        VkBufferUsageFlags usage{};
        VkMemoryPropertyFlags memory_properties{};
        bool pooled_memory{};
        VkQueueFlags shared_queue_flags{};
        bool operator==(const Descriptor& other) const
        {
            return size == other.size && usage == other.usage &&
                memory_properties == other.memory_properties && pooled_memory == other.pooled_memory &&
                shared_queue_flags == other.shared_queue_flags;
        }
    };
    struct Entry
    {
        Descriptor descriptor{};
        std::unique_ptr<BufferResource> resource;
        uint64_t last_used{};
        bool in_use{};
    };

    VkDevice device_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    BufferResourceDispatch dispatch_{};
    std::vector<Entry> entries_;
    uint64_t frame_{};
    static constexpr uint64_t EvictAfterFrames = 4;
};
}
