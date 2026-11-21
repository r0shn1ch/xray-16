#include "xrEngine/stdafx.h"
#include "FrameBufferPool.h"

namespace xray::render::vulkan
{
bool FrameBufferPool::initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferResourceDispatch& dispatch)
{
    reset();
    if (!device || !dispatch.create_buffer || !dispatch.destroy_buffer ||
        !dispatch.get_buffer_memory_requirements || !dispatch.allocate_memory ||
        !dispatch.free_memory || !dispatch.bind_buffer_memory || !dispatch.map_memory ||
        !dispatch.unmap_memory)
        return false;
    device_ = device;
    memory_ = memory;
    dispatch_ = dispatch;
    return true;
}

bool FrameBufferPool::acquire(VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags required_memory_properties, bool pooled_memory,
    BufferResource*& buffer, std::string& error, VkQueueFlags shared_queue_flags)
{
    buffer = nullptr;
    if (!device_ || !size || !usage)
    {
        error = "frame buffer pool needs an initialized device, size and usage";
        return false;
    }
    const Descriptor descriptor{size, usage, required_memory_properties, pooled_memory,
        shared_queue_flags};
    buffer = resources_.acquire(descriptor, [&]() -> std::unique_ptr<BufferResource> {
        auto resource = std::make_unique<BufferResource>();
        if (!resource->initialize(device_, size, usage, required_memory_properties,
                memory_, dispatch_, error, pooled_memory, shared_queue_flags))
            return {};
        return resource;
    });
    if (!buffer)
        return false;
    error.clear();
    return true;
}

bool FrameBufferPool::release(BufferResource* buffer)
{
    return resources_.release(buffer);
}

void FrameBufferPool::tick()
{
    resources_.tick();
}

void FrameBufferPool::reset()
{
    resources_.reset();
    device_ = VK_NULL_HANDLE;
    memory_ = {};
    dispatch_ = {};
}
}
