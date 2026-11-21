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
    for (Entry& entry : entries_)
        if (!entry.in_use && entry.descriptor == descriptor)
        {
            entry.in_use = true;
            entry.last_used = frame_;
            buffer = entry.resource.get();
            error.clear();
            return true;
        }

    auto resource = std::make_unique<BufferResource>();
    if (!resource->initialize(device_, size, usage, required_memory_properties,
            memory_, dispatch_, error, pooled_memory, shared_queue_flags))
        return false;
    buffer = resource.get();
    entries_.push_back({descriptor, std::move(resource), frame_, true});
    error.clear();
    return true;
}

bool FrameBufferPool::release(BufferResource* buffer)
{
    if (!buffer)
        return false;
    for (Entry& entry : entries_)
        if (entry.resource.get() == buffer && entry.in_use)
        {
            entry.in_use = false;
            entry.last_used = frame_;
            return true;
        }
    return false;
}

void FrameBufferPool::tick()
{
    ++frame_;
    for (size_t i = 0; i < entries_.size();)
    {
        const Entry& entry = entries_[i];
        if (!entry.in_use && frame_ - entry.last_used >= EvictAfterFrames)
        {
            entries_[i] = std::move(entries_.back());
            entries_.pop_back();
        }
        else
            ++i;
    }
}

void FrameBufferPool::reset()
{
    entries_.clear();
    device_ = VK_NULL_HANDLE;
    memory_ = {};
    dispatch_ = {};
    frame_ = 0;
}
}
