#include "FrameImageGraph.h"
#include "FrameBufferPool.h"

namespace xray::render::vulkan
{
bool FrameImageGraph::materialize_buffers(FrameBufferPool& pool, std::string& error)
{
    return materialize_buffers([&pool](const BufferDescription& description,
        TransientBuffer& output, std::string& failure) {
        vulkan::BufferResource* buffer = nullptr;
        if (!pool.acquire(description.size, description.usage, description.memory_properties,
                description.pooled_memory, buffer, failure, description.shared_queue_flags))
            return false;
        output.owner = std::shared_ptr<void>(buffer, [&pool](void* resource) {
            pool.release(static_cast<vulkan::BufferResource*>(resource));
        });
        output.handle = buffer->handle();
        output.sharing_mode = buffer->sharing_mode();
        output.queue_families = buffer->sharing_queue_families();
        return true;
    }, error);
}
}
